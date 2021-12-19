'''
brief: Sobolev wrapper model
Author: Steffen Schotthöfer
Version: 0.0
Date 05.08.2021
'''
from abc import ABC

import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from src import math


class EntropyModel(tf.keras.Model, ABC):
    """
    model that wraps entropy tools around a given core model to reconstruct moments, scale lagrange multipliers etc
    """
    # Sobolev implies, that the model outputs also its derivative
    core_model: tf.keras.Model
    enable_recons_u: bool
    poly_degree: int
    nq: int
    quad_pts: Tensor
    quad_weights: Tensor
    input_dim: int
    moment_basis: Tensor
    derivative_scaler_min: Tensor  # for scaled input data, we need to rescale the derivative to correclty reconstruct u
    derivative_scaler_max: Tensor
    derivative_scale_factor: Tensor
    regularization_gamma: Tensor  # @brief: Regularization Parameter for regularized entropy. =0 means non regularized
    regularization_gamma_vector: Tensor  # @brief: tensor of the form [0,gamma,gamma,...]
    input_dim: int  # @brief size of moment basis

    def __init__(self, core_model: tf.keras.Model, polynomial_degree: int = 1, spatial_dimension: int = 1,
                 reconstruct_u: bool = False, scaler_min: float = 0.0, scaler_max: float = 1.0,
                 scale_active: bool = True, subclass=False, gamma: float = 0.0, **opts):
        super(EntropyModel, self).__init__()
        # Member is only the model we want to wrap with sobolev execution
        self.core_model = core_model  # must be a compiled tensorflow model
        self.enable_recons_u = reconstruct_u
        # Create quadrature and momentBasis. Currently only for 1D problems
        self.poly_degree = polynomial_degree
        self.derivative_scaler_min = tf.constant(scaler_min, dtype=tf.float64)
        self.derivative_scaler_max = tf.constant(scaler_max, dtype=tf.float64)
        self.scale_active = scale_active
        self.derivative_scale_factor = tf.constant((scaler_max - scaler_min) * 0.5, dtype=tf.float64)
        self.regularization_gamma = tf.constant(gamma, dtype=tf.float64)

        if not subclass:
            print("Model output alpha will be scaled by factor " + str(self.derivative_scale_factor.numpy()))
        if spatial_dimension == 1:
            [quad_pts, quad_weights] = math.qGaussLegendre1D(20 * polynomial_degree)  # dims = nq
            m_basis = math.computeMonomialBasis1D(quad_pts, self.poly_degree)  # dims = (N x nq)
            self.nq = quad_weights.size  # = 20 * polyDegree
        elif spatial_dimension == 2:
            [quad_pts, quad_weights] = math.qGaussLegendre2D(20 * polynomial_degree)  # dims = nq
            self.nq = quad_weights.size  # is not 20 * polyDegree
            m_basis = math.computeMonomialBasis2D(quad_pts, self.poly_degree)  # dims = (N x nq)
        else:
            print("spatial dimension not yet supported for sobolev wrapper")
            exit()

        self.quad_pts = tf.constant(quad_pts, shape=(self.nq, spatial_dimension), dtype=tf.float64)  # dims = (ds x nq)
        self.quad_weights = tf.constant(quad_weights, shape=(1, self.nq), dtype=tf.float64)  # dims=(batchSIze x N x nq)
        self.input_dim = m_basis.shape[0]
        self.moment_basis = tf.constant(m_basis, shape=(self.input_dim, self.nq),
                                        dtype=tf.float64)  # dims=(batchSIze x N x nq)
        gamma_vec = gamma * np.ones(shape=(1, self.input_dim))
        gamma_vec[0, 0] = 0.0
        self.regularization_gamma_vector = tf.constant(gamma_vec, dtype=tf.float64, shape=(1, self.input_dim))

    def call(self, x: Tensor, training=False) -> list:
        """
        Defines the sobolev execution (does not return 0th order moment)
        input: x = [u_1,u_2,...,u_N]
        output: alpha = [alpha_1,...,alpha_N]
                u = [u_1,u_2,...,u_N]
        """

        alpha = self.core_model(x)
        if self.enable_recons_u:
            if self.scale_active:
                print("Scaled reconstruction of u and h enabled")
                # scale to [scaler_min, scaler_max]
                t1 = tf.add(tf.cast(alpha, dtype=tf.float64, name=None), 1)  # shift
                t2 = tf.math.scalar_mul(self.derivative_scale_factor, t1)  # scale
                alpha64 = tf.add(t2, self.derivative_scaler_min)  # shift
            else:
                print("Reconstruction of u and h enabled")
                alpha64 = tf.cast(alpha, dtype=tf.float64, name=None)
            alpha_complete = self.reconstruct_alpha(alpha64)
            u_complete = self.reconstruct_u(alpha_complete)
            u_res = u_complete[:, 1:]  # cutoff the 0th order moment, since it is 1 by construction

            # compute entropy functional h
            h_res = self.compute_h(u=u_complete, alpha=alpha_complete)
        else:
            print("Reconstruction of u and h disabled. Output 3 and 4 are meaningless")
            u_res = alpha
            h_res = alpha

        return [alpha, alpha, u_res, h_res]

    def call_scaled(self, u_non_normal: Tensor):
        """
        brief: neural network call, with non-normalized input.
        input: u_non_normal: tensor with non_normalized moments
        """
        u_0 = u_non_normal[:, 0]
        u_downscaled = self.scale_u(u_non_normal, tf.math.reciprocal(u_non_normal[:, 0]))  # downscaling
        u_reduced = u_downscaled[:, 1:]
        alpha = self.core_model(u_reduced)
        if self.scale_active:
            print("Scaled reconstruction of u and h enabled")
            # scale to [scaler_min, scaler_max]
            t1 = tf.add(tf.cast(alpha, dtype=tf.float64, name=None), 1)  # shift
            t2 = tf.math.scalar_mul(self.derivative_scale_factor, t1)  # scale
            alpha64 = tf.add(t2, self.derivative_scaler_min)  # shift
        else:
            print("Reconstruction of u and h enabled")
            alpha64 = tf.cast(alpha, dtype=tf.float64, name=None)
        alpha_complete = self.reconstruct_alpha(alpha64)
        u_complete = self.reconstruct_u(alpha_complete)
        u_rescaled = self.scale_u(u_complete, u_0)  # upscaling
        alpha_rescaled = self.scale_alpha(alpha_complete, u_0)
        # u_rescaled_recons = self.reconstruct_u(alpha_rescaled)
        h = self.compute_h(u_rescaled, alpha_rescaled)
        return [u_rescaled, alpha_rescaled, h]

    def reconstruct_alpha(self, alpha):
        """
        brief:  Reconstructs alpha_0 and then concats alpha_0 to alpha_1,... , from alpha1,...
                Only works for maxwell Boltzmann entropy so far.
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N-1)
               m    , dims = (N x nq)
               w    , dims = nq
        returns alpha_complete = [alpha_0,alpha], dim = (nS x N), where alpha_0 = - ln(<exp(alpha*m)>)
        """
        # Check the predicted alphas for +/- infinity or nan - raise error if found
        checked_alpha = tf.debugging.check_numerics(alpha,
                                                    message='input tensor checking error at alpha = ' + str(alpha),
                                                    name='checked')
        # Clip the predicted alphas below the tf.exp overflow threshold
        clipped_alpha = tf.clip_by_value(checked_alpha, clip_value_min=-50, clip_value_max=50, name='checkedandclipped')

        tmp = tf.math.exp(tf.tensordot(clipped_alpha, self.moment_basis[1:, :], axes=([1], [0])))  # tmp = alpha * m
        alpha_0 = -tf.math.log(tf.tensordot(tmp, self.quad_weights, axes=([1], [1])))  # ln(<tmp>)
        return tf.concat([alpha_0, alpha], axis=1)  # concat [alpha_0,alpha]

    def reconstruct_u(self, alpha):
        """
        brief: reconstructs u from alpha with regularization in mind
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N)
               m    , dims = (N x nq)
               w    , dims = nq
        returns u = <m*eta_*'(alpha*m)>, dim = (nS x N)
        """
        # Check the predicted alphas for +/- infinity or nan - raise error if found
        checked_alpha = tf.debugging.check_numerics(alpha, message='input tensor checking error', name='checked')
        # Clip the predicted alphas below the tf.exp overflow threshold
        clipped_alpha = tf.clip_by_value(checked_alpha, clip_value_min=-50, clip_value_max=50, name='checkedandclipped')

        # Currently only for maxwell Boltzmann entropy
        f_quad = tf.math.exp(tf.tensordot(clipped_alpha, self.moment_basis, axes=([1], [0])))  # exp(alpha*m)
        tmp = tf.math.multiply(f_quad, self.quad_weights)  # f*w
        u_rec = tf.tensordot(tmp, self.moment_basis[:, :], axes=([1], [1]))  # f * w * momentBasis
        alpha_regularization = tf.math.multiply(self.regularization_gamma_vector, alpha)
        return u_rec + alpha_regularization  # add regularization

    @staticmethod
    def scale_alpha(alpha, u_0):
        """
        Performs the up-scaling computation, s.t. alpha corresponds to a moment with u0 = u_0, instead of u0=1
        input: alpha = [alpha_0,...,alpha_N],  batch of lagrange multipliers of the zeroth moment, dim = (nsx(N+1)))
               u_0 = batch of zeroth moments, dim = (nsx1)
        output: alpha_scaled = alpha + [ln(u_0),0,0,...]
        """
        return tf.concat(
            [tf.reshape(tf.math.add(alpha[:, 0], tf.math.log(u_0)), shape=(alpha.shape[0], 1)), alpha[:, 1:]], axis=-1)

    @staticmethod
    def scale_u(u_orig, scale_values):
        """
        Up-scales a batch of normalized moments by its original zero order moment u_0
        input: u_orig = [u0_orig,...,uN_orig], original moments, dim=(nsx(N+1))
               scale_values = zero order moment, scaling factor, dim= (nsx1)
        output: u_scaled = u_orig*scale_values, dim(ns x(N+1)
        """
        return tf.math.multiply(u_orig, tf.reshape(scale_values, shape=(scale_values.shape[0], 1)))

    def compute_h(self, u, alpha):
        """
        brief: computes the entropy functional h on u and alpha with regularization in mind

        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N)
               u, dims = (nS x N)
        used members: m    , dims = (N x nq)
                    w    , dims = nq

        returns h = alpha*u - <eta_*(alpha*m)>
        """
        # Currently only for maxwell Boltzmann entropy
        f_quad = tf.math.exp(tf.tensordot(alpha, self.moment_basis, axes=([1], [0])))  # exp(alpha*m)
        entropy_pt1 = tf.tensordot(f_quad, self.quad_weights, axes=([1], [1]))  # f*w
        entropy_pt2 = tf.math.reduce_sum(tf.math.multiply(alpha, u), axis=1, keepdims=True)  # alpha*u
        # 0.5*gamma*alpha_r*alpha_r
        entropy_pt3 = 0.5 * self.regularization_gamma * tf.math.reduce_sum(tf.math.multiply(alpha[:, 1:], alpha[:, 1:]),
                                                                           axis=1, keepdims=True)
        return entropy_pt2 - entropy_pt1 - entropy_pt3  # negative of dual

    def compute_h_fast(self, u, alpha):
        """
        brief: computes the entropy functional h on u and alpha using that <exp(alpha*m)> = u_0 = 1 for normalized moments

        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N)
               u, dims = (nS x N)
        used members: m    , dims = (N x nq)
                    w    , dims = nq

        returns h = alpha*u - <eta_*(alpha*m)>
        """
        # Currently only for maxwell Boltzmann entropy
        # f_quad = tf.math.exp(tf.tensordot(alpha, self.moment_basis, axes=([1], [0])))  # exp(alpha*m)
        # tmp = tf.tensordot(f_quad, self.quad_weights, axes=([1], [1]))  # f*w
        tmp2 = tf.math.reduce_sum(tf.math.multiply(alpha, u), axis=1, keepdims=True)
        return tmp2 - u[:, 0]


class SobolevModel(EntropyModel):
    """
    model that wraps a sobolev layer around a given core model.
    Allows to train on derivatives of the core model
    """

    def __init__(self, core_model: tf.keras.Model, polynomial_degree: int = 1, spatial_dimension: float = 1.0,
                 reconstruct_u: bool = False, scaler_min: float = 0.0, scaler_max: float = 1.0,
                 scale_active: bool = True, gamma: float = 0.0, **opts):
        super(SobolevModel, self).__init__(core_model=core_model, polynomial_degree=polynomial_degree,
                                           spatial_dimension=spatial_dimension, reconstruct_u=reconstruct_u,
                                           scaler_min=scaler_min, scaler_max=scaler_max, scale_active=scale_active,
                                           subclass=True, gamma=gamma)
        self.derivative_scale_factor = tf.constant(scaler_max - scaler_min, dtype=tf.float64)
        print("Model output alpha and h will be scaled by factor " + str(self.derivative_scale_factor.numpy()))

    def call(self, x: Tensor, training=False) -> list:
        """
        Defines the sobolev execution (does not return 0th order moment)
        input: x = [u_1,u_2,...,u_N]
        output: h = entropy of u,alpha
                alpha = [alpha_1,...,alpha_N]
                u = [u_1,u_2,...,u_N]
        """
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            h = self.core_model(x)
        alpha = grad_tape.gradient(h, x)

        if self.enable_recons_u:
            if self.scale_active:
                print("Scaled reconstruction of u enabled")
                alpha64 = tf.math.scalar_mul(self.derivative_scale_factor, tf.cast(alpha, dtype=tf.float64, name=None))
            else:
                print("Reconstruction of u enabled")
                alpha64 = tf.cast(alpha, dtype=tf.float64, name=None)
            alpha_complete = self.reconstruct_alpha(alpha64)
            u_complete = self.reconstruct_u(alpha_complete)
            res = u_complete[:, 1:]  # cutoff the 0th order moment, since it is 1 by construction
        else:
            print("Reconstruction of u disabled. Output 3 is meaningless")
            res = alpha
        return [h, alpha, res]

    def call_derivative(self, x, training=False):
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.core_model(x)
        derivative_net = grad_tape.gradient(y, x)

        return derivative_net
