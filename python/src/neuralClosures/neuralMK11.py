'''
Network class "MK11" for the neural entropy closure.
MK7 ICNN with sobolev wrapper.
Author: Steffen Schotthöfer
Version: 0.0
Date 09.04.2020
'''
from .neuralBase import neuralBase
from .neuralBase import LossAndErrorPrintingCallback

import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
from tensorflow import Tensor
from src import math


class neuralMK11(neuralBase):
    '''
    MK4 Model: Train u to h and alpha
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, polyDegree=0, spatialDim=1, folderName="testFolder", lossCombi=0, width=10, depth=5,
                 normalized=False):
        if (folderName == "testFolder"):
            customFolderName = "MK11_N" + str(polyDegree) + "_D" + str(spatialDim)
        else:
            customFolderName = folderName

        super(neuralMK11, self).__init__(normalized, polyDegree, spatialDim, width, depth, lossCombi,
                                         customFolderName)

        self.model = self.createModel()

    def createModel(self):

        layerDim = self.modelWidth

        # Weight initializer
        # 1. This is a modified Kaiming inititalization with a first-order taylor expansion of the
        # softplus activation function (see S. Kumar "On Weight Initialization in
        # Deep Neural Networks").

        # Extra factor of (1/1.1) added inside sqrt to suppress inf for 1 dimensional inputs
        input_stddev = np.sqrt((1 / 1.1) * (1 / self.inputDim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        input_initializer = keras.initializers.RandomNormal(mean=0., stddev=input_stddev)
        # Weight initializer (uniform bounded)
        # initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        # initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        # Weight regularizer
        l1l2Regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)  # L1 + L2 penalties

        def convexLayer(layerInput_z: Tensor, netInput_x: Tensor, layerIdx=0, layer_dim=layerDim) -> Tensor:
            stddev = np.sqrt(
                (1 / 1.1) * (1 / layerDim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
            initializer = keras.initializers.RandomNormal(mean=0., stddev=stddev)

            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = layers.Dense(layer_dim, kernel_constraint=NonNeg(), activation=None,
                                               kernel_initializer=initializer,
                                               kernel_regularizer=l1l2Regularizer,
                                               use_bias=True, bias_initializer='zeros',
                                               name='non_neg_component_' + str(
                                                   layerIdx)
                                               )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = layers.Dense(layer_dim, activation=None,
                                         kernel_initializer=initializer,
                                         kernel_regularizer=l1l2Regularizer,
                                         use_bias=False, name='dense_component_' + str(layerIdx)
                                         )(netInput_x)
            # Wz+Wx+b
            intermediateSum = layers.Add(name='add_component_' + str(layerIdx))([weightedSum_x, weightedNonNegSum_z])

            # activation
            out = tf.keras.activations.softplus(intermediateSum)
            # batch normalization
            # out = layers.BatchNormalization(name='bn_' + str(layerIdx))(out)
            return out

        def convexLayerOutput(layerInput_z: Tensor, netInput_x: Tensor) -> Tensor:
            stddev = np.sqrt(
                (1 / 1.1) * (1 / 1) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
            initializer = keras.initializers.RandomNormal(mean=0., stddev=stddev)

            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = layers.Dense(1, kernel_constraint=NonNeg(), activation=None,
                                               kernel_initializer=initializer,
                                               kernel_regularizer=l1l2Regularizer,
                                               use_bias=True,
                                               bias_initializer='zeros'
                                               # name='in_z_NN_Dense'
                                               )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = layers.Dense(1, activation=None,
                                         kernel_initializer=initializer,
                                         kernel_regularizer=l1l2Regularizer,
                                         use_bias=False
                                         # name='in_x_Dense'
                                         )(netInput_x)
            # Wz+Wx+b
            intermediateSum = layers.Add()([weightedSum_x, weightedNonNegSum_z])

            # activation
            # out = tf.keras.activations.softplus(intermediateSum)
            # batch normalization
            # out = layers.BatchNormalization()(out)
            return intermediateSum

        ### build the core network with icnn closure architecture ###
        input_ = keras.Input(shape=(self.inputDim,))
        # First Layer is a std dense layer
        hidden = layers.Dense(layerDim, activation="softplus",
                              kernel_initializer=input_initializer,
                              kernel_regularizer=l1l2Regularizer,
                              bias_initializer='zeros',
                              name="first_dense"
                              )(input_)
        # other layers are convexLayers
        for idx in range(0, self.modelDepth):
            hidden = convexLayer(hidden, input_, layerIdx=idx)
        hidden = convexLayer(hidden, input_, layerIdx=self.modelDepth + 1, layer_dim=int(layerDim / 2))
        output_ = convexLayerOutput(hidden, input_)  # outputlayer

        # Create the core model
        coreModel = keras.Model(inputs=[input_], outputs=[output_], name="Icnn_closure")

        # build model
        model = sobolevModel(coreModel, polyDegree=self.polyDegree, name="sobolev_icnn_wrapper")

        batchSize = 2  # dummy entry
        model.build(input_shape=(batchSize, self.inputDim))

        # test
        a1 = tf.constant([[1], [2.5], [2]], shape=(3, 1), dtype=tf.float32)
        a0 = tf.constant([[2], [1.5], [3]], shape=(3, 1), dtype=tf.float32)
        a2 = tf.constant([[0, 0.5], [0, 1.5], [1, 2.5]], shape=(3, 2), dtype=tf.float32)
        a3 = tf.constant([[0, 1.5], [0, 2.5], [1, 3.5]], shape=(3, 2), dtype=tf.float32)

        print(tf.keras.losses.MeanSquaredError()(a3, a2))
        print(self.KL_divergence_loss(model.momentBasis, model.quadWeights)(a1, a0))
        print(self.custom_mse(a2, a3))
        model.compile(
            loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError(),
                  'output_3': tf.keras.losses.MeanSquaredError(),
                  'output_4': self.KL_divergence_loss(model.momentBasis, model.quadWeights)},  # self.custom_mse},
            loss_weights={'output_1': self.lossWeights[0], 'output_2': self.lossWeights[1],
                          'output_3': self.lossWeights[2], 'output_4': self.lossWeights[3]},
            optimizer=self.optimizer, metrics=['mean_absolute_error'])

        # model.summary()

        # tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True,
        # show_layer_names = True, rankdir = 'TB', expand_nested = True)

        return model

    def custom_mse(self, y_true, y_pred):

        # calculating squared difference between target and predicted values 
        loss = K.square(y_pred - y_true)  # (batch_size, 2)

        # multiplying the values with weights along batch dimension
        loss = loss * [0.3, 0.7]  # (batch_size, 2)

        # summing both loss values along batch dimension 
        loss = K.sum(loss, axis=1)  # (batch_size,)

        return loss

    def KL_divergence_loss(self, mB, qW):

        """
        KL divergence between f_u and f_true  using alpha and alpha_true.
        inputs: mB, moment Basis evaluted at quadPts, dim = (N x nq)
                quadWeights, dim = nq
        returns: KL_divergence function using mBasis and quadWeights
        """

        def reconstruct_alpha(alpha):
            """
            brief:  Reconstructs alpha_0 and then concats alpha_0 to alpha_1,... , from alpha1,...
                    Only works for maxwell Boltzmann entropy so far.
                    => copied from sobolev model code
            nS = batchSize
            N = basisSize
            nq = number of quadPts

            input: alpha, dims = (nS x N-1)
                   m    , dims = (N x nq)
                   w    , dims = nq
            returns alpha_complete = [alpha_0,alpha], dim = (nS x N), where alpha_0 = - ln(<exp(alpha*m)>)
            """
            # Check the predicted alphas for +/- infinity or nan - raise error if found
            checked_alpha = tf.debugging.check_numerics(alpha, message='input tensor checking error', name='checked')
            # Clip the predicted alphas below the tf.exp overflow threshold
            clipped_alpha = tf.clip_by_value(checked_alpha, clip_value_min=-50, clip_value_max=50,
                                             name='checkedandclipped')

            tmp = tf.math.exp(tf.tensordot(clipped_alpha, mB[1:, :], axes=([1], [0])))  # tmp = alpha * m
            alpha_0 = -tf.math.log(tf.tensordot(tmp, qW, axes=([1], [1])))  # ln(<tmp>)
            return tf.concat([alpha_0, alpha], axis=1)  # concat [alpha_0,alpha]

        def KL_divergence(y_true, y_pred):
            """
            brief: computes the Kullback-Leibler Divergence of the kinetic density w.r.t alpha given the kinetic density w.r.t
                    alpha_true
            input: alpha_true , dim= (ns,N)
                   alpha , dim = (ns, N+1)
            output: pointwise KL Divergence, dim  = ns x 1
            """

            # extend alpha_true to full dimension
            print("alpha_true")
            print(y_true)
            print("alpha")
            print(y_pred)
            alpha_true_recon = reconstruct_alpha(y_true)
            alpha_pred_recon = reconstruct_alpha(y_pred)
            diff = alpha_true_recon - alpha_pred_recon
            print("diff")

            # print(diff)
            # print("___")
            t1 = tf.math.exp(tf.tensordot(alpha_true_recon, mB, axes=([1], [0])))
            t2 = tf.tensordot(diff, mB, axes=([1], [0]))
            integrand = tf.math.multiply(t1, t2)
            # print(integrand.shape)
            # print("____")
            res = tf.tensordot(integrand, qW, axes=([1], [1]))
            # print(res.shape)
            # print("____")
            # res = tf.math.reduce_mean(res, axis=0, keepdims=False, name=None)
            # print(res.shape)
            # print("____")
            # res = tf.reshape(res, shape=())
            # print(res.shape)
            # print("____")
            # print("end:::::::::::::::end")
            return res

        return KL_divergence

    def call_training(self, val_split=0.1, epoch_size=2, batch_size=128, verbosity_mode=1, callback_list=[]):
        '''
        Calls training depending on the MK model
        '''
        xData = self.trainingData[0]
        # yData = [h,alpha,u, alpha (for KLDivergence)]
        yData = [self.trainingData[2], self.trainingData[1], self.trainingData[0], self.trainingData[1]]
        self.history = self.model.fit(x=xData, y=yData,
                                      validation_split=val_split, epochs=epoch_size,
                                      batch_size=batch_size, verbose=verbosity_mode,
                                      callbacks=callback_list, shuffle=True)
        return self.history

    def selectTrainingData(self):
        return [True, True, True]

    def trainingDataPostprocessing(self):
        return 0

    def callNetwork(self, u_complete):
        """
        brief: Only works for maxwell Boltzmann entropy so far.
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: u_complete, dims = (nS x N)
        returns: alpha_complete_predicted, dim = (nS x N)
                 u_complete_reconstructed, dim = (nS x N)
                 h_predicted, dim = (nS x 1)
        """
        u_reduced = u_complete[:, 1:]  # chop of u_0
        [h_predicted, alpha_predicted, u_0_predicted, tmp] = self.model(u_reduced)
        alpha_complete_predicted = self.model.reconstruct_alpha(alpha_predicted)
        u_complete_reconstructed = self.model.reconstruct_u(alpha_complete_predicted)

        return [u_complete_reconstructed, alpha_complete_predicted, h_predicted]

    def call_scaled(self, u_non_normal):

        """
        brief: Only works for maxwell Boltzmann entropy so far.
        Calls the network with non normalized moments. (first the moments get normalized, then the network gets called,
        then the upscaling mechanisms get called, then the original entropy gets computed.)
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: u_complete, dims = (nS x N)
        returns: [u,alpha,h], where
                 alpha_complete_predicted_scaled, dim = (nS x N)
                 u_complete_reconstructed_scaled, dim = (nS x N)
                 h_predicted_scaled, dim = (nS x 1)
        """
        u_non_normal = tf.constant(u_non_normal, dtype=tf.float32)
        u_downscaled = self.model.scale_u(u_non_normal, tf.math.reciprocal(u_non_normal[:, 0]))  # downscaling
        [u_complete_reconstructed, alpha_complete_predicted, h_predicted] = self.callNetwork(u_downscaled)
        u_rescaled = self.model.scale_u(u_complete_reconstructed, u_non_normal[:, 0])  # upscaling
        alpha_rescaled = self.model.scale_alpha(alpha_complete_predicted, u_non_normal[:, 0])  # upscaling
        h_rescaled = self.model.compute_h(u_rescaled, alpha_rescaled)

        return [u_rescaled, alpha_rescaled, h_rescaled]


class sobolevModel(tf.keras.Model):
    # Sobolev implies, that the model outputs also its derivative
    def __init__(self, coreModel, polyDegree=1, **opts):
        super(sobolevModel, self).__init__()
        # Member is only the model we want to wrap with sobolev execution
        self.coreModel = coreModel  # must be a compiled tensorflow model

        # Create quadrature and momentBasis. Currently only for 1D problems
        self.polyDegree = polyDegree
        self.nq = 100
        [quadPts, quadWeights] = math.qGaussLegendre1D(self.nq)  # dims = nq
        self.quadPts = tf.constant(quadPts, shape=(1, self.nq), dtype=tf.float32)  # dims = (batchSIze x N x nq)
        self.quadWeights = tf.constant(quadWeights, shape=(1, self.nq),
                                       dtype=tf.float32)  # dims = (batchSIze x N x nq)
        mBasis = math.computeMonomialBasis1D(quadPts, self.polyDegree)  # dims = (N x nq)
        self.inputDim = mBasis.shape[0]
        self.momentBasis = tf.constant(mBasis, shape=(self.inputDim, self.nq),
                                       dtype=tf.float32)  # dims = (batchSIze x N x nq)

    def call(self, x, training=False):
        """
        Defines the sobolev executio (does not return 0th order moment)
        input: x = [u_1,u_2,...,u_N]
        output: h = entropy of u,alpha
                alpha = [alpha_1,...,alpha_N]
                u = [u_1,u_2,...,u_N]
        """

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            h = self.coreModel(x)
        alpha = grad_tape.gradient(h, x)

        alpha_complete = self.reconstruct_alpha(alpha)
        u_complete = self.reconstruct_u(alpha_complete)
        return [h, alpha, u_complete[:, 1:], alpha]  # cutoff the 0th order moment, since it is 1 by construction

    def callDerivative(self, x, training=False):
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.coreModel(x)
        derivativeNet = grad_tape.gradient(y, x)

        return derivativeNet

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
        checked_alpha = tf.debugging.check_numerics(alpha, message='input tensor checking error', name='checked')
        # Clip the predicted alphas below the tf.exp overflow threshold
        clipped_alpha = tf.clip_by_value(checked_alpha, clip_value_min=-50, clip_value_max=50, name='checkedandclipped')

        tmp = tf.math.exp(tf.tensordot(clipped_alpha, self.momentBasis[1:, :], axes=([1], [0])))  # tmp = alpha * m
        alpha_0 = -tf.math.log(tf.tensordot(tmp, self.quadWeights, axes=([1], [1])))  # ln(<tmp>)
        return tf.concat([alpha_0, alpha], axis=1)  # concat [alpha_0,alpha]

    def reconstruct_u(self, alpha):
        """
        brief: reconstructs u from alpha
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
        f_quad = tf.math.exp(tf.tensordot(clipped_alpha, self.momentBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.math.multiply(f_quad, self.quadWeights)  # f*w
        return tf.tensordot(tmp, self.momentBasis[:, :], axes=([1], [1]))  # f * w * momentBasis

    def scale_alpha(self, alpha, u_0):
        """
        Performs the up-scaling computation, s.t. alpha corresponds to a moment with u0 = u_0, instead of u0=1
        input: alpha = [alpha_0,...,alpha_N],  batch of lagrange multipliers of the zeroth moment, dim = (nsx(N+1)))
               u_0 = batch of zeroth moments, dim = (nsx1)
        output: alpha_scaled = alpha + [ln(u_0),0,0,...]
        """
        return tf.concat(
            [tf.reshape(tf.math.add(alpha[:, 0], tf.math.log(u_0)), shape=(alpha.shape[0], 1)), alpha[:, 1:]], axis=-1)

    def scale_u(self, u_orig, scale_values):
        """
        Up-scales a batch of normalized moments by its original zero order moment u_0
        input: u_orig = [u0_orig,...,uN_orig], original moments, dim=(nsx(N+1))
               scale_values = zero order moment, scaling factor, dim= (nsx1)
        output: u_scaled = u_orig*scale_values, dim(ns x(N+1)
        """
        return tf.math.multiply(u_orig, tf.reshape(scale_values, shape=(scale_values.shape[0], 1)))

    def compute_h(self, u, alpha):
        """
        brief: computes the entropy functional h on u and alpha

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
        f_quad = tf.math.exp(tf.tensordot(alpha, self.momentBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.tensordot(f_quad, self.quadWeights, axes=([1], [1]))  # f*w
        tmp2 = tf.math.reduce_sum(tf.math.multiply(alpha, u), axis=1, keepdims=True)
        return tmp2 - tmp
