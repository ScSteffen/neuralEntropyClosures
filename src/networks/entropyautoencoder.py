'''
brief: Entropy autoencoder model
Author: Steffen Schotthöfer
Version: 1.0
Date 17.12.2021
'''

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow import Tensor
from tensorflow import keras as keras

from src import math
from src.networks.customlayers import MeanShiftLayer, DecorrelationLayer


class EntropyAutoEncoder(Model):
    nq: int  # @brief: number of quadrature points
    input_dim: int  # @brief: dimension of input vector - i.e. momnent vector lenght
    poly_degree: int  # @brief: order of the polynomial basis - also latent dimension
    model_depth: int  # @brief: amount of network building blocks
    model_width: int  # @brief: width of each block

    quad_pts: Tensor  # @brief: Tensor of quadrature points
    quad_weights: Tensor  # @brief: Tensorf of quadrature weights
    moment_basis: Tensor  # @brief: Tensor that represents the moment basis
    encoder: Model  # @brief: econder architecutre to map alppha*m to [f_q,...]
    decoder: Model  # @brief: decoder architecutre to map u=f*m*w to h

    def __init__(self, polynomial_degree: int = 1, spatial_dimension: int = 1, model_depth: int = 2,
                 model_width: int = 20):
        super(EntropyAutoEncoder, self).__init__()

        # initialize support variables
        self.model_depth = model_depth
        self.model_width = model_width
        self.poly_degree = polynomial_degree
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

        # build the  architecture
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def call(self, alpha):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def density_decoder(self) -> Model:
        """
        convex model that maps pointswise alpha*m(v_i) to f(v_i)
        """
        # Create support structures
        input_initializer = keras.initializers.LecunNormal()
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)  # L1 + L2 penalties
        input_ = keras.Input(shape=(1,))  # pointwise evaluation
        # First Layer is a std dense layer
        hidden = keras.layers.Dense(self.model_width, activation="relu", kernel_initializer=input_initializer,
                                    kernel_regularizer=l1l2_regularizer, use_bias=True,
                                    bias_initializer=input_initializer, bias_regularizer=l1l2_regularizer,
                                    name="layer_-1_input")(input_)
        for idx in range(0, self.model_depth):
            hidden = self.convex_layer(hidden, input_, layer_idx=idx, layer_dim=self.model_width)
        # output layer
        output_ = self.convex_layer(hidden, input_, layer_idx=self.model_depth, layer_dim=1)
        decoder = keras.Model(inputs=[input_], outputs=[output_], name="Density Decoder")
        return decoder

    def entropy_encoder(self) -> Model:
        """
        convex model that maps u to h(u)
        """
        input_initializer = keras.initializers.LecunNormal()
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)  # L1 + L2 penalties

        ### build the core network with icnn closure architecture ###
        input_ = keras.Input(shape=(self.input_dim,))

        # First Layer is a std dense layer
        hidden = keras.layers.Dense(self.model_width, activation="relu", kernel_initializer=input_initializer,
                                    kernel_regularizer=l1l2_regularizer, use_bias=True,
                                    bias_initializer=input_initializer,
                                    bias_regularizer=l1l2_regularizer, name="layer_-1_input")(input_)
        # other layers are convexLayers
        for idx in range(0, self.model_depth):
            hidden = self.convex_layer(hidden, input_, layer_idx=idx, layer_dim=self.model_width)
            # hidden = layers.BatchNormalization()(hidden)
        output_ = self.convex_output_layer(hidden, input_, layer_idx=self.model_depth)  # outputlayer
        # Create the core model
        encoder = keras.Model(inputs=[input_], outputs=[output_], name="Entropy Encoder")

        return encoder

    @staticmethod
    def convex_layer(layer_input_z: Tensor, nw_input_x: Tensor, layer_idx: int = 0, layer_dim: int = 10) -> Tensor:
        # Weight regularizer
        l2_regularizer_nn = tf.keras.regularizers.L1L2(l2=0.0001)  # L1 + L2 penalties
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)  # L1 + L2 penalties
        initializer = keras.initializers.LecunNormal()

        # Weighted sum of previous layers output plus bias
        weighted_non_neg_sum_z = keras.layers.Dense(units=layer_dim, kernel_constraint=keras.constraints.NonNeg(),
                                                    activation=None, kernel_initializer=initializer,
                                                    kernel_regularizer=l2_regularizer_nn, use_bias=True,
                                                    bias_initializer='zeros',
                                                    name='layer_' + str(layer_idx) + '_nn_component'
                                                    )(layer_input_z)
        # Weighted sum of network input
        weighted_sum_x = keras.layers.Dense(units=layer_dim, activation=None,
                                            kernel_initializer=initializer,
                                            kernel_regularizer=l1l2_regularizer,
                                            use_bias=False, name='layer_' + str(layer_idx) + '_dense_component'
                                            )(nw_input_x)
        # Wz+Wx+b
        intermediate_sum = keras.layers.Add(name='add_component_' + str(layer_idx))(
            [weighted_sum_x, weighted_non_neg_sum_z])
        # activation
        out = tf.keras.activations.relu(intermediate_sum)
        # out = tf.keras.activations.selu(intermediate_sum)
        # batch normalization
        # out = layers.BatchNormalization(name='bn_' + str(layer_idx))(out)
        return out

    @staticmethod
    def convex_output_layer(layer_input_z: Tensor, net_input_x: Tensor, layer_idx: int = 0) -> Tensor:
        # Weighted sum of previous layers output plus bias
        l2_regularizer_nn = tf.keras.regularizers.L1L2(l2=0.0001)  # L1 + L2 penalties
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)  # L1 + L2 penalties
        initializer = keras.initializers.LecunNormal()

        weighted_nn_sum_z: Tensor = keras.layers.Dense(1, kernel_constraint=keras.constraints.NonNeg(), activation=None,
                                                       kernel_initializer=initializer,
                                                       kernel_regularizer=l2_regularizer_nn,
                                                       use_bias=True,
                                                       bias_initializer='zeros',
                                                       name='layer_' + str(layer_idx) + 'nn_component'
                                                       )(layer_input_z)
        # Weighted sum of network input
        weighted_sum_x: Tensor = keras.layers.Dense(1, activation=None,
                                                    kernel_initializer=initializer,
                                                    kernel_regularizer=l1l2_regularizer,
                                                    use_bias=False,
                                                    name='layer_' + str(layer_idx) + 'dense_component'
                                                    )(net_input_x)
        # Wz+Wx+b
        out: Tensor = keras.layers.Add()([weighted_sum_x, weighted_nn_sum_z])
        return out
