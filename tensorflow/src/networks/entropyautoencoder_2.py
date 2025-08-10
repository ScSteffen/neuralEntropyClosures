'''
brief: Entropy autoencoder model
Author: Steffen Schotthöfer
Version: 1.0
Date 17.12.2021
'''

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import Tensor
from tensorflow import keras as keras

from src import math
from src.networks.customlayers import MeanShiftLayer, DecorrelationLayer


class EntropyAutoEncoder(Model):
    nq: int  # @brief: number of quadrature points
    input_dim: int  # @brief: dimension of input vector - i.e. momnent vector lenght
    poly_degree: int  # @brief: order of the polynomial basis - also latent dimension
    model_depth: int  # @brief: amount of resnet Blocks in each network

    quad_pts: Tensor  # @brief: Tensor of quadrature points
    quad_weights: Tensor  # @brief: Tensorf of quadrature weights
    moment_basis: Tensor  # @brief: Tensor that represents the moment basis
    encoder: Model  # @brief: econder architecutre to map alppha*m to [f_q,...]
    decoder: Model  # @brief: decoder architecutre to map u=f*m*w to h

    def __init__(self, polynomial_degree: int = 1, spatial_dimension: int = 1, model_depth: int = 2):
        super(EntropyAutoEncoder, self).__init__()

        # initialize support variables
        self.model_depth = model_depth
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

    def build_decoder(self) -> Model:
        """
        Method to build the entropy decoder model
        input: alpha*m
        output: f
        """
        initializer = keras.initializers.LecunNormal()
        l2_regularizer = tf.keras.regularizers.L2(l2=0.001)  # L1 + L2 penalties

        input_ = keras.Input(shape=(self.nq,))
        # input layers
        if self.input_decorrelation and self.input_dim > 1:
            hidden = MeanShiftLayer(input_dim=self.input_dim, mean_shift=self.mean_u, name="mean_shift")(input_)
            hidden = DecorrelationLayer(input_dim=self.input_dim, ev_cov_mat=self.cov_ev, name="decorrelation")(hidden)
            hidden = keras.layers.Dense(self.model_width, activation=None, kernel_initializer=initializer,
                                        use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                        bias_regularizer=l2_regularizer, name="layer_input")(hidden)
        else:
            hidden = keras.layers.Dense(self.model_width, activation=None, kernel_initializer=initializer,
                                        use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                        bias_regularizer=l2_regularizer, name="layer_input")(input_)
        # build resnet blocks
        dimension_difference = self.nq - self.input_dim
        if dimension_difference <= 0:
            print("vector length bigger than number of quad points. might hurt accuracy")
        dim_increase = float(dimension_difference) / float(self.model_depth)

        for idx in range(0, self.model_depth):
            current_depth = int(self.input_dim + dim_increase * idx)
            # hidden = self.residual_block(hidden, layer_dim=self.model_width, layer_idx=idx)
            hidden = keras.layers.Dense(units=current_depth, activation='relu', kernel_initializer=initializer,
                                        use_bias=True, bias_initializer=initializer,
                                        kernel_regularizer=l2_regularizer, bias_regularizer=l2_regularizer,
                                        name="encoder_hidden_" + str(idx))(hidden)
        # output layer
        output_ = hidden
        # keras.layers.Dense(units=self.nq, activation=None, kernel_initializer=initializer,
        # use_bias = True, bias_initializer = initializer, kernel_regularizer = l2_regularizer,
        # bias_regularizer = l2_regularizer, name = "layer_output")(hidden)
        # Create the core model
        decoder = keras.Model(inputs=[input_], outputs=[output_], name="Entropy Encoder")
        decoder.summary()

        return decoder

    def build_encoder(self) -> Model:
        """
        Method to build the decoder model
        """
        initializer = keras.initializers.LecunNormal()
        l2_regularizer = tf.keras.regularizers.L2(l2=0.001)  # L1 + L2 penalties

        input_ = keras.Input(shape=(self.input_dim,))
        # input layers
        if self.input_decorrelation and self.input_dim > 1:
            hidden = MeanShiftLayer(input_dim=self.input_dim, mean_shift=self.mean_u, name="mean_shift")(input_)
            hidden = DecorrelationLayer(input_dim=self.input_dim, ev_cov_mat=self.cov_ev, name="decorrelation")(hidden)
            hidden = keras.layers.Dense(self.model_width, activation=None, kernel_initializer=initializer,
                                        use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                        bias_regularizer=l2_regularizer, name="layer_input")(hidden)
        else:
            hidden = keras.layers.Dense(self.model_width, activation=None, kernel_initializer=initializer,
                                        use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                        bias_regularizer=l2_regularizer, name="layer_input")(input_)

        dimension_difference = 1 - self.nq
        dim_increase = float(dimension_difference) / float(self.model_depth)

        # build resnet blocks
        for idx in range(0, self.model_depth):
            current_depth = int(self.nq + dim_increase * idx)
            # hidden = self.residual_block(hidden, layer_dim=self.model_width, layer_idx=idx)
            hidden = keras.layers.Dense(units=current_depth, activation='relu', kernel_initializer=initializer,
                                        use_bias=True, bias_initializer=initializer,
                                        kernel_regularizer=l2_regularizer, bias_regularizer=l2_regularizer,
                                        name="decoder_hidden_" + str(idx))(hidden)
        # output layer
        output_ = hidden
        # keras.layers.Dense(units=self.nq, activation=None, kernel_initializer=initializer,
        # use_bias = True, bias_initializer = initializer, kernel_regularizer = l2_regularizer,
        # bias_regularizer = l2_regularizer, name = "layer_output")(hidden)
        # Create the core model
        encoder = keras.Model(inputs=[input_], outputs=[output_], name="Entropy Decoder")
        encoder.summary()

        return encoder

    @staticmethod
    def residual_block(x: tf.Tensor, layer_dim: int = 10, layer_idx: int = 0) -> tf.Tensor:
        initializer = keras.initializers.LecunNormal()

        # Weight regularizer
        l2_regularizer = tf.keras.regularizers.L2(l2=0.001)  # L1 + L2 penalties

        # ResNet architecture by https://arxiv.org/abs/1603.05027
        y = keras.layers.BatchNormalization()(x)  # 1) BN that normalizes each feature individually (axis=-1)
        y = keras.activations.selu(y)  # 2) activation
        # 3) layer without activation
        y = keras.layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                               bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                               bias_regularizer=l2_regularizer, name="block_" + str(layer_idx) + "_layer_0")(y)
        y = keras.layers.BatchNormalization()(y)  # 4) BN that normalizes each feature individually (axis=-1)
        y = keras.activations.selu(y)  # 5) activation
        # 6) layer
        y = keras.layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                               bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                               bias_regularizer=l2_regularizer, name="block_" + str(layer_idx) + "_layer_1")(y)
        # 7) add skip connection
        out = keras.layers.Add()([x, y])
        return out
