'''
Network class "MK11" for the neural entropy closure.
ICNN with sobolev wrapper.
Author: Steffen Schotthöfer
Version: 1.0
Date 09.04.2021
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
from tensorflow import Tensor

from src.networks.basenetwork import BaseNetwork
from src.networks.custommodels import SobolevModel
from src.networks.customlayers import MeanShiftLayer, DecorrelationLayer


class MK11Network(BaseNetwork):
    '''
    MK11 Model: Multi purpose sobolev based convex model
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, normalized: bool, input_decorrelation: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str = "", scale_active: bool = True):
        if save_folder == "":
            custom_folder_name = "MK11_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
        else:
            custom_folder_name = save_folder
        super(MK11Network, self).__init__(normalized=normalized, polynomial_degree=polynomial_degree,
                                          spatial_dimension=spatial_dimension, width=width, depth=depth,
                                          loss_combination=loss_combination, save_folder=custom_folder_name,
                                          input_decorrelation=input_decorrelation, scale_active=scale_active)

    def create_model(self) -> bool:
        # Weight initializer
        # 1. This is a modified Kaiming inititalization with a first-order taylor expansion of the
        # softplus activation function (see S. Kumar "On Weight Initialization in
        # Deep Neural Networks").
        # Extra factor of (1/1.1) added inside sqrt to suppress inf for 1 dimensional inputs
        # input_stddev: float = np.sqrt(
        #    (1 / 1.1) * (1 / self.inputDim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        input_stddev: float = np.sqrt(
            (1 / 1.1) * (1 / self.input_dim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        input_initializer = keras.initializers.RandomNormal(mean=0., stddev=input_stddev)
        # input_initializer = tf.keras.initializers.LecunNormal()
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)

        # keras.initializers.RandomNormal(mean=0., stddev=input_stddev)
        # Weight initializer (uniform bounded)
        # initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        # initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        # Weight regularizer
        l2_regularizer_nn = tf.keras.regularizers.L1L2(l2=0.001)  # L1 + L2 penalties
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)  # L1 + L2 penalties

        def convex_layer(layer_input_z: Tensor, nw_input_x: Tensor, layer_idx: int = 0, layer_dim: int = 10) -> Tensor:
            stddev = np.sqrt(
                (1 / 1.1) * (1 / layer_dim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
            initializer = keras.initializers.RandomNormal(mean=0., stddev=stddev)
            # initializer = tf.keras.initializers.LecunNormal()

            # Weighted sum of previous layers output plus bias
            weighted_non_neg_sum_z = layers.Dense(units=layer_dim, kernel_constraint=NonNeg(), activation=None,
                                                  kernel_initializer=initializer,
                                                  kernel_regularizer=l2_regularizer_nn,
                                                  use_bias=True, bias_initializer=initializer,
                                                  bias_regularizer=l1l2_regularizer,
                                                  name='layer_' + str(layer_idx) + 'nn_component'
                                                  )(layer_input_z)
            # Weighted sum of network input
            weighted_sum_x = layers.Dense(units=layer_dim, activation=None,
                                          kernel_initializer=initializer,
                                          kernel_regularizer=l1l2_regularizer,
                                          use_bias=False, name='layer_' + str(layer_idx) + 'dense_component'
                                          )(nw_input_x)
            # Wz+Wx+b
            intermediate_sum = layers.Add(name='add_component_' + str(layer_idx))(
                [weighted_sum_x, weighted_non_neg_sum_z])
            # activation
            out = tf.keras.activations.softplus(intermediate_sum)
            # out = tf.keras.activations.selu(intermediate_sum)
            # batch normalization
            out = layers.BatchNormalization(name='bn_' + str(layer_idx))(out)
            return out

        def convex_output_layer(layer_input_z: Tensor, net_input_x: Tensor, layer_idx: int = 0) -> Tensor:
            # Weighted sum of previous layers output plus bias
            weighted_nn_sum_z: Tensor = layers.Dense(1, kernel_constraint=NonNeg(), activation=None,
                                                     kernel_initializer=tf.keras.initializers.HeNormal(),
                                                     kernel_regularizer=l2_regularizer_nn,
                                                     use_bias=True, bias_regularizer=l1l2_regularizer,
                                                     bias_initializer=tf.keras.initializers.HeNormal(),
                                                     name='layer_' + str(layer_idx) + 'nn_component'
                                                     )(layer_input_z)
            # Weighted sum of network input
            weighted_sum_x: Tensor = layers.Dense(1, activation=None,
                                                  kernel_initializer=tf.keras.initializers.HeNormal(),
                                                  kernel_regularizer=l1l2_regularizer,
                                                  use_bias=False,
                                                  name='layer_' + str(layer_idx) + 'dense_component'
                                                  )(net_input_x)
            # Wz+Wx+b
            out: Tensor = layers.Add()([weighted_sum_x, weighted_nn_sum_z])
            if self.scale_active:  # if output is scaled, use relu.
                out = tf.keras.activations.relu(out)
            return out

        ### build the core network with icnn closure architecture ###
        input_ = keras.Input(shape=(self.input_dim,))
        if self.input_decorrelation:  # input data decorellation and shift
            hidden = MeanShiftLayer(input_dim=self.input_dim, mean_shift=self.mean_u, name="mean_shift")(input_)
            hidden = DecorrelationLayer(input_dim=self.input_dim, ev_cov_mat=self.cov_ev, name="decorrelation")(hidden)
            # First Layer is a std dense layer
            hidden = layers.Dense(self.model_width, activation="softplus", kernel_initializer=input_initializer,
                                  kernel_regularizer=l1l2_regularizer, use_bias=True,
                                  bias_initializer=input_initializer,
                                  bias_regularizer=l1l2_regularizer, name="layer_-1_input")(hidden)
        else:
            # First Layer is a std dense layer
            hidden = layers.Dense(self.model_width, activation="softplus", kernel_initializer=input_initializer,
                                  kernel_regularizer=l1l2_regularizer, use_bias=True,
                                  bias_initializer=input_initializer,
                                  bias_regularizer=l1l2_regularizer, name="layer_-1_input")(input_)
        # other layers are convexLayers
        for idx in range(0, self.model_depth):
            hidden = convex_layer(hidden, input_, layer_idx=idx, layer_dim=self.model_width)
            # hidden = layers.BatchNormalization()(hidden)
        hidden = convex_layer(hidden, input_, layer_idx=self.model_depth + 1, layer_dim=int(self.model_width / 2))
        pre_output = convex_output_layer(hidden, input_, layer_idx=self.model_depth + 2)  # outputlayer
        # scale ouput to range  (0,1) h = h_old*(h_max-h_min)+h_min
        # output_ = tf.add(tf.math.scalar_mul((h_max_tensor - h_min_tensor), pre_output), h_max_tensor)

        # Create the core model
        core_model = keras.Model(inputs=[input_], outputs=[pre_output], name="Icnn_closure")
        print("The core model overview")
        core_model.summary()
        print("The sobolev wrapped model overview")

        # build sobolev wrapper
        model = SobolevModel(core_model, polynomial_degree=self.poly_degree, spatial_dimension=self.spatial_dim,
                             reconstruct_u=bool(self.loss_weights[2]), scaler_max=self.scaler_max,
                             scaler_min=self.scaler_min, scale_active=self.scale_active, name="sobolev_icnn_wrapper")
        # build graph
        batch_size: int = 3  # dummy entry
        model.build(input_shape=(batch_size, self.input_dim))

        # test
        # a1 = tf.constant([[1], [2.5], [2]], shape=(3, 1), dtype=tf.float32)
        # a0 = tf.constant([[2], [1.5], [3]], shape=(3, 1), dtype=tf.float32)
        # a2 = tf.constant([[0, 0.5], [0, 1.5], [1, 2.5]], shape=(3, 2), dtype=tf.float32)
        # a3 = tf.constant([[0, 1.5], [0, 2.5], [1, 3.5]], shape=(3, 2), dtype=tf.float32)

        # print(tf.keras.losses.MeanSquaredError()(a3, a2))
        # print(self.KL_divergence_loss(model.momentBasis, model.quadWeights)(a1, a0))
        # print(self.custom_mse(a2, a3))
        model.compile(
            loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError(),
                  'output_3': tf.keras.losses.MeanSquaredError()},
            # 'output_4': self.KL_divergence_loss(model.momentBasis, model.quadWeights)},  # self.custom_mse},
            loss_weights={'output_1': self.loss_weights[0], 'output_2': self.loss_weights[1],
                          'output_3': self.loss_weights[2]},  # , 'output_4': self.lossWeights[3]},
            optimizer=self.optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

        # model.summary()

        # tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True,
        # show_layer_names = True, rankdir = 'TB', expand_nested = True)
        print("Weight data type:" + str(np.unique([w.dtype for w in model.get_weights()])))
        self.model = model
        return True

    def call_training(self, val_split: float = 0.1, epoch_size: int = 2, batch_size: int = 128, verbosity_mode: int = 1,
                      callback_list: list = []) -> list:
        '''
        Calls training depending on the MK model
        '''
        x_data = self.training_data[0]
        [h, alpha, u] = self.model(x_data[:1000])

        # y_data = [h,alpha,u, alpha (for KLDivergence)]
        y_data = [self.training_data[2], self.training_data[1], self.training_data[0]]  # , self.trainingData[1]]
        self.history = self.model.fit(x=x_data, y=y_data,
                                      validation_split=val_split, epochs=epoch_size,
                                      batch_size=batch_size, verbose=verbosity_mode,
                                      callbacks=callback_list, shuffle=True)

        [h, alpha, u] = self.model(x_data)

        return self.history

    def select_training_data(self):
        return [True, True, True]

    def call_network(self, u_complete):
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
        [u_complete_reconstructed, alpha_complete_predicted, h_predicted] = self.call_network(u_downscaled)
        u_rescaled = self.model.scale_u(u_complete_reconstructed, u_non_normal[:, 0])  # upscaling
        alpha_rescaled = self.model.scale_alpha(alpha_complete_predicted, u_non_normal[:, 0])  # upscaling
        h_rescaled = self.model.compute_h(u_rescaled, alpha_rescaled)

        return [u_rescaled, alpha_rescaled, h_rescaled]

    def call_scaled_64(self, u_non_normal):

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
        #
        #
        #
        u_reduced = u_downscaled[:, 1:]  # chop of u_0
        [h_predicted, alpha_predicted, u_0_predicted] = self.model(u_reduced)
        ### cast to fp64 ###
        alpha_predicted = tf.cast(alpha_predicted, dtype=tf.float64, name=None)
        mBasis = tf.cast(self.model.momentBasis, dtype=tf.float64, name=None)
        qWeights = tf.cast(self.model.quadWeights, dtype=tf.float64, name=None)
        # reconstruct alpha (with fp64)
        tmp = tf.math.exp(tf.tensordot(alpha_predicted, mBasis[1:, :], axes=([1], [0])))  # tmp = alpha * m
        alpha_0 = -tf.math.log(tf.tensordot(tmp, qWeights, axes=([1], [1])))  # ln(<tmp>)
        alpha_complete_predicted = tf.concat([alpha_0, alpha_predicted], axis=1)  # concat [alpha_0,alpha]
        #
        # reconstruct_u
        f_quad = tf.math.exp(tf.tensordot(alpha_complete_predicted, mBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.math.multiply(f_quad, qWeights)  # f*w
        u_complete_reconstructed = tf.tensordot(tmp, mBasis[:, :], axes=([1], [1]))
        #
        # upscale u
        u0 = tf.cast(u_non_normal[:, 0], dtype=tf.float64, name=None)
        u_rescaled = self.model.scale_u(u_complete_reconstructed, u0)  # upscaling
        alpha_rescaled = self.model.scale_alpha(alpha_complete_predicted, u0)  # upscaling
        #
        # compute h
        f_quad = tf.math.exp(tf.tensordot(alpha_rescaled, mBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.tensordot(f_quad, qWeights, axes=([1], [1]))  # f*w
        tmp2 = tf.math.reduce_sum(tf.math.multiply(alpha_rescaled, u_rescaled), axis=1, keepdims=True)
        h_rescaled = tmp2 - tmp

        return [u_rescaled, alpha_rescaled, h_rescaled]
