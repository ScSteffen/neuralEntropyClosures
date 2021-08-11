'''
Network class "MK14" for the neural entropy closure.
ICNN with sobolev wrapper.
Experimentation field for weight constraints
Author: Steffen Schotthöfer
Version: 1.0
Date 09.04.2021
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
from tensorflow import Tensor

from src.networks.basenetwork import BaseNetwork
from src.networks.sobolevmodel import SobolevModel
from src.networks.kernelconstraints import ClipByValueConstraint
from tensorflow.keras.constraints import NonNeg


class MK14Network(BaseNetwork):
    '''
    MK11 Model: Multi purpose sobolev based convex model
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, normalized: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str = ""):
        if save_folder == "":
            custom_folder_name = "MK14_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
        else:
            custom_folder_name = save_folder
        super(MK14Network, self).__init__(normalized=normalized, polynomial_degree=polynomial_degree,
                                          spatial_dimension=spatial_dimension, width=width, depth=depth,
                                          loss_combination=loss_combination, save_folder=custom_folder_name)

    def create_model(self) -> bool:
        input_initializer = tf.keras.initializers.LecunNormal()
        l2_regularizer_nn = tf.keras.regularizers.L1L2(l2=0.001)  # L1 + L2 penalties
        l1l2_regularizer = tf.keras.regularizers.L1L2(l1=0.0001, l2=0.0001)  # L1 + L2 penalties

        def shifted_convex_layer(layer_input_z: Tensor, nw_input_x: Tensor, layer_idx: int = 0,
                                 layer_dim: int = 10, input_shape: int = 10) -> Tensor:
            initializer = tf.keras.initializers.LecunNormal()
            shift_tensor = tf.ones(shape=(layer_dim, input_shape), dtype=tf.dtypes.float32, name=None)

            # Weighted sum of previous layers output plus bias
            weighted_non_neg_sum_z = layers.Dense(layer_dim,  # kernel_constraint=ClipByValueConstraint(-1.0),
                                                  activation=None, kernel_initializer=initializer,
                                                  kernel_regularizer=l2_regularizer_nn, use_bias=True,
                                                  bias_initializer='zeros',
                                                  name='non_neg_component_' + str(layer_idx))(layer_input_z)
            # Weighted sum of network input
            weighted_sum_x = layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                                          kernel_regularizer=l1l2_regularizer, use_bias=False,
                                          name='dense_component_' + str(layer_idx))(nw_input_x)
            # Wz+Wx+b
            intermediate_sum = layers.Add(name='add_component_' + str(layer_idx))(
                [weighted_sum_x, weighted_non_neg_sum_z])
            # input shift: ones*z == [sum(z),...]
            shift = tf.math.reduce_sum(layer_input_z, axis=1, keepdims=False,
                                       name='shift_component_' + str(layer_idx))
            # shift_bc = tf.broadcast_to(shift, shape=(None, layer_dim), name="broadcaster")

            # shift: Tensor = layers.Dot(axes=1, name='shift_component_' + str(layer_idx))(
            #   [layer_input_z, shift_tensor])
            # shift = tf.reshape(shift, shape=(shift.shape[1], layer_dim))
            # add to current layer: # Wz+Wx+b + 1z
            intermediate_sum = layers.Add(name='add_shift_' + str(layer_idx))(
                [intermediate_sum, shift])
            # activation
            out = tf.keras.activations.softplus(intermediate_sum)
            # out = tf.keras.activations.selu(intermediate_sum)
            # batch normalization
            # out = layers.BatchNormalization(name='bn_' + str(layerIdx))(out)
            return out

        def convex_output_layer(layer_input_z: Tensor, net_input_x: Tensor) -> Tensor:
            # stddev = np.sqrt(
            #    (1 / 1.1) * (1 / 1) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
            initializer = tf.keras.initializers.LecunNormal()
            # keras.initializers.RandomNormal(mean=0., stddev=stddev)

            # Weighted sum of previous layers output plus bias
            weighted_nn_sum_z: Tensor = layers.Dense(1, kernel_constraint=NonNeg(), activation=None,
                                                     kernel_initializer=initializer,
                                                     kernel_regularizer=l1l2_regularizer,
                                                     use_bias=True,
                                                     bias_initializer='zeros'
                                                     # name='in_z_NN_Dense'
                                                     )(layer_input_z)
            # Weighted sum of network input
            weighted_sum_x: Tensor = layers.Dense(1, activation=None, kernel_initializer=initializer,
                                                  kernel_regularizer=l1l2_regularizer,
                                                  use_bias=False
                                                  # name='in_x_Dense'
                                                  )(net_input_x)
            # Wz+Wx+b
            out: Tensor = layers.Add()([weighted_sum_x, weighted_nn_sum_z])
            # if self.h_max - self.h_min != 1.0:  # if output is scaled, use sigmoid.
            out = tf.keras.activations.relu(out)  # does not break convexity (Sarath Sivaprasad et al.)
            return out

        ### build the core network with icnn closure architecture ###
        input_ = keras.Input(shape=(self.inputDim,))
        # First Layer is a std dense layer
        hidden1 = layers.Dense(self.model_width, activation="selu", kernel_initializer=input_initializer,
                               kernel_regularizer=l1l2_regularizer, bias_initializer='zeros', name="first_dense")(
            input_)
        # other layers are convexLayers
        for idx in range(0, self.model_depth):
            hidden = shifted_convex_layer(hidden1, input_, layer_idx=idx, layer_dim=self.model_width,
                                          input_shape=self.model_width)
        hidden = shifted_convex_layer(hidden, input_, layer_idx=self.model_depth + 1,
                                      layer_dim=int(self.model_width / 2), input_shape=self.model_width)
        pre_output = convex_output_layer(hidden, input_)  # outputlayer
        # scale ouput to range  (0,1) h = h_old*(h_max-h_min)+h_min
        # output_ = tf.add(tf.math.scalar_mul((h_max_tensor - h_min_tensor), pre_output), h_max_tensor)

        # Create the core model
        core_model = keras.Model(inputs=[input_], outputs=[pre_output], name="Icnn_closure")

        # build sobolev wrapper
        model = SobolevModel(core_model, polynomial_degree=self.poly_degree, spatial_dimension=self.spatial_dim,
                             reconstruct_u=bool(self.loss_weights[2]), scale_factor=self.h_max - self.h_min,
                             name="sobolev_icnn_wrapper")
        # build graph
        batch_size: int = 2  # dummy entry
        model.build(input_shape=(batch_size, self.inputDim))

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
        # y_data = [h,alpha,u, alpha (for KLDivergence)]
        y_data = [self.training_data[2], self.training_data[1], self.training_data[0]]  # , self.trainingData[1]]
        self.history = self.model.fit(x=x_data, y=y_data,
                                      validation_split=val_split, epochs=epoch_size,
                                      batch_size=batch_size, verbose=verbosity_mode,
                                      callbacks=callback_list, shuffle=True)
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
