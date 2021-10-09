'''
Network class "MK15" for the neural entropy closure.
Dense feed forward network with ResNet structure for direct training on scaled alpha.
Includes an Entropy wrapper that reconstructs u.
Has a loss that enforeces monotonicity of the predictions empirically

Author: Steffen Schotthöfer
Version: 0.0
Date 13.08.2021
'''

import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.networks.basenetwork import BaseNetwork
from src.networks.customlosses import MonotonicFunctionLoss, RelativeMAELoss
from src.networks.custommodels import EntropyModel
from src.networks.customlayers import MeanShiftLayer, DecorrelationLayer


class MK15Network(BaseNetwork):

    def __init__(self, normalized: bool, input_decorrelation: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str = ""):
        if save_folder == "":
            custom_folder_name = "MK15_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
        else:
            custom_folder_name = save_folder
        super(MK15Network, self).__init__(normalized=normalized, polynomial_degree=polynomial_degree,
                                          spatial_dimension=spatial_dimension, width=width, depth=depth,
                                          loss_combination=loss_combination, save_folder=custom_folder_name,
                                          input_decorrelation=input_decorrelation)

    def create_model(self) -> bool:

        # Weight initializer
        initializer = keras.initializers.LecunNormal()

        # Weight regularizer
        l2_regularizer = tf.keras.regularizers.L2(l2=0.001)  # L1 + L2 penalties

        ### build the core network ###

        # Define Residual block

        def residual_block(x: tf.Tensor, layer_dim: int = 10, layer_idx: int = 0) -> tf.Tensor:
            # ResNet architecture by https://arxiv.org/abs/1603.05027
            y = keras.layers.BatchNormalization()(x)  # 1) BN that normalizes each feature individually (axis=-1)
            y = keras.activations.selu(y)  # 2) activation
            # 3) layer without activation
            y = layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                             bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                             bias_regularizer=l2_regularizer, name="block_" + str(layer_idx) + "_layer_0")(y)
            y = keras.layers.BatchNormalization()(y)  # 4) BN that normalizes each feature individually (axis=-1)
            y = keras.activations.selu(y)  # 5) activation
            # 6) layer
            y = layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                             bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                             bias_regularizer=l2_regularizer, name="block_" + str(layer_idx) + "_layer_1")(y)
            # 7) add skip connection
            out = keras.layers.Add()([x, y])
            return out

        input_ = keras.Input(shape=(self.input_dim,))
        if self.input_decorrelation and self.input_dim > 1:
            hidden = MeanShiftLayer(input_dim=self.input_dim, mean_shift=self.mean_u, name="mean_shift")(input_)
            hidden = DecorrelationLayer(input_dim=self.input_dim, ev_cov_mat=self.cov_ev, name="decorrelation")(hidden)
            hidden = layers.Dense(self.model_width, activation=None, kernel_initializer=initializer,
                                  use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                  bias_regularizer=l2_regularizer, name="layer_input")(hidden)
        else:
            hidden = layers.Dense(self.model_width, activation=None, kernel_initializer=initializer,
                                  use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                  bias_regularizer=l2_regularizer, name="layer_input")(input_)
        # build resnet blocks
        for idx in range(0, self.model_depth):
            hidden = residual_block(hidden, layer_dim=self.model_width, layer_idx=idx)
        # hidden = keras.layers.BatchNormalization()(hidden)  # BN that normalizes each feature individually (axis=-1)
        if self.scaler_max - self.scaler_min != 1.0:
            output_ = layers.Dense(self.input_dim, activation=None, kernel_initializer=initializer,
                                   use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                   bias_regularizer=l2_regularizer, name="layer_output")(hidden)
        else:
            output_ = layers.Dense(self.input_dim, activation=None, kernel_initializer=initializer,
                                   use_bias=True, bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                                   bias_regularizer=l2_regularizer, name="layer_output")(hidden)
        # Create the core model
        core_model = keras.Model(inputs=[input_], outputs=[output_], name="Direct_ResNet")
        core_model.summary()
        # build model
        model = EntropyModel(core_model, polynomial_degree=self.poly_degree, spatial_dimension=self.spatial_dim,
                             reconstruct_u=bool(self.loss_weights[2]),
                             scale_factor=(self.scaler_max - self.scaler_min) / 2.0,
                             name="entropy_wrapper")

        batch_size = 3  # dummy entry
        model.build(input_shape=(batch_size, self.input_dim))

        model.compile(
            loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': MonotonicFunctionLoss(),
                  'output_3': tf.keras.losses.MeanSquaredError(), 'output_4': tf.keras.losses.MeanSquaredError()},
            loss_weights={'output_1': self.loss_weights[0], 'output_2': self.loss_weights[1],
                          'output_3': self.loss_weights[2], 'output_4': self.loss_weights[2]},
            optimizer=self.optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

        # model.summary()
        # tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True,
        # show_layer_names = True, rankdir = 'TB', expand_nested = True)
        self.model = model
        return True

    def call_training(self, val_split=0.1, epoch_size=2, batch_size=128, verbosity_mode=1, callback_list=[]):
        '''
        Calls training depending on the MK model
        '''
        x_data = self.training_data[0]
        # print(self.model(self.training_data[1])[2] - tf.constant(self.training_data[0])) #sanity check
        y_data = [tf.constant(self.training_data[1], dtype=tf.float32),
                  tf.constant(self.training_data[0], dtype=tf.float32),
                  tf.constant(self.training_data[0], dtype=tf.float32),
                  tf.constant(self.training_data[2], dtype=tf.float32)]

        self.model.fit(x=x_data, y=y_data, validation_split=val_split, epochs=epoch_size,
                       batch_size=batch_size, verbose=verbosity_mode, callbacks=callback_list, shuffle=True)

        return self.history

    def select_training_data(self):
        return [True, True, True]

    def training_data_preprocessing(self, scaled_output: bool = False) -> bool:
        """
        Performs a scaling on the output data alpha. Sets a scale factor for the
        reconstruction of u during training and execution
        """
        self.scaler_max = 1.0
        self.scaler_min = 0.0
        # self.training_data.append(self.training_data[1])
        if scaled_output:
            new_len = self.training_data[1].shape[0] * self.training_data[1].shape[1]
            output = np.reshape(self.training_data[1], (new_len, 1))
            scaler = MinMaxScaler()
            scaler.fit(output)
            self.scaler_max = float(scaler.data_max_)
            self.scaler_min = float(scaler.data_min_)
            self.training_data[1] = 2 * self.training_data[1] / (self.scaler_max - self.scaler_min)
        print(
            "Output of network has internal scaling with alpha_max= " + str(self.scaler_max) + " and alpha_min= " + str(
                self.scaler_min))
        print("New alpha_min= " + str(self.training_data[1].min()) + ". New alpha_max= " + str(
            self.training_data[1].max()))
        return True

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
        alpha_predicted = self.model(u_reduced)
        alpha_complete_predicted = self.model.reconstruct_alpha(alpha_predicted)
        u_complete_reconstructed = self.model.reconstruct_u(alpha_complete_predicted)

        return [u_complete_reconstructed, alpha_complete_predicted]
