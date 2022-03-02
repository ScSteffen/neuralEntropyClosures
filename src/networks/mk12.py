'''
Network class "MK12" for the neural entropy closure.
As MK11, but resnet network instead of ICNN
Author: Steffen Schotthöfer
Version: 0.0
Date 09.01.2022
'''
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers

from src.networks.basenetwork import BaseNetwork
from src.networks.entropymodels import SobolevModel
from src.networks.customlayers import MeanShiftLayer, DecorrelationLayer


class MK12Network(BaseNetwork):

    def __init__(self, normalized: bool, input_decorrelation: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str = "", scale_active: bool = True,
                 gamma_lvl: int = 0):
        if save_folder == "":
            custom_folder_name = "MK12_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
        else:
            custom_folder_name = save_folder
        super(MK12Network, self).__init__(normalized=normalized, polynomial_degree=polynomial_degree,
                                          spatial_dimension=spatial_dimension, width=width, depth=depth,
                                          loss_combination=loss_combination, save_folder=custom_folder_name,
                                          input_decorrelation=input_decorrelation, scale_active=scale_active,
                                          gamma_lvl=gamma_lvl)

    def create_model(self) -> bool:

        # Weight initializer
        initializer = keras.initializers.LecunNormal()

        # Weight regularizer
        l2_regularizer = tf.keras.regularizers.L2(l2=0.0001)  # L1 + L2 penalties

        ### build the core network ###

        # Define Residual block
        def residual_block(x: tf.Tensor, layer_dim: int = 10, layer_idx: int = 0) -> tf.Tensor:
            # ResNet architecture by https://arxiv.org/abs/1603.05027
            y = keras.layers.BatchNormalization()(x)  # 1) BN that normalizes each feature individually (axis=-1)
            y = keras.activations.relu(y)  # 2) activation
            # 3) layer without activation
            y = layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                             bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                             bias_regularizer=l2_regularizer, name="block_" + str(layer_idx) + "_layer_0")(y)
            y = keras.layers.BatchNormalization()(y)  # 4) BN that normalizes each feature individually (axis=-1)
            y = keras.activations.relu(y)  # 5) activation
            # 6) layer
            y = layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                             bias_initializer=initializer, kernel_regularizer=l2_regularizer,
                             bias_regularizer=l2_regularizer, name="block_" + str(layer_idx) + "_layer_1")(y)
            # 7) add skip connection
            out = keras.layers.Add()([x, y])
            return out

        ### build the core network with icnn closure architecture ###
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
        # if self.scale_active:
        #    output_ = layers.Dense(1, activation="relu", kernel_initializer=initializer, name="dense_output",
        #                           kernel_regularizer=l2_regularizer, bias_initializer='zeros')(hidden)
        # else:
        output_ = layers.Dense(1, activation=None, kernel_initializer=initializer, name="dense_output",
                               kernel_regularizer=l2_regularizer, bias_initializer='zeros')(hidden)  # outputlayer
        # Create the core model
        core_model = keras.Model(inputs=[input_], outputs=[output_], name="ResNet_entropy_closure")

        print("The core model overview")
        core_model.summary()
        print("The sobolev wrapped model overview")

        # build model
        model = SobolevModel(core_model, polynomial_degree=self.poly_degree, spatial_dimension=self.spatial_dim,
                             reconstruct_u=bool(self.loss_weights[2]), scaler_max=self.scaler_max,
                             scaler_min=self.scaler_min, scale_active=self.scale_active,
                             gamma=self.regularization_gamma, name="sobolev_resnet_wrapper")

        # build graph
        batch_size: int = 3  # dummy entry
        model.build(input_shape=(batch_size, self.input_dim))

        model.compile(
            loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError(),
                  'output_3': tf.keras.losses.MeanSquaredError()},
            loss_weights={'output_1': self.loss_weights[0], 'output_2': self.loss_weights[1],
                          'output_3': self.loss_weights[2]},  # , 'output_4': self.lossWeights[3]},
            optimizer=self.optimizer, metrics=['mean_absolute_error', 'mean_squared_error'])

        self.model = model
        return True

    def call_training(self, val_split=0.1, epoch_size=2, batch_size=128, verbosity_mode=1, callback_list=[]):
        '''
        Calls training depending on the MK model
        '''
        # u_in = self.training_data[0][:10]
        # alpha_in = self.training_data[1][:10]
        # h_in = self.training_data[2][:10]
        # alpha_test = [[-0.391], [-0.2248], [1.9629]]
        # u_test = [[-0.39], [-0.2248], [1.962]]
        # [h, alpha, u] = self.model(alpha_test)

        x_data = self.training_data[0]
        y_data = [self.training_data[2], self.training_data[1], self.training_data[0]]  # , self.trainingData[1]]
        self.model.fit(x=x_data, y=y_data, validation_split=val_split, epochs=epoch_size, batch_size=batch_size,
                       verbose=verbosity_mode, callbacks=callback_list, shuffle=True)
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
        [h_predicted, alpha_predicted] = self.model(u_reduced)
        alpha_complete_predicted = self.model.reconstruct_alpha(alpha_predicted)
        u_complete_reconstructed = self.model.reconstruct_u(alpha_complete_predicted)

        return [u_complete_reconstructed, alpha_complete_predicted, h_predicted]
