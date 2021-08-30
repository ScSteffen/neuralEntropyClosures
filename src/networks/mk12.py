'''
Network class "MK12" for the neural entropy closure.
As MK11, but dense network instead of ICNN
Author: Steffen Schotthöfer
Version: 0.0
Date 09.04.2020
'''
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
import numpy as np

from src.networks.basenetwork import BaseNetwork
from src.networks.custommodels import SobolevModel
from tensorflow.keras.constraints import NonNeg


class MK12Network(BaseNetwork):

    def __init__(self, normalized: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str = ""):
        if save_folder == "":
            custom_folder_name = "MK12_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
        else:
            custom_folder_name = save_folder
        super(MK12Network, self).__init__(normalized=normalized, polynomial_degree=polynomial_degree,
                                          spatial_dimension=spatial_dimension, width=width, depth=depth,
                                          loss_combination=loss_combination, save_folder=custom_folder_name)

    def create_model(self) -> bool:

        layerDim = self.model_width

        # Weight initializer
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        input_stddev: float = np.sqrt(
            (1 / 1.1) * (1 / self.input_dim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        initializer_input = keras.initializers.RandomNormal(mean=0., stddev=input_stddev)
        stddev = np.sqrt(
            (1 / 1.1) * (1 / self.model_width) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        initializer = keras.initializers.RandomNormal(mean=0., stddev=stddev)
        # initializer = tf.keras.initializers.LecunNormal()
        # Weight regularizer
        l1l2Regularizer = tf.keras.regularizers.L1L2(l1=0.001, l2=0.0001)  # L1 + L2 penalties

        ### build the core network with icnn closure architecture ###
        input_ = keras.Input(shape=(self.input_dim,))
        # First Layer is a std dense layer
        hidden = layers.Dense(layerDim, activation="selu",
                              kernel_initializer=initializer_input,
                              kernel_regularizer=l1l2Regularizer,
                              bias_initializer='zeros',
                              name="first_dense"
                              )(input_)
        # other layers are convexLayers
        for idx in range(0, self.model_depth):
            hidden = layers.Dense(self.model_width, activation="selu",
                                  # kernel_constraint=NonNeg(),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=l1l2Regularizer,
                                  bias_initializer='zeros',
                                  name="dense_" + str(idx)
                                  )(hidden)
        output_ = layers.Dense(1, activation="relu",
                               kernel_initializer=initializer,
                               # kernel_regularizer=l1l2Regularizer,
                               bias_initializer='zeros',
                               name="dense_output"
                               )(hidden)  # outputlayer

        # Create the core model
        coreModel = keras.Model(inputs=[input_], outputs=[output_], name="Icnn_closure")

        # build model
        model = SobolevModel(coreModel, polynomial_degree=self.poly_degree, spatial_dimension=self.spatial_dim,
                             reconstruct_u=bool(self.loss_weights[2]), scale_factor=self.scaler_max - self.scaler_min,
                             name="sobolev_icnn_wrapper")

        batchSize = 2  # dummy entry
        model.build(input_shape=(batchSize, self.input_dim))

        # model.compile(loss=tf.keras.losses.MeanSquaredError(),
        #              # loss={'output_1': tf.keras.losses.MeanSquaredError()},
        #              # loss_weights={'output_1': 1, 'output_2': 0},
        #              optimizer='adam',
        #              metrics=['mean_absolute_error', 'mean_squared_error'])
        model.compile(
            loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError(),
                  'output_3': tf.keras.losses.MeanSquaredError()},
            loss_weights={'output_1': self.loss_weights[0], 'output_2': self.loss_weights[1],
                          'output_3': self.loss_weights[2]},  # , 'output_4': self.lossWeights[3]},
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
        y_data = [self.training_data[2], self.training_data[1], self.training_data[0]]  # , self.trainingData[1]]
        self.model.fit(x=x_data, y=y_data,
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
        [h_predicted, alpha_predicted] = self.model(u_reduced)
        alpha_complete_predicted = self.model.reconstruct_alpha(alpha_predicted)
        u_complete_reconstructed = self.model.reconstruct_u(alpha_complete_predicted)

        return [u_complete_reconstructed, alpha_complete_predicted, h_predicted]
