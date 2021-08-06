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

from src.networks.basenetwork import BaseNetwork
from src.networks.sobolevmodel import SobolevModel


class MK12Network(BaseNetwork):

    def __init__(self, scaled_output: bool, normalized: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str = ""):
        if save_folder == "":
            custom_folder_name = "MK12_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
        else:
            custom_folder_name = save_folder
        super(MK12Network, self).__init__(normalized=normalized, scaled_output=scaled_output,
                                          polynomial_degree=polynomial_degree,
                                          spatial_dimension=spatial_dimension, width=width, depth=depth,
                                          loss_combination=loss_combination, save_folder=custom_folder_name)
        self.create_model()

    def create_model(self) -> bool:

        layerDim = self.model_width

        # Weight initializer
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
        # Weight regularizer
        l1l2Regularizer = tf.keras.regularizers.L1L2(l1=0.0, l2=0.0)  # L1 + L2 penalties

        ### build the core network with icnn closure architecture ###
        input_ = keras.Input(shape=(self.inputDim,))
        # First Layer is a std dense layer
        hidden = layers.Dense(layerDim, activation="softplus",
                              kernel_initializer=initializer,
                              kernel_regularizer=l1l2Regularizer,
                              bias_initializer='zeros',
                              name="first_dense"
                              )(input_)
        # other layers are convexLayers
        for idx in range(0, self.model_depth):
            hidden = layers.Dense(self.model_width, activation="softplus",
                                  kernel_initializer=initializer,
                                  kernel_regularizer=l1l2Regularizer,
                                  bias_initializer='zeros',
                                  name="dense_" + str(idx)
                                  )(hidden)
        output_ = layers.Dense(1, activation="linear",
                               kernel_initializer=initializer,
                               kernel_regularizer=l1l2Regularizer,
                               bias_initializer='zeros',
                               name="dense_output"
                               )(hidden)  # outputlayer

        # Create the core model
        coreModel = keras.Model(inputs=[input_], outputs=[output_], name="Icnn_closure")

        # build model
        model = SobolevModel(coreModel, polyDegree=self.poly_degree, name="sobolev_icnn_wrapper")

        batchSize = 2  # dummy entry
        model.build(input_shape=(batchSize, self.inputDim))

        # model.compile(loss=tf.keras.losses.MeanSquaredError(),
        #              # loss={'output_1': tf.keras.losses.MeanSquaredError()},
        #              # loss_weights={'output_1': 1, 'output_2': 0},
        #              optimizer='adam',
        #              metrics=['mean_absolute_error', 'mean_squared_error'])
        model.compile(
            loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError()},
            loss_weights={'output_1': 1, 'output_2': 1},
            optimizer='adam',
            metrics=['mean_absolute_error'])

        # model.summary()
        # tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True,
        # show_layer_names = True, rankdir = 'TB', expand_nested = True)
        self.model = model
        return True

    def call_training(self, val_split=0.1, epoch_size=2, batch_size=128, verbosity_mode=1, callback_list=[]):
        '''
        Calls training depending on the MK model
        '''
        xData = self.training_data[0]
        yData = [self.training_data[2], self.training_data[1]]
        self.model.fit(x=xData, y=yData,
                       validation_split=val_split, epochs=epoch_size,
                       batch_size=batch_size, verbose=verbosity_mode,
                       callbacks=callback_list, shuffle=True)
        return self.history

    def select_training_data(self):
        return [True, True, True]

    def training_data_preprocessing(self):
        return 0

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
