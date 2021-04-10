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
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
from tensorflow import Tensor


class neuralMK11(neuralBase):
    '''
    MK4 Model: Train u to h and alpha
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, polyDegree=0, spatialDim=0, folderName="testFolder", optimizer='adam', width=10, depth=5,
                 normalized=False):
        if (folderName == "testFolder"):
            tempString = "MK11_N" + str(polyDegree) + "_D" + str(spatialDim)
        else:
            tempString = folderName

        self.polyDegree = polyDegree
        self.spatialDim = spatialDim
        self.modelWidth = width
        self.modelDepth = depth

        # --- Determine inputDim by MaxDegree ---
        if (spatialDim == 1):
            self.inputDim = polyDegree + 1
        elif (spatialDim == 3):
            if (self.polyDegree == 0):
                self.inputDim = 1
            elif (self.polyDegree == 1):
                self.inputDim = 4
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        elif (spatialDim == 2):
            if (self.polyDegree == 0):
                self.inputDim = 1
            elif (self.polyDegree == 1):
                self.inputDim = 3
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        else:
            raise ValueError("Saptial dimension other than 1,2 or 3 not supported atm")

        if normalized:
            self.inputDim = self.inputDim - 1

        self.opt = optimizer
        self.filename = "models/" + tempString

        self.model = self.createModel()

    def createModel(self):

        layerDim = self.modelWidth

        # Weight initializer
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        def convexLayer(layerInput_z: Tensor, netInput_x: Tensor, layerIdx=0) -> Tensor:
            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = layers.Dense(layerDim, kernel_constraint=NonNeg(), activation=None,
                                               kernel_initializer=initializerNonNeg,
                                               use_bias=True, bias_initializer='zeros',
                                               name='non_neg_component_' + str(layerIdx)
                                               )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = layers.Dense(layerDim, activation=None,
                                         kernel_initializer=initializer,
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
            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = layers.Dense(1, kernel_constraint=NonNeg(), activation=None,
                                               kernel_initializer=initializerNonNeg,
                                               use_bias=True,
                                               bias_initializer='zeros'
                                               # name='in_z_NN_Dense'
                                               )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = layers.Dense(1, activation=None,
                                         kernel_initializer=initializer,
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
                              kernel_initializer=initializer,
                              bias_initializer='zeros',
                              name="first_dense"
                              )(input_)
        # other layers are convexLayers
        for idx in range(0, self.modelDepth):
            hidden = convexLayer(hidden, input_, layerIdx=idx)
        output_ = convexLayerOutput(hidden, input_)  # outputlayer

        # Create the core model
        coreModel = keras.Model(inputs=[input_], outputs=[output_], name="Icnn_closure")

        # build model
        model = sobolevModel(coreModel, name="sobolev_icnn_wrapper")

        batchSize = 2  # dummy entry
        model.build(input_shape=(batchSize, self.inputDim))

        model.compile(
            loss='mean_absolute_error',
            optimizer='adam',
            metrics=['mean_absolute_error', 'mean_squared_error'])
        # model.compile(
        #    loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError()},
        #    loss_weights={'output_1': 1, 'output_2': 1},
        #    optimizer='adam',
        #    metrics=['mean_absolute_error'])

        # model.summary()

        # tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True,
        # show_layer_names = True, rankdir = 'TB', expand_nested = True)

        return model

    def trainModel(self, valSplit=0.1, epochCount=2, epochChunks=1, batchSize=500, verbosity=1, processingMode=0):
        '''
        Method to train network
        '''

        # Create callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.filename + '/best_model.h5', monitor='loss', mode='min',
                                                     save_best_only=True,
                                                     verbose=verbosity)  # , save_weights_only = True, save_freq = 50, verbose=0)

        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.0001, patience=10,
                                              verbose=1)
        # mc_checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=self.filename + '/model_saved',
        #                                         save_weights_only=False,
        #                                         verbose=1)

        # Split Training epochs
        miniEpoch = int(epochCount / epochChunks)

        for i in range(0, epochChunks):
            #  perform a batch doublication every 1/10th of the epoch count
            print("Current Batch Size: " + str(batchSize))

            # assemble callbacks
            callbackList = []
            csv_logger = self.createCSVLoggerCallback()
            if verbosity == 1:
                callbackList = [mc_best, csv_logger]
            else:
                callbackList = [mc_best, LossAndErrorPrintingCallback(), csv_logger]

            # start Training
            h = self.trainingData[2]
            alpha = self.trainingData[1]
            # u = self.trainingData[0]
            # trainDataY =          net_out = tf.stack([h, alpha], axis=1)[:, :, 0]

            self.history = self.model.fit(x=self.trainingData[0], y=[self.trainingData[2], self.trainingData[1]],
                                          validation_split=valSplit,
                                          epochs=miniEpoch,
                                          batch_size=batchSize,
                                          verbose=verbosity,
                                          callbacks=callbackList,
                                          )
            batchSize = 2 * batchSize

        self.concatHistoryFiles()

        return self.history

    def selectTrainingData(self):
        return [True, True, True]

    def trainingDataPostprocessing(self):
        return 0


class sobolevModel(tf.keras.Model):
    # Sobolev implies, that the model outputs also its derivative
    def __init__(self, coreModel, **opts):
        # tf.keras.backend.set_floatx('float64')  # Full precision training
        super(sobolevModel, self).__init__()

        # Member is only the model we want to wrap with sobolev execution
        self.coreModel = coreModel  # must be a compiled tensorflow model

    def call(self, x, training=False):
        """
        Defines the sobolev execution
        """

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.coreModel(x)
        derivativeNet = grad_tape.gradient(y, x)

        return [y, derivativeNet]
