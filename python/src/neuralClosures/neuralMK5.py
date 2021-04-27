'''
Derived network class "MK4" for the neural entropy closure.
It features the ICNN approach by Amos et al.
Author: Steffen Schotthöfer
Version: 0.0
Date 17.12.2020
'''
from .neuralBase import neuralBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import Tensor
from tensorflow.keras.constraints import NonNeg


class neuralMK5(neuralBase):
    '''
    MK4 Model: Train u to alpha
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, polyDegree=0, spatialDim=1, folderName="testFolder", lossCombi=0, width=10, depth=5,
                 normalized=False):
        if (folderName == "testFolder"):
            customFolderName = "MK1_N" + str(polyDegree) + "_D" + str(spatialDim)
        else:
            customFolderName = folderName

        super(neuralMK5, self).__init__(normalized, polyDegree, spatialDim, width, depth, lossCombi,
                                        customFolderName)

        self.model = self.createModel()

    def createModel(self):

        layerDim = 16

        # Weight initializer
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        # Number of basis functions used:
        input_ = keras.Input(shape=(self.inputDim,))

        ### Hidden layers ###
        # First Layer is a std dense layer
        hidden = layers.Dense(layerDim, activation="relu",
                              )(input_)
        # other layers are convexLayers
        hidden = layers.Dense(layerDim, activation="relu",
                              )(hidden)
        hidden = layers.Dense(layerDim, activation="relu",
                              )(hidden)
        hidden = layers.Dense(layerDim, activation="relu",
                              )(hidden)
        hidden = layers.Dense(layerDim, activation="relu",
                              )(hidden)

        output_ = layers.Dense(layerDim, activation="relu",
                               )(hidden)  # outputlayer

        # Create the model
        model = keras.Model(inputs=[input_], outputs=[output_], name="ICNN")
        # model.summary()

        # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
        model.compile(loss="mean_squared_error", optimizer=self.opt, metrics=['mean_absolute_error'])

        return model

    def selectTrainingData(self):
        return [True, False, True]
