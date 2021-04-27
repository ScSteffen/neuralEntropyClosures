'''
Derived network class "MK1" for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''
from .neuralBase import neuralBase
import tensorflow as tf
from tensorflow import keras


class neuralMK1(neuralBase):
    '''
    MK1 Model: Train u to alpha
    Training data generation: b) read solver data from file
    Loss function:  MSE between alpha and real_alpha
    '''

    def __init__(self, polyDegree=0, spatialDim=1, folderName="testFolder", lossCombi=0, width=10, depth=5,
                 normalized=False):
        if (folderName == "testFolder"):
            customFolderName = "MK1_N" + str(polyDegree) + "_D" + str(spatialDim)
        else:
            customFolderName = folderName

        super(neuralMK1, self).__init__(normalized, polyDegree, spatialDim, width, depth, lossCombi,
                                        customFolderName)

        self.model = self.createModel()

    def createModel(self):
        # inputDim = self.getIdxSphericalHarmonics(self.polyDegree, self.polyDegree) + 1

        model = keras.models.Sequential([
            keras.layers.Dense(256, activation='sigmoid', input_shape=(self.inputDim,)),
            keras.layers.Dense(512, activation='sigmoid'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(256, activation='sigmoid'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='sigmoid'),
            keras.layers.Dense(self.inputDim, )
        ], name="MK1closure")

        model.summary()
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.opt, metrics=['mean_absolute_error'])

        return model

    def selectTrainingData(self):
        return [True, True, False]

    def getIdxSphericalHarmonics(self, k, l):
        # Returns the global idx from spherical harmonics indices
        return l * l + k + l
