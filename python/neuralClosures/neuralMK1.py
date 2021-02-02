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

    def __init__(self, maxDegree_N=0, folderName= "testFolder"):
        if(folderName == "testFolder"):
            tempString = "MK1_N" + str(maxDegree_N)
        else:
            tempString=folderName
        self.maxDegree_N = maxDegree_N
        self.model = self.createModel()
        self.filename = "models/"+  tempString
        self.trainingData = ()

    def createModel(self):
        inputDim = self.getIdxSphericalHarmonics(self.maxDegree_N, self.maxDegree_N) + 1
        model = keras.models.Sequential([
            keras.layers.Dense(256, activation='sigmoid', input_shape=(inputDim,)),
            keras.layers.Dense(512, activation='sigmoid'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(256, activation='sigmoid'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(128, activation='sigmoid'),
            keras.layers.Dense(inputDim, )
        ], name="MK1closure")

        model.summary()
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mean_absolute_error'])

        return model

    def selectTrainingData(self):
        if len(self.trainingData) == 0:
            ValueError("Error: Training Data is an empty tuple.")
        if len(self.trainingData) < 3:
            ValueError("Error: Training Data Triple does not have length 3. Must consist of (u, alpha, h).")

        self.trainingData = (self.trainingData[0], self.trainingData[1]) #(u,alpha)

        return 0

    def getIdxSphericalHarmonics(self, k, l):
        # Returns the global idx from spherical harmonics indices
        return l * l + k + l