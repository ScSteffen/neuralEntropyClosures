'''
Derived network class "MK2" for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''
from .neuralBase import neuralBase
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras


class neuralMK2(neuralBase):
    '''
    MK2 Model: Train u to alpha
    Training data generation: c) sample u, train on entropy functional
    Loss function: Entropy functional derivative is loss
    '''

    def __init__(self, maxDegree_N=0, folderName= "testFolder"):
        if(folderName == "testFolder"):
            tempString = "MK1_N" + str(maxDegree_N)
        else:
            tempString=folderName
        self.maxDegree_N = maxDegree_N
        self.model = self.createModel()
        self.filename = "models/"+ tempString
        self.trainingData = ()

    def createModel(self):
        # Define the input
        input_ = keras.Input(shape=(1,))

        # Hidden layers
        hidden1 = keras.layers.Dense(4, activation="tanh")(input_)
        hidden2 = keras.layers.Dense(8, activation="tanh")(hidden1)
        hidden3 = keras.layers.Dense(32, activation="tanh")(hidden2)
        hidden4 = keras.layers.Dense(8, activation="tanh")(hidden3)
        hidden5 = keras.layers.Dense(4, activation="tanh")(hidden4)

        # Define the ouput
        output_ = keras.layers.Dense(1)(hidden5)

        # Create the model
        model = keras.Model(name="MK2closure", inputs=[input_], outputs=[output_])
        model.summary()

        # tf.keras.losses.MeanSquaredError()
        # custom_loss1d
        model.compile(loss=self.custom_loss1dMBPrime(), optimizer='adam')
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mean_absolute_error'])

        return model

    def selectTrainingData(self):
        if len(self.trainingData) == 0:
            ValueError("Error: Training Data is an empty tuple.")
        if len(self.trainingData) < 3:
            ValueError("Error: Training Data Triple does not have length 3. Must consist of (u, alpha, h).")

        self.trainingData = [self.trainingData[0], self.trainingData[0]] # (u,u)

        return 0

    ### helper functions

    # Custom Loss
    def custom_loss1dMB(self, u_input, alpha_pred):  # (label,prediciton)
        return 4 * math.pi * tf.math.exp(alpha_pred * np.sqrt(1 / (4 * np.pi))) - alpha_pred * u_input

    # Custom Loss
    def custom_loss1dMBPrime(self):  # (label,prediciton)
        def loss(u_input, alpha_pred):
            return 0.5 * tf.square(
                4 * math.pi * np.sqrt(1 / (4 * np.pi)) * tf.math.exp(alpha_pred * np.sqrt(1 / (4 * np.pi))) - u_input)

        return loss
