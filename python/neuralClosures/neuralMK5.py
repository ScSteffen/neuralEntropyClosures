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

    def __init__(self, maxDegree_N=0, folderName= "testFolder"):
        if(folderName == "testFolder"):
            tempString = "MK5_N" + str(maxDegree_N)
        else:
            tempString=folderName
        self.maxDegree_N = maxDegree_N
        # --- Determine inputDim by MaxDegree ---
        if (self.maxDegree_N == 0):
            self.inputDim = 1
        elif (self.maxDegree_N == 1):
            self.inputDim = 4
        else:
           raise ValueError("Polynomial degeree higher than 1 not supported atm")

        self.model = self.createModel()
        self.filename = "models/"+ tempString

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
        #model.summary()

        # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
        model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])

        return model

    def selectTrainingData(self):
        if len(self.trainingData) == 0:
            ValueError("Error: Training Data is an empty tuple.")
        if len(self.trainingData) < 3:
            ValueError("Error: Training Data Triple does not have length 3. Must consist of (u, alpha, h).")

        self.trainingData = [self.trainingData[0], self.trainingData[2]] # (u,h)

        return 0
