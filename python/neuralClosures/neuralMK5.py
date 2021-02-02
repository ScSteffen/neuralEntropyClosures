'''
Derived network class "MK5" for the neural entropy closure.
It features the ICNN approach by Amos et al.
Trains on entropy functional
For now, only one basis function!!!
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''
from .neuralBase import neuralBase
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor
from tensorflow.keras.constraints import NonNeg
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import sphericalquadpy as sqp
import csv


class neuralMK5(neuralBase):
    '''
    MK5 Model: Train u to h
    Training data generation: b) read solver data from file and compute h
    Loss function:  MSE between h(u) and real_h(u)
    '''

    def __init__(self, maxDegree_N=0, folderName= "testFolder"):
        if(folderName == "testFolder"):
            tempString = "MK1_N" + str(maxDegree_N)
        else:
            tempString=folderName
        self.maxDegree_N = maxDegree_N
        self.model = self.createModel()
        self.filename = "models/"+ tempString
        self.trainingData = ([0], [0])

    def createModel(self):

        # Model variables
        inputDim = self.getIdxSphericalHarmonics(self.maxDegree_N, self.maxDegree_N) + 1
        layerSize = 3

        # Weight initializer
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        # Define standard Residual block
        def ResNetBlock(x: Tensor) -> Tensor:
            y = keras.layers.Dense(20, activation="relu")(x)
            y = keras.layers.Dense(20, activation="relu")(y)
            y = keras.layers.Dense(20, activation="relu")(y)

            out = keras.layers.Add()([x, y])
            out = keras.layers.ReLU()(out)
            out = keras.layers.BatchNormalization()(out)
            return out

        # Convex Layer
        def ICLayer(layerInput_z: Tensor, netInput_x: Tensor) -> Tensor:
            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = keras.layers.Dense(layerSize, kernel_constraint=NonNeg(), activation=None,
                                                     kernel_initializer=initializerNonNeg,
                                                     use_bias=True,
                                                     bias_initializer='zeros'
                                                     # name='in_z_NN_Dense'
                                                     )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = keras.layers.Dense(layerSize, activation=None,
                                               kernel_initializer=initializer,
                                               use_bias=False
                                               # name='in_x_Dense'
                                               )(netInput_x)
            # Wz+Wx+b
            intermediateSum = keras.layers.Add()([weightedSum_x, weightedNonNegSum_z])
            # activation
            out = tf.keras.activations.softplus(intermediateSum)
            return out

        # Define the input = number of basis functions used
        input_ = keras.Input(shape=(inputDim,))

        # Hidden layers (First layer can be std)
        hidden = keras.layers.Dense(layerSize, activation="relu")(input_)

        # ICCN Layers
        hidden = ICLayer(hidden, input_)
        hidden = ICLayer(hidden, input_)
        hidden = ICLayer(hidden, input_)
        hidden = ICLayer(hidden, input_)

        # Define the ouput
        output_ = keras.layers.Dense(1)(hidden) # models the entropy

        # Create the model
        model = keras.Model(name="MK5_ICCN_onEntropy", inputs=[input_], outputs=[output_])
        model.summary()

        # alternative way of training
        # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mean_absolute_error'])

        return model

    def createTrainingData(self):
        filenameU = self.filename + "/trainingData_M0_u.csv"
        filenameH = self.filename + "/trainingData_M0_h.csv"

        # Load Alpha
        f = open(filenameH, 'r')
        hList = list()
        uList = list()

        # --- Load moments u ---
        with f:
            reader = csv.reader(f)

            for row in reader:
                numRow = []
                for word in row:
                    numRow.append(float(word))

                hList.append(numRow)

        f = open(filenameU, 'r')
        # --- Load entropy values ---
        with f:
            reader = csv.reader(f)

            for row in reader:
                numRow = []
                for word in row:
                    numRow.append(float(word))
                uList.append(numRow)

        self.trainingData = (np.asarray(uList), np.asarray(hList))
