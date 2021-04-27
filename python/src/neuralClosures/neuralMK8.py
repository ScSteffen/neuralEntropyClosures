'''
Same as MK7, but with relu activation
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


class neuralMK8(neuralBase):
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

        super(neuralMK8, self).__init__(normalized, polyDegree, spatialDim, width, depth, lossCombi,
                                        customFolderName)

        self.model = self.createModel()

    def createModel(self):

        layerDim = 20

        # Weight initializer
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        def convexLayer(layerInput_z: Tensor, netInput_x: Tensor) -> Tensor:
            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = layers.Dense(layerDim, kernel_constraint=NonNeg(), activation=None,
                                               kernel_initializer=initializerNonNeg,
                                               use_bias=True,
                                               bias_initializer='zeros'
                                               # name='in_z_NN_Dense'
                                               )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = layers.Dense(layerDim, activation=None,
                                         kernel_initializer=initializer,
                                         use_bias=False
                                         # name='in_x_Dense'
                                         )(netInput_x)
            # Wz+Wx+b
            intermediateSum = layers.Add()([weightedSum_x, weightedNonNegSum_z])

            # activation
            out = tf.keras.activations.relu(intermediateSum)
            # batch normalization
            # out = layers.BatchNormalization()(out)
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

        # Number of basis functions used:
        input_ = keras.Input(shape=(self.inputDim,))

        ### Hidden layers ###
        # First Layer is a std dense layer
        hidden = layers.Dense(layerDim, activation="relu",
                              kernel_initializer=initializer,
                              bias_initializer='zeros'
                              )(input_)
        # other layers are convexLayers
        hidden = convexLayer(hidden, input_)
        hidden = convexLayer(hidden, input_)
        hidden = convexLayer(hidden, input_)
        hidden = convexLayer(hidden, input_)
        hidden = convexLayer(hidden, input_)
        output_ = convexLayerOutput(hidden, input_)  # outputlayer

        # Create the model
        model = keras.Model(inputs=[input_], outputs=[output_], name="ICNN")
        # model.summary()

        # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
        model.compile(loss="mean_squared_error", optimizer=self.opt, metrics=['mean_absolute_error'])

        return model

    def selectTrainingData(self):
        return [True, False, True]

    def trainingDataPostprocessing(self):
        # find the maximum of u_0
        u0Max = max(self.trainingData[0][:, 0])
        self.trainingData[0] / u0Max
        print("Training Data Scaled")
        return 0
