#### MK 4 Networks ####
'''
Exploration of convex Networks on a simple example
It includes the ICNN techniques (Amos et al)
'''

### This is a script for the training of the
### Third NN approach

'''
Improvements:
1)  accepts u as a N-vector
2)  Generalized Loss function
3)  Adapted network layout
4)  RESNet Used as Netowork ( TODO )
'''

import csv
import multiprocessing
import pandas as pd
from joblib import Parallel, delayed

### imports
import numpy as np
# in-project imports
import legacyCode.nnUtils as nnUtils
import csv
# Tensorflow
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import initializers

import matplotlib.pyplot as plt

plt.style.use("kitish")


# ------  Code starts here --------

def main():
    # Training Parameters
    batchSize = 200
    epochCount = 200

    ### Dense Network
    filename = "legacyCode/models/ConvComparison_fcnn"

    model = create_modelMK4()
    model = trainModel(model, filename, batchSize, epochCount)
    # model.load_weights(filename + '/best_model.h5')
    # model = tf.keras.models.load_model(filename + '/model')

    ### Convex Network (nonnegative weights)
    model_nonneg = create_modelMK4_nonneg()
    filename = "legacyCode/models/ConvComparison_nonNeg"
    model_nonneg = trainModel(model_nonneg, filename, batchSize, epochCount)

    # model_nonneg.load_weights(filename + '/best_model.h5')
    # model_nonneg = tf.keras.models.load_model(filename + '/model')

    ### Convex Network ICNN architecture
    model_ICNN = create_modelMK4_ICNN()
    filename = "legacyCode/models/ConvComparison_ICNN"
    model_ICNN = trainModel(model_ICNN, filename, batchSize, epochCount)
    # model_nonneg.load_weights(filename + '/best_model.h5')
    # model_ICNN = tf.keras.models.load_model(filename + '/model')

    evaluateModel(model, model_nonneg, model_ICNN)

    printWeights(model)
    print("----")
    printWeights(model_nonneg)
    return 0


def printWeights(model):
    for layer in model.layers:
        weights = layer.get_weights()  # list of numpy arrays
        print(weights)
        # if weights:
        #    plt.plot(weights)

    # plt.ylabel('weight value')
    # plt.xlabel('weight index')
    # plt.show()

    return 0


def evaluateModel(model, model2, model3):
    x = np.arange(-12, 12, 0.001)

    y = createTrainingData(x)

    predictions = model.predict(x)
    predictions2 = model2.predict(x)
    predictions3 = model3.predict(x)

    plt.plot(x, y)
    plt.plot(x, predictions)
    plt.plot(x, predictions2)
    plt.plot(x, predictions3)

    plt.ylabel('function value')
    plt.xlabel('input value')
    # plt.ylim([30.9,31])
    plt.legend(['quadratic function', 'FCNN', 'naive convex', 'ICNN'])
    plt.show()

    return 0


def trainModel(model, filename, batchSize, epochCount):
    ### 0) Set variables #######################################################

    # Name of modelDirectory
    # filename = "models/Mk4_nnM_1"
    filenameAlpha = "trainingData_M1_alpha.csv"
    filenameU = "trainingData_M1_u.csv"

    ### 1)  Generate Training Data #############################################

    print("Create Training Data")
    # build training data!
    x = np.arange(-1.0, 1.0, 0.0001)
    y = createTrainingData(x)

    ### 2) Create Model ########################################################
    # print("Create Model")

    # Load weights
    # model.load_weights(filename + '/best_model.h5')

    ### 3) Setup Training and Train the model ##################################

    # Create Early Stopping callback
    es = EarlyStopping(monitor='loss', mode='min', min_delta=0.000000001, patience=500,
                       verbose=10)  # loss == custom_loss1dMBPrime by model definition
    mc_best = ModelCheckpoint(filename + '/best_model.h5', monitor='loss', mode='min', save_best_only=True)
    mc_500 = ModelCheckpoint(filename + '/model_quicksave.h5', monitor='loss', mode='min', save_best_only=False,
                             save_freq=500)

    # Train the model
    print("Train Model")
    history = model.fit(x, y, validation_split=0.01, epochs=epochCount, batch_size=batchSize, verbose=1,
                        callbacks=[es, mc_best, mc_500])  # batch size = 900000

    # View History
    # nnUtils.print_history(history.history)

    ### 4) Save trained model and history ########################################

    print("Save model and history")
    nnUtils.save_training(filename, model, history)
    print("Training successfully saved")

    # load history
    history1 = nnUtils.load_trainHistory(filename)
    # print history as a check
    # nnUtils.print_history(history1)

    print("Training Sequence successfully finished")
    return model


### Build the network:
def create_modelMK4():
    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05
    # Number of basis functions used:

    # Weight initializer
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
    #### input layer ####
    input_ = keras.Input(shape=(1,))
    # Hidden layers
    # hidden = layers.BatchNormalization()(input_)
    '''

    hidden = layers.Dense(3,kernel_constraint=NonNeg(), activation="relu")(input_)
    hidden = layers.Dense(3,kernel_constraint=NonNeg(), activation="relu")(hidden)
    hidden = layers.Dense(3, kernel_constraint=NonNeg(), activation="relu")(hidden)

    '''
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones')(input_)
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones'
                          )(hidden)
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones'
                          )(hidden)

    # Define the output
    output_ = layers.Dense(1,
                           kernel_initializer=initializer,
                           bias_initializer='ones'
                           )(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def create_modelMK4_nonneg():
    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05

    # Define LayerDimensions
    layerDim = 3

    # Weight initializer
    initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)

    input_ = keras.Input(shape=(1,))

    # Hidden layers
    # hidden = layers.BatchNormalization()(input_)

    hidden = layers.Dense(layerDim, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(input_)
    hidden = layers.Dense(layerDim, kernel_constraint=NonNeg(), activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(hidden)
    hidden = layers.Dense(layerDim, kernel_constraint=NonNeg(), activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(hidden)

    # Define the ouput
    output_ = layers.Dense(1, kernel_constraint=NonNeg(),
                           kernel_initializer=initializer,
                           bias_initializer='zeros'
                           )(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def create_modelMK4_ICNN():
    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05

    # Define LayerDimensions
    # inputDim = 1
    layerDim = 3

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
        out = tf.keras.activations.softplus(intermediateSum)
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
        out = tf.keras.activations.softplus(intermediateSum)
        # batch normalization
        # out = layers.BatchNormalization()(out)
        return out

    # Number of basis functions used:

    input_ = keras.Input(shape=(1,))

    ### Hidden layers ###
    # First Layer is a std dense layer
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(input_)
    # other layers are convexLayers
    hidden = convexLayer(hidden, input_)
    hidden = convexLayer(hidden, input_)
    output_ = convexLayerOutput(hidden, input_)  # outputlayer

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def createTrainingData(x):
    return x  # 0.5 * x * x


if __name__ == '__main__':
    main()
