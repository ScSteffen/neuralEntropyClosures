### imports
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import initializers
# import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from src.utils import finiteDiff, integrate, loadData

plt.style.use("kitish")


# ------  Code starts here --------

def main():
    # --- Set Parameters ---
    batchSize = 5000
    epochCount = 100000

    filenameFCNN = "legacyCode/models/EntropyConvComparison_fcnn"
    filenameNonNeg = "legacyCode/models/EntropyConvComparison_nonNeg"
    filenameICNN = "legacyCode/models/EntropyConvComparison_ICNN"

    filenameU = "trainingData_M0_u.csv"
    filenameH = "trainingData_M0_h.csv"
    filenameAlpha = "trainingData_M0_alpha.csv"

    # filenameUClean = "trainingData_M0_u_clean.csv"
    # filenameAlphaClean = "trainingData_M0_alpha_clean.csv"
    # filenameHClean = "trainingData_M0_h_clean.csv"

    filenameUCleanTrain = "data/0_stage/trainingData_M0_u_cleanTrain.csv"
    filenameAlphaCleanTrain = "data/0_stage/trainingData_M0_alpha_cleanTrain.csv"
    filenameHCleanTrain = "data/0_stage/trainingData_M0_h_cleanTrain.csv"

    filename = "data/0_stage/DataGen_M0_0_100.csv"

    filenameUCleanTest = "trainingData_M0_u_cleanTest.csv"
    filenameAlphaCleanTest = "trainingData_M0_alpha_cleanTest.csv"
    filenameHCleanTest = "trainingData_M0_h_cleanTest.csv"

    # --- Load and Preprocess Data ---

    print("Load Training Data")
    # (u,h) = loadTrainingData(filenameUCleanTrain,filenameHCleanTrain)
    # (uTest, alphaTest, hTest) = loadTrainingData(filenameUCleanTest, filenameHCleanTest)
    (u, alpha, h) = loadData(filename)
    # (u,alpha,h) = loadTrainingData(filenameUCleanTrain,filenameAlphaCleanTrain,filenameHCleanTrain)
    # print("Clean Training Data")
    # (u,alpha,h) = cleanTrainingData(u,alpha,h)
    # print("Store Cleaned Training Data")
    # storeTrainingData(u,alpha,h,filenameUCleanTrain,filenameAlphaCleanTrain,filenameHCleanTrain)

    # --- Fully Connected Network ---
    model = create_modelMK5()
    # model = tf.keras.models.load_model(filenameFCNN + '/model')
    model.load_weights(filenameFCNN + '/best_model.h5')
    # model = trainModel(model,u,h, filenameFCNN, batchSize, epochCount)

    # --- Convex Network (nonnegative weights) ---
    # model_nonneg = create_modelMK5_nonneg()
    # model_nonneg = tf.keras.models.load_model(filenameNonNeg + '/model')
    # model_nonneg.load_weights(filenameNonNeg + '/best_model.h5')
    # model_nonneg = trainModel(model_nonneg,u,h,filenameNonNeg, batchSize, epochCount)

    # --- Convex Network (ICNN architecture) ---
    model_ICNN = create_modelMK5_ICNN()
    # model_ICNN = tf.keras.models.load_model(filenameICNN + '/model')
    model_ICNN.load_weights(filenameICNN + '/best_model.h5')
    # model_ICNN = trainModel(model_ICNN,u,h, filenameICNN, batchSize, epochCount)

    # --- Model evaluation ---

    # evaluateModel(u,h, model, model_nonneg, model_ICNN)

    # printDerivative(model, u,alpha,h)
    # printDerivative(model_nonneg, u,alpha,h)
    printDerivative(model_ICNN, model, u, alpha, h)

    # printDerivative(model_ICNN)

    # printWeights(model)
    # print("----")
    # printWeights(model_nonneg)
    return 0


def create_model_MK9_poly():  # Build the network:

    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05
    # Number of basis functions used:

    # Weight initializer
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
    #### input layer ####
    input_ = keras.Input(shape=(1,))
    # Hidden layers
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
    '''
    hidden = layers.Dense(10, activation="softplus")(input_)
    hidden = layers.Dense(10, activation="softplus")(hidden)
    hidden = layers.Dense(10, activation="softplus")(hidden)
    hidden = layers.Dense(10, activation="softplus")(hidden)
    output_ = layers.Dense(1, activation=None)(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_], name="FCNN")
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


if __name__ == '__main__':
    main()
