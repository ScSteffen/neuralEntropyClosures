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


def printDerivative(model, model2, u, alpha, h):
    # x = np.arange(-100.0, 100.0, 0.001)
    # tmp = np.reshape(x,(x.shape[0],1))
    x_model = tf.Variable(u)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(x_model, training=False)  # same as model.predict(x)

    gradients = tape.gradient(predictions, x_model)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions2 = model2(x_model, training=False)  # same as model.predict(x)

    gradients2 = tape.gradient(predictions2, x_model)

    # np.gradient(x_model, predictions)
    # m_n phd student: paris (teddy picard?)

    # Gradient
    # print(grads)

    # plot model predictions and derivatives
    plt.plot(u, predictions)
    plt.plot(u, predictions2, '-.')
    plt.plot(u, h, '--')
    plt.ylabel('entropy')
    plt.xlabel('moment')
    plt.legend(['ICNN', 'std Network', 'Target Fct'])
    # plt.legend(['Model Derivative', 'Target Derivative'])
    # plt.ylim([-15, 20])
    # plt.xlim([0, 50])
    plt.show()

    plt.plot(u, gradients)
    plt.plot(u, gradients2, '-.')
    plt.plot(u, alpha, '--')
    plt.ylabel('lagrangian')
    plt.xlabel('moment')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    plt.legend(['ICNN Derivative', 'std Network Derivative', 'Target Derivative'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0,20])
    # plt.xlim([0,50])
    plt.show()

    integratedGradients = integrate(u, gradients)
    plt.plot(u, integratedGradients)
    plt.plot(u, h, '--')
    plt.ylabel('function value')
    plt.xlabel('input value')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    plt.legend(['Integrated Gradients', 'Target Funktion'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0, 20])
    plt.show()

    finDiff = finiteDiff(u, h)
    plt.plot(u, finDiff)
    plt.plot(u, gradients)
    plt.plot(u, alpha, '--')
    plt.ylabel('function value')
    plt.xlabel('input value')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    plt.legend(['Finite Difference', 'Model Derivative', 'alpha'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0, 50])
    plt.show()

    return gradients


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


def evaluateModel(x, y, model, model2, model3):
    # --- Get Data ----
    u = x[0::3]
    h = y[0::3]

    predictions = model.predict(u)
    predictions2 = model2.predict(u)
    predictions3 = model3.predict(u)

    plt.plot(u, h)
    plt.plot(u, predictions)
    plt.plot(u, predictions3)
    plt.plot(u, predictions2)

    plt.ylabel('function value')
    plt.xlabel('input value')
    # plt.ylim([30.9,31])
    plt.legend(['Entropy', 'FCNN', 'naive convex', 'ICNN'])
    plt.show()

    return 0


def trainModel(model, u, h, filename, batchSize, epochCount):
    ### 0) Set variables #######################################################

    ### 1)  Generate Training Data #############################################

    ### 2) Create Model ########################################################
    print("Create Model")

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
    history = model.fit(u, h, validation_split=0.01, epochs=epochCount, batch_size=batchSize, verbose=1,
                        callbacks=[mc_best])  # batch size = 900000

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


def create_modelMK5():  # Build the network:

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


def create_modelMK5_nonneg():
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
    model = keras.Model(inputs=[input_], outputs=[output_], name="NNNN")
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def create_modelMK5_ICNN():
    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05

    # Define LayerDimensions
    # inputDim = 1
    layerDim = 10

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
        # out = tf.keras.activations.softplus(intermediateSum)
        # batch normalization
        # out = layers.BatchNormalization()(out)
        return intermediateSum

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
    model = keras.Model(inputs=[input_], outputs=[output_], name="ICNN")
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def loadTrainingData_DataGen(filename):
    '''
    :param filename: name of file with training Data generated with KiT-RT framework
    :return: (u,alpha,h) as the training parameters
    '''

    # Load Alpha
    f = open(filename, 'r')
    alphaList = list()
    uList = list()
    hList = list()

    with f:
        reader = csv.reader(f)
        for row in reader:
            numRowU = []
            numRowAlpha = []
            numRowH = []
            word_idx = 0
            for word in row:
                # skip first index, which is date and time info
                if word_idx == 1:  # hardcoded... careful
                    numRowU.append(float(word))
                if word_idx == 2:
                    numRowAlpha.append(float(word))
                if word_idx == 3:
                    numRowH.append(float(word))
                word_idx = word_idx + 1
            uList.append(numRowU)
            alphaList.append(numRowAlpha)
            hList.append(numRowH)

    # print("Data loaded!")
    # return (np.asarray(uList),np.asarray(alphaList), np.asarray(hList))

    t = np.asarray(uList)
    xIn = list(np.arange(-3, 3, 0.001))
    # tmp = np.reshape(xIn, (xIn.shape[0], 1))
    y = []
    dy = []
    x = []
    for (xItem) in xIn:
        x.append([xItem])
        y.append([0.5 * xItem * xItem - 1])
        dy.append([xItem])
    xArr = np.asarray(x)
    return (np.asarray(x), np.asarray(dy), np.asarray(y),)


def loadTrainingData(filenameU, filenameAlpha, filenameH):
    hList = list()
    uList = list()
    alphaList = list()

    # --- Load moments u ---
    f = open(filenameU, 'r')
    with f:
        reader = csv.reader(f)

        for row in reader:
            numRow = []
            for word in row:
                numRow.append(float(word))
            uList.append(numRow)
    f.close()
    # --- Load entropy values ---
    f = open(filenameH, 'r')
    with f:
        reader = csv.reader(f)

        for row in reader:
            numRow = []
            for word in row:
                numRow.append(float(word))

            hList.append(numRow)
    f.close()
    # --- Load alpha values ---
    f = open(filenameAlpha, 'r')
    with f:
        reader = csv.reader(f)

        for row in reader:
            numRow = []
            for word in row:
                numRow.append(float(word))

            alphaList.append(numRow)
    f.close()

    return (np.asarray(uList), np.asarray(alphaList), np.asarray(hList))


def cleanTrainingData(u, alpha, h):
    # brief: removes unrealistic values of the training set. sorts training set
    # input: uTrain.shape     = (setSize, basisSize) Moment vector
    #        alphaTrain.shape = (setSize, basisSize) lagrange multilpier vector
    #        hTrain.shape     = (setSize, 1) entropy
    # output: uTrain.shape     = (setSize, basisSize) Moment vector
    #         alphaTrain.shape = (setSize, basisSize) lagrange multilpier vector
    #         hTrain.shape     = (setSize, 1) entropy

    setSize = u.shape[0]

    def entryMarker(idx):
        # --- mark entries, that should be deleted ---
        keepEntry = True

        # check if first moment is too big ==> unrealistic value
        if (u[idx, 0] > 7):
            keepEntry = False

        if (u[idx, 0] < 4):
            keepEntry = False

        if (idx % 100000 == 0):  # Progress info
            print("Status: {:.2f} percent".format(idx / setSize * 100))

        return keepEntry

    # --- parallelize data generation ---
    num_cores = multiprocessing.cpu_count()
    print("Starting data cleanup using " + str(num_cores) + " cores")
    deletionList = Parallel(n_jobs=num_cores)(
        delayed(entryMarker)(i) for i in range(0, setSize))  # (u,  h)

    # --- delete entries ---
    u = u[deletionList]
    h = h[deletionList]
    alpha = alpha[deletionList]

    # --- sort remaining entries
    zipped_lists = zip(u, alpha, h)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    listU, listAlpha, listH = [list(tuple) for tuple in tuples]

    return (np.asarray(listU), np.asarray(listAlpha), np.asarray(listH))


def storeTrainingData(u, alpha, h, filenameU, filenameAlpha, filenameH):
    # store u
    f = open(filenameU, 'w')
    with f:
        writer = csv.writer(f)
        for row in u:
            writer.writerow(row)
    f.close()
    # store alpha
    f = open(filenameAlpha, 'w')
    with f:
        writer = csv.writer(f)
        for row in alpha:
            writer.writerow(row)
    f.close()
    # store h
    f = open(filenameH, 'w')
    with f:
        writer = csv.writer(f)
        for row in h:
            writer.writerow(row)
    f.close()
    return 0


if __name__ == '__main__':
    main()
