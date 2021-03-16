'''
Accumulation of utility functions
Date: 15.03.2021
Author: Steffen Schotthöfer
'''

import numpy as np
import time
import pandas as pd
import tensorflow as tf


def finiteDiff(x, y):
    '''
    :param x: Function Argument
    :param y: Function value = f(x)
    :return: df/dx at all points x
    '''

    grad = np.zeros(x.shape)

    grad[0] = (y[1] - y[0]) / (x[1] - x[0])

    for i in range(0, x.shape[0] - 1):
        grad[i + 1] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])

    return grad


def integrate(x, y):
    '''
    :param x: function argument
    :param y: = f(x)
    :return: integrate y over span of x
    '''

    integral = np.zeros(x.shape)

    for i in range(0, x.shape[0] - 1):
        integral[i + 1] = integral[i] + (x[i + 1] - x[i]) * y[i + 1]

    return integral


def loadData(filename, inputDim, selectedCols=[True, True, True]):
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: trainingData = [u,alpha,h]
    '''

    trainingData = []

    print("Loading Data from location: " + filename)
    # determine which cols correspond to u, alpha and h
    uCols = list(range(1, inputDim + 1))
    alphaCols = list(range(inputDim + 1, 2 * inputDim + 1))
    hCol = [2 * inputDim + 1]

    # selectedCols = self.selectTrainingData() #outputs a boolean triple.
    start = time.time()
    if selectedCols[0] == True:
        df = pd.read_csv(filename, usecols=[i for i in uCols])
        uNParray = df.to_numpy()
        trainingData.append(uNParray)
    if selectedCols[1] == True:
        df = pd.read_csv(filename, usecols=[i for i in alphaCols])
        alphaNParray = df.to_numpy()
        trainingData.append(alphaNParray)
    if selectedCols[2] == True:
        df = pd.read_csv(filename, usecols=[i for i in hCol])
        hNParray = df.to_numpy()
        trainingData.append(hNParray)

    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))

    return trainingData


def evaluateModel(model, input):
    '''Evaluates the model at input'''
    # x = input
    # print(x.shape)
    # print(x)
    return model.predict(input)


def evaluateModelDerivative(model, input):
    '''Evaluates model derivatives at input'''

    x_model = tf.Variable(input)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(x_model, training=False)  # same as model.predict(x)

    gradients = tape.gradient(predictions, x_model)
    return gradients


def loadTFModel(filename):
    '''Loads a .h5 file to memory'''
    nn = tf.keras.models.load_model(filename)
    return nn
