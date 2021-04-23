'''
Accumulation of utility functions
Date: 15.03.2021
Author: Steffen Schotthöfer
'''

import numpy as np
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os


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


def plot1D(x, ys, labels=[], name='defaultName', log=True, linetypes=[], ):
    plt.clf()
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h',
                     'H',
                     '+', 'x', 'D', 'd', '|', '_']
        linetypes = linetypes[0:len(labels)]

    for y, lineType in zip(ys, linetypes):
        plt.plot(x, y, lineType)
    plt.legend(labels)

    if (log):
        plt.yscale('log')

    # plt.show()
    plt.savefig("figures/" + name + ".png", dpi=150)
    print("Figure successfully saved to file: " + str("figures/" + name + ".png"))
    return 0


def shuffleTrainData(x, y, mode="random"):
    c = list(zip(x, y))

    random.shuffle(c)

    x, y = zip(*c)

    return [np.asarray(x), np.asarray(y)]


def writeConfigFile(options, neuralClosureModel):
    # create String to create a python runscript
    runScript = "python callNeuralClosure.py \\  \n"
    runScript = runScript + "--alphasampling " + str(int(options.alphasampling)) + " \\  \n"
    runScript = runScript + "--batch " + str(options.batch) + " \\ \n"
    runScript = runScript + "--epochChunk " + str(options.epochchunk) + "\\  \n"
    runScript = runScript + "--degree " + str(options.degree) + " \\  \n"
    runScript = runScript + "--epoch " + str(options.epoch) + " \\  \n"
    runScript = runScript + "--folder " + str(options.folder) + "\\  \n"
    runScript = runScript + "--loadModel " + str(1) + " \\  \n"  # force to load
    runScript = runScript + "--model " + str(options.model) + " \\  \n"
    runScript = runScript + "--normalized " + str(int(options.normalized)) + " \\  \n"
    runScript = runScript + "--optimizer " + str(options.optimizer) + " \\  \n"
    runScript = runScript + "--processingmode " + str(options.processingmode) + "\\  \n"
    runScript = runScript + "--spatialDimension " + str(options.spatialDimension) + "\\  \n"
    runScript = runScript + "--training " + str(options.training) + " \\  \n"
    runScript = runScript + "--verbosity " + str(options.verbosity) + " \\  \n"
    runScript = runScript + "--networkwidth " + str(options.networkwidth) + " \\  \n"
    runScript = runScript + "--networkdepth " + str(options.networkdepth)

    # Getting filename
    rsFile = neuralClosureModel.filename + '/runScript_001_'
    count = 0

    while os.path.isfile(rsFile + '.sh'):
        count += 1
        rsFile = neuralClosureModel.filename + '/runScript_' + str(count).zfill(3) + '_'

    rsFile = rsFile + '.sh'

    print("Writing config to " + rsFile)
    f = open(rsFile, "w")
    f.write(runScript)
    f.close()

    # Print chosen options to csv
    d = {'alphasampling': [options.alphasampling],
         'batch': [options.batch],
         'epochChunk': [options.epochchunk],
         'degree': [options.degree],
         'epoch': [options.epoch],
         'folder': [options.folder],
         'loadmodel': [options.loadmodel],
         'model': [options.model],
         'normalized moments': [options.normalized],
         'optimizer': [options.optimizer],
         'processingmode': [options.processingmode],
         'spatial Dimension': [options.spatialDimension],
         'verbosity': [options.verbosity],
         'training': [options.training],
         'network width': [options.networkwidth],
         'network depth': [options.networkdepth]}

    df = pd.DataFrame(data=d)
    count = 0
    cfgFile = neuralClosureModel.filename + '/config_001_'

    while os.path.isfile(cfgFile + '.csv'):
        count += 1
        cfgFile = neuralClosureModel.filename + '/config_' + str(count).zfill(3) + '_'

    cfgFile = cfgFile + '.csv'
    df.to_csv(cfgFile, index=False)

    return True
