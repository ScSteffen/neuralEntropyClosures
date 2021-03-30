'''
This is the script that gets called from the C++ KiT-RT method MLOptimizer.cpp
It initializes and loads a neural Closure
The call method performs a prediction
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''

### imports ###
from src.neuralClosures.configModel import initNeuralClosure
import numpy as np
import tensorflow as tf
import os
import pandas as pd

from optparse import OptionParser

### global variable ###
neuralClosureModel = 0  # bm.initNeuralClosure(0,0)


### function definitions ###
def initModelCpp(input):
    '''
    input: string array consisting of [modelNumber,maxDegree_N, folderName]
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    folderName: Path to the folder containing the neural network model
    '''

    print("|-------------------- Tensorflow initialization Log ------------------")
    print("|")

    modelNumber = input[0]
    maxDegree_N = input[1]

    # --- Transcribe the modelNumber and MaxDegree to the correct model folder --- #
    folderName = "neuralClosure_M" + str(maxDegree_N) + "_MK" + str(modelNumber)

    global neuralClosureModel
    neuralClosureModel = initNeuralClosure(modelNumber, maxDegree_N, folderName)
    neuralClosureModel.loadModel()
    neuralClosureModel.model.summary()
    print("|")
    print("| Tensorflow neural closure initialized.")
    print("|")
    return 0


### function definitions ###
def initModel(modelNumber=1, polyDegree=0, spatialDim=3, folderName="testFolder", optimizer='adam', width=10, height=5):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    global neuralClosureModel
    neuralClosureModel = initNeuralClosure(modelNumber, polyDegree, spatialDim, folderName, optimizer, width, height)

    return 0


def callNetwork(input):
    '''
    # Input: input.shape = (nCells,nMaxMoment), nMaxMoment = 9 in case of MK3
    # Output: Gradient of the network wrt input
    '''
    # predictions = neuralClosureModel.model.predict(input)

    x_model = tf.Variable(input)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = neuralClosureModel.model(x_model, training=False)  # same as neuralClosureModel.model.predict(x)

    gradients = tape.gradient(predictions, x_model)

    return gradients


def callNetworkBatchwise(inputNetwork):
    # Transform npArray to tfEagerTensor
    x_model = tf.Variable(inputNetwork)

    # Compute Autodiff tape
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = neuralClosureModel.model(x_model, training=False)  # same as model.predict(x)

    # Compute the gradients
    gradients = tape.gradient(predictions, x_model)

    # ---- Convert gradients from eagerTensor to numpy array and then to flattened c array ----

    # Note: Use inputNetwork as array, since a newly generated npArray seems to cause a Segfault in cpp
    (dimCell, dimBase) = inputNetwork.shape

    for i in range(0, dimCell):
        for j in range(0, dimBase):
            inputNetwork[i, j] = gradients[i, j]

    return inputNetwork


def main():
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-b", "--batch", dest="batch", default=1000,
                      help="batch size", metavar="BATCH")
    parser.add_option("-c", "--epochChunk", dest="epochchunk", default=1,
                      help="number of epoch chunks", metavar="EPOCHCHUNK")
    parser.add_option("-d", "--degree", dest="degree", default=0,
                      help="max degree of moment", metavar="DEGREE")
    parser.add_option("-e", "--epoch", dest="epoch", default=1000,
                      help="epoch count for neural network", metavar="EPOCH")
    parser.add_option("-f", "--folder", dest="folder", default="testFolder",
                      help="folder where the model is stored", metavar="FOLDER")
    parser.add_option("-l", "--loadModel", dest="loadmodel", default=1,
                      help="load model weights from file", metavar="LOADING")
    parser.add_option("-m", "--model", dest="model", default=1,
                      help="choice of network model", metavar="MODEL")
    parser.add_option("-n", "--normalized", dest="normalized", default=0,
                      help="train on normalized moments", metavar="NORMALIZED")
    parser.add_option("-o", "--optimizer", dest="optimizer", default="Adam",
                      help="optimizer choice", metavar="OPTIMIZER")
    parser.add_option("-p", "--processingmode", dest="processingmode", default=1,
                      help="gpu mode (1). cpu mode (0) ", metavar="PROCESSINGMODE")
    parser.add_option("-s", "--spatialDimension", dest="spatialDimension", default=3,
                      help="spatial dimension of closure", metavar="SPATIALDIM")
    parser.add_option("-t", "--training", dest="training", default=1,
                      help="training mode (1) execution mode (0)", metavar="TRAINING")
    parser.add_option("-v", "--verbosity", dest="verbosity", default=1,
                      help="output verbosity keras (0 or 1)", metavar="VERBOSITY")
    parser.add_option("-w", "--networkwidth", dest="networkwidth", default=10,
                      help="width of each network layer", metavar="WIDTH")
    parser.add_option("-x", "--networkheight", dest="networkheight", default=5,
                      help="height of the network", metavar="HEIGHT")

    (options, args) = parser.parse_args()
    options.degree = int(options.degree)
    options.spatialDimension = int(options.spatialDimension)
    options.model = int(options.model)
    options.epoch = int(options.epoch)
    options.epochchunk = int(options.epochchunk)
    options.batch = int(options.batch)
    options.verbosity = int(options.verbosity)
    options.loadmodel = int(options.loadmodel)
    options.training = int(options.training)
    options.processingmode = int(options.processingmode)
    options.normalized = int(options.normalized)
    options.networkwidth = int(options.networkwidth)
    options.networkheight = int(options.networkheight)

    # --- End Option Parsing ---

    # witch to CPU mode, if wished
    if options.processingmode == 0:
        # Set CPU as available physical device
        # Set CPU as available physical device
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if tf.test.gpu_device_name():
            print('GPU found. Using GPU')
        else:
            print("Disabled GPU. Using CPU")

    # --- initialize model
    print("Initialize model")
    initModel(modelNumber=options.model, polyDegree=options.degree, spatialDim=options.spatialDimension,
              folderName=options.folder,
              optimizer=options.optimizer, width=options.networkwidth, height=options.networkheight)
    neuralClosureModel.model.summary()

    # Print chosen options to file
    d = {'degree': [options.degree],
         'spatial Dimension': [options.spatialDimension],
         'model': [options.model],
         'epoch': [options.epoch],
         'epochChunk': [options.epochchunk],
         'batchsize': [options.batch],
         'verbosity': [options.verbosity],
         'loadmodel': [options.loadmodel],
         'training': [options.training],
         'folder': [options.folder],
         'optimizer': [options.optimizer],
         'processingmode': [options.processingmode],
         'normalized moments': [options.normalized]}

    df = pd.DataFrame(data=d)
    count = 0
    cfgFile = neuralClosureModel.filename + '/config_001_'
    while os.path.isfile(cfgFile + '.csv'):
        count += 1
        cfgFile = neuralClosureModel.filename + '/config_' + str(count).zfill(3) + '_'

    cfgFile = cfgFile + '.csv'

    print("Writing config to " + cfgFile)

    df.to_csv(cfgFile, index=False)

    if (options.loadmodel == 1 or options.training == 0):
        # in execution mode the model must be loaded.
        # load model weights
        neuralClosureModel.loadModel()
    else:
        print("Start training with new weights")

    if (options.training == 1):
        # create training Data
        neuralClosureModel.loadTrainingData(normalizedMoments=options.normalized)
        # train model
        neuralClosureModel.trainModel(valSplit=0.01, epochCount=options.epoch, epochChunks=options.epochchunk,
                                      batchSize=options.batch, verbosity=options.verbosity,
                                      processingMode=options.processingmode)
        # save model
        neuralClosureModel.saveModel()

    # --- in execution mode,  callNetwork or callNetworkBatchwise get called from c++ directly ---
    return 0


if __name__ == '__main__':
    main()
