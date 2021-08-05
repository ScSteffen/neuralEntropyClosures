'''
This is the script that gets called from the C++ KiT-RT method MLOptimizer.cpp
It initializes and loads a neural Closure
The call method performs a prediction
Author: Steffen Schotthoefer
Version: 0.0
Date 29.10.2020
'''

### imports ###
# internal modules
import numpy as np

from src.neuralClosures.configModel import init_neural_closure
from src import utils

# python modules
import tensorflow as tf
import os
from optparse import OptionParser
import time
import statistics


### global variable ###

# neuralClosureModel = 0  # bm.initNeuralClosure(0,0)


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
    neuralClosureModel = init_neural_closure(modelNumber, maxDegree_N, folderName)
    neuralClosureModel.loadModel()
    neuralClosureModel.model.summary()
    print("|")
    print("| Tensorflow neural closure initialized.")
    print("|")
    return 0


### function definitions ###
def initModel(modelNumber=1, polyDegree=0, spatialDim=3, folderName="testFolder", lossCombi=0, width=10, depth=5,
              normalized=False):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    global neuralClosureModel
    neuralClosureModel = init_neural_closure(network_mk=modelNumber, poly_degree=polyDegree, spatial_dim=spatialDim,
                                             folder_name=folderName, loss_combination=lossCombi, nw_depth=depth,
                                             nw_width=width, normalized=normalized)

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
    parser.add_option("-a", "--alphasampling", dest="alphasampling", default=0,
                      help="uses data sampled in alpha", metavar="ALPHA")
    parser.add_option("-b", "--batch", dest="batch", default=128,
                      help="batch size", metavar="BATCH")
    parser.add_option("-c", "--curriculum", dest="curriculum", default=1,
                      help="training curriculum", metavar="EPOCHCHUNK")
    parser.add_option("-d", "--degree", dest="degree", default=0,
                      help="max degree of moment", metavar="DEGREE")
    parser.add_option("-e", "--epoch", dest="epoch", default=1000,
                      help="epoch count for neural network", metavar="EPOCH")
    parser.add_option("-f", "--folder", dest="folder", default="testFolder",
                      help="folder where the model is stored", metavar="FOLDER")
    parser.add_option("-l", "--loadModel", dest="loadmodel", default=1,
                      help="load model weights from file", metavar="LOADING")
    parser.add_option("-m", "--model", dest="model", default=11,
                      help="choice of network model", metavar="MODEL")
    parser.add_option("-n", "--normalized", dest="normalized", default=0,
                      help="train on normalized moments", metavar="NORMALIZED")
    parser.add_option("-o", "--objective", dest="objective", default=0,
                      help="choice of loss functions:\n 0=[h]\n 1 =[h,alpha]\n 2=[h,alpha,u]\n3=[rel_entropy_h]",
                      metavar="OBJECTIVE")
    parser.add_option("-p", "--processingmode", dest="processingmode", default=1,
                      help="gpu mode (1). cpu mode (0) ", metavar="PROCESSINGMODE")
    parser.add_option("-s", "--spatialDimension", dest="spatialDimension", default=3,
                      help="spatial dimension of closure", metavar="SPATIALDIM")
    parser.add_option("-t", "--training", dest="training", default=1,
                      help="execution mode (0) training mode (1)  analysis mode (2) re-save mode (3)",
                      metavar="TRAINING")
    parser.add_option("-v", "--verbosity", dest="verbosity", default=1,
                      help="output verbosity keras (0 or 1)", metavar="VERBOSITY")
    parser.add_option("-w", "--networkwidth", dest="networkwidth", default=10,
                      help="width of each network layer", metavar="WIDTH")
    parser.add_option("-x", "--networkdepth", dest="networkdepth", default=5,
                      help="height of the network", metavar="HEIGHT")

    (options, args) = parser.parse_args()
    options.objective = int(options.objective)
    options.alphasampling = int(options.alphasampling)
    options.degree = int(options.degree)
    options.spatialDimension = int(options.spatialDimension)
    options.model = int(options.model)
    options.epoch = int(options.epoch)
    options.curriculum = int(options.curriculum)
    options.batch = int(options.batch)
    options.verbosity = int(options.verbosity)
    options.loadmodel = int(options.loadmodel)
    options.training = int(options.training)
    options.processingmode = int(options.processingmode)
    options.normalized = bool(int(options.normalized))
    options.networkwidth = int(options.networkwidth)
    options.networkdepth = int(options.networkdepth)

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
              folderName=options.folder, normalized=options.normalized,
              lossCombi=options.objective, width=options.networkwidth, depth=options.networkdepth)
    neuralClosureModel.model.summary()

    # Save options and runscript to file
    utils.writeConfigFile(options, neuralClosureModel)

    if options.loadmodel == 1 or options.training == 0 or options.training == 2:
        # in execution mode the model must be loaded.
        # load model weights
        neuralClosureModel.loadModel()
    else:
        print("Start training with new weights")

    if options.training == 1:
        # create training Data
        trainingMode = True
        neuralClosureModel.load_training_data(shuffleMode=trainingMode,
                                              alphasampling=options.alphasampling,
                                              normalizedData=neuralClosureModel.normalized)  # normalizedData=False)

        # normalize data (experimental)
        # neuralClosureModel.normalizeData()
        # train model
        neuralClosureModel.config_start_training(valSplit=0.1, epochCount=options.epoch, curriculum=options.curriculum,
                                                 batchSize=options.batch, verbosity=options.verbosity,
                                                 processingMode=options.processingmode)
        # save model
        neuralClosureModel.saveModel()

    elif options.training == 2:
        print("Analysis mode entered.")
        print("Evaluate Model on normalized data...")
        neuralClosureModel.load_training_data(shuffleMode=False, loadAll=True, normalizedData=True)
        [u, alpha, h] = neuralClosureModel.getTrainingData()
        neuralClosureModel.evaluate_model_normalized(u, alpha, h)
        print("Evaluate Model on non-normalized data...")
        neuralClosureModel.load_training_data(shuffleMode=False, loadAll=True, normalizedData=False)
        [u, alpha, h] = neuralClosureModel.getTrainingData()
        neuralClosureModel.evaluateModel(u, alpha, h)
    elif options.training == 3:
        print(
            "Re-Save mode entered.")  # if training was not finished, models are not safed to .pb. this can be done here
        neuralClosureModel.load_training_data(shuffleMode=False,
                                              alphasampling=options.alphasampling,
                                              normalizedData=neuralClosureModel.normalized)

        # normalize data (experimental)
        # neuralClosureModel.normalizeData()
        # train model

        neuralClosureModel.model(neuralClosureModel.trainingData[0])
        # save model
        neuralClosureModel.saveModel()

    elif options.training == 4:
        # timing measurement
        # startup
        u_in = tf.zeros([2, neuralClosureModel.inputDim], tf.float32)
        [u, alpha, h] = neuralClosureModel.model(u_in)

        u_in = tf.ones([1000000, neuralClosureModel.inputDim], tf.float32)

        # u_tf = tf.constant(u_in)
        totduration = 0
        durations = []
        for i in range(0, 100):
            print("Start computation")
            start = time.perf_counter()
            [u, alpha, h] = neuralClosureModel.model(u_in)
            end = time.perf_counter()
            totduration += end - start
            durations.append(end - start)
            print("Model executed. Elapsed time: " + str(end - start) + " in iteration " + str(i) + ".")
        avg = totduration / 100
        print("Average duration: " + str(avg) + " seconds")
        stddev = statistics.stdev(durations)
        print("Standard deviation:" + str(stddev) + "")
    else:
        # --- in execution mode,  callNetwork or callNetworkBatchwise get called from c++ directly ---
        print("pure execution mode")
    print("Neural Entropy Closure Suite finished successfully.")
    return 0


if __name__ == '__main__':
    main()
