'''
This is the script that gets called from the C++ KiT-RT method MLOptimizer.cpp
It initializes and loads a neural Closure
The call method performs a prediction
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''

### imports ###
from neuralClosures.configModel import initNeuralClosure
import numpy as np
import tensorflow as tf

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

    modelNumber = input[0]
    maxDegree_N = input[1]

    # --- Transcribe the modelNumber and MaxDegree to the correct model folder --- #
    folderName = "neuralClosure_M" + str(maxDegree_N) + "_MK" + str(modelNumber)

    global neuralClosureModel
    neuralClosureModel = initNeuralClosure(modelNumber, maxDegree_N, folderName)
    neuralClosureModel.loadModel()
    neuralClosureModel.model.summary()
    print("| Tensorflow neural closure initialized.")

    return 0


### function definitions ###
def initModel(modelNumber=1, maxDegree_N=0, folderName = "testFolder"):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    global neuralClosureModel
    neuralClosureModel = initNeuralClosure(modelNumber, maxDegree_N, folderName)

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
        predictions = neuralClosureModel.model(x_model, training=False)  # same as model.predict(x)

    gradients = tape.gradient(predictions, x_model)


    return gradients

def callNetworkBatchwise(input):

    print(type(input))
    #print(input)
    inputNetwork = np.reshape(input, (input.shape[0], 1))
    #inputNP = np.asarray(input)
    #predictions = neuralClosureModel.model.predict(inputNP)

    #print(inputNP.shape)
    #print(inputNP)

    x_model = tf.Variable(inputNetwork)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = neuralClosureModel.model(x_model, training=False)  # same as model.predict(x)

    gradients = tape.gradient(predictions, x_model)

    # ---- Convert gradients from eagerTensor to numpy array and then to flattened c array ----

    gradNP =  np.reshape(gradients.numpy(), (input.shape[0]))
    return gradNP

    #print(gradients)
    #print(gradNP)
    #print(predictions)

    #size = gradients.shape[0]*gradients.shape[1]
    #test = np.zeros(size)
    #for i in  range(0,size):
    #    test[i] = predictions.flatten(order='C')[i]
    #return (predictions, gradients)
    #return test

def main():
    # Tests

    initModelCpp([4,0])
    # test
    nnIN = np.arange(0.5, 5, 0.5)
    test= callNetworkBatchwise(nnIN)
    print(test)

    #print("her")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-d", "--degree", dest="degree",default=0,
                      help="max degree of moment", metavar="DEGREE")
    parser.add_option("-m", "--model", dest="model", default=1,
                      help="choice of network model", metavar="MODEL")
    parser.add_option("-e", "--epoch", dest="epoch", default=1000,
                      help="epoch count for neural network", metavar="EPOCH")
    parser.add_option("-b", "--batch", dest="batch", default=1000,
                      help="batch size", metavar="BATCH")
    parser.add_option("-v", "--verbosity", dest="verbosity", default=1,
                      help="output verbosity keras (0 or 1)", metavar="VERBOSITY")
    parser.add_option("-l", "--loadModel", dest="loadmodel", default=1,
                      help="load model weights from file", metavar="LOADING")
    parser.add_option("-f", "--folder", dest="folder",default="testFolder",
                      help="folder with training data and where the model is stored", metavar="FOLDER")
    parser.add_option("-t", "--training", dest="training", default=1,
                      help="training mode (1) execution mode (0)", metavar="TRAINING")

    (options, args) = parser.parse_args()
    options.degree = int(options.degree)
    options.model = int(options.model)
    options.epoch = int(options.epoch)
    options.batch = int(options.batch)
    options.verbosity = int(options.verbosity)
    options.loadmodel = int(options.loadmodel)
    options.training = int(options.training)

    # --- End Option Parsing ---


    # --- initialize model
    initModel(modelNumber=options.model, maxDegree_N=options.degree, folderName = options.folder)
    neuralClosureModel.model.summary()

    if(options.loadmodel == 1 or options.training == 0):
        # in execution mode the model must be loaded.
        # load model weights
        neuralClosureModel.loadModel()
    else:
        print("Start training with new weights")


    if(options.training == 1):
        # create training Data
        neuralClosureModel.createTrainingData()
        neuralClosureModel.selectTrainingData()
        # train model
        neuralClosureModel.trainModel(valSplit=0.01, epochCount=options.epoch, batchSize=options.batch, verbosity = options.verbosity)
        # save model
        neuralClosureModel.saveModel()

    # --- in execution mode,  callNetwork or callNetworkBatchwise get called from c++ directly ---
    return 0


if __name__ == '__main__':
    main()
