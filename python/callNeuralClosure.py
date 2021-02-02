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

from optparse import OptionParser


### global variable ###
neuralClosureModel = 0  # bm.initNeuralClosure(0,0)


### function definitions ###
def initModel(modelNumber=1, maxDegree_N=0, folderName = "testFolder"):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    global neuralClosureModel
    neuralClosureModel = initNeuralClosure(modelNumber, maxDegree_N, folderName)

    return 0

def call_network(input):
    '''
    # Input: input.shape = (nCells,nMaxMoment), nMaxMoment = 9 in case of MK3
    '''
    predictions = neuralClosureModel.model.predict(input)

    return predictions

def call_networkBatchwise(input):

    #print(input)
    inputNP = np.asarray(input)
    #print(inputNP.shape)
    #print(inputNP)

    predictions = neuralClosureModel.model.predict(inputNP)

    #print(predictions)

    size = predictions.shape[0]*predictions.shape[1]
    test = np.zeros(size)
    for i in  range(0,size):
        test[i] = predictions.flatten(order='C')[i]
    return test

def main():
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-d", "--degree", dest="degree",default=0,
                      help="max degree of moment", metavar="DEGREE")
    parser.add_option("-m", "--model", dest="model", default=1,
                      help="choice of network model", metavar="MODEL")
    parser.add_option("-e", "--epoch", dest="epoch", default=1000,
                      help="epoch count for neural network", metavar="EPCOH")
    parser.add_option("-b", "--batch", dest="batch", default=1000,
                      help="batch size", metavar="BATCH")
    parser.add_option("-v", "--verbosity", dest="verbosity", default=1,
                      help="output verbosity keras (0 or 1)", metavar="VERBOSITY")
    parser.add_option("-l", "--loadModel", dest="loadmodel", default=1,
                      help="load model weights from file", metavar="LOADING")
    parser.add_option("-f", "--folder", dest="folder",default="testFolder",
                      help="folder with training data and where the model is stored", metavar="FOLDER")
    (options, args) = parser.parse_args()
    options.degree = int(options.degree)
    options.model = int(options.model)
    options.epoch = int(options.epoch)
    options.batch = int(options.batch)
    options.verbosity = int(options.verbosity)
    options.loadmodel = int(options.loadmodel)

    # --- End Option Parsing ---


    # --- initialize model
    initModel(modelNumber=options.model, maxDegree_N=options.degree, folderName = options.folder)

    # load model weights
    if(options.loadmodel == 1):
        neuralClosureModel.loadModel()

    # create training Data
    neuralClosureModel.createTrainingData()
    ## Test some stufff DELETE
    #(u,h) =  neuralClosureModel.trainingData
    #print(u)
    #print(h)
    # load model
    #neuralClosureModel.loadModel()
    # train model
    neuralClosureModel.trainModel(valSplit=0.01, epochCount=options.epoch, batchSize=options.batch, verbosity = options.verbosity)

    # save model
    neuralClosureModel.saveModel()

    # plot training history
    #neuralClosureModel.plotTrainingHistory()

    # initialize_network()
    # print(neuralClosureModel.computePrediction([[1]]))

    return 0


if __name__ == '__main__':
    main()
