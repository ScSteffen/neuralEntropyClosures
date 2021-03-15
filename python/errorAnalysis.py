'''
Script to conduct error Analysis of the Training
Author: Steffen Schotthöfer
Date: 15.03.2021
'''

from src.neuralClosures.configModel import initNeuralClosure
import src.utils as utils

import matplotlib.pyplot as plt

plt.style.use("kitish")


def main():
    filenameModel = "/models/ "
    filenameData = "/data/1_stage/1D/Monomial_"
    inputDim = 2

    # Load Model
    model = utils.loadTFModel(filenameModel)

    # Load Data
    [u, alpha, h] = utils.loadData(filenameData)

    # Model Predictions
    h_pred = utils.evaluateModel(model, u)
    alpha_pred = utils.evaluateModel(model, u)

    # plot errors
    # h - h_pred over u

    return 0


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

    neuralClosureModel = initNeuralClosure(modelNumber, maxDegree_N, folderName)
    neuralClosureModel = tf.keras.models.load_model(file)

    neuralClosureModel.model.summary()
    print("|")
    print("| Tensorflow neural closure initialized.")
    print("|")
    return 0


if __name__ == '__main__':
    main()
