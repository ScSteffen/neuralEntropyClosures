'''
Script to Preprocess Network choices and create the correct network
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''

### imports ###
from .neuralMK1 import neuralMK1
from .neuralMK2 import neuralMK2
from .neuralMK3 import neuralMK3
from .neuralMK4 import neuralMK4
from .neuralMK5 import neuralMK5
from .neuralMK6 import neuralMK6
from .neuralMK7 import neuralMK7
from .neuralMK8 import neuralMK8
from .neuralMK9 import neuralMK9
from .neuralMK10 import neuralMK10
from .neuralMK11 import neuralMK11


### global functions ###
def initNeuralClosure(modelNumber=1, polyDegree=0, spatialDim=3, folderName="testFolder", optimizer='adam', width=10,
                      depth=5, normalized=False):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    # Catch obvious errors
    if (polyDegree < 0):
        ValueError("Negative number of basis functions not possible")
    if (spatialDim < 0 or spatialDim > 3):
        ValueError("Spatial dimension must be between 1 and 3.")
    if (width < 1):
        ValueError("Model width must be bigger than 0.")
    if (depth < 1):
        ValueError("Model depth must be bigger than 0.")

    msg = "Chosen Model: MK" + str(modelNumber) + ", Degree " + str(polyDegree)
    print(msg)

    # Create the correct network
    if (modelNumber == 1):
        neuralClosureModel = neuralMK1(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 2):
        neuralClosureModel = neuralMK2(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 3):
        neuralClosureModel = neuralMK3(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 4):
        neuralClosureModel = neuralMK4(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 5):
        neuralClosureModel = neuralMK5(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 6):
        neuralClosureModel = neuralMK6(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 7):
        neuralClosureModel = neuralMK7(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 8):
        neuralClosureModel = neuralMK8(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 9):
        neuralClosureModel = neuralMK9(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                       optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 10):
        neuralClosureModel = neuralMK10(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                        optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    elif (modelNumber == 11):
        neuralClosureModel = neuralMK11(polyDegree=polyDegree, spatialDim=spatialDim, folderName=folderName,
                                        optimizer=optimizer, width=width, depth=depth, normalized=normalized)
    else:
        ValueError("No network fits your preferences!")

    print("Neural closure model created")

    return neuralClosureModel
