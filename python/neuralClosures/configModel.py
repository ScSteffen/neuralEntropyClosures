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


### global functions ###
def initNeuralClosure(modelNumber=1, maxDegree_N=0, folderName = "testFolder"):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    # Create the correct network
    if (maxDegree_N < 0):
        ValueError("Negative number of basis functions not possible")

    elif modelNumber == 1:  # MK1 model
        neuralClosureModel = neuralMK1(maxDegree_N,folderName)

    elif modelNumber == 2:  # Mk2 model
        if maxDegree_N > 0:
            ValueError("Model MK1 is constructed only for maximum degree 0 of the spherical harmonics")
        neuralClosureModel = neuralMK2(maxDegree_N,folderName)

    elif modelNumber == 3:  # MK3 model
        if maxDegree_N > 1:
            ValueError("Model MK3 is constructed only for maximum degree 1 of the spherical harmonics (at the moment)")

        neuralClosureModel = neuralMK3(maxDegree_N,folderName)

    elif modelNumber == 4:  # MK4 model ICNN
        if maxDegree_N > 1:
            ValueError("Model MK4 is constructed only for maximum degree 1 of the spherical monomials (at the moment)")

        neuralClosureModel = neuralMK4(maxDegree_N,folderName)

    elif modelNumber == 5:  # MK4 model ICNN with entropy target
        if maxDegree_N > 0:
            ValueError("Model MK5 is constructed only for maximum degree 0 of the spherical harmonics (at the moment)")

        neuralClosureModel = neuralMK5(maxDegree_N,folderName)

    else:
        ValueError("No network fits your preferences!")

    print("Neural closure model created")

    return neuralClosureModel
