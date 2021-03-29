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


### global functions ###
def initNeuralClosure(modelNumber=1, polyDegree=0, spatialDim=3, folderName="testFolder", optimizer='adam', width=10,
                      height=5):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    msg = "Chosen Model: MK" + str(modelNumber) + ", Degree " + str(polyDegree)
    print(msg)
    # Create the correct network

    if (spatialDim == 3):
        if (polyDegree < 0):
            ValueError("Negative number of basis functions not possible")
        elif modelNumber == 1:  # MK1 model
            neuralClosureModel = neuralMK1(polyDegree, folderName, optimizer)

        elif modelNumber == 2:  # Mk2 model
            if polyDegree > 0:
                ValueError("Model MK1 is constructed only for maximum degree 0 of the spherical harmonics")
            neuralClosureModel = neuralMK2(polyDegree, folderName, optimizer)

        elif modelNumber == 3:  # MK3 model
            if polyDegree > 1:
                ValueError(
                    "Model MK3 is constructed only for maximum degree 1 of the spherical harmonics (at the moment)")

            neuralClosureModel = neuralMK3(polyDegree, folderName, optimizer)

        elif modelNumber == 4:  # MK4 model ICNN
            if polyDegree > 1:
                ValueError(
                    "Model MK4 is constructed only for maximum degree 1 of the spherical monomials (at the moment)")

            neuralClosureModel = neuralMK4(polyDegree, folderName, optimizer)

        elif modelNumber == 5:  # MK5 model Dense with entropy target (direct comparison with model MK4)
            if polyDegree > 1:
                ValueError(
                    "Model MK5 is constructed only for maximum degree 1 of the spherical monomials (at the moment)")

            neuralClosureModel = neuralMK5(polyDegree, folderName, optimizer)

        elif modelNumber == 6:  # MK4 model ICNN with input normalization
            if polyDegree > 1:
                ValueError(
                    "Model MK6 is constructed only for maximum degree 1 of the spherical monomials (at the moment)")

            neuralClosureModel = neuralMK6(polyDegree, folderName, optimizer)

        elif modelNumber == 7:  # MK4 model ICNN with input normalization and deeper layout
            if polyDegree > 1:
                ValueError(
                    "Model MK7 is constructed only for maximum degree 1 of the spherical monomials (at the moment)")

            neuralClosureModel = neuralMK7(polyDegree, spatialDim, folderName, optimizer, width, height)

        elif modelNumber == 8:  # MK4 model ICNN with input normalization and deeper layout and relu activation
            if polyDegree > 1:
                ValueError(
                    "Model MK8 is constructed only for maximum degree 1 of the spherical monomials (at the moment)")

            neuralClosureModel = neuralMK8(polyDegree, folderName, optimizer)
    elif spatialDim == 1:
        if modelNumber == 7:  # MK4 model ICNN with input normalization and deeper layout
            if polyDegree > 3:
                ValueError(
                    "Model MK7 is constructed only for maximum degree 3 of the spherical monomials (at the moment)")
            neuralClosureModel = neuralMK7(polyDegree, spatialDim, folderName, optimizer, width, height)

        if modelNumber == 9:  # MK4 model ICNN with input normalization and deeper layout
            if polyDegree > 3:
                ValueError(
                    "Model MK7 is constructed only for maximum degree 3 of the spherical monomials (at the moment)")
            neuralClosureModel = neuralMK9(polyDegree, spatialDim, folderName, optimizer)

    elif spatialDim == 2:
        if modelNumber == 7:  # MK4 model ICNN with input normalization and deeper layout
            if polyDegree > 1:
                ValueError(
                    "Model MK7 is constructed only for maximum degree 3 of the spherical monomials (at the moment)")
            neuralClosureModel = neuralMK7(polyDegree, spatialDim, folderName, optimizer, width, height)
    else:
        print("No network fits your preferences!")

    print("Neural closure model created")

    return neuralClosureModel
