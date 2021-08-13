'''
Script to Preprocess Network choices and create the correct network
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''

### imports ###
from src.networks.mk11 import MK11Network
from src.networks.mk12 import MK12Network
from src.networks.mk13 import MK13Network
from src.networks.mk14 import MK14Network
from src.networks.mk15 import MK15Network

from src.networks.basenetwork import BaseNetwork


def init_neural_closure(network_mk: int = 1, poly_degree: int = 0, spatial_dim: int = 3,
                        folder_name: str = "testFolder", loss_combination: int = 0, nw_width: int = 10,
                        nw_depth: int = 5, normalized: bool = True):
    '''
    params: network_mk = Defines the used network model, i.e. MK1, MK2...
            poly_degree = Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
            spatial_dim = spatial dimension of the closure
            folder_name = name of the folder in which the model is saved
            loss_combination =  combination of used network losses
            nw_depth = number of layers of the network
            nw_width = width of the hidden layers
            normalized =  if true, the network uses normalized data
            scaled_output = if true, the range of the entropy functional is scaled to [0,1] and the training data is scaled accordingly
    returns: a fully configured neural network
    '''

    # Catch obvious errors
    if poly_degree < 0:
        ValueError("Negative number of basis functions not possible")
        exit(1)
    if spatial_dim < 0 or spatial_dim > 3:
        ValueError("Spatial dimension must be between 1 and 3.")
        exit(1)
    if nw_width < 1:
        ValueError("Model width must be bigger than 0.")
        exit(1)
    if nw_depth < 1:
        ValueError("Model depth must be bigger than 0.")
        exit(1)

    msg = "Chosen Model: MK " + str(network_mk) + ", Degree " + str(poly_degree)
    print(msg)
    neural_closure_model: BaseNetwork
    # Create the correct network
    if network_mk < 11:
        print("This model is deprecated. Visit branch <deprecated_models> to try them.")
        exit(1)
    elif network_mk == 11:
        neural_closure_model = MK11Network(polynomial_degree=poly_degree, spatial_dimension=spatial_dim,
                                           save_folder=folder_name, loss_combination=loss_combination,
                                           width=nw_width, depth=nw_depth, normalized=normalized)
    elif network_mk == 12:
        neural_closure_model = MK12Network(polynomial_degree=poly_degree, spatial_dimension=spatial_dim,
                                           save_folder=folder_name, loss_combination=loss_combination,
                                           width=nw_width, depth=nw_depth, normalized=normalized)
    elif network_mk == 13:
        neural_closure_model = MK13Network(polynomial_degree=poly_degree, spatial_dimension=spatial_dim,
                                           save_folder=folder_name, loss_combination=loss_combination,
                                           width=nw_width, depth=nw_depth, normalized=normalized)
    elif network_mk == 14:
        neural_closure_model = MK14Network(polynomial_degree=poly_degree, spatial_dimension=spatial_dim,
                                           save_folder=folder_name, loss_combination=loss_combination,
                                           width=nw_width, depth=nw_depth, normalized=normalized)
    elif network_mk == 15:
        neural_closure_model = MK15Network(polynomial_degree=poly_degree, spatial_dimension=spatial_dim,
                                           save_folder=folder_name, loss_combination=loss_combination,
                                           width=nw_width, depth=nw_depth, normalized=normalized)
    else:
        print("No available network fits your preferences!")
        exit()
    print("Neural closure model created")
    return neural_closure_model
