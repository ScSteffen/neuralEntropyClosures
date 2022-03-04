# neuralEntropy package for deep learning entropy closures for the boltzmann equation

A Project to construct some neural networks to solve the minimal entropy problem.

## Packages needed

* If you use conda, please consult the installer script '''package_installer.sh'''. If you use pip, consult '''requirements.txt'''. 
* A ready to use docker container is provided [here] (https://hub.docker.com/repository/docker/scsteffen/neural_entropy). One needs the nvidia-docker installed (tutorial [here] (https://www.tensorflow.org/install/docker) and [here] (https://github.com/NVIDIA/nvidia-docker).

## Available, tested network models ##

* MK11: ICNN model with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses normalized data. Variable
  network size. Can reconstruct u. Different network call possibilities
* MK12: ResNet model with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses normalized data. Variable
  network size. Can reconstruct u. Different network call possibilities (Not Convex!)
* MK15: ResNet model that directly approximates alpha and reconstructs u and h. Equipped with Monotonicity loss

## How to use ## 

Preliminary: Execute package_installer.sh to install all neccessary python packages.


### Training ### 

To perform an end to end training for a network, call the "callNeuralClosure.py".

Options:

* -a (--alphasampling): Determines sampling strategy
* -b (--batch): Determines batch size
* -c (--curriculum): Determines training curriculum
* -d (--degree): Determines degree of the basis functions (monomials)
* -e (--epoch): Determines number of epochs
* -f (--folder): Determines subfolder of "models"
* -l (--load_model): Determines if programm loads existing model weights
* -m (--model): Choice of model version. It is recommended to use version 11
* -n (--normalized): Determine if training happens on normalized data (recommended)
* -o (--objective): Determines choice of training objective
* -p (--processingmode): Determine to train on CPU or on GPU (if installed)
* -s (--spatialDimension): Determines spatial dimension of closure (1,2 or 3)
* -t (--training): Determine training mode
* -v (--verbosity): Determine output verbosity
* -w (--networkWidth): Determine width of a convex layer
* -x (--networkDepth): Determine depth of the convex block (number of convex hidden layers)

Type  "callNeuralClosure.py --help" for information on the options
The runScript.sh provides a template for quick bash execution.

### Employing a network with KiT-RT C++ framework ###.

Use the [KiT-RT](https://github.com/CSMMLab/KiT-RT) kinetic simulation suite. 

