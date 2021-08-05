# neuralEntropy package for deep learning entropy closures for the boltzmann equation

A Project to construct some neural networks to solve the minimal entropy problem.

## Packages needed

* Tensorflow v2.2.0
* Pandas
* Numpy

## Available network models ##

* MK11: ICNN model with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses normalized data. Variable
  network size. Can reconstruct u. Different network call possibilities
* MK12: Dense model (for comparison) with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses normalized
  data. Variable network size. Can reconstruct u. Different network call possibilities
* MK13: [Test bench for MK11] ICNN model with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses
  normalized data. Variable network size. Can reconstruct u. Different network call possibilities

## How to use ## 

Preliminary: Make sure to install tensorflow (with or without GPU acceleration) on your machine.

This small network suite is build to solve the minimal entropy closure for the Boltzmann equation. The problem is the
following:

### Training ### 

To perform an end to end training for a network, call the "callNeuralClosure.py".

Options:

* -a (--alphasampling): Determines sampling strategy
* -b (--batch): Determines batch size
* -c (--curriculum): Determines training curriculum
* -d (--degree): Determines degree of the basis functions (monomials)
* -e (--epoch): Determines number of epochs
* -f (--folder): Determines subfolder of "models"
* -l (--loadModel): Determines if programm loads existing model weights
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

### Employing a network with KiT-RT C++ framework ###.

Use the [KiT-RT](https://github.com/CSMMLab/KiT-RT) kinetic simulation suite. 

