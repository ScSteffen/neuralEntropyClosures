# neuralEntropyClosures

The repo contains neural network models and supplementary code for the minimal entropy closure of the Boltzmann moment
system

Computational Backbone for the following publications:

* [Neural network-based, structure-preserving entropy closures for the Boltzmann moment system](https://arxiv.org/abs/2201.10364) (
  Accepted at ICML 2022 with title: Structure Preserving Neural Networks: A Case Study in the Entropy Closure of the
  Boltzmann Equation)
* [A structure-preserving surrogate model for the closure of the moment system of the Boltzmann equation using convex deep neural networks](https://arxiv.org/abs/2106.09445) (
  Accepted to AIAA Aviation Forum 2021, Conference Manuscript)

## Content

* [Installation](#Installation)
* [Available network models](#Models)
* [Training](#Training)
* [Usage with solver](#Solver)
* [Citation Info](#Cite)

## Installation

Preliminary: Execute package_installer.sh to install all neccessary python packages. Or use

```
pip install -r requirements.txt
```

## Models

* MK11: ICNN model with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses normalized data. Variable
  network size. Can reconstruct u. Different network call possibilities
* MK12: ResNet model with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses normalized data. Variable
  network size. Can reconstruct u. Different network call possibilities (Not Convex!)
* MK13: ICNN-ResNet model with sobolev core to train the mapping u to h, alpha (RMSE loss). Uses normalized data.
  Variable
  network size. Can reconstruct u. Different network call possibilities
* MK15: ResNet model that directly approximates alpha and reconstructs u and h. Equipped with Monotonicity loss

## Training

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

## Solver

Use the [KiT-RT](https://github.com/CSMMLab/KiT-RT) kinetic simulation suite.

## Cite

```
@article{neuralEntropyClosure,
  author = {Schotthöfer, Steffen},
  title = {neuralEntropyClosure - Github Repository},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ScSteffen/neuralEntropyClosures}},
  commit = {af8e47386115b89a16f0f105e8302b650f47e68d}
}
``` 
