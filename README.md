# neuralEntropy package for deep learning entropy closures for the boltzmann equation

A Project to construct some neural networks to solve the minimal entropy problem. 


## How to use ## 

Preliminary: Make sure to install tensorflow (with or without GPU acceleration) on your machine. 

This small network suite is build to solve the minimal entropy closure for the Boltzmann equation. 
The problem is the following: 

$\alpha* = argmin_{\alpha} h(\alpha;u) = argmin_{\alpha} {\alpha*u - <\eta(\alpha*m)>}$

where $\alpha$ is the lagrange multiplier, $u$ the moment vector, $m$ the polynomial basis and $\eta$ the entropy of the entropy closure problem.
	
### Training ### 

To perform an end to end training for a network, call the "callNeuralClosure.py"

Options: 

* -d (--degree): Degree of the polynomial basis
* -m (--model): Model verion (1,2,...)
* -e (--epochs): Number of epochs for training
* -b (--batch): Batch size for training
* -v (--verbosity): Determines trianing output verbosity. If 0, there is still epoch summary, but not live update for batches
* -l (--loadModel): 0 = new weight initialization, 1 = load model weights from .h5 file
* -f (--folder): folder where the model is stored
* -t (--training): 1 = training mode, 0 = execution mode

### Model overview
Notation: 

 * MK1 : small dense network training $u\rightarrow\alpha$
 * MK2 : train $u\rightarrow \alpha$, where the loss is $h$
 * MK3 : resnet-like architecture  training $u\rightarrow\alpha$
 * MK4 : ICNN training $u\rightarrow h$
 * MK5 : Dense net training $u\rightarrow h$
 * MK6 : ICNN training $u\rightarrow h$ with scaled down imput (relative moments)

### Employing a network with KiT-RT C++ framework ###

Use the KiT-RT kinetic simulation suite (to be published soon). 

### Using a network in the RTSN codebase (beta) ###

1) Copy the folder containing the saved model into the /python folder of the RTSN codebase. 
2) Copy the correct "call_NN_MK<X>.py" file into the /python folder of the RTSN codebase. 
3) Select "ENTROPY_OPTIMIZER = ML" as an option in the .cfg file
4) Make sure that "MAX_MOMENT_SOLVER" specifies the correct maximum degree of basis functions, for which the network is trained. (In future this will be automatized). 
5) Run RTSN yourConfig.cfg 
