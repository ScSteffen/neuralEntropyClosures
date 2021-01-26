(not up to date) 

# EntropyTrainer

A Project to construct some neural networks to solve the minimal entropy problem. 


## How to use ## 

Preliminary: Make sure to install tensorflow (with or without GPU acceleration) on your machine. 

There are several versions available for the networks, i.e. MK1,MK2,... 
	
### Training ### 

To perform an end to end training for a network, call the "train_NN_MK<X>.py" script, where X is the version to train. 

### Employing a network ###

A network can be called for the testing and enrollment stage using the call scripts call scripts "call_NN_MK<X>.py" where X is the version of the network. 

### Using a network in the RTSN codebase (beta) ###

1) Copy the folder containing the saved model into the /python folder of the RTSN codebase. 
2) Copy the correct "call_NN_MK<X>.py" file into the /python folder of the RTSN codebase. 
3) Select "ENTROPY_OPTIMIZER = ML" as an option in the .cfg file
4) Make sure that "MAX_MOMENT_SOLVER" specifies the correct maximum degree of basis functions, for which the network is trained. (In future this will be automatized). 
5) Run RTSN yourConfig.cfg 


