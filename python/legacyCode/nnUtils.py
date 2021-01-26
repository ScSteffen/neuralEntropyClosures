'''
Author: Steffen Schotthöfer
Date: 12.09.2020
Description: Collection of all utility functions used for
             creation of training data, plotting and preprocessing
'''

import json
import matplotlib.pyplot as plt
#plt.style.use("kitish")                      #or "kitishnotex" to avoid text rendering with TeX.

# ------  Code starts here --------

# history processing
def print_history(history):
    fig, axs = plt.subplots(2)
    #print(history.history.keys())

    axs[0].plot(history['mean_absolute_error'])
    axs[0].plot(history['val_mean_absolute_error'])
    axs[0].set_title('model accuracy')
    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel('epoch')
    axs[0].set_yscale("log")
    axs[0].legend(['train mean abs error', 'val mean abs error'], loc='upper right')
    # axs[0].show()

    # summarize history for loss
    axs[1].plot(history['loss'])
    axs[1].plot(history['val_loss'])
    axs[1].set_title('model loss')
    axs[1].set_ylabel('loss')
    axs[1].set_xlabel('epoch')
    axs[1].set_yscale("log")
    axs[1].legend(['train mse', 'val mse'], loc='upper right')
    # axs[1].show()
    plt.show()
    return 0

# training data processing
def plot_trainingData(uTrain):
    # input must have dim 4 or higher

    fig, axs = plt.subplots(4)
    axs[0].plot(uTrain[:, 0])
    axs[0].set_title('$m_0^0$')
    axs[0].set_ylabel('value')
    axs[0].set_xlabel('sample id')

    axs[1].plot(uTrain[:, 1])
    axs[1].set_title('$m_1^-1$')
    axs[1].set_ylabel('value')
    axs[1].set_xlabel('sample id')

    axs[2].plot(uTrain[:, 2])
    axs[2].set_title('$m_1^0$')
    axs[2].set_ylabel('value')
    axs[2].set_xlabel('sample id')

    axs[3].plot(uTrain[:, 3])
    axs[3].set_title('$m_1^1$')
    axs[3].set_ylabel('value')
    axs[3].set_xlabel('sample id')

    #ax.plot(uTrain[:,4], label="$m_4$")
    #ax.plot(uTrain[:,5], label="$m_5$")
    #ax.plot(uTrain[:,6], label="$m_6$")
    #ax.plot(uTrain[:,7], label="$m_7$")
    #ax.plot(uTrain[:,8], label="$m_8$")

    fig.legend()
    #plt.set_title(r"First moments of kinetic density.")
    #fig.set_ylabel("Values")
    #fig.set_xlabel(r"Sample ID.")
    fig.savefig("trainingData.png")
    plt.show()
    return 0

# save history
def save_training(filename, model, history):
    model.save(filename + '/model')

    with open(filename + '/model_history.json', 'w') as file:
        json.dump(history.history, file)
    return 0

# load history
def load_trainHistory(filename):
    with open(filename + '/model_history.json') as json_file:
        history = json.load(json_file)
    return history

### helper functions
def getGlobalIdx(l,k):
    return l*l+k+l
