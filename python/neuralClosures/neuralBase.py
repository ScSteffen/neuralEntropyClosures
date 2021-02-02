'''
Base Network class for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''

### imports ###
# import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np
import csv


### class definitions ###
class neuralBase:

    # Used member functions
    # trainingData = [u,alpha,h]
    # model
    # filename
    # maxDegree_N

    # Member functions

    def __init__(self, maxDegree_N=0):
        # self.model = self.createModel(self)
        # self.filename = "models/MK1_N" + maxDegree_N
        self.trainingData = ()
        self.maxDegree_N = maxDegree_N
        self.inputDim = 1 # Must be overwritten by child classes!

    def createModel(self):
        pass

    def computePrediction(self, input):
        return self.model.predict(input)

    def printWeights(self):
        for layer in self.model.layers:
            weights = layer.get_weights()  # list of numpy arrays
            print(weights)
            print("---------------------------------------------")

    def showModel(self):
        self.model.summary()
        return 0

    def trainModel(self, valSplit=0.1, epochCount=2, batchSize=500, verbosity=1):
        '''
        Method to train network
        '''
        # Create callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.filename + '/model_saved', monitor='loss', mode='min',
                                                     save_best_only=True, save_weights_only = False, save_freq = 50, verbose=0)
        #mc_checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=self.filename + '/model_saved',
        #                                         save_weights_only=False,
        #                                         verbose=1)
        self.history = self.model.fit(self.trainingData[0], self.trainingData[1],
                                      validation_split=valSplit,
                                      epochs=epochCount,
                                      batch_size=batchSize,
                                      verbose=verbosity,
                                      callbacks=[mc_best, LossAndErrorPrintingCallback()],
                                      )
        return self.history

    '''
    def plotTrainingHistory(self):
        
        #Method to plot the training data
        
        fig, axs = plt.subplots(2)

        axs[0].plot(self.model.history.history['mean_absolute_error'])
        axs[0].plot(self.model.history.history['val_mean_absolute_error'])
        axs[0].set_title('absolute error')
        axs[0].set_ylabel('error')
        axs[0].set_xlabel('epoch')
        axs[0].set_yscale("log")
        axs[0].legend(['train mean abs error', 'val mean abs error'], loc='upper right')

        # summarize history for loss
        axs[1].plot(self.model.history.history['loss'])
        axs[1].plot(self.model.history.history['val_loss'])
        axs[1].set_title('model loss: mse')
        axs[1].set_ylabel('error')
        axs[1].set_xlabel('epoch')
        axs[1].set_yscale("log")
        axs[1].legend(['train mse', 'val mse'], loc='upper right')
        # axs[1].show()
        plt.show()
        return 0
    '''

    def saveModel(self):
        self.model.save(self.filename + '/model')

        with open(self.filename + '/trainingHistory.json', 'w') as file:
            json.dump(self.model.history.history, file)
        return 0

    def loadModel(self, filename = None):
        usedFileName = self.filename + '/model_saved'
        if filename != None:
            usedFileName = filename
        self.model = tf.keras.models.load_model(filename)
        print("Model from file loaded")
        return 0

    def createTrainingData(self):
        # Loads the training data generated by "generateTrainingData.py"

        filename = "data/1_stage/Monomial_M" + str(self.maxDegree_N) + ".csv"
        # Load Alpha
        f = open(filename, 'r')
        alphaList = list()
        uList = list()
        hList = list()
        with f:
            reader = csv.reader(f)
            for row in reader:
                numRowU = []
                numRowAlpha = []
                numRowH = []
                word_idx = 0
                for word in row:
                    # skip first index, which is date and time info
                    if word_idx > 0 and word_idx < self.inputDim + 1:  # hardcoded... careful
                        numRowU.append(float(word))
                    if word_idx > self.inputDim and word_idx < 2 * self.inputDim + 1:
                        numRowAlpha.append(float(word))
                    if word_idx == 2 * self.inputDim + 1:
                        numRowH.append(float(word))
                    word_idx = word_idx + 1
                uList.append(numRowU)
                alphaList.append(numRowAlpha)
                hList.append(numRowH)

        print("Data loaded")
        self.trainingData = (np.asarray(uList), np.asarray(alphaList), np.asarray(hList))

    def selectTrainingData(self):
        pass

class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    #def on_train_batch_end(self, batch, logs=None):
    #    print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    #def on_test_batch_end(self, batch, logs=None):
    #    print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )

