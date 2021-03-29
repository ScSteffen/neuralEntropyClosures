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
import pandas as pd
from os import path, makedirs, walk
import time


### class definitions ###
class neuralBase:

    # Used member functions
    # trainingData = [u,alpha,h]
    # model
    # filename
    # maxDegree_N

    # Member functions

    def __init__(self, polyDegree=0):
        # self.model = self.createModel(self)
        # self.filename = "models/MK1_N" + maxDegree_N
        self.trainingData = []
        self.polyDegree = polyDegree
        self.inputDim = 1  # Must be overwritten by child classes!
        self.spatialDim = 3  # Must be overwritten by child classes!

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

    def trainModel(self, valSplit=0.1, epochCount=2, epochChunks=1, batchSize=500, verbosity=1, processingMode=0):
        '''
        Method to train network
        '''

        # Set full precision training for CPU training
        if processingMode == 0:
            tf.keras.backend.set_floatx('float32')

        # Create callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.filename + '/best_model.h5', monitor='loss', mode='min',
                                                     save_best_only=True,
                                                     verbose=verbosity)  # , save_weights_only = True, save_freq = 50, verbose=0)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.0001, patience=10,
                                              verbose=1)
        # mc_checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=self.filename + '/model_saved',
        #                                         save_weights_only=False,
        #                                         verbose=1)

        # Split Training epochs
        miniEpoch = int(epochCount / epochChunks)

        for i in range(0, epochChunks):
            #  perform a batch doublication every 1/10th of the epoch count
            print("Current Batch Size: " + str(batchSize))

            # assemble callbacks
            callbackList = []
            csv_logger = self.createCSVLoggerCallback()
            if verbosity == 1:
                callbackList = [mc_best, csv_logger]
            else:
                callbackList = [mc_best, LossAndErrorPrintingCallback(), csv_logger]

            # start Training
            self.history = self.model.fit(self.trainingData[0], self.trainingData[1],
                                          validation_split=valSplit,
                                          epochs=miniEpoch,
                                          batch_size=batchSize,
                                          verbose=verbosity,
                                          callbacks=callbackList,
                                          )
            batchSize = 2 * batchSize

        self.concatHistoryFiles()

        return self.history

    def concatHistoryFiles(self):
        '''
        concatenates the historylogs (works only for up to 10 logs right now)
        assumes that all files in the folder correspond to current training
        '''

        if not path.exists(self.filename + '/historyLogs/'):
            ValueError("Folder <historyLogs> does not exist.")

        historyLogs = []

        for (dirpath, dirnames, filenames) in walk(self.filename + '/historyLogs/'):
            historyLogs.extend(filenames)
            break
        print("Found logs:")
        historyLogs.sort()
        print(historyLogs)

        historyLogsDF = []
        count = 0
        for log in historyLogs:
            historyLogsDF.append(pd.read_csv(self.filename + '/historyLogs/' + log))

        totalDF = pd.concat(historyLogsDF, ignore_index=True)

        # postprocess:
        numEpochs = len(totalDF.index)
        totalDF['epoch'] = np.arange(numEpochs)
        # write
        totalDF.to_csv(self.filename + '/historyLogs/CompleteHistory.csv', index=False)
        return 0

    def createCSVLoggerCallback(self):
        '''
        dynamically creates a csvlogger
        '''
        # check if dir exists
        if not path.exists(self.filename + '/historyLogs/'):
            makedirs(self.filename + '/historyLogs/')

        # checkfirst, if history file exists.
        logFile = self.filename + '/historyLogs/history_001_'
        count = 1
        while path.isfile(logFile + '.csv'):
            count += 1
            logFile = self.filename + '/historyLogs/history_' + str(count).zfill(3) + '_'

        logFile = logFile + '.csv'
        # create logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(logFile)

        return csv_logger

    def saveModel(self):
        self.model.save(self.filename + '/model')

        with open(self.filename + '/trainingHistory.json', 'w') as file:
            json.dump(self.model.history.history, file)
        return 0

    def loadModel(self, filename=None):
        usedFileName = self.filename
        if filename != None:
            usedFileName = filename

        usedFileName = usedFileName + '/best_model.h5'

        if path.exists(usedFileName) == False:
            ValueError("Model does not exists at this path: " + usedFileName)
        self.model.load_weights(usedFileName)
        print("Model loaded from file ")
        return 0

    def loadTrainingData(self, normalizedMoments=0):
        # Loads the training data generated by "generateTrainingData.py"
        self.trainingData = []
        filename = "data/1_stage/" + str(self.spatialDim) + "D/Monomial_M" + str(self.polyDegree) + "_" + str(
            self.spatialDim) + "D.csv"
        if normalizedMoments == 1:
            filename = "data/1_stage/" + str(self.spatialDim) + "D/Monomial_M" + str(self.polyDegree) + "_" + str(
                self.spatialDim) + "D_normal.csv"

        print("Loading Data from location: " + filename)
        # determine which cols correspond to u, alpha and h
        uCols = list(range(1, self.inputDim + 1))
        alphaCols = list(range(self.inputDim + 1, 2 * self.inputDim + 1))
        hCol = [2 * self.inputDim + 1]

        # selectedCols = self.selectTrainingData() #outputs a boolean triple.

        selectedCols = [True, False, True]

        start = time.time()
        if selectedCols[0] == True:
            df = pd.read_csv(filename, usecols=[i for i in uCols])
            uNParray = df.to_numpy()
            self.trainingData.append(uNParray)
        if selectedCols[1] == True:
            df = pd.read_csv(filename, usecols=[i for i in alphaCols])
            alphaNParray = df.to_numpy()
            self.trainingData.append(alphaNParray)
        if selectedCols[2] == True:
            df = pd.read_csv(filename, usecols=[i for i in hCol])
            hNParray = df.to_numpy()
            self.trainingData.append(hNParray)

        end = time.time()
        print("Data loaded. Elapsed time: " + str(end - start))

        return True

    def selectTrainingData(self):
        pass

    def trainingDataPostprocessing(self):
        return 0

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


class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):
    # def on_train_batch_end(self, batch, logs=None):
    #    print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    # def on_test_batch_end(self, batch, logs=None):
    #    print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))

    def on_epoch_end(self, epoch, logs=None):
        print(
            "The average loss for epoch {} is {:7.2f} "
            "and mean absolute error is {:7.2f}.".format(
                epoch, logs["loss"], logs["mean_absolute_error"]
            )
        )
