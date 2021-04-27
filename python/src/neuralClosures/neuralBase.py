'''
Base Network class for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''

### imports ###
# python modules
import tensorflow as tf
import numpy as np
import pandas as pd
from os import path, makedirs, walk
import time

# intern modules
from src import utils


### class definitions ###
class neuralBase:

    def __init__(self, normalized, polyDegree, spatialDim, width, depth, lossCombi, customFolderName):
        self.normalized = normalized
        self.polyDegree = polyDegree
        self.spatialDim = spatialDim
        self.modelWidth = width
        self.modelDepth = depth
        self.optimizer = 'adam'
        self.filename = "models/" + customFolderName

        # --- Determine loss combination ---
        if lossCombi == 0:
            self.lossWeights = [1, 0, 0, 0]
        if lossCombi == 1:
            self.lossWeights = [1, 1, 0, 0]
        if lossCombi == 2:
            self.lossWeights = [1, 1, 1, 0]
        if lossCombi == 3:
            self.lossWeights = [1, 1, 1, 1]
        else:
            self.lossWeights = [1, 0, 0, 0]

        # --- Determine inputDim by MaxDegree ---
        if (spatialDim == 1):
            self.inputDim = polyDegree + 1
        elif (spatialDim == 3):
            if (self.polyDegree == 0):
                self.inputDim = 1
            elif (self.polyDegree == 1):
                self.inputDim = 4
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        elif (spatialDim == 2):
            if (self.polyDegree == 0):
                self.inputDim = 1
            elif (self.polyDegree == 1):
                self.inputDim = 3
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        else:
            raise ValueError("Saptial dimension other than 1,2 or 3 not supported atm")

        self.csvInputDim = self.inputDim  # only for reading csv data

        if normalized:
            self.inputDim = self.inputDim - 1

    def createModel(self):
        pass

    def callNetwork(self, u):
        """
        Brief: This does not reconstruct u, but returns original u. Careful here!

        # Input: input.shape = (nCells, nMaxMoment), nMaxMoment = 9 in case of MK3
        # Output: Gradient of the network wrt input
        """
        x_model = tf.Variable(u)

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_model, training=False)  # same as neuralClosureModel.model.predict(x)

        gradients = tape.gradient(predictions, x_model)

        return [x_model, gradients, predictions]

    def simple_callNetwork(self, input):
        """
        Simple prediction, that just calls the model
        """
        return self.model.predict(input)

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
                                          shuffle=True
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
        """
        Saves best model to .pb file
        """
        # load best h5 file
        usedFileName = self.filename
        self.model.load_weights(self.filename + '/best_model.h5')
        self.model.save(self.filename + '/best_model')
        print("Model successfully saved to file and .h5")
        # with open(self.filename + '/trainingHistory.json', 'w') as file:
        #    json.dump(self.model.history.history, file)
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

    def printWeights(self):
        for layer in self.model.layers:
            weights = layer.get_weights()  # list of numpy arrays
            print(weights)
            print("---------------------------------------------")

    def showModel(self):
        self.model.summary()
        return 0

    def loadTrainingData(self, shuffleMode=False, alphasampling=False, loadAll=False):
        """
        Loads the trianing data
        params: normalizedMoments = load normalized data  (u_0=1)
                shuffleMode = shuffle loaded Data  (yes,no)
                alphasampling = use data uniformly sampled in the space of Lagrange multipliers.
        return: True, if loading successful
        """
        self.trainingData = []

        ### Create trainingdata filename"
        filename = "data/" + str(self.spatialDim) + "D/Monomial_M" + str(self.polyDegree) + "_" + str(
            self.spatialDim) + "D"
        if self.normalized:
            filename = "data/" + str(self.spatialDim) + "D/Monomial_M" + str(self.polyDegree) + "_" + str(
                self.spatialDim) + "D_normal"
        if alphasampling:
            filename = filename + "_alpha"
        filename = filename + ".csv"

        print("Loading Data from location: " + filename)
        # determine which cols correspond to u, alpha and h
        uCols = list(range(1, self.csvInputDim + 1))
        alphaCols = list(range(self.csvInputDim + 1, 2 * self.csvInputDim + 1))
        hCol = [2 * self.csvInputDim + 1]

        selectedCols = self.selectTrainingData()  # outputs a boolean triple.

        # selectedCols = [True, False, True]

        start = time.time()
        if selectedCols[0] == True:
            df = pd.read_csv(filename, usecols=[i for i in uCols])
            uNParray = df.to_numpy()
            if self.normalized and not loadAll:
                # ignore first col of u
                uNParray = uNParray[:, 1:]

            self.trainingData.append(uNParray)
        if selectedCols[1] == True:
            df = pd.read_csv(filename, usecols=[i for i in alphaCols])
            alphaNParray = df.to_numpy()
            if self.normalized and not loadAll:
                # ignore first col of alpha
                alphaNParray = alphaNParray[:, 1:]
            self.trainingData.append(alphaNParray)
        if selectedCols[2] == True:
            df = pd.read_csv(filename, usecols=[i for i in hCol])
            hNParray = df.to_numpy()
            self.trainingData.append(hNParray)

        # shuffle data
        if (shuffleMode):
            indices = np.arange(hNParray.shape[0])
            np.random.shuffle(indices)
            for idx in range(len(self.trainingData)):
                self.trainingData[idx] = self.trainingData[idx][indices]

        end = time.time()
        print("Data loaded. Elapsed time: " + str(end - start))

        return True

    def getTrainingData(self):
        return self.trainingData

    def selectTrainingData(self):
        pass

    def trainingDataPostprocessing(self):
        return 0

    def evaluateModel(self, u_test, alpha_test, h_test):
        """
        brief: runs a number of tests and evalutations to determine test errors of the model.
        input: u_test, dim (nS, N)
               alpha_test, dim (nS, N)
               h_test, dim(nS,1)
        return: True, if run successfully. Prints several plots and pictures to file.
        """

        [u_pred, alpha_pred, h_pred] = self.callNetwork(u_test)

        # create the loss functions
        def pointwiseDiff(trueSamples, predSamples):
            """
            brief: computes the squared 2-norm for each sample point
            input: trueSamples, dim = (ns,N)
                   predSamples, dim = (ns,N)
            returns: mse(trueSamples-predSamples) dim = (ns,)
            """
            loss_val = tf.keras.losses.mean_squared_error(trueSamples, predSamples)
            return loss_val

        def meanDiff(trueSamples, predSamples):
            """
            brief: computes the average of pointwiseDiff over all samples
            input: trueSamples, dim = (ns,N)
                   predSamples, dim = (ns,N)
            returns: avg(mse(trueSamples-predSamples)) dim = (1,)
            """
            loss_val = tf.keras.losses.MeanSquaredError()(trueSamples, predSamples)
            return loss_val

        diff_h = pointwiseDiff(h_test, h_pred)
        diff_alpha = pointwiseDiff(alpha_test, alpha_pred)
        diff_u = pointwiseDiff(u_test, u_pred)

        utils.plot1D(u_test[:, 1], [h_pred, h_test], ['h pred', 'h'], 'h_over_u', log=False)
        utils.plot1D(u_test[:, 1], [alpha_pred[:, 1], alpha_test[:, 1]], ['alpha1 pred', 'alpha1 true'],
                     'alpha1_over_u1',
                     log=False)
        utils.plot1D(u_test[:, 1], [alpha_pred[:, 0], alpha_test[:, 0]], ['alpha0 pred', 'alpha0 true'],
                     'alpha0_over_u1',
                     log=False)
        utils.plot1D(u_test[:, 1], [u_pred[:, 0], u_test[:, 0]], ['u0 pred', 'u0 true'], 'u0_over_u1', log=False)
        utils.plot1D(u_test[:, 1], [u_pred[:, 1], u_test[:, 1]], ['u1 pred', 'u1 true'], 'u1_over_u1', log=False)
        utils.plot1D(u_test[:, 1], [diff_alpha, diff_h, diff_u], ['difference alpha', 'difference h', 'difference u'],
                     'errors', log=True)

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
