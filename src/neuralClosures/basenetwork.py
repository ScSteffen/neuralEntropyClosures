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
class BaseNetwork:
    normalized: bool  # Determines if model works with normalized data
    poly_degree: int  # Degree of basis function polynomials
    spatial_dim: int  # spatial dimension of problem
    inputDim: int  # dimension of network input
    model_width: int  # width of the hidden layers
    model_depth: int  # number of hidden layers (or ICNN blocks)
    optimizer: str  # choice of optimizer
    folder_name: str  # place where model and logs are stored
    history: list  # list of training history objects
    loss_weights: list  # one hot vector to enable/disable loss functions
    model: tf.keras.Model  # the neural network model
    training_data: list  # list of ndarrays containing the training data

    def __init__(self, normalized: bool, polynomial_degree: int, spatial_dimension: int, width: int, depth: int,
                 loss_combination: int, save_folder: str):
        self.normalized = normalized
        self.poly_degree: int = polynomial_degree
        self.spatial_dim: int = spatial_dimension
        self.model_width: int = width
        self.model_depth: int = depth
        self.optimizer: str = 'adam'
        self.folder_name: str = "models/" + save_folder
        self.history: list = []

        # --- Determine loss combination ---
        if loss_combination == 0:
            self.loss_weights = [1, 0, 0, 0]
        elif loss_combination == 1:
            self.loss_weights = [1, 1, 0, 0]
        elif loss_combination == 2:
            self.loss_weights = [1, 1, 1, 0]
        elif loss_combination == 3:
            self.loss_weights = [0, 0, 0, 1]
        else:
            self.loss_weights = [1, 0, 0, 0]

        # --- Determine inputDim by MaxDegree ---
        if spatial_dimension == 1:
            self.inputDim = polynomial_degree + 1
        elif spatial_dimension == 3:
            if self.poly_degree == 0:
                self.inputDim = 1
            elif self.poly_degree == 1:
                self.inputDim = 4
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        elif spatial_dimension == 2:
            if self.poly_degree == 0:
                self.inputDim = 1
            elif self.poly_degree == 1:
                self.inputDim = 3
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        else:
            raise ValueError("Saptial dimension other than 1,2 or 3 not supported atm")

        self.csvInputDim = self.inputDim  # only for reading csv data

        if self.normalized:
            self.inputDim = self.inputDim - 1

    def create_model(self) -> tf.keras.Model:
        pass

    def call_network(self, u) -> list:
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

    def call_scaled(self, u_non_normal):
        """
        Brief: By default the same behaviour as callNetwork.
              This method is subclassed by all networks, that only take normalized moments (MK11 and newer)
        """
        return self.call_network(u_non_normal)

    def config_start_training(self, valSplit=0.1, epochCount=2, curriculum=1, batchSize=500, verbosity=1,
                              processingMode=0):
        '''
        Method to train network
        '''

        # Set double precision training for CPU training #TODO
        if processingMode == 0:
            tf.keras.backend.set_floatx('float32')
        elif processingMode == 1:
            tf.keras.backend.set_floatx('float32')

        # Create callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.folder_name + '/best_model.h5', monitor='loss', mode='min',
                                                     save_best_only=True,
                                                     verbose=verbosity)  # , save_weights_only = True, save_freq = 50, verbose=0)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.0001, patience=10,
                                              verbose=1)

        if curriculum == 0:  # Epoch chunk training
            # mc_checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=self.filename + '/model_saved',
            #                                         save_weights_only=False,
            #                                         verbose=1)
            epochChunks = 4
            # Split Training epochs
            mini_epoch = int(epochCount / epochChunks)

            for i in range(0, curriculum):
                print("Training with increasing batch size")
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
                self.history = self.call_training(val_split=valSplit, epoch_size=epochCount, batch_size=batchSize,
                                                  verbosity_mode=verbosity, callback_list=callbackList)
                batchSize = 2 * batchSize

            self.concat_history_files()

        elif curriculum == 1:  # learning rate scheduler
            print("Training with learning rate scheduler")
            # We only use this at the moment
            initial_lr = float(1e-3)
            drop_rate = (epochCount / 3)
            stop_tol = 1e-7
            mt_patience = int(epochCount / 10)
            min_delta = stop_tol / 10

            # specific callbacks
            def step_decay(epoch):
                step_size = initial_lr * np.power(10, (-epoch / drop_rate))
                return step_size

            LR = tf.keras.callbacks.LearningRateScheduler(step_decay)
            HW = HaltWhenCallback('val_loss', stop_tol)
            ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                  verbose=1, patience=mt_patience, min_delta=min_delta)
            csv_logger = self.createCSVLoggerCallback()

            if verbosity == 1:
                callbackList = [mc_best, csv_logger, LR, HW, ES]
            else:
                callbackList = [mc_best, LossAndErrorPrintingCallback(), csv_logger, LR, HW, ES]

            # start Training
            self.history = self.call_training(val_split=valSplit, epoch_size=epochCount, batch_size=batchSize,
                                              verbosity_mode=verbosity, callback_list=callbackList)

        return self.history

    def call_training(self, val_split: float = 0.1, epoch_size: int = 2, batch_size: int = 128, verbosity_mode: int = 1,
                      callback_list: list = []) -> list:
        '''
        Calls training depending on the MK model
        '''
        xData = self.training_data[0]
        yData = self.training_data[1]
        self.model.fit(x=xData, y=yData, validation_split=val_split, epochs=epoch_size, batch_size=batch_size,
                       verbose=verbosity_mode, callbacks=callback_list, shuffle=True)
        return self.history

    def concat_history_files(self):
        '''
        concatenates the historylogs (works only for up to 10 logs right now)
        assumes that all files in the folder correspond to current training
        '''

        if not path.exists(self.folder_name + '/historyLogs/'):
            ValueError("Folder <historyLogs> does not exist.")

        historyLogs = []

        for (dirpath, dirnames, filenames) in walk(self.folder_name + '/historyLogs/'):
            historyLogs.extend(filenames)
            break
        print("Found logs:")
        historyLogs.sort()
        print(historyLogs)

        historyLogsDF = []
        count = 0
        for log in historyLogs:
            historyLogsDF.append(pd.read_csv(self.folder_name + '/historyLogs/' + log))

        totalDF = pd.concat(historyLogsDF, ignore_index=True)

        # postprocess:
        numEpochs = len(totalDF.index)
        totalDF['epoch'] = np.arange(numEpochs)
        # write
        totalDF.to_csv(self.folder_name + '/historyLogs/CompleteHistory.csv', index=False)
        return 0

    def createCSVLoggerCallback(self):
        '''
        dynamically creates a csvlogger
        '''
        # check if dir exists
        if not path.exists(self.folder_name + '/historyLogs/'):
            makedirs(self.folder_name + '/historyLogs/')

        # checkfirst, if history file exists.
        logFile = self.folder_name + '/historyLogs/history_001_'
        count = 1
        while path.isfile(logFile + '.csv'):
            count += 1
            logFile = self.folder_name + '/historyLogs/history_' + str(count).zfill(3) + '_'

        logFile = logFile + '.csv'
        # create logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(logFile)

        return csv_logger

    def saveModel(self):
        """
        Saves best model to .pb file
        """
        # load best h5 file
        usedFileName = self.folder_name
        self.model.load_weights(self.folder_name + '/best_model.h5')
        self.model.save(self.folder_name + '/best_model')
        print("Model successfully saved to file and .h5")
        # with open(self.filename + '/trainingHistory.json', 'w') as file:
        #    json.dump(self.model.history.history, file)
        return 0

    def loadModel(self, filename=None):
        usedFileName = self.folder_name
        if filename != None:
            usedFileName = filename

        usedFileName = usedFileName + '/best_model.h5'

        if path.exists(usedFileName) == False:
            print("Model does not exists at this path: " + usedFileName)
            exit(1)
        self.model.load_weights(usedFileName)
        print("Model loaded from file ")
        return 0

    def printWeights(self):
        for layer in self.model.layers:
            weights = layer.get_weights()  # list of numpy arrays
            print(weights)
            print("---------------------------------------------")

    def show_model(self):
        self.model.summary()
        return 0

    def load_training_data(self, shuffleMode=False, alphasampling=0, loadAll=False, normalizedData=False):
        """
        Loads the trianing data
        params: normalizedMoments = load normalized data  (u_0=1)
                shuffleMode = shuffle loaded Data  (yes,no)
                alphasampling = use data uniformly sampled in the space of Lagrange multipliers.
        return: True, if loading successful
        """
        self.training_data = []

        ### Create trainingdata filename"
        filename = "data/" + str(self.spatial_dim) + "D/Monomial_M" + str(self.poly_degree) + "_" + str(
            self.spatial_dim) + "D"
        if normalizedData:
            filename = "data/" + str(self.spatial_dim) + "D/Monomial_M" + str(self.poly_degree) + "_" + str(
                self.spatial_dim) + "D_normal"
        if alphasampling == 1:
            filename = filename + "_alpha"
        filename = filename + ".csv"

        print("Loading Data from location: " + filename)
        # determine which cols correspond to u, alpha and h
        uCols = list(range(1, self.csvInputDim + 1))
        alphaCols = list(range(self.csvInputDim + 1, 2 * self.csvInputDim + 1))
        hCol = [2 * self.csvInputDim + 1]

        selectedCols = self.select_training_data()  # outputs a boolean triple.

        # selectedCols = [True, False, True]

        start = time.perf_counter()
        if selectedCols[0] == True:
            df = pd.read_csv(filename, usecols=[i for i in uCols])
            uNParray = df.to_numpy()
            if normalizedData and not loadAll:
                # ignore first col of u
                uNParray = uNParray[:, 1:]

            self.training_data.append(uNParray)
        if selectedCols[1] == True:
            df = pd.read_csv(filename, usecols=[i for i in alphaCols])
            alphaNParray = df.to_numpy()
            if normalizedData and not loadAll:
                # ignore first col of alpha
                alphaNParray = alphaNParray[:, 1:]
            self.training_data.append(alphaNParray)
        if selectedCols[2] == True:
            df = pd.read_csv(filename, usecols=[i for i in hCol])
            hNParray = df.to_numpy()
            self.training_data.append(hNParray)

        # shuffle data
        if (shuffleMode):
            indices = np.arange(hNParray.shape[0])
            np.random.shuffle(indices)
            for idx in range(len(self.training_data)):
                self.training_data[idx] = self.training_data[idx][indices]

        end = time.perf_counter()
        print("Data loaded. Elapsed time: " + str(end - start))

        return True

    def getTrainingData(self):
        return self.training_data

    def select_training_data(self):
        pass

    def training_data_postprocessing(self):
        return 0

    def evaluate_model_normalized(self, u_test, alpha_test, h_test):
        """
        brief: runs a number of tests and evalutations to determine test errors of the model.
        input: u_test, dim (nS, N)
               alpha_test, dim (nS, N)
               h_test, dim(nS,1)
        return: True, if run successfully. Prints several plots and pictures to file.
        """

        [u_pred, alpha_pred, h_pred] = self.call_network(u_test)

        # create the loss functions
        def pointwise_diff(true_samples, pred_samples):
            """
            brief: computes the squared 2-norm for each sample point
            input: trueSamples, dim = (ns,N)
                   predSamples, dim = (ns,N)
            returns: mse(trueSamples-predSamples) dim = (ns,)
            """
            loss_val = tf.keras.losses.mean_squared_error(true_samples, pred_samples)
            return loss_val

        diff_h = pointwise_diff(h_test, h_pred)
        diff_alpha = pointwise_diff(alpha_test, alpha_pred)
        diff_u = pointwise_diff(u_test, u_pred)
        np.linspace(0, 1, 10)
        utils.plot1D([np.linspace(0, 1, 10)], [np.linspace(0, 1, 10), 2 * np.linspace(0, 1, 10)], ['t1', 't2'],
                     'test', log=False)

        utils.plot1D([u_test[:, 1]], [h_pred, h_test], ['h pred', 'h'], 'h_over_u', log=False)
        utils.plot1D([u_test[:, 1]], [alpha_pred[:, 1], alpha_test[:, 1]], ['alpha1 pred', 'alpha1 true'],
                     'alpha1_over_u1',
                     log=False)
        utils.plot1D([u_test[:, 1]], [alpha_pred[:, 0], alpha_test[:, 0]], ['alpha0 pred', 'alpha0 true'],
                     'alpha0_over_u1',
                     log=False)
        utils.plot1D([u_test[:, 1]], [u_pred[:, 0], u_test[:, 0]], ['u0 pred', 'u0 true'], 'u0_over_u1', log=False)
        utils.plot1D([u_test[:, 1]], [u_pred[:, 1], u_test[:, 1]], ['u1 pred', 'u1 true'], 'u1_over_u1', log=False)
        utils.plot1D([u_test[:, 1]], [diff_alpha, diff_h, diff_u], ['difference alpha', 'difference h', 'difference u'],
                     'errors', log=True)

        return 0

    def evaluateModel(self, u_test, alpha_test, h_test):
        """
        brief: runs a number of tests and evalutations to determine test errors of the model.
        input: u_test, dim (nS, N)
               alpha_test, dim (nS, N)
               h_test, dim(nS,1)
        return: True, if run successfully. Prints several plots and pictures to file.
        """

        # normalize data
        [u_pred_scaled, alpha_pred_scaled, h_pred_scaled] = self.call_scaled(u_test)

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

        # compute errors
        # diff_h = pointwiseDiff(h_test, h_pred)
        # diff_alpha = pointwiseDiff(alpha_test, alpha_pred)
        diff_u = pointwiseDiff(u_test, u_pred_scaled)

        # print losses
        utils.scatterPlot2D(u_test, diff_u, name="err in u over u", log=False, show_fig=False)
        # utils.plot1D(u_test[:, 1], [u_pred[:, 0], u_test[:, 0]], ['u0 pred', 'u0 true'], 'u0_over_u1', log=False)
        # utils.plot1D(u_test[:, 1], [u_pred[:, 1], u_test[:, 1]], ['u1 pred', 'u1 true'], 'u1_over_u1', log=False)

        return 0


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


class HaltWhenCallback(tf.keras.callbacks.Callback):
    def __init__(self, quantity, tol):
        """
        Should be used in conjunction with
        the saving criterion for the model; otherwise
        training will stop without saving the model with quantity <= tol
        """
        super(HaltWhenCallback, self).__init__()
        if type(quantity) == str:
            self.quantity = quantity
        else:
            raise TypeError('HaltWhen(quantity,tol); quantity must be a string for a monitored quantity')
        self.tol = tol

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1:
            if logs.get(self.quantity) < self.tol:
                print('\n\n', self.quantity, ' has reached', logs.get(self.quantity), ' < = ', self.tol,
                      '. End Training.')
                self.model.stop_training = True
        else:
            pass
