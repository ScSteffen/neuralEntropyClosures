'''
Base Network class for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''

import csv
import time
from os import path, makedirs, walk

import numpy as np
import pandas as pd
### imports ###
# python modules
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# intern modules
from src import utils
from src.networks.customcallbacks import HaltWhenCallback, LossAndErrorPrintingCallback, LearningRateSchedulerWithWarmup


### class definitions ###


class BaseNetwork:
    normalized: bool  # Determines if model works with normalized data
    poly_degree: int  # Degree of basis function polynomials
    spatial_dim: int  # spatial dimension of problem
    input_dim: int  # dimension of network input
    model_width: int  # width of the hidden layers
    model_depth: int  # number of hidden layers (or ICNN blocks)
    optimizer: str  # choice of optimizer
    folder_name: str  # place where model and logs are stored
    history: list  # list of training history objects
    loss_weights: list  # one hot vector to enable/disable loss functions
    model: tf.keras.Model  # the neural network model
    model_legacy: tf.keras.Model  # the neural network model
    # list of ndarrays containing the training data: [u,alpha,h,h_max,h_min]
    training_data: list
    scaler_max: float  # for output scaling
    scaler_min: float  # for output scaling
    scale_active: bool  # flag if output scaling is active
    mean_u: np.ndarray  # mean value of the input moments
    cov_u: np.ndarray  # covariance of input moments
    cov_ev: np.ndarray  # eigenvalues of cov matrix of input moments
    input_decorrelation: bool  # flag to turn on decorrelation of input variables
    # regularization parameter for regularized entropy closures
    regularization_gamma: float
    loss_comp_dict: dict = {0: [1, 0, 0, 0], 1: [1, 1, 0, 0], 2: [1, 1, 10, 0],
                            3: [0, 0, 0, 1]}  # hash table for loss combination
    # hash table for input dimension depending on polyDegree
    input_dim_dict_2D: dict = {1: 3, 2: 6, 3: 10, 4: 15, 5: 21}
    input_dim_dict_3D_sh: dict = {1: 4, 2: 9, 3: 16}
    input_dim_dict_2D_sh: dict = input_dim_dict_2D
    input_dim_dict_1D_sh: dict = {1: 2}

    rotated: bool

    def __init__(self, normalized: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str, input_decorrelation: bool,
                 scale_active: bool, gamma_lvl: int, basis: str = "monomial", rotated=False):
        if gamma_lvl == 0:
            self.regularization_gamma = 0.0
        else:
            self.regularization_gamma = 10 ** (-1.0 * gamma_lvl)
        self.model_legacy = None
        self.scale_active = scale_active
        self.normalized = normalized
        self.input_decorrelation = input_decorrelation
        if self.input_decorrelation:
            print("Model is build with mean shift and decorrelation layers.")
        self.poly_degree: int = polynomial_degree
        self.spatial_dim: int = spatial_dimension
        self.model_width: int = width
        self.model_depth: int = depth
        self.optimizer: str = 'adam'
        self.folder_name: str = "models/" + save_folder
        self.history: list = []
        self.scaler_max = 1.0  # default is no scaling
        self.scaler_min = 0.0  # default is no scaling
        self.basis = basis
        self.rotated = rotated
        # --- Determine loss combination ---
        if loss_combination < 4:
            self.loss_weights = self.loss_comp_dict[loss_combination]
        else:
            print("Error. Loss combination not supported.")
            exit(1)

        # --- Determine inputDim by MaxDegree ---
        if self.basis == "monomial":
            if spatial_dimension == 1:
                self.input_dim = polynomial_degree + 1
            elif spatial_dimension == 2:
                if self.poly_degree > 5:
                    print(
                        "Polynomial degeree higher than 5 not supported atm")
                    exit(1)
                self.input_dim = self.input_dim_dict_2D[self.poly_degree]
            else:
                raise ValueError(
                    "Saptial dimension other than 1 or 2 not supported atm")
        elif self.basis == "spherical_harmonics":
            if spatial_dimension == 3:
                self.input_dim = self.input_dim_dict_3D_sh[self.poly_degree]
            elif spatial_dimension == 2:
                self.input_dim = self.input_dim_dict_2D_sh[self.poly_degree]
            elif spatial_dimension == 1:
                self.input_dim = self.input_dim_dict_1D_sh[self.poly_degree]
            else:
                raise ValueError(
                    "Saptial dimension other than 2 or 3 not supported atm")
        else:
            raise ValueError("Basis >" + str(self.basis) + "< not supported")

        if rotated and self.poly_degree == 1:
            self.input_dim -= 1
        else:
            self.rotated = False  # only change architecture for m1

        self.csvInputDim = self.input_dim  # only for reading csv data

        if self.normalized:
            self.input_dim = self.input_dim - 1

        self.mean_u = np.zeros(shape=(self.input_dim,), dtype=float)
        self.cov_u = np.zeros(
            shape=(self.input_dim, self.input_dim), dtype=float)
        self.cov_ev = np.zeros(
            shape=(self.input_dim, self.input_dim), dtype=float)

    def create_model(self) -> bool:
        pass

    def call_network(self, u_complete) -> list:
        """
        Brief: This does not reconstruct u, but returns original u. Careful here!

        # Input: input.shape = (nCells, nMaxMoment), nMaxMoment = 9 in case of MK3
        # Output: Gradient of the network wrt input
        """
        x_model = tf.Variable(u_complete)

        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # same as neuralClosureModel.model.predict(x)
            predictions = self.model(x_model, training=False)

        gradients = tape.gradient(predictions, x_model)

        return [x_model, gradients, predictions]

    def call_scaled(self, u_non_normal):
        """
        Brief: By default the same behaviour as call_network.
              This method is subclassed by all networks, that only take normalized moments (MK11 and newer)
        """
        return self.call_network(u_non_normal)

    def call_scaled_64(self, u_non_normal: np.ndarray, legacy_mode=False):
        """
        Brief: Same as call_scaled, but all variables are cast to fp64
        By default just calls call_scaled
        """
        return self.call_scaled(u_non_normal)

    def config_start_training(self, val_split: float = 0.1, epoch_count: int = 2, curriculum: int = 1,
                              batch_size: int = 500, verbosity: int = 1, processing_mode: int = 0):
        '''
        Method to train network
        '''

        # print scaling data to file.
        scaling_file_name = self.folder_name + '/scaling_data/min_max_scaler.csv'
        if not path.exists(self.folder_name + '/scaling_data'):
            makedirs(self.folder_name + '/scaling_data')
        with open(scaling_file_name, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([self.scaler_min, self.scaler_max])

        # Set double precision training for CPU training #TODO
        if processing_mode == 0:
            tf.keras.backend.set_floatx('float32')
        elif processing_mode == 1:
            tf.keras.backend.set_floatx('float32')

        # Create callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.folder_name + '/best_model', monitor='val_loss', mode='min',
                                                     save_best_only=True, verbose=verbosity)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.0001, patience=10,
                                              verbose=1)

        if curriculum == 0:  # Epoch chunk training
            # mc_checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=self.filename + '/model_saved',
            #                                         save_weights_only=False,
            #                                         verbose=1)
            epochChunks = 4
            # Split Training epochs
            mini_epoch = int(epoch_count / epochChunks)

            for i in range(0, curriculum):
                print("Training with increasing batch size")
                #  perform a batch doublication every 1/10th of the epoch count
                print("Current Batch Size: " + str(batch_size))

                # assemble callbacks
                callbackList = []
                csv_logger = self.create_csv_logger_cb()
                if verbosity == 0:
                    callbackList = [
                        mc_best, LossAndErrorPrintingCallback(), csv_logger]
                else:
                    callbackList = [mc_best, csv_logger]

                # start Training
                self.history = self.call_training(val_split=val_split, epoch_size=epoch_count, batch_size=batch_size,
                                                  verbosity_mode=verbosity, callback_list=callbackList)
                batch_size = 2 * batch_size

            self.concat_history_files()

        elif curriculum >= 1:  # learning rate scheduler
            print("Training with learning rate scheduler with warmup of 5 epochs")
            # We only use this at the moment
            initial_lr = float(0.005)
            drop_rate = (epoch_count / 3)
            stop_tol = 4e-6
            mt_patience = int(epoch_count / 10)
            min_delta = stop_tol / 10

            # specific callbacks
            def step_decay(epoch):
                # step_size = initial_lr * np.power(10, (-epoch / drop_rate))
                # return step_size
                # Initial learning rate
                end_lr = 0.00005  # Final learning rate
                total_epochs = min(100, epoch_count)  # Total number of epochs

                if epoch < total_epochs:
                    return initial_lr - (epoch / total_epochs) * (initial_lr - end_lr)
                else:
                    return end_lr

            # TODO LR SCHEDULER
            LR = LearningRateSchedulerWithWarmup(
                warmup_epochs=5, lr_schedule=step_decay)
            HW = HaltWhenCallback('val_output_3_loss', stop_tol)
            ES = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                                  verbose=1, patience=mt_patience, min_delta=min_delta)
            csv_logger, tensorboard_logger = self.create_csv_logger_cb()

            if verbosity == 1:
                callbackList = [mc_best, csv_logger,
                                tensorboard_logger, HW, LR]  # , ES]  # LR,
            else:
                callbackList = [mc_best, LossAndErrorPrintingCallback(), csv_logger, tensorboard_logger,
                                HW, LR]  # , ES]  # LR,

            # start Training
            self.history = self.call_training(val_split=val_split, epoch_size=epoch_count, batch_size=batch_size,
                                              verbosity_mode=verbosity, callback_list=callbackList)
            print("Model saved to location: " + self.folder_name)

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
        for log in historyLogs:
            historyLogsDF.append(pd.read_csv(
                self.folder_name + '/historyLogs/' + log))

        totalDF = pd.concat(historyLogsDF, ignore_index=True)

        # postprocess:
        numEpochs = len(totalDF.index)
        totalDF['epoch'] = np.arange(numEpochs)
        # write
        totalDF.to_csv(self.folder_name +
                       '/historyLogs/CompleteHistory.csv', index=False)
        return 0

    def create_csv_logger_cb(self):
        '''
        dynamically creates a csvlogger and tensorboard logger
        '''
        # check if dir exists
        if not path.exists(self.folder_name + '/historyLogs/'):
            makedirs(self.folder_name + '/historyLogs/')

        # checkfirst, if history file exists.
        logName = self.folder_name + '/historyLogs/history_001_'
        count = 1
        while path.isfile(logName + '.csv'):
            count += 1
            logName = self.folder_name + \
                '/historyLogs/history_' + str(count).zfill(3) + '_'

        logFile = logName + '.csv'
        # create logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(logFile)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=logName, histogram_freq=1)
        return csv_logger, tensorboard_callback

    def save_model(self):
        """
        Saves best model to .pb file
        """
        # self.model.load_weights(self.folder_name + '/best_model.h5')
        self.model.save(self.folder_name + '/best_model', save_format='tf')
        print("Model successfully saved to file and .h5")
        # with open(self.filename + '/trainingHistory.json', 'w') as file:
        #    json.dump(self.model.history.history, file)
        return 0

    def load_model(self, file_name=None):
        used_file_name = self.folder_name
        if file_name != None:
            used_file_name = file_name

        # read scaling data
        scaling_file_name = used_file_name + '/scaling_data/min_max_scaler.csv'
        if not path.exists(scaling_file_name):
            print("Scaling Data is missing. Expected in: " + scaling_file_name)
            exit(1)
        with open(scaling_file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                scaling_data = row
        self.scaler_min = float(scaling_data[0])
        self.scaler_max = float(scaling_data[1])
        self.create_model(rotated=rotated)
        used_file_name = used_file_name + '/best_model/'

        if not path.exists(used_file_name):
            print("Model does not exists at this path: " + used_file_name)
            exit(1)
        model = tf.keras.models.load_model(used_file_name, custom_objects={
                                           "CustomModel": self.model})
        self.model.load_weights(used_file_name)
        print("Model loaded from file ")
        return 0

    def print_weights(self):
        for layer in self.model.layers:
            weights = layer.get_weights()  # list of numpy arrays
            print(weights)
            print("---------------------------------------------")

    def show_model(self) -> bool:
        self.model.summary()
        return True

    def load_training_data(self, shuffle_mode: bool = False, sampling: int = 0, load_all: bool = False,
                           normalized_data: bool = False, train_mode: bool = False, gamma_level: int = 0,
                           rotated=False) -> bool:
        """
        Loads the training data
        params: shuffle_mode = shuffle loaded Data  (yes,no)
                sampling : 0 = momnents uniform, 1 = alpha uniform, 2 = alpha uniform
                load_all : If true, loads moments of order zero
                normalized_data : load normalized data  (u_0=1)
                train_mode : If true, computes data statistics to build the preprocessing head for model
                gamma_level: specifies the level of regularization of the data

        return: True, if loading successful
        """
        self.training_data = []

        if self.basis == "monomial":
            # Create trainingdata filename"
            filename = "data/" + str(self.spatial_dim) + "D/Monomial_M" + str(self.poly_degree) + "_" + str(
                self.spatial_dim) + "D"
            if normalized_data:
                filename = "data/" + str(self.spatial_dim) + "D/Monomial_M" + str(self.poly_degree) + "_" + str(
                    self.spatial_dim) + "D_normal"
        elif self.basis == "spherical_harmonics":
            # Create trainingdata filename"
            filename = "data/" + str(self.spatial_dim) + "D/SphericalHarmonics_M" + str(self.poly_degree) + "_" + str(
                self.spatial_dim) + "D"
            if normalized_data:
                filename = "data/" + str(self.spatial_dim) + "D/SphericalHarmonics_M" + str(
                    self.poly_degree) + "_" + str(
                    self.spatial_dim) + "D_normal"
        else:
            raise ValueError("Not supported basis: " + self.basis)

        # add sampling information
        if sampling == 1:
            filename = filename + "_alpha"
        elif sampling == 2:
            filename = filename + "_gaussian"
        # add regularization information
        if gamma_level > 0:
            filename = filename + "_gamma" + str(gamma_level)
        # add rotation informtaion
        if rotated:
            filename = filename + "_rot"
            # self.input_dim = self.input_dim - 1
            # self.csvInputDim = self.csvInputDim - 1
        filename = filename + ".csv"

        print("Loading Data from location: " + filename)
        # determine which cols correspond to u, alpha and h
        u_cols = list(range(1, self.csvInputDim + 1))
        alpha_cols = list(
            range(self.csvInputDim + 1, 2 * self.csvInputDim + 1))
        h_col = [2 * self.csvInputDim + 1]

        # outputs a boolean triple.
        selected_cols = self.select_training_data()

        # selected_cols = [True, False, True]

        start = time.perf_counter()
        if selected_cols[0]:
            df = pd.read_csv(filename, usecols=[i for i in u_cols])
            u_ndarray = df.to_numpy()
            if normalized_data and not load_all:
                # ignore first col of u
                u_ndarray = u_ndarray[:, 1:]
            self.training_data.append(u_ndarray)
        if selected_cols[1]:
            df = pd.read_csv(filename, usecols=[i for i in alpha_cols])
            alpha_ndarray = df.to_numpy()
            if normalized_data and not load_all:
                # ignore first col of alpha
                alpha_ndarray = alpha_ndarray[:, 1:]
            self.training_data.append(alpha_ndarray)
        if selected_cols[2]:
            df = pd.read_csv(filename, usecols=[i for i in h_col])
            h_ndarray = df.to_numpy()
            self.training_data.append(h_ndarray)

        # shuffle data
        if shuffle_mode:
            indices = np.arange(h_ndarray.shape[0])
            np.random.shuffle(indices)
            for idx in range(len(self.training_data)):
                self.training_data[idx] = self.training_data[idx][indices]

        end = time.perf_counter()
        print("Data loaded. Elapsed time: " + str(end - start))
        if selected_cols[0] and self.input_decorrelation:
            print("Computing input data statistics")
            self.mean_u = np.mean(u_ndarray, axis=0)
            print("Training data mean (of u) is")
            print(self.mean_u)
            print("Training data covariance (of u) is")
            self.cov_u = np.cov(u_ndarray, rowvar=False)
            print(self.cov_u)
            if self.input_dim > 1:
                [_, self.cov_ev] = np.linalg.eigh(self.cov_u)
            else:
                self.cov_ev = self.cov_u  # 1D case
            print(
                "Shifting the data accordingly if network architecture is MK11, MK12, MK13 or MK15...")
        else:
            print("Warning: Mean of training data moments was not computed")

        print("Entropy statistics: Max: " + str(np.max(h_ndarray)) +
              " Min: " + str(np.min(h_ndarray)))
        print("Langrange multiplier statistics: Max: " + str(np.max(np.linalg.norm(alpha_ndarray, axis=1))) +
              " Min: " + str(np.min(np.linalg.norm(alpha_ndarray, axis=1))))
        print("Moment statistics: Max: " + str(np.max(np.linalg.norm(u_ndarray, axis=1))) +
              " Min: " + str(np.min(np.linalg.norm(u_ndarray, axis=1))))
        return True

    def training_data_preprocessing(self, scaled_output: bool = False, model_loaded: bool = False) -> bool:
        """
        Performs a scaling on the output data (h) and scales alpha correspondingly. Sets a scale factor for the
        reconstruction of u during training and execution
        params:            scaled_output: bool = Determines if entropy is scaled to range (0,1) (other outputs scaled accordingly)
                           model_loaded: bool = Determines if models is loaded from file. Then scaling data from file is used
        returns: True if terminated successfully
        """
        self.scale_active = scaled_output
        if scaled_output:
            if not model_loaded:
                scaler = MinMaxScaler()
                scaler.fit(self.training_data[2])
                self.scaler_max = float(scaler.data_max_)
                self.scaler_min = float(scaler.data_min_)
            # scale to [0,1]
            self.training_data[2] = (
                self.training_data[2] - self.scaler_min) / (
                self.scaler_max - self.scaler_min)
            # scale correspondingly
            self.training_data[1] = self.training_data[1] / \
                (self.scaler_max - self.scaler_min)
            print("Output of network has internal scaling with h_max=" + str(self.scaler_max) + " and h_min=" + str(
                self.scaler_min))
            print("New h_min= " + str(self.training_data[2].min()) + ". New h_max= " + str(
                self.training_data[2].max()))
            print("New alpha_min= " + str(self.training_data[1].min()) + ". New alpha_max= " + str(
                self.training_data[1].max()))
        else:
            self.scaler_max = 1.0
            self.scaler_min = 0.0
        return True

    def get_training_data(self):
        return self.training_data

    def select_training_data(self):
        pass

    def evaluate_model_normalized(self, u_test, alpha_test, h_test):
        """
        brief: runs a number of tests and evalutations to determine test errors of the model.
        input: u_test, dim (nS, N)
               alpha_test, dim (nS, N)
               h_test, dim(nS,1)
        return: True, if run successfully. Prints several plots and pictures to file.
        """

        [u_pred, alpha_pred, h_pred] = self.call_network(u_test)

        # alpha = self.call_network(u_test)

        # create the loss functions
        def pointwise_diff(true_samples, pred_samples):
            """
            brief: computes the squared 2-norm for each sample point
            input: trueSamples, dim = (ns,N)
                   predSamples, dim = (ns,N)
            returns: mse(trueSamples-predSamples) dim = (ns,)
            """
            loss_val = tf.keras.losses.mean_squared_error(
                true_samples, pred_samples)
            return loss_val

        diff_h = pointwise_diff(h_test, h_pred)
        tmp = tf.reshape(alpha_pred[:, 1], shape=(
            tf.shape(alpha_pred[:, 2]).numpy()[0], 1), name=None)
        diff_alpha = pointwise_diff(alpha_test, tmp)
        tmp = tf.reshape(u_pred[:, 1], shape=(
            tf.shape(u_pred[:, 2]).numpy()[0], 1), name=None)
        diff_u = pointwise_diff(u_test, tmp)

        with open(self.folder_name + "/test_results.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            row_list = []
            for i in range(self.csvInputDim):
                row_list.append("u_" + str(i))
            for i in range(self.csvInputDim):
                row_list.append("alpha_" + str(i))
            writer.writerow(row_list + ["u", "alpha", "h"])

            for idx in range(len(diff_h.numpy())):
                r = np.concatenate((u_test[idx], alpha_test[idx], diff_u[idx].numpy().reshape((1,)),
                                    diff_alpha[idx].numpy().reshape((1,)), diff_h[idx].numpy().reshape((1,))))
                writer.writerow(r)

        # only for M_2 1D closure
        if self.poly_degree == 2 and self.spatial_dim == 1:
            utils.scatter_plot_2d(x_in=u_test[:, 1:], z_in=diff_u, lim_x=(-1, 1), lim_y=(0, 1), lim_z=(1e-5, 10),
                                  title=r"$|u-u_\theta|_2$ over ${u^r}$",
                                  folder_name=self.folder_name, name="err_u_over_u", show_fig=False, log=True)
            utils.scatter_plot_2d(x_in=u_test[:, 1:], z_in=diff_alpha, lim_x=(-1, 1), lim_y=(0, 1), lim_z=(1e-5, 10),
                                  title=r"$|\alpha_u-\alpha_\theta|_2$ over ${u^r}$",
                                  folder_name=self.folder_name, name="err_alpha_over_u", show_fig=False, log=True)
            utils.scatter_plot_2d(x_in=u_test[:, 1:], z_in=diff_h, lim_x=(-1, 1), lim_y=(0, 1), lim_z=(1e-5, 10),
                                  title=r"$|h-h_\theta|_2$ over ${u^r}$",
                                  folder_name=self.folder_name, name="err_h_over_u", show_fig=False, log=True)
            utils.scatter_plot_2d(x_in=alpha_test[:, 1:], z_in=diff_alpha, lim_x=(-20, 20), lim_y=(-20, 20),
                                  lim_z=(1e-5, 10),
                                  title=r"$|\alpha_u-\alpha_\theta|_2$ over ${u^r}$",
                                  folder_name=self.folder_name, name="err_alpha_over_u", show_fig=False, log=True)
            utils.scatter_plot_2d(x_in=alpha_test[:, 1:], z_in=diff_h, lim_x=(-20, 20), lim_y=(-20, 20),
                                  lim_z=(1e-5, 10),
                                  title=r"$|h-h_\theta|_2$ over ${u^r}$",
                                  folder_name=self.folder_name, name="err_h_over_u", show_fig=False, log=True)

        if self.poly_degree == 1 and self.spatial_dim == 1:
            np.linspace(0, 1, 10)
            utils.plot_1d([np.linspace(0, 1, 10)], [np.linspace(0, 1, 10), 2 * np.linspace(0, 1, 10)], ['t1', 't2'],
                          'test', log=False)
            utils.plot_1d([u_test[:, 1]], [h_pred, h_test], [
                'h pred', 'h'], 'h_over_u', log=False)
            utils.plot_1d([u_test[:, 1]], [alpha_pred[:, 1], alpha_test[:, 1]], ['alpha1 pred', 'alpha1 true'],
                          'alpha1_over_u1',
                          log=False)
            utils.plot_1d([u_test[:, 1]], [alpha_pred[:, 0], alpha_test[:, 0]], ['alpha0 pred', 'alpha0 true'],
                          'alpha0_over_u1',
                          log=False)
            utils.plot_1d([u_test[:, 1]], [u_pred[:, 0], u_test[:, 0]], ['u0 pred', 'u0 true'], 'u0_over_u1',
                          log=False)
            utils.plot_1d([u_test[:, 1]], [u_pred[:, 1], u_test[:, 1]], ['u1 pred', 'u1 true'], 'u1_over_u1',
                          log=False)
            utils.plot_1d([u_test[:, 1]], [diff_alpha, diff_h, diff_u],
                          ['difference alpha', 'difference h', 'difference u'],
                          'errors', log=True)
        return 0

    def evaluate_model(self, u_test, alpha_test, h_test):
        """
        brief: runs a number of tests and evalutations to determine test errors of the model.
        input: u_test, dim (nS, N)
               alpha_test, dim (nS, N)
               h_test, dim(nS,1)
        return: True, if run successfully. Prints several plots and pictures to file.
        """

        # normalize data
        [u_pred_scaled, alpha_pred_scaled,
         h_pred_scaled] = self.call_scaled(u_test)

        # create the loss functions
        def pointwise_diff(true_samples, pred_samples):
            """
            brief: computes the squared 2-norm for each sample point
            input: trueSamples, dim = (ns,N)
                   predSamples, dim = (ns,N)
            returns: mse(trueSamples-predSamples) dim = (ns,)
            """
            loss_val = tf.keras.losses.mean_squared_error(
                true_samples, pred_samples)
            return loss_val

        # compute errors
        # diff_h = pointwise_diff(h_test, h_pred)
        # diff_alpha = pointwise_diff(alpha_test, alpha_pred)
        diff_u = pointwise_diff(u_test, u_pred_scaled)

        # print losses
        utils.scatter_plot_2d(
            u_test, diff_u, name="err in u over u", log=False, show_fig=False)
        # utils.plot1D(u_test[:, 1], [u_pred[:, 0], u_test[:, 0]], ['u0 pred', 'u0 true'], 'u0_over_u1', log=False)
        # utils.plot1D(u_test[:, 1], [u_pred[:, 1], u_test[:, 1]], ['u1 pred', 'u1 true'], 'u1_over_u1', log=False)

        return 0

    """
    def normalize_data(self):
    
        # load data
        #
        [u_t, alpha_t, h_t] = self.training_data
        mBasis = tf.cast(self.model.momentBasis, dtype=tf.float64, name=None)
        qWeights = tf.cast(self.model.quadWeights, dtype=tf.float64, name=None)
        #
        #
        u_non_normal = tf.constant(u_t, dtype=tf.float64)
        alpha_non_normal = tf.constant(alpha_t, dtype=tf.float64)
        h_non_normal = tf.constant(h_t, dtype=tf.float64)
    
        # scale u and alpha
        u_normal = self.model.scale_u(u_non_normal, tf.math.reciprocal(u_non_normal[:, 0]))  # downscaling
        alpha_normal = self.model.scale_alpha(alpha_non_normal, tf.math.reciprocal(u_non_normal[:, 0]))
    
        # compute h
        f_quad = tf.math.exp(tf.tensordot(alpha_normal, mBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.tensordot(f_quad, qWeights, axes=([1], [1]))  # f*w
        tmp2 = tf.math.reduce_sum(tf.math.multiply(alpha_normal, u_normal), axis=1, keepdims=True)
        h_normal = tmp2 - tmp
    
        self.training_data = [u_normal[:, 1:], alpha_normal[:, 1:], h_normal]
        return 0
    """
