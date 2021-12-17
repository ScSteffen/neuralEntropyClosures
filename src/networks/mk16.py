'''
Network class "MK16" for the neural entropy closure.
Encoder Decoder Architecture, that is tasked to find an entropy closure on its own

Author: Steffen Schotthöfer
Version: 0.0
Date 17.12.2021
'''

'''
Network class "MK15" for the neural entropy closure.
Dense feed forward network with ResNet structure for direct training on scaled alpha.
Includes an Entropy wrapper that reconstructs u.
Has a loss that enforeces monotonicity of the predictions empirically

Author: Steffen Schotthöfer
Version: 0.0
Date 13.08.2021
'''
import csv
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from os import path
from sklearn.preprocessing import MinMaxScaler

from src.networks.basenetwork import BaseNetwork
from src.networks.entropyautoencoder import EntropyAutoEncoder


class MK16Network(BaseNetwork):

    def __init__(self, normalized: bool, input_decorrelation: bool, polynomial_degree: int, spatial_dimension: int,
                 width: int, depth: int, loss_combination: int, save_folder: str = "", scale_active: bool = True,
                 gamma_lvl: int = 0):
        if save_folder == "":
            custom_folder_name = "MK16_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
        else:
            custom_folder_name = save_folder
        if normalized:
            print("Error. Not implemented for normalized data")
            # exit(1)

        super(MK16Network, self).__init__(normalized=normalized, polynomial_degree=polynomial_degree,
                                          spatial_dimension=spatial_dimension, width=width, depth=depth,
                                          loss_combination=loss_combination, save_folder=custom_folder_name,
                                          input_decorrelation=input_decorrelation, scale_active=scale_active,
                                          gamma_lvl=gamma_lvl)

        # hacked in
        self.input_dim += 1

    def create_model(self) -> bool:
        model = EntropyAutoEncoder(polynomial_degree=self.poly_degree, spatial_dimension=self.spatial_dim,
                                   model_depth=self.model_depth, model_width=self.model_width)
        batch_size = 3  # dummy entry
        building_tensor = tf.ones(shape=(batch_size, self.input_dim), dtype=tf.float32)
        test_out = model(building_tensor)
        # model.build(input_shape=(batch_size, self.input_dim))
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=self.optimizer,
                      metrics=['mean_absolute_error', 'mean_squared_error'])
        self.model = model
        return True

    def call_training(self, val_split=0.1, epoch_size=2, batch_size=128, verbosity_mode: int = 1,
                      callback_list: list = []):
        '''
        Calls training depending on the MK model
        '''

        # u_tf = tf.constant(self.training_data[0][:50, :])
        alpha_tf = tf.constant(self.training_data[1][:50, :])
        res = self.model(alpha_tf)
        # h_tf = tf.constant(self.training_data[2][:50, :])
        # [a_res, a_res, u_res, h_res] = self.model(alpha_tf)

        x_data = self.training_data[1]
        y_data = tf.constant(self.training_data[1], dtype=tf.float32)
        self.model.fit(x=x_data, y=y_data, validation_split=val_split, epochs=epoch_size,
                       batch_size=batch_size, verbose=verbosity_mode, callbacks=callback_list, shuffle=True)
        return self.history

    def select_training_data(self):
        return [True, True, True]

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
                # sup norm scaling
                new_len = self.training_data[1].shape[0] * self.training_data[1].shape[1]
                output = np.reshape(self.training_data[1], (new_len, 1))
                scaler = MinMaxScaler()
                scaler.fit(output)
                self.scaler_max = float(scaler.data_max_)
                self.scaler_min = float(scaler.data_min_)
            # scale to [-1,1]
            self.training_data[1] = -1.0 + 2 / (self.scaler_max - self.scaler_min) * (
                    self.training_data[1] - self.scaler_min)
            print(
                "Output of network has internal scaling with alpha_max= " + str(
                    self.scaler_max) + " and alpha_min= " + str(
                    self.scaler_min))
            print("New alpha_min= " + str(self.training_data[1].min()) + ". New alpha_max= " + str(
                self.training_data[1].max()))
        else:
            self.scaler_max = 1.0
            self.scaler_min = 0.0
        return True

    def call_network(self, u_complete, legacy_mode=False):
        """
        brief: Only works for maxwell Boltzmann entropy so far.
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: u_complete, dims = (nS x N)
        returns: alpha_complete_predicted, dim = (nS x N)
                 u_complete_reconstructed, dim = (nS x N)
                 h_predicted, dim = (nS x 1)
        """
        u_reduced = u_complete[:, 1:]  # chop of u_0
        [alpha_predicted, mono_loss, u_predicted, h_prediced] = self.model(u_reduced)
        alpha_complete_predicted = self.model.reconstruct_alpha(tf.cast(alpha_predicted, dtype=tf.float64))
        u_complete_reconstructed = self.model.reconstruct_u(alpha_complete_predicted)

        return [u_complete_reconstructed, alpha_complete_predicted, h_prediced]

    def call_scaled_64(self, u_non_normal: np.ndarray, legacy_mode=False) -> list:
        """
        brief: Only works for maxwell Boltzmann entropy so far.
        Calls the network with non normalized moments. (first the moments get normalized, then the network gets called,
        then the upscaling mechanisms get called, then the original entropy gets computed.)
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: u_complete, dims = (nS x N)
        returns: [u,alpha,h], where
                 alpha_complete_predicted_scaled, dim = (nS x N)
                 u_complete_reconstructed_scaled, dim = (nS x N)
                 h_predicted_scaled, dim = (nS x 1)
        """
        u_non_normal = tf.constant(u_non_normal, dtype=tf.float64)
        return self.model.call_scaled(u_non_normal)

    def load_model(self, file_name=None):
        # TODO

        return 0

    def load_training_data(self, shuffle_mode: bool = False, sampling: int = 0, load_all: bool = False,
                           normalized_data: bool = False, train_mode: bool = False, gamma_level: int = 0) -> bool:
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
        load_all = True  ### Only difference to original laod_training_data!

        self.training_data = []

        ### Create trainingdata filename"
        filename = "data/" + str(self.spatial_dim) + "D/Monomial_M" + str(self.poly_degree) + "_" + str(
            self.spatial_dim) + "D"
        if normalized_data:
            filename = "data/" + str(self.spatial_dim) + "D/Monomial_M" + str(self.poly_degree) + "_" + str(
                self.spatial_dim) + "D_normal"
        # add sampling information
        if sampling == 1:
            filename = filename + "_alpha"
        elif sampling == 2:
            filename = filename + "_gaussian"
        # add regularization information
        if gamma_level > 0:
            filename = filename + "_gamma" + str(gamma_level)

        filename = filename + ".csv"

        print("Loading Data from location: " + filename)
        # determine which cols correspond to u, alpha and h
        u_cols = list(range(1, self.csvInputDim + 1))
        alpha_cols = list(range(self.csvInputDim + 1, 2 * self.csvInputDim + 1))
        h_col = [2 * self.csvInputDim + 1]

        selected_cols = self.select_training_data()  # outputs a boolean triple.

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
        if selected_cols[0] and not train_mode:
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
            print("Shifting the data accordingly if network architecture is MK15 or newer...")
        else:
            print("Warning: Mean of training data moments was not computed")
        return True
