"""
brief: class that performs training data statistics
author: Steffen Schotthöfer
date: 26.08.21
"""

import numpy as np
import tensorflow as tf


class DataStatistics:
    data: np.ndarray  # data to perform statistical analysis. dim =
    data_mean_free: np.ndarray  # mean substracted from data
    cov_matrix: np.ndarray  # covariance matrix of the dataset
    mean_vector: np.ndarray  # mean vector of the dataset
    cov_ev_trafo_mat: np.ndarray  # transpose of eigenvector matrix of cov matrix

    def __init__(self, data: np.ndarray):
        self.data = data
        self.data_mean_free = self.data - self.get_mean()

    def get_mean(self):
        self.mean_vector = np.mean(self.data, axis=0)
        return self.mean_vector

    def get_cov(self):
        self.cov_matrix = np.cov(self.data_mean_free, rowvar=False)
        return self.cov_matrix

    def compute_ev_cov(self):
        [w, v] = np.linalg.eigh(self.cov_matrix)
        self.cov_ev_trafo_mat = v
        return 0

    def transform_data(self):
        data = tf.constant(self.data)
        trafo = tf.constant(self.cov_ev_trafo_mat)
        trafoT = tf.constant(self.cov_ev_trafo_mat.T)
        data_res = tf.matmul(trafo, data, transpose_b=True)
        data_r2 = tf.matmul(data, trafoT)
        print(data_res)
        print(data_r2)
        test_cov = np.cov(data_res, rowvar=True)
        print(test_cov)
        test_cov = np.cov(data_r2, rowvar=False)
        print(test_cov)

        return 0
