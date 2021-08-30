"""
brief: class that performs training data statistics
author: Steffen Schotthöfer
date: 26.08.21
"""

import numpy as np


class DataStatistics:
    data: np.ndarray  # data to perform statistical analysis. dim =

    def __init__(self, data: np.ndarray):
        self.data = data

    def get_mean(self):
        return np.mean(self.data, axis=0)

    def get_cov(self):
        return np.cov(self.data, rowvar=False)
