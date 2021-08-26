"""
brief: class that performs training data statistics
author: Steffen Schotthöfer
date: 26.08.21
"""

import numpy as np
import scipy


class DataStatistics:
    data: np.ndarray  # data to perform statistical analysis. dim =

    def __init__(self, data: np.ndarray):
        self.data = data

    def get_mean(self):
        return scipy.mean(self.data, axis=1)
