import numpy as np
import math


def find_nearest(array, value):
    """
    Given a sorted array and a value, find the index of the array such that
    array[idx] is the closest to value.
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1
    else:
        return idx
