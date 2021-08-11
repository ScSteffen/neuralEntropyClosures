"""
Brief: classes that overwrite kernel.constraints from tensorflow
Author: Steffen Schotthöfer
Date: 10.08.2021
"""
import tensorflow as tf


class AbsWeightConstraint(tf.keras.constraints.Constraint):
    """
    Applies the absolute value to weights
    """

    def __init__(self):
        self.ref_value = 0.0

    def __call__(self, w):
        return tf.math.abs(w)

    def get_config(self):
        return {'ref_value': self.ref_value}


class ClipByValueConstraint(tf.keras.constraints.Constraint):
    """
    sets all weights below clip_low to the value clip_low
    """

    def __init__(self, clip_l):
        self.ref_value = 0.0

    def __call__(self, w):
        return tf.math.abs(w)

    def get_config(self):
        return {'ref_value': self.ref_value}
