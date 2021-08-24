"""
brief: Selection of custom subclassed losses
"""

from tensorflow.keras.losses import Loss
import tensorflow as tf


class MonotonicFunctionLoss(Loss):
    """
       for all elements i in a batch, the model computes
         1/N sum_(i)(sum_(i!=j) l2( dot(alpha_i-alpha_j, u_i-u_j)) )
    """

    def call(self, u_true, alpha_pred):
        ns = tf.shape(u_true)[0]
        # ns_int = u_true.shape.as_list()[0]
        ns_f = tf.cast(ns, dtype=tf.float32)
        loss = tf.constant([0.0], dtype=tf.float32)
        for i in range(ns):
            t1 = tf.math.subtract(u_true, u_true[i, :])
            t2 = tf.math.subtract(alpha_pred, alpha_pred[i, :])
            t3 = t1 * t2
            t4 = tf.reduce_mean(tf.keras.activations.relu(-1 * tf.reduce_sum(t3, 1)))
            loss += t4
        return tf.divide(loss, ns_f)
