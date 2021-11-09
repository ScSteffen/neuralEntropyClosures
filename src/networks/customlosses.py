"""
brief: Selection of custom subclassed losses
"""

from tensorflow.keras.losses import Loss
import tensorflow as tf
from keras import backend


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


class RelativeMAELoss(Loss):
    """
    Computes the relative mean absolute  error between `y_true` and `y_pred`.
    `loss =  (y_true - y_pred) / (y_true)
    """

    def call(self, y_true, y_pred):
        # y_pred = tf.convert_to_tensor(y_pred)
        # y_true = tf.cast(y_true, y_pred.dtype)
        # t1 = tf.math.subtract(y_true, y_pred)
        # t2 = t1 * t1
        # t1 = tf.math.squared_difference(y_pred, y_true)
        # t2 = tf.math.squared_difference(y_true, 2 * y_true)  # tf.square(y_pred)
        # t3 = tf.divide(t1, t2)

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        diff = tf.abs((y_true - y_pred) / backend.maximum(tf.abs(y_true), backend.epsilon()))
        return backend.mean(diff, axis=-1)


def kl_divergence_loss(m_b, q_w):
    """
    KL divergence between f_u and f_true  using alpha and alpha_true.
    inputs: mB, moment Basis evaluted at quadPts, dim = (N x nq)
            quadWeights, dim = nq
    returns: KL_divergence function using mBasis and quadWeights
    """

    def reconstruct_alpha(alpha):
        """
        brief:  Reconstructs alpha_0 and then concats alpha_0 to alpha_1,... , from alpha1,...
                Only works for maxwell Boltzmann entropy so far.
                => copied from sobolev model code
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N-1)
               m    , dims = (N x nq)
               w    , dims = nq
        returns alpha_complete = [alpha_0,alpha], dim = (nS x N), where alpha_0 = - ln(<exp(alpha*m)>)
        """
        # Check the predicted alphas for +/- infinity or nan - raise error if found
        checked_alpha = tf.debugging.check_numerics(alpha, message='input tensor checking error', name='checked')
        # Clip the predicted alphas below the tf.exp overflow threshold
        clipped_alpha = tf.clip_by_value(checked_alpha, clip_value_min=-50, clip_value_max=50,
                                         name='checkedandclipped')

        tmp = tf.math.exp(tf.tensordot(clipped_alpha, m_b[1:, :], axes=([1], [0])))  # tmp = alpha * m
        alpha_0 = -tf.math.log(tf.tensordot(tmp, q_w, axes=([1], [1])))  # ln(<tmp>)
        return tf.concat([alpha_0, alpha], axis=1)  # concat [alpha_0,alpha]

    def kl_divergence(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        brief: computes the Kullback-Leibler Divergence of the kinetic density w.r.t alpha given the kinetic density w.r.t
                alpha_true
        input: alpha_true , dim= (ns,N)
               alpha , dim = (ns, N+1)
        output: pointwise KL Divergence, dim  = ns x 1
        """

        # extend alpha_true to full dimension
        alpha_true_recon = reconstruct_alpha(y_true)
        alpha_pred_recon = reconstruct_alpha(y_pred)
        # compute KL_divergence
        diff = alpha_true_recon - alpha_pred_recon
        t1 = tf.math.exp(tf.tensordot(alpha_true_recon, m_b, axes=([1], [0])))
        t2 = tf.tensordot(diff, m_b, axes=([1], [0]))
        integrand = tf.math.multiply(t1, t2)
        return tf.tensordot(integrand, q_w, axes=([1], [1]))

    return kl_divergence
