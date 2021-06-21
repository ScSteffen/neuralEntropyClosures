"""
author: Steffen Schotthöfer
date: 11.05.2020
brief: Evaluate Kullback Leibner Divergence as objective function
"""

### imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

import src.utils as utils
import src.math as math

# import tensorflow.keras.backend as K
import matplotlib.pyplot as plt


# plt.style.use("kitish")


# ------  Code starts here --------

def main():
    ## generate some alphas.
    alpha_1_true = [[-1]]
    ns = 400
    alpha_1 = np.linspace(-50, 50, ns)

    alpha_1 = tf.constant(alpha_1, shape=(ns, 1), dtype=tf.float32)
    alpha_1_true = tf.constant(alpha_1_true, shape=(1, 1), dtype=tf.float32)
    # reconstruct solutions
    eTools = math.EntropyTools(1)
    alpha = eTools.reconstruct_alpha(alpha_1)
    alpha_true = eTools.reconstruct_alpha(alpha_1_true)

    KL_div = eTools.KL_divergence(alpha_true, alpha)  # + eTools.KL_divergence(alpha, alpha_true)
    alpha_true_extend = tf.repeat(alpha_true, ns, axis=0)
    h = eTools.compute_h(eTools.reconstruct_u(alpha), alpha)
    h_true = eTools.compute_h(eTools.reconstruct_u(alpha_true_extend), alpha_true_extend)
    u = eTools.reconstruct_u(alpha)
    MSE = pointwiseDiff(h, h_true)
    utils.plot1D([alpha_1[:, 0]], [KL_div[:, 0], MSE], labels=['KL_Divergence', 'MSE'], name="kl_divergence",
                 show_fig=True, log=False)
    utils.plot1D([u[:, 1]], [KL_div[:, 0], MSE], labels=['KL_Divergence', 'MSE'],
                 name="kl_divergence",
                 show_fig=True, log=False)
    return 0


def pointwiseDiff(trueSamples, predSamples):
    """
    brief: computes the squared 2-norm for each sample point
    input: trueSamples, dim = (ns,N)
           predSamples, dim = (ns,N)
    returns: mse(trueSamples-predSamples) dim = (ns,)
    """
    loss_val = tf.keras.losses.mean_squared_error(trueSamples, predSamples)
    return loss_val


if __name__ == '__main__':
    main()
