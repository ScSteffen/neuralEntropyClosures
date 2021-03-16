'''
Script to conduct error Analysis of the Training
Author: Steffen Schotthöfer
Date: 15.03.2021
'''

from src.neuralClosures.configModel import initNeuralClosure
import src.utils as utils
import src.math as math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("kitish")


def main():
    filenameModel = "models/01_errorAnalysis_M1_1D/best_model.h5"
    filenameData = "data/1_stage/1D/Monomial_M1_1D_normal.csv"
    inputDim = 2

    # Load Model
    model = utils.loadTFModel(filenameModel)

    # Load Data
    [u, alpha, h] = utils.loadData(filenameData, inputDim)

    # Model Predictions
    h_pred = utils.evaluateModel(model, u)
    alpha_pred = utils.evaluateModelDerivative(model, u)

    # plot errors
    # h - h_pred over u
    # plot1D(u[:, 1], abs/(h - h_pred)/h))
    # alpha - alpha_pred
    # alphaErr = np.linalg.norm(alpha - alpha_pred, 1)
    # print(alpha[:, 1] - alpha_pred[:, 1])
    x = u[:, 1]
    ys = [alpha[:, 1], alpha_pred[:, 1]]
    labels = ["alpha 1", "alpha 1 pred"]
    # plot1D(x, ys, labels,'alpha_0')
    ys = [alpha[:, 0], alpha_pred[:, 0]]
    labels = ["alpha 0", "alpha 0 pred"]
    # plot1D(x, ys, labels, 'alpha_0')
    ys = [relDifferenceScalar(alpha[:, 0], alpha_pred[:, 0]), relDifferenceScalar(alpha[:, 1], alpha_pred[:, 1]),
          relDifferenceScalar(h, h_pred)]
    labels = ["diff alpha0", "diff alpha1", "diff h"]
    # plot1D(x, ys, labels, 'differences')

    # Compare u and reconstructed u
    [mu, w] = math.qGaussLegendre1D(100)  # Create quadrature
    mBasis = math.computeMonomialBasis1D(mu, 1)  # Create basis

    # Sanity check: reconstruct u with original alpha
    erg = math.reconstructU(alpha, mBasis, w)
    err = relDifference(u, erg)
    erg_neural = math.reconstructU(alpha_pred, mBasis, w)
    err_neural = relDifference(u, erg_neural)
    # plot1D(x, [err, err_neural], ['err', 'err neural'], 'Error in U')

    # plot error in u and error in alpha
    err_alpha = relDifference(alpha_pred, alpha)
    plot1D(x, [err_alpha, err_neural], ['errAlpha', 'errU'], 'Error in network prediction')

    # do error analysis at one point
    u_spec = u[25000]
    alpha_spec = alpha[25000]
    # disturb alpha in direction e = [0,1] with different stepsize

    us = np.repeat(u_spec[np.newaxis, :], 20, axis=0)
    alphas = np.repeat(alpha_spec[np.newaxis, :], 20, axis=0)
    alphas_dist = np.copy(alphas)
    delta = 1

    for i in range(0, 20):
        alphas_dist[i, 1] = alphas_dist[i, 1] * (1 + delta)
        delta = delta / 2

    u_rec_spec = math.reconstructU(alphas_dist, mBasis, w)

    err_alpha = relDifference(alphas_dist, alphas)
    err_u = relDifference(u_rec_spec, us)
    plot1D(range(0, 20), [err_alpha, err_u], ['errAlpha', 'errU'], 'Disturbed_alpha')

    return 0


def plot1D(x, ys, labels=[], name='defaultName'):
    plt.clf()
    for y in ys:
        plt.plot(x, y)
    plt.legend(labels)
    plt.yscale('log')
    plt.savefig("figures/" + name + ".png")
    return 0


def relDifferenceScalar(x1, x2):
    '''
    input: x1,x2: dim ns
    returns: rel difference vector (dim ns)
    '''
    return abs((x1 - x2) / np.maximum(abs(x1), abs(x2)))


def relDifference(x1, x2):
    '''
    input: x1,x2: dim nsxN
    returns: rel difference vector (dim ns)
    '''
    absDiff = np.linalg.norm((x1 - x2), axis=1, ord=1)
    normalization = np.maximum(np.linalg.norm(x1, axis=1, ord=1), np.linalg.norm(x2, axis=1, ord=1))
    return absDiff / normalization


if __name__ == '__main__':
    main()
