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

    ### Compare u and reconstructed u
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
    # plot1D(x, [err_alpha, err_neural], ['errAlpha', 'errU'], 'Error in network prediction')

    ### Error Analysis of the reconstruction
    # errorAnalysisU(u, alpha, mBasis, w)
    # errorAnalysisf(alpha, mBasis, w)
    # errorAnalysisUM0()
    errorAnalysisUM1_normal()

    return 0


def errorAnalysisf(alpha, mBasis, w):
    # do error analysis at one for f (L1 error)
    alpha_spec = alpha[25000]
    # disturb alpha in direction e = [0,1] with different stepsize

    alphas = np.repeat(alpha_spec[np.newaxis, :], 50, axis=0)
    alphas_dist = np.copy(alphas)
    delta = np.power(0.5, 20)

    for i in range(0, 50):
        alphas_dist[i, 1] = alphas_dist[i, 1] * (1 + delta)
        delta = delta * 2

    f_err = math.reconstructL1F(alphas_dist - alphas, mBasis, w)
    # f = math.reconstructL1F(alphas, mBasis, w)
    err_alpha = relDifference(alphas_dist, alphas)
    plot1D(range(0, 50), [err_alpha, f_err], ['errAlpha', 'err f'], 'err in f')
    return 0


def errorAnalysisU(u, alpha, mBasis, w):
    # do error analysis at one point
    u_spec = u[25000]
    alpha_spec = alpha[25000]
    # disturb alpha in direction e = [0,1] with different stepsize

    us = np.repeat(u_spec[np.newaxis, :], 50, axis=0)
    alphas = np.repeat(alpha_spec[np.newaxis, :], 50, axis=0)
    alphas_dist = np.copy(alphas)
    delta = np.power(0.5, 20)

    for i in range(0, 50):
        alphas_dist[i, 1] = alphas_dist[i, 1] * (1 + delta)
        delta = delta * 2

    u_rec_spec = math.reconstructU(alphas_dist, mBasis, w)

    err_alpha = relDifference(alphas_dist, alphas)
    err_u = relDifference(u_rec_spec, us)
    plot1D(range(0, 50), [err_alpha, err_u], ['errAlpha', 'errU'], 'Disturbed_alpha')
    return 0


def errorAnalysisUM0():
    """
    plots u0 over alpha0 in [0,1]
    """
    [mu, w] = math.qGaussLegendre1D(100)  # Create quadrature
    mBasis = math.computeMonomialBasis1D(mu, 0)  # Create basis

    alphas = np.arange(0, 1, 0.001).reshape((1000, 1))
    us = 2 * np.exp(alphas)
    us_rec = math.reconstructU(alphas, mBasis, w)
    plot1D(alphas, [us, us_rec], ['u', 'u rec'], 'u and u rec', log=False)

    return 0


def errorAnalysisUM1_normal():
    """
    Creates random deviations of a given starting alpha, and plots error of u over the deviation
    """
    # Basis creation
    [mu, w] = math.qGaussLegendre1D(1000)  # Create quadrature
    mBasis = math.computeMonomialBasis1D(mu, 1)  # Create basis
    
    alphas = np.arange(0, 1, 0.001).reshape((1000, 1))
    alphas1 = np.ones((1000, 1))
    alphas = np.concatenate((alphas1, alphas), axis=1)

    # us = 2 * np.exp(alphas)
    us_rec = math.reconstructU(alphas, mBasis, w)
    us_orig = np.repeat(us_rec[0][np.newaxis, :], us_rec.shape[0], axis=0)
    relDiff = relDifference(us_orig, us_rec, maxMode=False)
    plot1D(alphas[:, 1], [relDiff], ['relErr'], 'relErrU M1 normal', log=False)

    return 0


def plot1D(x, ys, labels=[], name='defaultName', log=True):
    plt.clf()
    lineTypes = ['-', '--', '-.', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H',
                 '+', 'x', 'D', 'd', '|', '_']
    lineTypes = lineTypes[0:len(labels)]
    for y, lineType in zip(ys, lineTypes):
        plt.plot(x, y, lineType)
    plt.legend(labels)

    if (log):
        plt.yscale('log')

    # plt.show()
    plt.savefig("figures/" + name + ".png")
    return 0


def relDifferenceScalar(x1, x2):
    '''
    input: x1,x2: dim ns
    returns: rel difference vector (dim ns)
    '''
    return abs((x1 - x2) / np.maximum(abs(x1), abs(x2)))


def relDifference(x1, x2, maxMode=True):
    '''
    input: x1,x2: dim nsxN
    returns: rel difference vector (dim ns)
    '''
    absDiff = np.linalg.norm((x1 - x2), axis=1, ord=1)
    if maxMode == True:
        normalization = np.maximum(np.linalg.norm(x1, axis=1, ord=1), np.linalg.norm(x2, axis=1, ord=1))
    else:
        normalization = np.linalg.norm(x1, axis=1, ord=1)
    return absDiff / normalization


if __name__ == '__main__':
    main()
