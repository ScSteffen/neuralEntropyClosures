"""
Author: Steffen Schotthöfer
Date: 21.04.2020
Brief: Script to test the optimal convex data sampling alogrithm
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import tensorflow as tf

from src.neuralClosures.configModel import initNeuralClosure
from src import utils
from src import math


def quad_func(x):
    return x * x


def quad_func_grad(x):
    return 2 * x


def exp_func(x):
    return np.exp(x)  # / np.exp(3)


def get_errors(s_x, function, grad, tol):
    eL = []
    tL = []
    for i in range(len(s_x) - 1):
        t1 = 1 / (grad(s_x[i + 1]) - grad(s_x[i])) * (function(s_x[i]) - s_x[i] * grad(s_x[i]) - (
                function(s_x[i + 1]) - s_x[i + 1] * grad(s_x[i + 1])))
        e1 = function(t1) - (function(s_x[i]) + (t1 - s_x[i]) * grad(s_x[i]))

        # error in case that the approximation overshoots
        tmp1 = function(s_x[i + 1])
        tmp2 = function(s_x[i])
        grad_intermediate = (tmp1 - tmp2) / (s_x[i + 1] - s_x[i])

        def F(x):
            return 0.5 * (grad_intermediate - grad(x)) * (grad_intermediate - grad(x))

        result = scipy.optimize.minimize_scalar(method="bounded", fun=F, bounds=[s_x[i], s_x[i + 1]])
        if not result.success:
            ValueError("Optimization unsuccessful!")
            print(grad_intermediate)
            print((s_x[i + 1] - s_x[i]) / 2)
            print("----")

        t2 = float(result.x)
        e2 = function(s_x[i]) + grad_intermediate * (t2 - s_x[i]) - function(t2)

        if e1 > e2:
            eL.append(e1)
            tL.append(t1)
        else:
            eL.append(e2)
            tL.append(t2)

    return [eL, tL]


def sample_data_entropy_M1(u_min, u_max, tol):
    """
    performs the smart sampling algorithm on the entropy closure problem
    """

    s_x = [u_min, u_max]
    e_max = tol
    j = 0
    while e_max >= tol:

        e_max = 0
        i = 0

        [eL, tL] = get_errors(s_x, function, grad, 0.0001)
        plt.semilogy(tL, eL, '*')
        plt.savefig("figures/errors_" + str(j).zfill(3) + ".png")
        plt.clf()

        while i < len(s_x) - 1:

            # error, in case that the approximation undershoots
            t1 = 1 / (grad(s_x[i + 1]) - grad(s_x[i])) * (function(s_x[i]) - s_x[i] * grad(s_x[i]) - (
                    function(s_x[i + 1]) - s_x[i + 1] * grad(s_x[i + 1])))
            e1 = function(t1) - (function(s_x[i]) + (t1 - s_x[i]) * grad(s_x[i]))

            # error in case that the approximation overshoots
            grad_intermediate = (function(s_x[i + 1]) - function(s_x[i])) / (s_x[i + 1] - s_x[i])

            def F(x):
                return 0.5 * (grad_intermediate - grad(x)) * (grad_intermediate - grad(x))

            result = scipy.optimize.minimize_scalar(method="bounded", fun=F, bounds=[s_x[i], s_x[i + 1]])
            if not result.success:
                ValueError("Optimization unsuccessful!")
                print(grad_intermediate)
                print((s_x[i + 1] - s_x[i]) / 2)
                print("----")

            t2 = float(result.x)
            e2 = function(s_x[i]) + grad_intermediate * (t2 - s_x[i]) - function(t2)

            if e1 > e2:
                t = t1
                e = e1
                # print("undershoot worse")
            else:
                t = t2
                e = e2
                # print("overshoot worse")

            if e > e_max:
                e_max = e
            if e > tol:
                s_x.insert(i + 1, t)
                i = i + 1
            i = i + 1
        j = j + 1
    return s_x


def main():
    # x = sample_data(-10, 10, 0.0001, quad_func, quad_func_grad)

    # test the entropyTools
    batchSize = 200
    N = 1
    alpha_1 = np.linspace(-50, 50, batchSize)
    alpha_1 = alpha_1.reshape((batchSize, N))
    entropy_tools = math.EntropyTools(N)
    alpha_1 = entropy_tools.convert_to_tensorf(alpha_1)
    alpha = entropy_tools.reconstruct_alpha(alpha_1)
    u = entropy_tools.reconstruct_u(alpha)
    h = entropy_tools.compute_h(u, alpha)

    utils.plot1D(u[:, 1], [h], ['h'], 'sanity_check', log=False, folder_name="figures")
    # Looking good
    u_lower = -0.999
    u_upper = 0.999
    tolerance = 0.1

    [u_train, alpha_train, h_train] = sample_data_entropy_M1(u_lower, u_upper, tolerance)

    return 0


def pointwiseDiff(trueSamples, predSamples):
    """
    brief: computes the squared 2-norm for each sample point
    input: trueSamples, dim = (ns,N)
           predSamples, dim = (ns,N)
    returns: mse(trueSamples-predSamples) dim = (ns,)
    """
    err = []
    for i in range(trueSamples.shape[0]):
        err.append(np.abs(trueSamples[i] - predSamples[i]))
    loss_val = np.asarray(err)
    return loss_val


if __name__ == '__main__':
    main()
