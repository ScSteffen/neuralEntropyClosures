"""
Author: Steffen Schotthöfer
Date: 21.04.2020
Brief: Script to test the optimal convex data sampling alogrithm
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import solve

import scipy.optimize


def quad_func(x):
    return x * x


def quad_func_grad(x):
    return 2 * x


def exp_func(x):
    return np.exp(x)


def sample_data(min_x, max_x, tol, function, grad):
    """
    performs the sampling algorithm
    """

    s_x = [min_x, max_x]
    e_max = tol
    j = 0
    while e_max >= tol:

        e_max = 0
        i = 0

        [eL, tL] = get_errors(s_x, function, grad)
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
                return grad(x) - grad_intermediate

            print(grad_intermediate)
            print((s_x[i + 1] - s_x[i]) / 2)
            print("----")
            
            t2 = float(scipy.optimize.broyden1(F, (s_x[i + 1] - s_x[i]) / 2, f_tol=1e-8))
            e2 = function(s_x[i]) + grad_intermediate * (t2 - s_x[i]) - function(t2)

            if e1 > e2:
                t = t1
                e = e1
                print("undershoot worse")
            else:
                t = t2
                e = e2
                print("overshoot worse")

            if e > e_max:
                e_max = e
            if e > tol:
                s_x.insert(i + 1, t)
                i = i + 1
            i = i + 1
        j = j + 1
    return s_x


def get_errors(s_x, function, grad):
    eL = []
    tL = []
    for i in range(len(s_x) - 1):
        t1 = 1 / (grad(s_x[i + 1]) - grad(s_x[i])) * (function(s_x[i]) - s_x[i] * grad(s_x[i]) - (
                function(s_x[i + 1]) - s_x[i + 1] * grad(s_x[i + 1])))
        e1 = function(t1) - (function(s_x[i]) + (t1 - s_x[i]) * grad(s_x[i]))

        # error in case that the approximation overshoots
        grad_intermediate = (function(s_x[i + 1]) - function(s_x[i])) / (s_x[i + 1] - s_x[i])

        def F(x):
            return grad(x) - grad_intermediate

        t2 = float(scipy.optimize.broyden1(F, (s_x[i + 1] - s_x[i]) / 2, f_tol=1e-8))
        e2 = function(s_x[i]) + grad_intermediate * (t2 - s_x[i]) - function(t2)

        if e1 > e2:
            eL.append(e1)
            tL.append(t1)
        else:
            eL.append(e2)
            tL.append(t2)

    return [eL, tL]


def main():
    z = np.linspace(0.1, 10, 100)
    xB = np.log(z)

    # x = sample_data(-10, 10, 0.0001, quad_func, quad_func_grad)
    x = sample_data(0, 10, 0.0001, exp_func, exp_func)

    y = []
    for i in range(len(x)):
        y.append(quad_func(x[i]))

    plt.plot(x, y, '*')
    plt.savefig("figures/sampling.png")
    plt.clf()

    # plot errors
    [e, tL] = get_errors(x, quad_func, quad_func_grad)
    plt.plot(tL, e, '*')
    plt.savefig("figures/errors.png")
    plt.clf()

    # show distance between sampling points
    d = []
    for i in range(len(x) - 1):
        d.append(abs(x[i + 1] - x[i]))

    plt.semilogy(x[1:], d, '*')
    plt.savefig("figures/distance.png")
    return 0


if __name__ == '__main__':
    main()
