"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import tensorflow as tf
from src.math import EntropyTools
from optparse import OptionParser
import matplotlib.pyplot as plt


def main():
    print("---------- Start Boundary Sampling Suite ------------")
    print("| Prototype only for 1 spatial dimension and 2 moments")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-d", "--degree", dest="degree", default=0,
                      help="max degree of moment", metavar="DEGREE")
    parser.add_option("-n", "--num_samples", dest="num_samples", default=10,
                      help="number of samples", metavar="DEGREE")

    (options, args) = parser.parse_args()
    options.degree = int(options.degree)
    options.num_samples = int(options.num_samples)

    # --- sample boundary kinetic densities using monreals Lem3.12 and 3.13 ---
    n = 10000
    alpha_max = 50000
    alpha_1 = np.linspace(-alpha_max, alpha_max, n)  # np.array([800])  #
    # h = entropy_of_alpha(alpha_1)
    h_analytic = entropy_of_alpha_analytic_long_double(alpha_1)  # entropy_of_alpha_analytic(alpha_1)
    # print(h)
    limit = np.log(np.abs(alpha_1)) - 2
    # plt.plot(alpha_1, h)
    plt.plot(alpha_1, h_analytic, '-')
    plt.plot(alpha_1, limit, '-.')
    plt.legend(["entropy", "log(|alpha|)-2"])
    # plt.show()
    plt.savefig("figures/entropylimit")
    t = np.abs(h_analytic - limit)
    plt.plot(alpha_1, t, '-.')
    plt.yscale("log")
    plt.show()

    """
    et1 = EntropyTools(polynomial_degree=1, spatial_dimension=1)
    n = 10000
    tmp = np.linspace(-700, 700, n).tolist()
    alpha1 = tf.constant(tmp, shape=(n, 1), dtype=tf.float64)
    alpha_comp = et1.reconstruct_alpha(alpha1)
    f = et1.compute_kinetic_density(alpha_comp)
    u = et1.compute_u(f)
    h = et1.compute_h(u, alpha_comp)
    h_primal = et1.compute_h_primal(f)
    integral = et1.integrate_f(f).numpy()
    t = h_primal.numpy()
    t2 = h.numpy()
    print(alpha_comp)
    print(u)
    print(h)
    print(h_primal)

    plt.plot(u[:, 1:].numpy(), h.numpy())
    plt.show()
    plt.plot(f[0, :])
    plt.show()
    print("here")
    """
    # for i in range(options.num_samples):
    # upper boundary of realizable set

    return True


def entropy_of_alpha(alpha_1: np.ndarray) -> np.ndarray:
    """
    brief: computes the m1 1d entropy for normalized moments depending on alpha
    input: alpha_1: (ns,)
    """
    et = EntropyTools(polynomial_degree=1, spatial_dimension=1)
    m = et.momentBasis.numpy()[1:, :]
    m = m.reshape((m.shape[1],))
    w = et.quadWeights.numpy().reshape((m.shape[0],))
    # alpha_1 *
    t1 = np.exp(np.outer(alpha_1, m))
    print(np.isnan(t1))
    t2 = np.matmul(t1, w)
    print(np.isnan(t2))

    f = t1 / t2[:, None]
    eta = f * np.log(f) - f

    # t3 = np.matmul(f, w)
    h = np.matmul(eta, w)
    return h


def entropy_of_alpha_analytic_limit(alpha_1: np.ndarray) -> np.ndarray:
    """
    brief: computes the m1 1d entropy for normalized moments depending on alpha
    input: alpha_1: (ns,)
    """
    h = -2 + alpha_1 * (np.exp(alpha_1) + np.exp(-alpha_1)) / (np.exp(alpha_1) - np.exp(-alpha_1))
    h2 = np.log(alpha_1 / (np.exp(alpha_1) - np.exp(-alpha_1)))
    return h + h2


def entropy_of_alpha_analytic(alpha_1: np.ndarray) -> np.ndarray:
    """
    brief: computes the m1 1d entropy for normalized moments depending on alpha
    input: alpha_1: (ns,)
    """
    h = -2 + alpha_1 * (np.exp(alpha_1) + np.exp(-alpha_1)) / (np.exp(alpha_1) - np.exp(-alpha_1))
    h2 = np.log(alpha_1 / (np.exp(alpha_1) - np.exp(-alpha_1)))
    return h + h2


def entropy_of_alpha_analytic_long_double(alpha_1: np.ndarray) -> np.ndarray:
    """
    brief: computes the m1 1d entropy for normalized moments depending on alpha
    input: alpha_1: (ns,)
    """
    alpha_1 = alpha_1.astype(np.clongdouble)
    h = -2 + alpha_1 * (np.exp(alpha_1) + np.exp(-alpha_1)) / (np.exp(alpha_1) - np.exp(-alpha_1))
    h2 = np.log(alpha_1 / (np.exp(alpha_1) - np.exp(-alpha_1)))
    return h + h2


if __name__ == '__main__':
    main()
