"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_density_function2D, load_solution, plot_1d, load_data, scatter_plot_2d_N2, scatter_plot_2d


def maxwellian2D(x, y, rho, T):
    return rho / (2 * np.pi * T) * np.exp(-rho / (2 * T) * (x ** 2 + y ** 2))


def main():
    print("---------- Start Result Illustration Suite ------------")
    [x, y, w, kinetic_f] = load_density_function2D("xxx.csv")
    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    out = plt.scatter(x, y, c=kinetic_f)
    cbar = fig.colorbar(out, ax=ax, extend='both')
    # plt.show()
    sum = 0
    sum2 = 0
    rec_U = 0.0
    rec_T = 0
    kin = []
    nq = kinetic_f.shape[0]
    w2 = 25.0 / nq
    for i in range(kinetic_f.shape[0]):
        kin.append(maxwellian2D(x[0, i], y[0, i], 1, 1))
        sum += kinetic_f[i] * w[0, i]
        sum2 += (x[0, i] ** 2 + y[0, i] ** 2) / 2 * kinetic_f[i] * w[0, i]
        rec_U += maxwellian2D(x[0, i], y[0, i], 1, 2) * w[0, i]
        rec_T += (x[0, i] ** 2 + y[0, i] ** 2) / 2 * maxwellian2D(x[0, i], y[0, i], 1, 2) * w[0, i]  # * w[0, i]
    print(rec_U)
    print(rec_T)
    print("...")
    print(sum)
    print(sum2)
    print("_---")
    kin = np.asarray(kin)

    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    out = plt.scatter(x, y, c=kin)
    cbar = fig.colorbar(out, ax=ax, extend='both')
    # plt.show()

    # plt.ylim(0, 1)
    # plt.xlim(-5, 5)
    # plt.show()
    #  plt.savefig("test_a10_ev5")
    # for i in range(int(kinetic_f.shape[0] / 5)):
    #    kinetic_list = [kinetic_f[i + 0], kinetic_f[i + 1], kinetic_f[i + 2], kinetic_f[i + 3], kinetic_f[i + 4]]
    #    plot_1d(x, kinetic_list, show_fig=False, log=False, name='kinetics_kond3_' + str(i).zfill(3), ylim=[0, 3],
    #            xlim=[x[0, 0], x[0, -1]])

    return True


if __name__ == '__main__':
    main()
