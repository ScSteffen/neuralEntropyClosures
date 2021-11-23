"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import load_density_function, load_solution, plot_1d, load_data, scatter_plot_2d_N2, scatter_plot_2d


def main():
    print("---------- Start Result Illustration Suite ------------")

    [x, _, kinetic_f] = load_density_function("test_a10_ev5.csv")

    for i in range(kinetic_f.shape[0]):
        plt.plot(x[0], kinetic_f[i, :])

    plt.ylim(0, 1)
    plt.xlim(-5, 5)
    # plt.show()
    plt.savefig("test_a10_ev5")
    # for i in range(int(kinetic_f.shape[0] / 5)):
    #    kinetic_list = [kinetic_f[i + 0], kinetic_f[i + 1], kinetic_f[i + 2], kinetic_f[i + 3], kinetic_f[i + 4]]
    #    plot_1d(x, kinetic_list, show_fig=False, log=False, name='kinetics_kond3_' + str(i).zfill(3), ylim=[0, 3],
    #            xlim=[x[0, 0], x[0, -1]])

    return True


if __name__ == '__main__':
    main()
