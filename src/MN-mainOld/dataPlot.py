'''
brief: Scripts for postprocessing
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src import utils

plt.style.use("kitish")


def postprocessData():
    dataframeList = []
    ax = plt.subplot()

    name = "m1_t0.7.csv"
    x = np.linspace(0, 1, 100)
    df = pd.read_csv(name)
    data = df.to_numpy()
    utils.plot1D([x, x, x, x],
                 [data[:, 2], data[:, 3], data[:, 0], data[:, 1]],
                 [r'$u_{0}$', r'$u_{1}$', r'$u_{0,\theta}$', r'$u_{1,\theta}$'], linetypes=['-', '-', '--', '--'],
                 name='000_1DM1_07', folder_name="plots", log=False, show_fig=False,
                 xlabel=r"$u_1$")
    name = "m2_t0.7.csv"
    x = np.linspace(0, 1, 100)
    df = pd.read_csv(name)
    data = df.to_numpy()
    utils.plot1D([x, x, x, x, x, x],
                 [data[:, 3], data[:, 4], data[:, 5], data[:, 0], data[:, 1], data[:, 2]],
                 [r'$u_{0}$', r'$u_{1}$', r'$u_{2}$', r'$u_{0,\theta}$', r'$u_{1,\theta}$', r'$u_{2,\theta}$'],
                 linetypes=['-', '-', '-', '--', '--', '--'],
                 name='000_1DM2_07', folder_name="plots", log=False, show_fig=False,
                 xlabel=r"$u_1$")
    # , ylim=[0, 0.2])
    # utils.plot1D([data[:, 0], data[:, 0], data[:, 0], data[:, 0]],
    #             [data[:, 2], data[:, 3], data[:, 4], data[:, 5]],
    #             [r'MRE($u(t_i),u_{\theta}(t_i)$)', r'MRE($u_0(t_i),u_{0\theta}(t_i)$)',
    #              r'MRE($u_1(t_i),u_{1\theta}(t_i)$)',
    #              r'MRE($u_2(t_i),u_{2\theta}(t_i)$)'],
    #             '000_1DM2_05', folder_name="plots", log=False, show_fig=False,
    #             xlabel=r"$u_1$", ylim=[0, 0.03])
    # df.plot(x='iter', y=['meanRe', 'meanAe', 'meanAu0', 'meanAu1', 'meanAu2'])
    # plt.title("Mean relative errors over time step. Different norms")
    # plt.legend(['l2 norm of u', 'l1 norm of u', 'l1 norm of u0', 'l1 norm of u1', 'l1 norm of u2'])
    # plt.xlabel("time step")
    # plt.xlim(0, 1500)
    # plt.show()
    return 0


if __name__ == '__main__':
    postprocessData()
