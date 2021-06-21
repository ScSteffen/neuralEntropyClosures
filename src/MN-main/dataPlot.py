'''
brief: Scripts for postprocessing
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src import utils
import tensorflow as tf

plt.style.use("kitish")


def postprocessData():
    # imported = tf.saved_model.load("1D_M1/tfmodel")
    # imported.summary()

    dataframeList = []
    ax = plt.subplot()

    name = "m1_entropy.csv"
    x = np.linspace(0, 1, 100)
    df = pd.read_csv(name)
    data = df.to_numpy()
    utils.plot1D([data[:, 0], data[:, 0]],
                 [data[:, 2], data[:, 1]],
                 [r'$h$', r'$h_\theta$'], linetypes=['-', '--'],
                 name='000_1DM1_entropy', folder_name="plots", log=False, show_fig=False,
                 xlabel=r"t")

    name = "m1_t0.7.csv"
    x = np.linspace(0, 1, 100)
    df = pd.read_csv(name)
    data = df.to_numpy()
    x = data[:, 0]
    utils.plot1D([x, x, x, x],
                 [data[:, 3], data[:, 4], data[:, 1], data[:, 2]],
                 [r'$u_{0}$', r'$u_{1}$', r'$u_{0,\theta}$', r'$u_{1,\theta}$'], linetypes=['-', '-', '--', '--'],
                 name='000_1DM1_07', folder_name="plots", log=False, show_fig=False,
                 xlabel=r"$u_1$")

    t = data[:, 3:5]
    t2 = data[:, 1:3]
    t3 = data
    data2 = data
    relErrCompl = np.linalg.norm(data[:, 3:5] - data[:, 1:3], axis=1, ord=1)  # / np.linalg.norm(data[:, 1:3], axis=1,
    #                                         ord=1)
    relErrComplU0 = np.abs(data[:, 3] - data[:, 1])  # / np.abs(data[:, 1])
    relErrComplU1 = np.abs(data[:, 4] - data[:, 2])  # / np.abs(data[:, 2])

    distR = np.abs(data[:, 2] / data[:, 1])

    # print(relErrCompl)
    utils.plot1D([x, x, x],
                 [relErrCompl, relErrComplU0, relErrComplU1],
                 [r"RE$(u_\theta(x,t_f),u(x,t_f))$ ", r"RE$(u_{0\theta}(x,t_f),u_0(x,t_f))$ ",
                  r"RE$(u_{1\theta}(x,t_f),u_1(x,t_f))$ "],
                 name='000_1DM1_errABS', folder_name="plots", log=True, show_fig=False,
                 xlabel=r"$x$", ylim=[1e-7, 0.1])
    utils.plot1D([x, x, x],
                 [relErrCompl, relErrComplU0, relErrComplU1],
                 [r"$||u_\theta(x,t_f)-u(x,t_f)||_1$ ", r"$||u_{0\theta}(x,t_f)-u_0(x,t_f)||_1$ ",
                  r"$||u_{1\theta}(x,t_f)-u_1(x,t_f)||_1$ "],
                 name='000_1DM1_errABS', folder_name="plots", log=True, show_fig=False,
                 xlabel=r"$x$", ylim=[1e-8, 0.1])

    print(relErrCompl.mean())

    #########################
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

    t = data[:, 3:6]
    t2 = data[:, 0:3]
    t4 = data
    relErr = np.linalg.norm(data[:, 3:6] - data[:, 0:3], axis=1, ord=1) / np.linalg.norm(data[:, 0:3], axis=1, ord=1)

    relErrCompl2 = np.linalg.norm(data[:, 3:6] - data[:, 0:3], axis=1, ord=1)  # / np.linalg.norm(data[:, 0:3], axis=1,
    #                 ord=1)
    relErrComplU0 = np.abs(data[:, 3] - data[:, 0])  # / np.abs(data[:, 0])
    relErrComplU1 = np.abs(data[:, 4] - data[:, 1])  # / np.abs(data[:, 1])
    relErrComplU2 = np.abs(data[:, 5] - data[:, 2])  # / np.abs(data[:, 2])

    utils.plot1D([x[80:], x[80:]],
                 [data[80:, 4], data[80:, 1]], ["ref", r"theta "],
                 name='000_1DM2_er2r', folder_name="plots", log=False, show_fig=False,
                 xlabel=r"$x$")

    # print(relErrCompl)
    utils.plot1D([x, x, x, x],
                 [relErrCompl2, relErrComplU0, relErrComplU1, relErrComplU2],
                 [r"$||u_\theta(x,t_f)-u(x,t_f)||_1$ ", r"$||u_{0\theta}(x,t_f)-u_0(x,t_f)||_1$ ",
                  r"$||u_{1\theta}(x,t_f)-u_1(x,t_f)||_1$ ", r"$||u_{2\theta}(x,t_f)-u_2(x,t_f)||_1$ "],
                 name='000_1DM2_errASBS', folder_name="plots", log=True, show_fig=False,
                 xlabel=r"$x$", ylim=[1e-8, 0.1])

    utils.plot1D([x, x, ],
                 [relErrCompl, relErrCompl2],
                 [r"RE$(u_\theta(x,t_f),u(x,t_f))$ , $M_1$ ", r"RE$(u_{0\theta}(x,t_f),u_0(x,t_f))$ , $M_2$ "],
                 name='000_1DM2_errTOG', folder_name="plots", log=False, show_fig=False,
                 xlabel=r"$x$", ylim=[0.0, 0.3])

    utils.plot1D([x, x, ],
                 [relErrCompl, relErrCompl2],
                 [r"RE$(u_\theta(x,t_f),u(x,t_f))$ , $M_1$ ", r"RE$(u_{0\theta}(x,t_f),u_0(x,t_f))$ , $M_2$ "],
                 name='000_1DM2_errTOG', folder_name="plots", log=False, show_fig=False,
                 xlabel=r"$x$", ylim=[0.0, 0.3])

    print(relErr)
    print(relErr.mean())

    normUM1 = np.linalg.norm(data2[:, 3:5], axis=1)
    normUM2 = np.linalg.norm(data[:, 3:6], axis=1)
    utils.plot1D([x, x, x, x, x],  # , x, x],
                 [normUM1, normUM2, data[:, 3], data[:, 4], data[:, 5]],  # , data2[:, 2], data2[:, 3]],
                 ["m1", "M2", "M2 u0", "M2 u1", "M2 u2"],  # , "M1 u0", "M1 u1"],
                 name='000_1DM2_Value', folder_name="plots", log=True, show_fig=False,
                 xlabel=r"$x$")
    utils.plot1D([x, x, x, x],  # , x, x],
                 [normUM1, normUM2, data2[:, 2], data2[:, 3]],
                 ["m1", "M2", "M2 u0", "M2 u2"],  # , "M1 u0", "M1 u1"],
                 name='000_1DM2_Valu2e', folder_name="plots", log=True, show_fig=False,
                 xlabel=r"$x$")

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
