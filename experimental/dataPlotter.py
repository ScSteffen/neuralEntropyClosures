'''
small script to plot cpp output
'''

import csv

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import time
from src import utils

plt.style.use("kitish")


def main():
    # Plot normalized Data
    # [u, alpha, h] = loadTrainingData("data/1D/Monomial_M1_1D_normal.csv", 2)
    # utils.plot1D(u[:, 1], [h], ['h'], 'validation_check', log=False, folder_name="figures")

    # plot1DM0Data(u, alpha, h)

    # Plots of 1D M0 Data
    # [u, alpha, h] = loadTrainingData("disc_Moments.csv", 3)
    # plot2DMoments(u, h)

    # [u, alpha, h] = loadTrainingData("data/1D/Monomial_M1_1D_normal_stdnormal.csv", 2)
    # u = u[1500:2000, :]
    # h = h[1500:2000]

    # plot_h_over_n1(u[:, 1], alpha[:, 1])
    # plot_h_over_n1_n2_n3(u, h)
    # plot_h_over_n1_n2(u[:, [0, 1, 3]], h)

    # Normalized Plots
    # [u, alpha, h] = loadTrainingData("data/1D/Monomial_M3_1D_normal_alpha.csv", 4)
    # u = u[1500:2000, :]
    # h = h[1500:2000]
    # plot_h_over_n1_n2_n3(u, h)
    # plot_h_over_n1_n2(u[:, [0, 1, 3]], h)
    # plot_h_over_n1_n2(u[:, [0, 1, 2]], h)

    # Normalized Plots
    [u, alpha, h] = loadTrainingData("data/1D/Monomial_M2_1D_normal_gaussian2.csv", 3)
    utils.scatter_plot_2d(x_in=u[:, 1:], z_in=h, lim_x=(-1, 1), lim_y=(0, 1), lim_z=(-1, 4),
                          title=r"$h$ over ${\mathcal{R}^r}$",
                          folder_name="delete", name="h_over_u__alpha_gauss", show_fig=False, log=False)
    utils.scatter_plot_2d(x_in=alpha[:, 1:], z_in=h, lim_x=(-30, 30), lim_y=(-30, 30), lim_z=(-1, 4),
                          title=r"$h$ over ${\alpha_u^r}$",
                          folder_name="delete", name="h_over_alpha__alpha_gauss", show_fig=False, log=False)
    plot_h_over_n1_n2(alpha, h, lim_x=(-40, 40), lim_y=(-40, 40),
                      label_x=r"$\alpha_1^r$", label_y=r"$\alpha_2^r$",
                      title=r"$h$ over $\alpha^r$")
    # Normalized Plots
    # [u, alpha, h] = loadTrainingData("realizable_set_pictures/N2_alpha.csv", 3)
    # plot_h_over_n1_n2(u, h)

    # [u, alpha, h] = loadTrainingData("data/1_stage/1D/Monomial_M3_D1_normalized.csv", 4)
    # plotHoverNormalized_N2_N3(u, h)

    # Plots of 1D M0 Data
    # [u, alpha, h] = loadTrainingData("data/1D/Monomial_M0_1D.csv", 1)
    # plot1DM0Data(u, alpha, h)

    # Plots of 1D M1 Data

    # [u, alpha, h] = loadTrainingData("data/1D/Monomial_M1_1D_small.csv", 2)
    # plot1DM1Data(alpha, alpha, h)

    return 0


def loadTrainingData(filename, lenU: int):
    # Loads the training data generated by "generateTrainingData.py"
    trainingData = []

    # determine which cols correspond to u, alpha and h
    uCols = list(range(1, lenU + 1))
    alphaCols = list(range(lenU + 1, 2 * lenU + 1))
    hCol = [2 * lenU + 1]

    # selectedCols = self.selectTrainingData() #outputs a boolean triple.

    selectedCols = [True, True, True]

    start = time.time()
    if selectedCols[0] == True:
        df = pd.read_csv(filename, usecols=[i for i in uCols])
        uNParray = df.to_numpy()
        trainingData.append(uNParray)
    if selectedCols[1] == True:
        df = pd.read_csv(filename, usecols=[i for i in alphaCols])
        alphaNParray = df.to_numpy()
        trainingData.append(alphaNParray)
    if selectedCols[2] == True:
        df = pd.read_csv(filename, usecols=[i for i in hCol])
        hNParray = df.to_numpy()
        trainingData.append(hNParray)

    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))

    return trainingData


def plot_h_over_n1(u, h):
    fig = plt.figure(figsize=(4.7, 4), dpi=400)
    plt.plot(u, h, '*')
    # plt.xlim(-1, 1)
    # plt.ylim(0, 1)

    plt.savefig("N1_alpha.png", dpi=150)
    plt.show()

    return 0


def plot_h_over_n1_n2_n3(u, h):
    fig = plt.figure(figsize=(4.7, 4), dpi=400)
    ax = fig.add_subplot(111, projection='3d')
    # ax.grid(True, linestyle='-', color='0.75')
    x = u[:, 1]
    y = u[:, 2]
    z = u[:, 3]
    out = ax.scatter(x, y, z, s=6, c=h, cmap=cm.hot)  # , vmin=-1.5, vmax=3.0)
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    # plt.zlim(-1, 1)
    ax.set_title(r"$h$ over $\overline{\mathcal{R}}$", fontsize=14)
    ax.set_xlabel(r"$u_1^r$")
    ax.set_ylabel(r"$u_2^r$")
    ax.set_zlabel(r"$u_3^r$")
    ax.set_aspect('auto')
    # ax.set_xlabel('N1')
    # ax.set_ylabel('N2')
    # ax.set_zlabel('h')
    # pos_neg_clipped = ax.imshow(z)
    cbar = fig.colorbar(out, ax=ax, extend='both')
    plt.savefig("N3_alpha.png", dpi=150)
    # plt.show()

    return 0


def plotHoverNormalized_N2_N3(u, h):
    '''
    Plot h over relative moments corresponding to Monreals diss
    '''

    # Fixing random state for reproducibility
    np.random.seed(19680801)

    def randrange(n, vmin, vmax):
        '''
        Helper function to make an array of random numbers having shape (n, )
        with each number distributed Uniform(vmin, vmax).
        '''
        return (vmax - vmin) * np.random.rand(n) + vmin

    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')
    ax.grid(True, linestyle='-', color='0.75')
    x = u[:, 2]
    y = u[:, 3]
    z = h
    out = ax.scatter(x, y, s=20, c=z, cmap=cm.jet);

    ax.set_title("h over N2 and N3", fontsize=14)
    ax.set_xlabel("N2", fontsize=12)
    ax.set_ylabel("N3", fontsize=12)
    # ax.set_xlabel('N1')
    # ax.set_ylabel('N2')
    # ax.set_zlabel('h')
    # pos_neg_clipped = ax.imshow(z)
    cbar = fig.colorbar(out, ax=ax, extend='both')
    plt.show()

    return 0


def plot_h_over_n1_n2(u, h, lim_x=(-1, 1), lim_y=(0, 1), label_x=r"$u_1^r$", label_y=r"$u_2^r$",
                      title=r"$h^n$ over ${\mathcal{R}^r}$"):
    '''
    Plot h over relative moments corresponding to Monreals diss
    '''

    fig = plt.figure(figsize=(4.7, 4), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    x = u[:, 1]
    y = u[:, 2]
    z = h
    out = ax.scatter(x, y, s=6, c=z, cmap=cm.hot)  # , vmin=-1.5, vmax=3.0)
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect('auto')
    cbar = fig.colorbar(out, ax=ax, extend='both')
    plt.savefig("N2_alpha.png", dpi=150)
    return 0


def plot2DMoments(u, h):
    '''
        Plot h over two dimensions of u1
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')
    ax.grid(True, linestyle='-', color='0.75')
    x = u[:, 1]
    y = u[:, 2]
    z = h
    out = ax.scatter(x, y, s=5, c=z, cmap=cm.jet);

    ax.set_title("h over u1 (2D)", fontsize=14)
    ax.set_xlabel("u1 1", fontsize=12)
    ax.set_ylabel("u1 2", fontsize=12)
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    # ax.set_xlabel('N1')
    # ax.set_ylabel('N2')
    # ax.set_zlabel('h')
    # pos_neg_clipped = ax.imshow(z)
    cbar = fig.colorbar(out, ax=ax, extend='both')

    circle1 = plt.Circle((0, 0), 1, color='k', fill=False)
    ax.add_patch(circle1)
    ax.plot()
    plt.show()
    return 0


def exampleplot(u, h):
    x = u[:, 1]
    y = u[:, 2]
    z = h

    xx, yy = np.meshgrid(x, y, sparse=True)
    z = np.sin(xx ** 2 + yy ** 2) / (xx ** 2 + yy ** 2)

    fig = plt.contourf(x, y, z)

    plt.show()


def plot1DM0Data(u, alpha, h):
    integratedGradients = integrate(u, alpha)
    # plt.plot(u, integratedGradients)
    plt.plot(u, h, '--')
    plt.ylabel('h')
    plt.xlabel('u')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    plt.legend(['h'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0, 20])
    plt.show()

    finDiff = finiteDiff(u, h)
    plt.plot(u, finDiff)
    plt.plot(u, alpha, '--')
    plt.ylabel('alpha')
    plt.xlabel('u')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    plt.legend(['fd h', 'alpha'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0, 50])
    plt.show()

    plt.plot(alpha, h, '--')
    plt.ylabel('h')
    plt.xlabel('alpha')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    # plt.legend(['fd h', 'alpha'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0, 50])
    plt.show()

    plt.plot(alpha, u, '--')
    plt.ylabel('u')
    plt.xlabel('alpha')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    # plt.legend(['fd h', 'alpha'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0, 50])
    plt.show()

    plt.plot(h, alpha,
             '--')
    plt.ylabel('alpha')
    plt.xlabel('h')
    # plt.legend(['Model', 'Model Derivative', 'Target Fct', 'Target Derivative'])
    # plt.legend(['fd h', 'alpha'])
    # plt.legend(['Model ','Target Function'])
    # plt.ylim([0, 50])
    plt.show()

    return 0


def plot1DM1Data(u, alpha, h):
    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection='3d')
    ax.grid(True, linestyle='-', color='0.75')
    x = u[:, 0]
    y = u[:, 1]
    z = h
    out = ax.scatter(x, y, s=20, c=z, cmap=cm.jet);

    ax.set_title("h over u0 and u1", fontsize=14)
    ax.set_xlabel("u0", fontsize=12)
    ax.set_ylabel("u1", fontsize=12)
    # ax.set_xlabel('N1')
    # ax.set_ylabel('N2')
    # ax.set_zlabel('h')
    # pos_neg_clipped = ax.imshow(z)
    cbar = fig.colorbar(out, ax=ax, extend='both')
    plt.show()

    return 0


def integrate(x, y):
    '''
    :param x: function argument
    :param y: = f(x)
    :return: integrate y over span of x
    '''

    integral = np.zeros(x.shape)

    for i in range(0, x.shape[0] - 1):
        integral[i + 1] = integral[i] + (x[i + 1] - x[i]) * y[i + 1]

    return integral


def finiteDiff(x, y):
    '''
    :param x: Function Argument
    :param y: Function value = f(x)
    :return: df/dx at all points x
    '''

    grad = np.zeros(x.shape)

    grad[0] = (y[1] - y[0]) / (x[1] - x[0])

    for i in range(0, x.shape[0] - 1):
        grad[i + 1] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])

    return grad


if __name__ == '__main__':
    main()
