'''
Accumulation of utility functions
Date: 15.03.2021
Author: Steffen Schotthöfer
'''

import numpy as np
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib import cm
import seaborn as sns
import os
from pathlib import Path
import git


# plt.style.use("kitish")


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


def load_data(filename: str, input_dim: int, selected_cols: list = [True, True, True]) -> list:
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: training_data = [u,alpha,h]
    '''

    training_data = []

    print("Loading Data from location: " + filename)
    # determine which cols correspond to u, alpha and h
    u_cols = list(range(1, input_dim + 1))
    alpha_cols = list(range(input_dim + 1, 2 * input_dim + 1))
    h_col = [2 * input_dim + 1]

    # selectedCols = self.selectTrainingData() #outputs a boolean triple.
    start = time.time()
    if selected_cols[0]:
        df = pd.read_csv(filename, usecols=[i for i in u_cols])
        uNParray = df.to_numpy()
        training_data.append(uNParray)
    if selected_cols[1]:
        df = pd.read_csv(filename, usecols=[i for i in alpha_cols])
        alphaNParray = df.to_numpy()
        training_data.append(alphaNParray)
    if selected_cols[2]:
        df = pd.read_csv(filename, usecols=[i for i in h_col])
        hNParray = df.to_numpy()
        training_data.append(hNParray)

    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))

    return training_data


def load_density_function(filename: str) -> list:
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: training_data = [u,alpha,h]
    '''
    print("Loading Data from location: " + filename)
    start = time.time()
    df = pd.read_csv(filename, header=None)
    df = df.drop(df.columns[0], axis=1)
    data = df.to_numpy()
    x = data[0, :].reshape((1, len(data[0, :])))
    weights = data[1, :].reshape((1, len(data[0, :])))
    f_kinetic = data[2:, :]
    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))
    return [x, weights, f_kinetic]


def load_solution(filename: str) -> list:
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: training_data = [u,alpha,h]
    '''
    print("Loading Data from location: " + filename)
    start = time.time()
    df = pd.read_csv(filename)
    data = df.to_numpy()
    t = data.shape[1] / 2
    u_neural = data[:, :int(data.shape[1] / 2)]
    u_ref = data[:, int(data.shape[1] / 2):]
    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))
    return [u_neural, u_ref]


def evaluateModel(model, input):
    '''Evaluates the model at input'''
    # x = input
    # print(x.shape)
    # print(x)
    return model.predict(input)


def evaluateModelDerivative(model, input):
    '''Evaluates model derivatives at input'''

    x_model = tf.Variable(input)

    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(x_model, training=False)  # same as model.predict(x)

    gradients = tape.gradient(predictions, x_model)
    return gradients


def loadTFModel(filename):
    '''Loads a .h5 file to memory'''
    nn = tf.keras.models.load_model(filename)
    return nn


def plot_1d(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
            show_fig=False,
            xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    symbol_size = 1.5
    marker_size = 4
    marker_width = 0.5
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    if len(xs) == 1:
        x = xs[0]
        for y, lineType in zip(ys, linetypes):
            for i in range(y.shape[1]):
                if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                    if colors[i] == 'k':
                        plt.plot(x, y[:, i], 'w' + lineType, linewidth=symbol_size, markersize=marker_size,
                                 markeredgewidth=marker_width, markeredgecolor='k')
                    else:
                        plt.plot(x, y[:, i], colors[i] + lineType, linewidth=symbol_size, markersize=marker_size,
                                 markeredgewidth=marker_width, markeredgecolor='k')
                else:
                    plt.plot(x, y[:, i], colors[i] + lineType, linewidth=symbol_size)
        if labels != None:
            plt.legend(labels)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType in zip(xs, ys, linetypes):
            plt.plot(x, y, lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')
    if loglog:
        plt.yscale('log')
        plt.xscale('log')
    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    plt.close()
    return 0


def plot_1dv2(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False,
              xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['r', 'g', 'b', 'k']
    symbol_size = 3
    marker_size = 4
    marker_width = 0.5
    if len(xs) == 1:
        x = xs[0]
        i = 0
        for y, lineType in zip(ys, linetypes):
            if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                if colors[i] == 'k':
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor='k')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor='k')
            else:
                plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size)
            i += 1
        if labels != None:
            plt.legend(labels, loc="upper left")
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType, color in zip(xs, ys, linetypes, colors):
            plt.plot(x, y, color + lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')
    if loglog:
        plt.yscale('log')
        plt.xscale('log')
    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    plt.close()
    return 0


def plot_inflow(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
                show_fig=False,
                xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)

    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 1.2
    marker_size = 4
    marker_width = 0.5

    linetypes = ['-', 'o', '^']
    linetypes_out = ['-', '--', '-.']
    case_y = ys[0]

    for j in range(len(case_y)):
        u_j = case_y[j]
        plt.plot(xs[0], u_j, colors[0] + linetypes_out[j], linewidth=symbol_size)
    for j in range(len(ys[0])):
        for i in range(1, len(ys)):
            case_y = ys[i]
            u_j = case_y[:][j]
            plt.plot(xs[0], u_j, colors[i] + linetypes[i], linewidth=symbol_size,
                     markersize=marker_size, markeredgewidth=marker_width, markeredgecolor='k')
    if log:
        plt.yscale('log')
    if loglog:
        plt.yscale('log')
        plt.xscale('log')
    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    if labels != None:
        if len(ys[0]) == 3:
            labels = [r"$u_0$", r"$u_1$", r"$u_2$", "convex", "monotonic"]
        else:
            labels = [r"$u_0$", r"$u_1$", "convex", "monotonic"]
        plt.legend(labels, loc="upper right")
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    plt.close()
    return 0


def plot_1dv3(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False,
              xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    if len(xs) == 1:
        x = xs[0]
        for y, color in zip(ys, colors):
            for i in range(y.shape[1]):
                plt.plot(x, y[:, i], color + linetypes[i], linewidth=symbol_size)
        if labels is not None:
            plt.legend(labels)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType in zip(xs, ys, linetypes):
            plt.plot(x, y, lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')
    if loglog:
        plt.yscale('log')
        plt.xscale('log')
    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    plt.close()
    return 0


def plot_1dv4(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False,
              xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'k', 'r', 'g', 'b']
    symbol_size = 0.7
    if len(xs) == 1:
        x = xs[0]
        i = 0
        for y, lineType in zip(ys, linetypes):
            if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                if colors[i] == 'k':
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=2.5,
                             markeredgewidth=0.5, markeredgecolor='k')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=2.5,
                             markeredgewidth=0.5, markeredgecolor='k')
            else:
                plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size)
            i += 1
        if labels != None:
            plt.legend(labels)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType, color in zip(xs, ys, linetypes, colors):
            plt.plot(x, y, color + lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')
    if loglog:
        plt.yscale('log')
        plt.xscale('log')
    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=12)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=12)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    plt.close()
    return 0


def scatter_plot_2d(x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                    lim_z: tuple = (0, 1), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                    title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                    folder_name: str = "figures", show_fig: bool = False, color_map: int = 0):
    '''
    brief: Compute a scatter plot
    input: x_in = [x1,x2] function arguments
           y_in = function values
    return: True if exit successfully
    '''
    # choose colormap
    if color_map == 1:
        c_map = cm.summer
    else:
        c_map = cm.inferno

    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    # ax = fig.add_subplot(111)  # , projection='3d')
    x = x_in[:, 0]
    y = x_in[:, 1]
    z = z_in
    if log:
        out = plt.scatter(x, y, s=6, c=z, cmap=c_map, norm=colors.LogNorm(vmin=lim_z[0], vmax=lim_z[1]))
    else:
        out = plt.scatter(x, y, s=6, c=z, cmap=c_map, vmin=lim_z[0], vmax=lim_z[1])
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    # ax.set_title(title, fontsize=14)
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    # fig.set_aspect('auto')
    cbar = fig.colorbar(out, pad=0.02)
    if show_fig:
        plt.show()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    plt.close(fig)
    return 0


def scatter_plot_2d_N2(x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                       lim_z: tuple = (0, 1), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                       title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                       folder_name: str = "figures", show_fig: bool = False, color_map: int = 0):
    '''
    brief: Compute a scatter plot
    input: x_in = [x1,x2] function arguments
           y_in = function values
    return: True if exit successfully
    '''
    # choose colormap
    if color_map == 1:
        c_map = cm.summer
    else:
        c_map = cm.inferno

    plt.plot()
    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')

    u1 = np.linspace(-1, 1, 100)
    u2 = u1 * u1
    u2_top = np.ones(100)
    ax.plot(u1, u2, 'k--')
    ax.plot(u1, u2_top, 'k--')

    x = x_in[:, 0]
    y = x_in[:, 1]
    z = z_in
    if log:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map, norm=colors.LogNorm(vmin=lim_z[0], vmax=lim_z[1]))
    else:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map, vmin=lim_z[0], vmax=lim_z[1])
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    # ax.set_title(title, fontsize=14)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect('auto')
    cbar = fig.colorbar(out, pad=0.02)
    if show_fig:
        plt.show()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    plt.close(fig)
    return 0


def plot_flowfield(x, y, z, name="reference_M1_2D", z_min=0.5, z_max=2.5, contour=True, logscale=False):
    # --- plot ---
    c_map = cm.hot
    cLevel = 1000
    # 1) rel error
    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=400)
    # filled contours
    if logscale:
        im = plt.imshow(z, extent=[x[0], x[-1], y[0], y[-1]], cmap=c_map, norm=colors.LogNorm(vmin=z_min, vmax=z_max))
    else:
        im = ax.contourf(x, y, z, levels=cLevel, cmap=c_map, vmin=z_min, vmax=z_max)
    # contour lines
    if contour:
        im2 = ax.contour(x, y, z, colors='k', vmin=z_min, vmax=z_max)
    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(im, ax=ax, pad=0.02)
    plt.savefig("paper_data/2D_M1/" + name + ".png", dpi=500)
    plt.close(fig)
    return 0


def write_config_file(options, neural_closure_model):
    # create String to create a python runscript
    runScript = "python callNeuralClosure.py \\\n"
    runScript = runScript + "--sampling=" + str(int(options.sampling)) + " \\\n"
    runScript = runScript + "--batch=" + str(options.batch) + " \\\n"
    runScript = runScript + "--curriculum=" + str(options.curriculum) + " \\\n"
    runScript = runScript + "--degree=" + str(options.degree) + " \\\n"
    runScript = runScript + "--epoch=" + str(options.epoch) + " \\\n"
    runScript = runScript + "--folder=" + str(options.folder) + " \\\n"
    runScript = runScript + "--loadModel=" + str(1) + " \\\n"  # force to load
    runScript = runScript + "--model=" + str(options.model) + " \\\n"
    runScript = runScript + "--normalized=" + str(int(options.normalized)) + " \\\n"
    runScript = runScript + "--scaledOutput=" + str(int(options.scaledOutput)) + " \\\n"
    runScript = runScript + "--decorrInput=" + str(int(options.decorrInput)) + " \\\n"
    runScript = runScript + "--objective=" + str(options.objective) + " \\\n"
    runScript = runScript + "--processingmode=" + str(options.processingmode) + " \\\n"
    runScript = runScript + "--spatialDimension=" + str(options.spatial_dimension) + " \\\n"
    runScript = runScript + "--training=" + str(options.training) + " \\\n"
    runScript = runScript + "--verbosity=" + str(options.verbosity) + " \\\n"
    runScript = runScript + "--networkwidth=" + str(options.networkwidth) + " \\\n"
    runScript = runScript + "--networkdepth=" + str(options.networkdepth)

    # Getting filename
    rsFile = neural_closure_model.folder_name + '/runScript_001_'
    count = 0

    # create directory if it does not exist
    make_directory(neural_closure_model.folder_name)

    while os.path.isfile(rsFile + '.sh'):
        count += 1
        rsFile = neural_closure_model.folder_name + '/runScript_' + str(count).zfill(3) + '_'

    rsFile = rsFile + '.sh'

    print("Writing config to " + rsFile)
    f = open(rsFile, "w")
    f.write(runScript)
    f.close()

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    print("Current git checkout: " + str(sha))
    # Print chosen options to csv
    d = {'git_version': [sha],
         'sampling': [options.sampling],
         'batch': [options.batch],
         'curriculum': [options.curriculum],
         'degree': [options.degree],
         'epoch': [options.epoch],
         'folder': [options.folder],
         'loadModel': [options.loadmodel],
         'model': [options.model],
         'normalized moments': [options.normalized],
         'decorrelate inputs': [options.decorrInput],
         'scaled outputs': [options.scaledOutput],
         'objective': [options.objective],
         'processingmode': [options.processingmode],
         'spatial Dimension': [options.spatial_dimension],
         'verbosity': [options.verbosity],
         'training': [options.training],
         'network width': [options.networkwidth],
         'network depth': [options.networkdepth]}

    count = 0
    cfg_file = neural_closure_model.folder_name + '/config_001_'

    while os.path.isfile(cfg_file + '.csv'):
        count += 1
        cfg_file = neural_closure_model.folder_name + '/config_' + str(count).zfill(3) + '_'
    cfg_file = cfg_file + '.csv'
    pd.DataFrame.from_dict(data=d, orient='index').to_csv(cfg_file, header=False, sep=';')
    return True


def make_directory(path_to_directory):
    if not os.path.exists(path_to_directory):
        p = Path(path_to_directory)
        p.mkdir(parents=True)
    return 0
