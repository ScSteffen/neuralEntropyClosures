'''
Accumulation of utility functions
Date: 15.03.2021
Author: Steffen Schotthöfer
'''

import os
import time
from datetime import date
from pathlib import Path

import git
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import cm
from matplotlib import colors
from matplotlib.colors import ListedColormap
from matplotlib.ticker import StrMethodFormatter


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


def load_data(filename: str, data_dim: int, selected_cols: list = [True, True, True]) -> list:
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: training_data = [u,alpha,h]
    '''

    training_data = []

    print("Loading Data from location: " + filename)
    # determine which cols correspond to u, alpha and h
    u_cols = list(range(1, data_dim + 1))
    alpha_cols = list(range(data_dim + 1, 2 * data_dim + 1))
    h_col = [2 * data_dim + 1]

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


def load_density_function2D(filename: str) -> list:
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
    y = data[1, :].reshape((1, len(data[0, :])))

    weights = data[3, :].reshape((1, len(data[0, :])))
    f_kinetic = data[4, :]
    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))
    return [x, y, weights, f_kinetic]


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


def plot_1d(xs, ys, labels=None, name='defaultName', log=True, folder_name="figures", linetypes=None, show_fig=False,
            xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    plt.clf()
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("ticks")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    if len(xs) == 1:
        x = xs[0]
        for y, lineType in zip(ys, linetypes):
            for i in range(y.shape[1]):
                if colors[i] == 'k' and lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                    colors[i] = 'w'
                plt.plot(x, y[:, i], colors[i] + lineType, linewidth=symbol_size, markersize=2.5,
                         markeredgewidth=0.5, markeredgecolor=colors[i])
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
    plt.title(title, fontsize=14)
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=400)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".pdf"))
    return 0


def plot_1dv2(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False, xlim=None, ylim=None, xlabel=None, ylabel=None, ticks=None, symbol_size=3.0,
              marker_size=2.5,
              legend_pos=None, font_size=20, xticks=None):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=500)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h', 'H', '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("ticks")
    colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']
    symbol_size = symbol_size
    if len(xs) == 1:
        x = xs[0]
        i = 0
        for y, lineType in zip(ys, linetypes):
            if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                if colors[i] == 'k':
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=0.5, markeredgecolor='w')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=0.5, markeredgecolor=colors[i])
            else:
                plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=marker_size, )
            i += 1
        if labels:
            if legend_pos:
                plt.legend(labels, loc=legend_pos, fontsize=int(0.75 * font_size))
            else:
                plt.legend(labels, fontsize=int(0.5 * font_size))

    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType, color in zip(xs, ys, linetypes, colors):
            plt.plot(x, y, color + lineType, linewidth=symbol_size, markersize=marker_size)
        if labels:
            if legend_pos:
                plt.legend(labels, loc=legend_pos, fontsize=int(0.75 * font_size))
            else:
                plt.legend(labels, fontsize=int(0.5 * font_size))
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
        plt.xlabel(xlabel, fontsize=font_size)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    if ticks:
        plt.xticks(ticks[0])
        plt.yticks(ticks[1])
    if xticks:
        plt.xticks(xticks, fontsize=int(0.6 * font_size))
        plt.yticks(fontsize=int(0.6 * font_size))
    # plt.xticks(fontsize=int(0.5 * font_size))
    # plt.yticks(fontsize=int(0.5 * font_size))

    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".pdf"))
    plt.close()
    return 0


def plot_1dv2_thic(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures",
                   linetypes=None, show_fig=False, xlim=None, ylim=None, xlabel=None, ylabel=None, ticks=None,
                   xticks=None, legend_pos=None, black_first=False, font_size=20, symbol_size=1, marker_size=5):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """

    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h', 'H', '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("ticks")
    colors = ['r', 'g', 'b', 'k']
    if black_first:
        colors = ['k', 'r', 'g', 'b']
    symbol_size = symbol_size
    marker_size = marker_size
    marker_width = 0.1

    x = xs[0]
    for y, lineType, color in zip(ys, linetypes, colors):
        plt.plot(x, y, color + lineType, linewidth=symbol_size, markersize=marker_size,
                 markeredgewidth=marker_width, markeredgecolor=color)
    if labels:
        if legend_pos:
            plt.legend(labels, loc=legend_pos, fontsize=int(0.75 * font_size))
        else:
            plt.legend(labels, fontsize=int(0.75 * font_size))
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
        plt.xlabel(xlabel, fontsize=font_size)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)

    plt.yticks(fontsize=int(0.7 * font_size))
    plt.xticks(fontsize=int(0.7 * font_size))
    if ticks:
        plt.xticks(ticks[0])
        plt.yticks(ticks[1])
    if xticks:
        plt.xticks(xticks, fontsize=int(0.7 * font_size))
        plt.yticks(fontsize=int(0.7 * font_size))

    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".pdf"))
    plt.close()
    return 0


def scatter_plot_2d(x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                    lim_z: tuple = (0, 1), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                    title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                    folder_name: str = "figures", show_fig: bool = False, color_map: int = 0, marker_size=6,
                    axis_formatter=False, font_size=20, xticks=None, yticks=None):
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
        out = plt.scatter(x, y, s=marker_size, c=z, cmap=c_map, norm=colors.LogNorm(vmin=lim_z[0], vmax=lim_z[1]))
    else:
        out = plt.scatter(x, y, s=marker_size, c=z, cmap=c_map, vmin=lim_z[0], vmax=lim_z[1])
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    # ax.set_title(title, fontsize=14)
    plt.xlabel(label_x, fontsize=font_size)
    plt.ylabel(label_y, fontsize=font_size)
    if xticks:
        plt.xticks(xticks, fontsize=int(0.7 * font_size))
    else:
        plt.xticks(fontsize=int(0.7 * font_size))
    if yticks:
        plt.yticks(yticks, fontsize=int(0.7 * font_size))
    else:
        plt.yticks(fontsize=int(0.7 * font_size))
    # fig.set_aspect('auto')
    cbar = fig.colorbar(out, pad=0.02)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(int(0.5 * font_size))

    if show_fig:
        plt.show()
    if axis_formatter:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))  # 1 decimal places
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))  # 1 decimal places

    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    plt.close(fig)
    print("Saved image at: " + folder_name + "/" + name + ".pdf")
    return 0


def scatter_plot_2d_N2(x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                       lim_z: tuple = (0, 1), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                       title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                       folder_name: str = "figures", show_fig: bool = False, color_map: int = 0, marker_size=6,
                       axis_formatter=False, font_size=20, xticks=None, yticks=None):
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
    # plot boundary
    u1 = np.linspace(-1, 1, 100)
    u2 = u1 * u1
    u2_top = np.ones(100)
    plt.plot(u1, u2, 'k--')
    plt.plot(u1, u2_top, 'k--')
    # ax = fig.add_subplot(111)  # , projection='3d')
    x = x_in[:, 0]
    y = x_in[:, 1]
    z = z_in
    if log:
        out = plt.scatter(x, y, s=marker_size, c=z, cmap=c_map, norm=colors.LogNorm(vmin=lim_z[0], vmax=lim_z[1]))
    else:
        out = plt.scatter(x, y, s=marker_size, c=z, cmap=c_map, vmin=lim_z[0], vmax=lim_z[1])
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    # ax.set_title(title, fontsize=14)
    plt.xlabel(label_x, fontsize=font_size)
    plt.ylabel(label_y, fontsize=font_size)
    if xticks:
        plt.xticks(xticks, fontsize=int(0.7 * font_size))
    else:
        plt.xticks(fontsize=int(0.7 * font_size))
    if yticks:
        plt.yticks(yticks, fontsize=int(0.7 * font_size))
    else:
        plt.yticks(fontsize=int(0.7 * font_size))
    # fig.set_aspect('auto')
    cbar = fig.colorbar(out, pad=0.02)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(int(0.5 * font_size))

    if show_fig:
        plt.show()
    if axis_formatter:
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))  # 1 decimal places
        plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))  # 1 decimal places

    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
    plt.close(fig)
    print("Saved image at: " + folder_name + "/" + name + ".pdf")
    return 0


def scatter_plot_2d_N2_old(x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                           lim_z: tuple = (0, 1), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                           title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                           folder_name: str = "figures", show_fig: bool = False, marker_size=6, color_map: int = 0):
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

    u1 = np.linspace(-1, 1, 100)
    u2 = u1 * u1
    u2_top = np.ones(100)
    plt.plot(u1, u2, 'k--')
    plt.plot(u1, u2_top, 'k--')

    x = x_in[:, 0]
    y = x_in[:, 1]
    z = z_in
    if log:
        out = plt.scatter(x, y, s=marker_size, c=z, cmap=c_map, norm=colors.LogNorm(), vmin=lim_z[0], vmax=lim_z[1])
    else:
        out = plt.scatter(x, y, s=marker_size, c=z, cmap=c_map, vmin=lim_z[0], vmax=lim_z[1])
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    # plt.set_aspect('auto')
    cbar = fig.colorbar(out, pad=0.02)
    if show_fig:
        plt.show()

    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
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
    curr_date = date.today()
    # Print chosen options to csv
    d = {'date': [curr_date],
         'git_version': [sha],
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


def plot_flowfield(x, y, z, name="reference_M1_2D", z_min=0.5, z_max=2.5, contour=True, logscale=False, font_size=16,
                   xticks=None, yticks=None, err_plot=False):
    # --- plot ---
    c_map = cm.inferno
    cLevel = 1000
    # 1) rel error
    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=500)
    # filled contours
    if logscale:
        im = plt.imshow(z, extent=[x[0], x[-1], y[0], y[-1]], cmap=c_map, norm=colors.LogNorm(vmin=z_min, vmax=z_max))
    else:
        im = ax.contourf(x, y, z, levels=cLevel, cmap=c_map, vmin=z_min, vmax=z_max)
    # contour lines
    if contour:
        im2 = ax.contour(x, y, z, colors='k', vmin=z_min, vmax=z_max)
    # plt.xlabel(r"$x_1$", fontsize=int(0.75 * font_size))
    # plt.ylabel(r"$x_2$", fontsize=int(0.75 * font_size))

    if xticks:
        plt.xticks(xticks, fontsize=int(0.7 * font_size))
    else:
        plt.xticks(fontsize=int(0.7 * font_size))
    if yticks:
        plt.yticks(yticks, fontsize=int(0.7 * font_size))
    else:
        plt.yticks(fontsize=int(0.7 * font_size))

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    if err_plot:
        cbar = fig.colorbar(im, ax=ax, pad=0.02, ticks=[1e-5, 1e-4, 1e-3, 1e-2])

    else:
        cbar = fig.colorbar(im, ax=ax, pad=0.02, ticks=[0.5, 1.0, 1.5, 2.0, 2.5])
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(int(0.7 * font_size))
    plt.tight_layout()
    plt.savefig("paper_data/paper1/illustration/2D_M1/" + name + ".png", dpi=500)
    plt.close(fig)
    return 0


def beautify_img(load_name: str, xlabel: str = None, ylabel: str = None, xticks: list = None, yticks: list = None,
                 font_size: int = 14, folder_name: str = "figures", name: str = 'defaultName', c_map: str = "inferno",
                 cbar_ticks: list = None, cbar_log: bool = False, img_size: list = [0, 1, 0, 1]) -> bool:
    plt.clf()
    if cbar_ticks:
        fig = plt.figure(figsize=(5.8, 4.7), dpi=500)
    else:
        fig = plt.figure(figsize=(4.7, 4.7), dpi=500)

    ax = plt.axes()

    sns.set_theme()
    sns.set_style("white")

    if c_map == "RdGy":
        viridis = cm.get_cmap('viridis', 256)
        # newcolors = viridis(np.linspace(0, 1, 256))
        newcolors2 = np.asarray(
            [[77, 12, 23], [83, 12, 21], [88, 11, 20], [94, 11, 18], [100, 10, 15], [106, 11, 14], [112, 12, 13],
             [118, 13, 11], [124, 14, 10], [130, 16, 9], [135, 20, 8], [141, 24, 7], [147, 27, 7], [152, 31, 6],
             [157, 34, 5], [161, 37, 5], [166, 41, 4], [170, 44, 4], [175, 48, 3], [180, 51, 3], [185, 55, 2],
             [189, 58, 1], [192, 62, 1], [195, 66, 1], [198, 70, 0], [201, 74, 0], [205, 79, 1], [210, 85, 1],
             [214, 90, 2], [219, 96, 3], [222, 101, 4], [224, 105, 4], [227, 110, 5], [229, 115, 7], [231, 119, 8],
             [232, 123, 9], [232, 127, 10], [233, 131, 11], [234, 134, 13], [235, 138, 15], [235, 142, 17],
             [236, 145, 19], [236, 149, 21], [237, 153, 23], [238, 156, 26], [238, 160, 29], [239, 164, 31],
             [240, 168, 35], [240, 172, 39], [241, 176, 43], [241, 180, 46], [241, 184, 50], [242, 188, 54],
             [242, 191, 58], [243, 195, 62], [244, 198, 66], [244, 202, 71], [245, 206, 76], [246, 209, 81],
             [246, 213, 86], [246, 216, 92], [247, 219, 98], [247, 222, 104], [247, 225, 110], [247, 227, 115],
             [248, 230, 121], [248, 232, 126], [248, 234, 132], [249, 236, 137], [249, 238, 142], [250, 240, 147],
             [250, 242, 152], [250, 243, 156], [250, 245, 161], [250, 246, 165], [250, 248, 170], [250, 249, 175],
             [251, 250, 180], [251, 251, 186], [252, 252, 191], [252, 252, 197], [252, 252, 203], [251, 252, 209],
             [251, 252, 215], [241, 241, 226], [238, 238, 221], [234, 234, 216], [230, 230, 211], [227, 227, 206],
             [223, 223, 202], [219, 219, 197], [216, 215, 193], [212, 211, 189], [209, 207, 185], [206, 204, 182],
             [203, 201, 179], [200, 197, 175], [197, 194, 172], [194, 191, 169], [191, 188, 166], [188, 185, 163],
             [185, 182, 160], [182, 179, 157], [179, 176, 154], [177, 173, 152], [174, 170, 149], [171, 167, 146],
             [168, 163, 143], [166, 160, 140], [163, 157, 138], [160, 154, 135], [157, 151, 132], [154, 148, 129],
             [151, 145, 127], [148, 142, 124], [145, 139, 121], [143, 136, 119], [140, 133, 116], [138, 130, 114],
             [135, 127, 112], [133, 124, 109], [131, 122, 107], [129, 119, 105], [126, 116, 103], [124, 114, 101],
             [122, 112, 99], [120, 109, 97], [118, 107, 95], [116, 105, 93], [114, 103, 92], [112, 101, 90],
             [111, 100, 89], [109, 98, 87], [107, 96, 85], [106, 94, 84], [104, 92, 82], [102, 90, 81], [101, 88, 79],
             [99, 86, 78], [97, 84, 76], [95, 82, 75], [93, 80, 73], [92, 79, 72], [90, 77, 70], [89, 76, 69],
             [87, 74, 67], [85, 72, 65], [84, 70, 64], [82, 68, 62], [80, 66, 61], [79, 65, 59], [77, 63, 57],
             [75, 61, 56], [73, 59, 54], [71, 57, 52], [69, 55, 51], [68, 54, 49], [66, 52, 48], [64, 50, 46],
             [63, 48, 45], [61, 47, 44], [60, 45, 43], [58, 44, 42], [57, 42, 40], [55, 40, 38], [53, 39, 37],
             [51, 37, 35], [50, 40, 43], [48, 43, 51], [45, 46, 59], [40, 49, 68], [39, 51, 72], [40, 53, 74],
             [42, 55, 76], [43, 56, 78], [45, 58, 80], [47, 60, 82], [48, 62, 84], [50, 63, 86], [51, 65, 88],
             [53, 67, 90], [54, 69, 92], [55, 71, 95], [57, 73, 97], [58, 75, 99], [60, 77, 101], [61, 78, 103],
             [63, 80, 105], [64, 82, 108], [66, 85, 110], [68, 87, 112], [69, 89, 115], [71, 91, 117], [73, 93, 119],
             [75, 95, 122], [77, 97, 124], [78, 99, 126], [80, 102, 129], [82, 104, 131], [84, 106, 133],
             [86, 108, 135], [88, 110, 138], [90, 112, 140], [92, 114, 142], [95, 116, 144], [97, 119, 147],
             [99, 121, 149], [101, 123, 151], [103, 125, 153], [105, 128, 156], [107, 130, 158], [109, 132, 160],
             [111, 134, 162], [114, 137, 165], [116, 139, 167], [118, 142, 170], [120, 144, 172], [123, 147, 175],
             [125, 149, 177], [127, 152, 180], [129, 154, 182], [132, 157, 185], [134, 159, 187], [137, 162, 189],
             [139, 164, 191], [141, 166, 194], [144, 169, 196], [146, 171, 198], [149, 174, 200], [151, 176, 203],
             [154, 179, 205], [157, 182, 207], [159, 184, 209], [162, 187, 211], [164, 189, 213], [167, 192, 215],
             [170, 195, 218], [172, 197, 220], [175, 200, 222], [177, 202, 224], [179, 204, 225], [182, 207, 227],
             [184, 209, 229], [186, 211, 231], [188, 213, 232], [191, 216, 234], [193, 218, 236], [195, 220, 238],
             [198, 223, 240], [201, 225, 242], [204, 228, 244], [207, 231, 245], [212, 234, 247], [217, 237, 248],
             [222, 241, 250], [227, 244, 251], ])
        newcolors2 = newcolors2 / 256
        newcolors2 = np.concatenate([newcolors2, np.ones([256, 1])], axis=1)

        c_map = ListedColormap(newcolors2[:160, :])

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=font_size)
        plt.xticks(fontsize=int(0.7 * font_size))
        if xticks is not None:
            plt.xticks(xticks, fontsize=int(0.7 * font_size))
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)
        plt.yticks(fontsize=int(0.7 * font_size))
        if yticks is not None:
            plt.yticks(yticks, fontsize=int(0.7 * font_size))

    if cbar_ticks:
        if cbar_log:
            im = plt.imshow(np.array([cbar_ticks]), cmap=c_map,
                            norm=colors.LogNorm(vmin=cbar_ticks[0], vmax=cbar_ticks[-1]))
        else:
            im = plt.imshow(np.array([cbar_ticks]), cmap=c_map)

    img = plt.imread(load_name)
    plt.imshow(img, extent=img_size)

    if cbar_ticks is not None:
        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        cbar = plt.colorbar(im, cax=cax)  # Similar to fig.colorbar(im, cax = cax)

        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(int(0.7 * font_size))

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)

    # plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500, bbox_inches='tight', pad_inches=0)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".pdf"))
    plt.close()
    plt.clf()

    return True


def plot_1dv4(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False,
              xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    symbol_size = 2
    marker_size = 6
    marker_width = 0.1

    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    sns.set_theme()
    sns.set_style("ticks")

    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    colors = ['k', 'r', 'g', 'b']
    if len(xs) == 1:
        x = xs[0]
        i = 0
        for y, lineType in zip(ys, linetypes):
            if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                if colors[i] == 'k':
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor='w')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor='w')
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
        plt.xlabel(xlabel, fontsize=14)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".pdf"))
    plt.close()
    return 0


def plot_inflow(xs, ys, name='defaultName', folder_name="figures", xlim=[0, 1], xlabel=None, ylabel=None, font_size=16,
                xticks=None):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)

    sns.set_theme()
    sns.set_style("ticks")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 1.2
    marker_size = 4
    marker_width = 0.1

    case_y = ys[0]

    plt.plot(xs[0], case_y[0], "k-", linewidth=symbol_size, label=r"reference")
    plt.plot(xs[0], case_y[1], "k-", linewidth=symbol_size)
    if len(case_y) == 3:
        plt.plot(xs[0], case_y[2], "k-", linewidth=symbol_size)

    # Convex
    case_y = ys[1]
    plt.plot(xs[0], case_y[0], "or", linewidth=symbol_size,
             markersize=marker_size, markeredgewidth=marker_width, markeredgecolor='r', label="ICNN")
    plt.plot(xs[0], case_y[1], "or", linewidth=symbol_size,
             markersize=marker_size, markeredgewidth=marker_width, markeredgecolor='r')
    if len(case_y) == 3:
        plt.plot(xs[0], case_y[2], "or", linewidth=symbol_size,
                 markersize=marker_size, markeredgewidth=marker_width, markeredgecolor='r')
    # Monotonic
    case_y = ys[2]
    plt.plot(xs[0], case_y[0], "^g", linewidth=symbol_size,
             markersize=marker_size, markeredgewidth=marker_width, markeredgecolor='g', label="IMNN")
    plt.plot(xs[0], case_y[1], "^g", linewidth=symbol_size,
             markersize=marker_size, markeredgewidth=marker_width, markeredgecolor='g')
    if len(case_y) == 3:
        plt.plot(xs[0], case_y[2], "^g", linewidth=symbol_size,
                 markersize=marker_size, markeredgewidth=marker_width, markeredgecolor='g')

    texts = []
    if len(case_y) == 3:
        texts.append(plt.text(x=0.1, y=0.52, s=r"$u_0$", size="x-large"))
        texts.append(plt.text(x=0.1, y=0.22, s=r"$u_1$", size="x-large"))
        texts.append(plt.text(x=0.1, y=0.12, s=r"$u_2$", size="x-large"))
    else:
        texts.append(plt.text(x=0.1, y=0.38, s=r"$u_0$", size="x-large"))
        texts.append(plt.text(x=0.1, y=0.15, s=r"$u_1$", size="x-large"))

    if xticks:
        plt.xticks(xticks, fontsize=int(0.7 * font_size))
        plt.yticks(fontsize=int(0.7 * font_size))

    # adjust_text(texts, only_move={'texts': 'y'})
    plt.xlabel(xlabel, fontsize=font_size)
    plt.ylabel(ylabel, fontsize=font_size)
    plt.xlim(xlim[0], xlim[1])
    plt.legend(loc="upper right", fontsize=font_size)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".pdf"))
    plt.close()
    return 0


def plot_wide(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False, xlim=None, ylim=None, xlabel=None, ylabel=None, legend_pos="upper right",
              black_first=False, font_size=20):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(11.5, 4.7), dpi=500)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("ticks")
    colors = ['r', 'g', 'b', 'k']
    if black_first:
        colors = ['k', 'r', 'g', 'b']
    symbol_size = 4
    marker_size = 8
    marker_width = 0.1
    if len(xs) == 1:
        x = xs[0]
        i = 0
        for y, lineType in zip(ys, linetypes):
            if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                if colors[i] == 'k':
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor='w')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor=colors[i])
            else:
                plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size)
            i += 1
        if labels != None:
            plt.legend(labels, loc=legend_pos, fontsize=font_size)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType, color in zip(xs, ys, linetypes, colors):
            plt.plot(x, y, color + lineType, linewidth=symbol_size)
        plt.legend(labels, fontsize=font_size)  # , prop={'size': 6})
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
        plt.xlabel(xlabel, fontsize=font_size)
    plt.xticks([0, 2, 4, 6, 8, 10], fontsize=int(font_size * 0.75))
    plt.yticks([-5.2, -5, -4.8, -4.6], fontsize=int(font_size * 0.75))
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".pdf"))
    plt.close()
    return 0
