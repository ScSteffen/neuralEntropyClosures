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
from matplotlib import colors
from matplotlib import cm
import seaborn as sns
import os
from pathlib import Path
import git
from datetime import date
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
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=2.5,
                             markeredgewidth=0.5, markeredgecolor='w')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=2.5,
                             markeredgewidth=0.5, markeredgecolor=colors[i])
            else:
                plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size)
            i += 1
        if labels:
            if legend_pos:
                plt.legend(labels, loc=legend_pos, fontsize=int(0.75 * font_size))
            else:
                plt.legend(labels, fontsize=int(0.75 * font_size))

    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType, color in zip(xs, ys, linetypes, colors):
            plt.plot(x, y, color + lineType, linewidth=symbol_size)
        plt.legend(labels, fontsize=int(0.75 * font_size))  # , prop={'size': 6})
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
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=font_size)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    if ticks:
        plt.xticks(ticks[0])
        plt.yticks(ticks[1])

    if xticks:
        plt.xticks(xticks, fontsize=int(0.7 * font_size))
        plt.yticks(fontsize=int(0.7 * font_size))

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
    plt.savefig(folder_name + "/" + name + ".pdf", dpi=500)
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
                 font_size: int = 14, folder_name: str = "figures", name: str = 'defaultName',
                 cbar_ticks: list = None, cbar_log: bool = False, img_size: list = [0, 1, 0, 1]) -> bool:
    plt.clf()
    fig = plt.figure(figsize=(5.8, 4.7), dpi=500)
    ax = plt.axes()

    sns.set_theme()
    sns.set_style("ticks")

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
            im = plt.imshow(np.array([cbar_ticks]), cmap="inferno",
                            norm=colors.LogNorm(vmin=cbar_ticks[0], vmax=cbar_ticks[-1]))
        else:
            im = plt.imshow(np.array([cbar_ticks]), cmap="inferno")

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
