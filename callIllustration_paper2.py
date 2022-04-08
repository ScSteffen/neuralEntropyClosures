"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 5.4.2022
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import plot_1d


def main():
    print("---------- Start Result Illustration Suite ------------")

    mk11_m3_2d_g0 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g0/historyLogs/history_002_.csv")
    mk11_m3_2d_g1 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g1/historyLogs/history_002_.csv")
    mk11_m3_2d_g2 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g2/historyLogs/history_002_.csv")
    mk11_m3_2d_g3 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g3/historyLogs/history_002_.csv")

    n_epochs = mk11_m3_2d_g0.shape[0]
    plot_1d([mk11_m3_2d_g0["epoch"].to_numpy()],
            [mk11_m3_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1),
             mk11_m3_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1),
             mk11_m3_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1),
             mk11_m3_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)],
            labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
            name="loss_mk11_m3_h_gammas", log=True, folder_name="paper_data/paper2/illustrations", show_fig=False,
            xlabel="epochs", ylabel="loss", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1d([mk11_m3_2d_g0["epoch"].to_numpy()],
            [mk11_m3_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1),
             mk11_m3_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1),
             mk11_m3_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)],
            labels=[r"$h_\theta^n$", r"$\alpha_{u,\theta}^n$", r"$u_\theta^n$"],
            name="loss_mk11_m3_2d", log=True, folder_name="paper_data/paper2/illustrations", show_fig=False,
            xlabel="epochs", ylabel="loss", title="", xlim=[0, 2000], ylim=[1e-5, 1e0])
    return True


def load_history_file(filename: str):
    """
    loads training history
    """
    df = pd.read_csv(filename)

    return df


def plot_1dx(x, ys, labels=None, name='defaultName', log=True, folder_name="figures", linetypes=None, show_fig=False,
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
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7

    x
    count = 0
    for y, lineType in zip(ys, linetypes):
        plt.plot(x, y, colors + lineType, linewidth=symbol_size, markersize=2.5,
                 markeredgewidth=0.5, markeredgecolor='k')
    if labels is not None:
        plt.legend(labels)
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
    plt.savefig(folder_name + "/" + name + ".png", dpi=400)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    return 0


if __name__ == '__main__':
    main()
