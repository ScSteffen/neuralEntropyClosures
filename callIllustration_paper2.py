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
import tensorflow as tf
from src.utils import plot_1d, plot_1dv2, scatter_plot_2d
from src.networks.configmodel import init_neural_closure
from src.math import EntropyTools


def main():
    print("---------- Start Result Illustration Suite ------------")

    # 1) Training performance
    # print_training_performance()

    # 2) Tests for realizable set
    # test_on_realizable_set_m2()

    # 3) Print memory-computational cost study
    # print_comp_efficiency_memory()

    # 4) Print cross sections
    # print_cross_sections()

    # 5) Print method errors
    print_method_errors()
    return True


def test_on_realizable_set_m2():
    mk11_m2_2d_g0 = init_neural_closure(network_mk=11, poly_degree=2, spatial_dim=2, folder_name="tmp",
                                        loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                        input_decorrelation=True, scale_active=False, gamma_lvl=0)
    mk11_m2_2d_g0.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g0/")

    mk11_m2_2d_g1 = init_neural_closure(network_mk=13, poly_degree=2, spatial_dim=2, folder_name="tmp",
                                        loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                        input_decorrelation=True, scale_active=False, gamma_lvl=1)
    mk11_m2_2d_g1.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g1_v2/")

    mk11_m2_2d_g2 = init_neural_closure(network_mk=11, poly_degree=2, spatial_dim=2, folder_name="tmp",
                                        loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                        input_decorrelation=True, scale_active=False, gamma_lvl=2)
    mk11_m2_2d_g2.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g2/")

    mk11_m2_2d_g3 = init_neural_closure(network_mk=11, poly_degree=2, spatial_dim=2, folder_name="tmp",
                                        loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                        input_decorrelation=True, scale_active=False, gamma_lvl=3)
    mk11_m2_2d_g3.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g3/")

    # define curve in set of lagrange multipliers
    [u_recons, alpha_recons, h_recons, t_param] = get_lagrange_curve(poly_degree=2, spatial_dim=2, dim=5, n=100)
    out_g0 = mk11_m2_2d_g0.model(u_recons[:, 1:])
    out_g1 = mk11_m2_2d_g1.model(u_recons[:, 1:])
    out_g2 = mk11_m2_2d_g2.model(u_recons[:, 1:])
    out_g3 = mk11_m2_2d_g3.model(u_recons[:, 1:])
    scatter_plot_2d(u_recons[:, 1:3], h_recons, lim_x=(-1, 1), lim_y=(-1, 1), lim_z=(0, 6), label_x="", label_y="",
                    title="", name="moment_path", log=False,
                    folder_name="paper_data/paper2/illustrations/model_testing")
    ys = [h_recons[:, 0], out_g0[0][:, 0], out_g1[0][:, 0], out_g2[0][:, 0], out_g3[0][:, 0]]
    plot_1dv2(xs=[t_param], ys=ys,
              labels=["h", "g0", "g1", "g2", "g3"],
              name="entropyplot2", log=False, loglog=False, folder_name="paper_data/paper2/illustrations/model_testing",
              linetypes=["-", "--", "--", "--", "--"], show_fig=False, xlim=None, ylim=None, xlabel="u curve",
              ylabel="entropy", title="")

    u_input = get_moment_curve(dim=5, n=100)
    out_g0 = mk11_m2_2d_g0.model(u_input)
    out_g1 = mk11_m2_2d_g1.model(u_input)
    out_g2 = mk11_m2_2d_g2.model(u_input)
    out_g3 = mk11_m2_2d_g3.model(u_input)
    ys = [out_g1[0][:, 0], out_g1[0][:, 0], out_g1[0][:, 0], out_g1[0][:, 0]]
    plot_1dv2(xs=[u_input[:, 0]], ys=ys,
              labels=["g0", "g1", "g2", "g3"],
              name="entropyplot3", log=False, loglog=False, folder_name="paper_data/paper2/illustrations/model_testing",
              linetypes=["--", "--", "--", "--"], show_fig=False, xlim=None, ylim=None, xlabel="u curve",
              ylabel="entropy", title="")
    return 0


def get_lagrange_curve(poly_degree: int, spatial_dim: int, dim: int, n: int) -> list:
    alpha_coeff = np.asarray([0.2, 0.2, 0.2, 0.0, 0.0])
    alphas = np.zeros(shape=(n, dim))
    i = 0
    for t in np.linspace(-10, 10, n):
        alphas[i, :] = t * alpha_coeff
        i += 1
    et = EntropyTools(polynomial_degree=poly_degree, spatial_dimension=spatial_dim)
    alpha_recons = et.reconstruct_alpha(tf.constant(alphas))
    u_recons = et.reconstruct_u(alpha_recons)
    h_recons = et.compute_h(u_recons, alpha_recons)
    return [u_recons, alpha_recons, h_recons, np.linspace(0, 10, n)]


def get_moment_curve(dim: int, n: int) -> np.ndarray:
    u_coeff = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
    u_s = np.zeros(shape=(n, dim))
    i = 0
    for t in np.linspace(-1.2, 1.2, n):
        u_s[i, :] = t * u_coeff
        i += 1
    return u_s


def print_training_performance():
    # --------------- M2 2D -----------------------
    mk11_m2_2d_g0 = load_history_file("paper_data/paper2/2D_M2/mk11_m2_2d_g0/historyLogs/history_002_.csv")
    mk11_m2_2d_g1 = load_history_file("paper_data/paper2/2D_M2/mk11_m2_2d_g1_v2/historyLogs/history_002_.csv")
    mk11_m2_2d_g2 = load_history_file("paper_data/paper2/2D_M2/mk11_m2_2d_g2/historyLogs/history_002_.csv")
    mk11_m2_2d_g3 = load_history_file("paper_data/paper2/2D_M2/mk11_m2_2d_g3/historyLogs/history_002_.csv")

    n_epochs = mk11_m2_2d_g0.shape[0]
    plot_1dv2([mk11_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m2_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m2_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk11_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m2_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m2_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk11_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m2_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m2_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk13_m2_2d_g0 = load_history_file("paper_data/paper2/2D_M2/mk13_m2_2d_g0/historyLogs/history_002_.csv")
    mk13_m2_2d_g1 = mk13_m2_2d_g0  # load_history_file("paper_data/paper2/2D_M2/mk13_m2_2d_g1/historyLogs/history_002_.csv")
    mk13_m2_2d_g2 = load_history_file("paper_data/paper2/2D_M2/mk13_m2_2d_g2/historyLogs/history_002_.csv")
    mk13_m2_2d_g3 = load_history_file("paper_data/paper2/2D_M2/mk13_m2_2d_g3/historyLogs/history_002_.csv")

    n_epochs = mk13_m2_2d_g0.shape[0]
    plot_1dv2([mk13_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m2_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m2_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk13_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m2_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m2_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk13_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m2_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m2_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk12_m2_2d_g0 = load_history_file("paper_data/paper2/2D_M2/mk12_m2_2d_g0/historyLogs/history_001_.csv")
    mk12_m2_2d_g1 = load_history_file("paper_data/paper2/2D_M2/mk12_m2_2d_g1/historyLogs/history_001_.csv")
    mk12_m2_2d_g2 = load_history_file("paper_data/paper2/2D_M2/mk12_m2_2d_g2/historyLogs/history_001_.csv")
    mk12_m2_2d_g3 = load_history_file("paper_data/paper2/2D_M2/mk12_m2_2d_g3/historyLogs/history_001_.csv")

    n_epochs = mk12_m2_2d_g0.shape[0]
    plot_1dv2([mk12_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m2_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m2_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk12_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m2_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m2_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk12_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m2_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m2_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    # --------------- M3 2D -----------------------
    mk11_m3_2d_g0 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g0/historyLogs/history_002_.csv")
    mk11_m3_2d_g1 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g1/historyLogs/history_002_.csv")
    mk11_m3_2d_g2 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g2/historyLogs/history_002_.csv")
    mk11_m3_2d_g3 = load_history_file("paper_data/paper2/2D_M3/mk11_M3_2D_g3/historyLogs/history_002_.csv")

    n_epochs = mk11_m3_2d_g0.shape[0]
    plot_1dv2([mk11_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m3_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m3_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk11_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m3_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m3_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk11_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m3_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m3_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk13_m3_2d_g0 = load_history_file("paper_data/paper2/2D_M3/mk13_M3_2D_g0/historyLogs/history_002_.csv")
    mk13_m3_2d_g1 = load_history_file("paper_data/paper2/2D_M3/mk13_M3_2D_g1/historyLogs/history_002_.csv")
    mk13_m3_2d_g2 = load_history_file("paper_data/paper2/2D_M3/mk13_M3_2D_g2/historyLogs/history_002_.csv")
    mk13_m3_2d_g3 = load_history_file("paper_data/paper2/2D_M3/mk13_M3_2D_g3/historyLogs/history_002_.csv")

    n_epochs = mk13_m3_2d_g0.shape[0]
    plot_1dv2([mk13_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m3_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m3_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk13_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m3_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m3_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk13_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m3_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m3_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk12_m3_2d_g0 = load_history_file("paper_data/paper2/2D_M3/mk12_M3_2D_g0/historyLogs/history_001_.csv")
    mk12_m3_2d_g1 = load_history_file("paper_data/paper2/2D_M3/mk12_M3_2D_g1/historyLogs/history_001_.csv")
    mk12_m3_2d_g2 = load_history_file("paper_data/paper2/2D_M3/mk12_M3_2D_g2/historyLogs/history_001_.csv")
    mk12_m3_2d_g3 = load_history_file("paper_data/paper2/2D_M3/mk12_M3_2D_g3/historyLogs/history_001_.csv")

    n_epochs = mk12_m3_2d_g0.shape[0]
    plot_1dv2([mk12_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m3_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m3_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk12_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m3_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m3_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk12_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m3_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m3_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    # --------------- M4 2D -----------------------
    mk11_m4_2d_g0 = load_history_file("paper_data/paper2/2D_M4/mk11_m4_2d_g0/historyLogs/history_003_.csv")
    mk11_m4_2d_g1 = load_history_file("paper_data/paper2/2D_M4/mk11_m4_2d_g1/historyLogs/history_002_.csv")
    mk11_m4_2d_g2 = load_history_file("paper_data/paper2/2D_M4/mk11_m4_2d_g2/historyLogs/history_002_.csv")
    mk11_m4_2d_g3 = load_history_file("paper_data/paper2/2D_M4/mk11_m4_2d_g3_v2/historyLogs/history_002_.csv")

    n_epochs = mk11_m4_2d_g0.shape[0]
    plot_1dv2([mk11_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m4_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m4_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk11_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m4_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m4_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk11_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m4_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m4_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk13_m4_2d_g0 = load_history_file("paper_data/paper2/2D_M4/mk13_m4_2d_g0/historyLogs/history_002_.csv")
    mk13_m4_2d_g1 = load_history_file("paper_data/paper2/2D_M4/mk13_m4_2d_g1/historyLogs/history_002_.csv")
    mk13_m4_2d_g2 = load_history_file("paper_data/paper2/2D_M4/mk13_m4_2d_g2/historyLogs/history_002_.csv")
    mk13_m4_2d_g3 = load_history_file("paper_data/paper2/2D_M4/mk13_m4_2d_g3/historyLogs/history_002_.csv")

    n_epochs = mk13_m4_2d_g0.shape[0]
    plot_1dv2([mk13_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m4_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m4_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk13_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m4_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m4_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk13_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m4_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m4_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk12_m4_2d_g0 = load_history_file("paper_data/paper2/2D_M4/mk12_m4_2d_g0/historyLogs/history_001_.csv")
    mk12_m4_2d_g1 = load_history_file("paper_data/paper2/2D_M4/mk12_m4_2d_g1/historyLogs/history_001_.csv")
    mk12_m4_2d_g2 = load_history_file("paper_data/paper2/2D_M4/mk12_m4_2d_g2/historyLogs/history_001_.csv")
    mk12_m4_2d_g3 = load_history_file("paper_data/paper2/2D_M4/mk12_m4_2d_g3/historyLogs/history_001_.csv")
    n_epochs = mk12_m4_2d_g0.shape[0]
    plot_1dv2([mk12_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m4_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m4_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk12_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m4_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m4_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk12_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m4_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m4_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    # ------------------ M5 2D -----------------------------

    mk11_m5_2d_g0 = load_history_file("paper_data/paper2/2D_M5/mk11_M5_2D_g0/historyLogs/history_002_.csv")
    mk11_m5_2d_g1 = load_history_file("paper_data/paper2/2D_M5/mk11_M5_2D_g1/historyLogs/history_004_.csv")
    mk11_m5_2d_g2 = load_history_file("paper_data/paper2/2D_M5/mk11_M5_2D_g2/historyLogs/history_004_.csv")
    mk11_m5_2d_g3 = load_history_file("paper_data/paper2/2D_M5/mk11_M5_2D_g3/historyLogs/history_004_.csv")

    n_epochs = mk11_m5_2d_g0.shape[0]
    plot_1dv2([mk11_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m5_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m5_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk11_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m5_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m5_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk11_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m5_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m5_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk13_m5_2d_g0 = load_history_file("paper_data/paper2/2D_M5/mk13_M5_2D_g0/historyLogs/history_002_.csv")
    mk13_m5_2d_g1 = load_history_file("paper_data/paper2/2D_M5/mk13_M5_2D_g1/historyLogs/history_004_.csv")
    mk13_m5_2d_g2 = load_history_file("paper_data/paper2/2D_M5/mk13_M5_2D_g2/historyLogs/history_004_.csv")
    mk13_m5_2d_g3 = load_history_file("paper_data/paper2/2D_M5/mk13_M5_2D_g3/historyLogs/history_004_.csv")

    n_epochs = mk13_m5_2d_g0.shape[0]
    plot_1dv2([mk13_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m5_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m5_h_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk13_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m5_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m5_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk13_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m5_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m5_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])

    mk12_m5_2d_g0 = load_history_file("paper_data/paper2/2D_M5/mk12_M5_2D_g0/historyLogs/history_001_.csv")
    mk12_m5_2d_g1 = load_history_file("paper_data/paper2/2D_M5/mk12_M5_2D_g1/historyLogs/history_001_.csv")
    mk12_m5_2d_g2 = load_history_file("paper_data/paper2/2D_M5/mk12_M5_2D_g2/historyLogs/history_001_.csv")
    mk12_m5_2d_g3 = load_history_file("paper_data/paper2/2D_M5/mk12_M5_2D_g3/historyLogs/history_001_.csv")

    n_epochs = mk12_m5_2d_g0.shape[0]
    plot_1dv2([mk12_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m5_2d_g0["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g3["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g2["val_output_1_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g1["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m5_h_gammas", log=True, folder_name="paper_data/paper2/illustrations", show_fig=False,
              xlabel="epochs", ylabel="loss h", title="", xlim=[0, 2000], ylim=[1e-6, 1e-2])

    plot_1dv2([mk12_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m5_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m5_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations", show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", title="", xlim=[0, 2000], ylim=[1e-5, 1e-0])

    plot_1dv2([mk12_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m5_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m5_u_gammas", log=True, folder_name="paper_data/paper2/illustrations", show_fig=False,
              xlabel="epochs", ylabel="loss u", title="", xlim=[0, 2000], ylim=[1e-7, 1e-2])
    return 0


def print_cross_sections():
    names = ["linesource_M2_g0", "linesource_M2_g1", "linesource_M2_g2", "linesource_M2_g3", "linesource_M4_g1",
             "linesource_N2_g0", "linesource_N2_g1", "linesource_N2_g2", "linesource_N2_g3",
             "linesource_N2_g0_r", "linesource_N2_g1_r", "linesource_N2_g2_r", "linesource_N2_g3_r",
             "linesource_N2_g0_resnet", "linesource_N2_g1_resnet", "linesource_N2_g2_resnet", "linesource_N2_g3_resnet",
             "linesource_N2_g0_r_resnet", "linesource_N2_g1_r_resnet", "linesource_N2_g2_r_resnet",
             "linesource_N2_g3_r_resnet", "linesource_M4_g1",
             "linesource_M3_g0", "linesource_M3_g1", "linesource_M3_g3",
             "linesource_N3_g0", "linesource_N3_g1", "linesource_N3_g2", "linesource_N3_g3", ]

    for name in names:
        print_single_xs(name)
    return 0


def print_single_xs(name):
    load_name = "paper_data/paper2/linesource/cross_sections/"
    save_name = "paper_data/paper2/illustrations/cross_sections/"
    df_analytic = pd.read_csv(load_name + "linesource_analytic.csv")
    x_analytic = df_analytic["Points:0"].to_numpy()
    y_analytic = df_analytic["Points:1"].to_numpy()
    xy = np.vstack([x_analytic, y_analytic]).T
    radius_analytic = np.linalg.norm(xy, axis=1)
    radius_analytic[:int(len(radius_analytic) / 2)] = -radius_analytic[:int(len(radius_analytic) / 2)]

    df_name_vert = pd.read_csv(load_name + name + "_vert.csv")
    x_analytic = df_name_vert["Points:0"].to_numpy()
    y_analytic = df_name_vert["Points:1"].to_numpy()
    xy = np.vstack([x_analytic, y_analytic]).T
    radius_vert = np.linalg.norm(xy, axis=1)
    radius_vert[:int(len(radius_vert) / 2)] = -radius_vert[:int(len(radius_vert) / 2)]

    df_name_45 = pd.read_csv(load_name + name + ".csv")
    x_analytic = df_name_45["Points:0"].to_numpy()
    y_analytic = df_name_45["Points:1"].to_numpy()
    xy = np.vstack([x_analytic, y_analytic]).T
    radius_45 = np.linalg.norm(xy, axis=1)
    radius_45[:int(len(radius_vert) / 2)] = -radius_45[:int(len(radius_45) / 2)]

    # t = np.sum(df_analytic["analytic radiation flux density"].to_numpy())
    # t2 = np.sum(df_name_vert["radiation flux density"].to_numpy())
    # t3 = np.sum(df_name_45["radiation flux density"].to_numpy())

    # ratio = t / t2
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    plt.plot(radius_analytic, df_analytic["analytic radiation flux density"] / 2, "-k")
    plt.plot(radius_vert, df_name_vert["radiation flux density"], "-b")
    plt.plot(radius_45, df_name_45["radiation flux density"], "-.r")
    plt.xlim([-1, 1])
    plt.legend(["analytic", "vertical", "diagonal"], loc="upper left")
    plt.xlabel("radius")
    plt.ylabel("scalar flux")
    plt.savefig(save_name + "xs_" + name + ".png", dpi=500)
    plt.clf()


def print_comp_efficiency_memory():
    # list of tuples in format (time,sys_size)
    data = [(105.570747, 6), (131.381264, 12), (285.899369, 20), (547.719602, 49), (1014.543704, 72), (356.75354, 100),
            (1648.625912, 400), (4119.824155, 900), (30975.63164, 1600), (83928.613993, 2500), (16140.171279, 6),
            (31281.166781, 10), (1658.046172, 6), (1884.270762, 10),
            (2094.88887, 15), (2515.013316, 6), (1652.840298, 6), (1895.105454, 10), ]
    # list of method names
    names = [r"$P_2$", r"$P_3$", r"$P_5$", r"$P_7$", r"$P_9$", r"$S_{10}$", r"$S_{20}$", r"$S_{30}$", r"$S_{40}$",
             r"$S_{50}$", r"$M_2^{\gamma_3}$", r"$M_3^{\gamma_3}$", r"$\mathcal{N}_{2}$", r"$\mathcal{N}_{3}$",
             r"$\mathcal{N}_{4}$", r"$\mathcal{N}_{2}^{{rot}}$",
             r"$M_{2,\mathcal{N}}^{\gamma_3}$", r"$M_{3,\mathcal{N}}^{\gamma_3}$"]

    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    fig, ax = plt.subplots()

    for i in range(len(data) - 2):
        # circle = plt.Circle(data[i], radius=1)
        plt.scatter(data[i][1], data[i][0], s=10, facecolors='red', edgecolors='red')
        # ax.add_patch(circle)
        label = ax.annotate(names[i], xy=(data[i][1], data[i][0]), fontsize=15, ha="center")

    # ax.axis('off')
    # ax.set_aspect('equal')
    # ax.autoscale_view()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("system size")
    plt.ylabel("time [s]")
    # plt.show()
    plt.savefig("paper_data/paper2/illustrations/hohlraum/methods.png", dpi=500)

    return 0


def print_method_errors():
    # list of tuples in format (time,sys_size)
    # data = [(105.570747, 6),  # P2
    #        (131.381264, 12),  # P3
    #        (285.899369, 20),  # P5
    #        (547.719602, 49),  # P7
    #        (1014.543704, 72),  # P9
    #        (356.75354, 100),  # S10
    #        (1648.625912, 400),  # S20
    #        (4119.824155, 900),  # S30
    #        (30975.63164, 1600),  # S40
    #        (83928.613993, 2500),  # S50
    #        (16140.171279, 6),  # M2_g3
    #        (31281.166781, 10),  # M3_g3
    #        (1658.046172, 6),  # N2_g3
    #        (1884.270762, 10),  # N3_g3
    #        (2094.88887, 15),  # N4_g3
    #        (2515.013316, 6),  # N2_g3_rot
    #        (1652.840298, 6), (1895.105454, 10), ]
    df = pd.read_csv("paper_data/paper2/hohlraum/errors/method_errors.csv", delimiter=";")
    # list of method names
    # names = [r"$P_2$", r"$P_3$", r"$P_5$", r"$P_7$", r"$P_9$", r"$S_{10}$", r"$S_{20}$", r"$S_{30}$", r"$S_{40}$",
    #         r"$S_{50}$", r"$M_2^{\gamma_3}$", r"$M_3^{\gamma_3}$", r"$\mathcal{N}_{2}$", r"$\mathcal{N}_{3}$",
    #         r"$\mathcal{N}_{4}$", r"$\mathcal{N}_{2}^{{rot}}$",
    #         r"$M_{2,\mathcal{N}}^{\gamma_3}$", r"$M_{3,\mathcal{N}}^{\gamma_3}$"]

    names = [r"$M_2^{\gamma_3}$",
             r"$M_3^{\gamma_3}$",
             r"$\mathcal{N}_2$",
             r"$\mathcal{N}_2^{\gamma_1}$",
             r"$\mathcal{N}_2^{\gamma_2}$",
             r"$\mathcal{N}_2^{\gamma_3}$",
             r"$\mathcal{N}_2^{r}$",
             r"$\mathcal{N}_2^{\gamma_1,r}$",
             r"$\mathcal{N}_2^{\gamma_2,r}$",
             r"$\mathcal{N}_2^{\gamma_3,r}$",
             r"$\mathcal{N}_3$",
             r"$\mathcal{N}_3^{\gamma_1}$",
             r"$\mathcal{N}_3^{\gamma_2}$",
             r"$\mathcal{N}_3^{\gamma_3}$",
             r"$\mathcal{N}_4$",
             r"$\mathcal{N}_4^{\gamma_1}$",
             r"$\mathcal{N}_4^{\gamma_2}$",
             r"$\mathcal{N}_4^{\gamma_3}$",
             r"$S_{10}$",
             r"$S_{20}$",
             r"$S_{30}$",
             r"$S_{40}$",
             r"$S_{50}$",
             r"$P_{2}$",
             r"$P_{3}$",
             r"$P_{5}$",
             r"$P_{7}$",
             r"$P_{9}$"]
    errors = df["spatial_error"].to_numpy(dtype=np.float)
    sys_size = df["sys_size"].to_numpy()
    timing = df["timing"].to_numpy()

    indices_ref = [0, 1, 18, 19, 20, 21, 22, 23, 24, 25, 26]

    indices_N2 = [2, 4, 5]
    indices_N2_rot = [6, 8, 9]
    indices_N3 = [10, 12, 13]
    indices_N4 = [14, 16, 17]
    errors_N2_mean = np.mean(errors[indices_N2])
    errors_N2_std = np.std(errors[indices_N2])
    errors_N3_mean = np.mean(errors[indices_N3])
    errors_N3_std = np.std(errors[indices_N3])
    errors_N2_r_mean = np.mean(errors[indices_N2_rot])
    errors_N2_r_std = np.std(errors[indices_N2_rot])
    errors_N4_mean = np.mean(errors[indices_N4])
    errors_N4_std = np.std(errors[indices_N4])

    # a) sysSize over errror

    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    fig, ax = plt.subplots()

    plt.scatter(sys_size[2], errors_N2_mean, s=10, facecolors='blue', edgecolors='blue')
    ax.errorbar(sys_size[2], y=errors_N2_mean,
                yerr=errors_N2_std,
                alpha=0.5,
                ecolor='blue',
                capsize=10)

    plt.scatter(sys_size[7], errors_N2_r_mean, s=10, facecolors='green', edgecolors='green')
    ax.errorbar(sys_size[7], y=errors_N2_r_mean,
                yerr=errors_N2_std,
                alpha=0.5,
                ecolor='green',
                capsize=10)

    plt.scatter(sys_size[11], errors_N3_mean, s=10, facecolors='cyan', edgecolors='cyan')
    ax.errorbar(sys_size[11], y=errors_N3_mean,
                yerr=errors_N2_std,
                alpha=0.5,
                ecolor='cyan',
                capsize=10)

    plt.scatter(sys_size[14], errors_N4_mean, s=10, facecolors='orange', edgecolors='orange')
    ax.errorbar(sys_size[14], y=errors_N4_mean,
                yerr=errors_N4_std,
                alpha=0.5,
                ecolor='orange',
                capsize=10)
    plt.legend([r"ICNN $M_2$", r"ICNN $M_2$, rotated", r"ICNN $M_3$", r"ICNN $M_4$"])

    for i in indices_ref:
        # circle = plt.Circle(data[i], radius=1)
        plt.scatter(sys_size[i], errors[i], s=10, facecolors='red', edgecolors='red')
        # ax.add_patch(circle)
        label = ax.annotate(names[i], xy=(sys_size[i], errors[i]), fontsize=15, ha="center")

    plt.xscale("log")
    plt.yscale("linear")
    plt.xlabel("system size")
    plt.ylabel(r'$|M_N -S_{50}|$')
    # plt.show()
    plt.savefig("paper_data/paper2/illustrations/hohlraum/sys_size_vs_error.png", dpi=500)
    plt.clf()

    # b) timing over errror
    sns.set_theme()
    sns.set_style("white")
    fig, ax = plt.subplots()
    plt.scatter(timing[2], errors_N2_mean, s=10, facecolors='blue', edgecolors='blue')
    ax.errorbar(timing[2], y=errors_N2_mean,
                yerr=errors_N2_std,
                alpha=0.5,
                ecolor='blue',
                capsize=10)

    plt.scatter(timing[7], errors_N2_r_mean, s=10, facecolors='green', edgecolors='green')
    ax.errorbar(timing[7], y=errors_N2_r_mean,
                yerr=errors_N2_std,
                alpha=0.5,
                ecolor='green',
                capsize=10)

    plt.scatter(timing[11], errors_N3_mean, s=10, facecolors='cyan', edgecolors='cyan')
    ax.errorbar(timing[11], y=errors_N3_mean,
                yerr=errors_N2_std,
                alpha=0.5,
                ecolor='cyan',
                capsize=10)

    plt.scatter(timing[14], errors_N4_mean, s=10, facecolors='orange', edgecolors='orange')
    ax.errorbar(timing[14], y=errors_N4_mean,
                yerr=errors_N4_std,
                alpha=0.5,
                ecolor='orange',
                capsize=10)
    plt.legend([r"ICNN $M_2$", r"ICNN $M_2$, rotated", r"ICNN $M_3$", r"ICNN $M_4$"])

    for i in indices_ref:
        # circle = plt.Circle(data[i], radius=1)
        plt.scatter(timing[i], errors[i], s=10, facecolors='red', edgecolors='red')
        # ax.add_patch(circle)
        label = ax.annotate(names[i], xy=(timing[i], errors[i]), fontsize=15, ha="center")

        # ax.axis('off')
    # ax.set_aspect('equal')
    # ax.autoscale_view()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("simulation time")
    plt.ylabel(r'$|M_N -S_{50}|$')
    # plt.show()
    plt.savefig("paper_data/paper2/illustrations/hohlraum/timing_vs_error.png", dpi=500)
    plt.clf()

    # c) sys size vs timing
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    fig, ax = plt.subplots()

    plt.scatter(sys_size[2], timing[2], s=10, facecolors='red', edgecolors='red')
    label = ax.annotate(names[2], xy=(sys_size[2], timing[2],), fontsize=15, ha="center")
    plt.scatter(sys_size[7], timing[7], s=10, facecolors='red', edgecolors='red')
    label = ax.annotate(names[6], xy=(sys_size[7], timing[7],), fontsize=15, ha="center")
    plt.scatter(sys_size[11], timing[11], s=10, facecolors='red', edgecolors='red')
    label = ax.annotate(names[10], xy=(sys_size[11], timing[11],), fontsize=15, ha="center")
    plt.scatter(sys_size[14], timing[14], s=10, facecolors='red', edgecolors='red')
    label = ax.annotate(names[14], xy=(sys_size[14], timing[14],), fontsize=15, ha="center")

    # plt.legend([r"ICNN $M_2$", r"ICNN $M_2$, rotated", r"ICNN $M_3$", r"ICNN $M_4$"])

    for i in indices_ref:
        # circle = plt.Circle(data[i], radius=1)
        plt.scatter(sys_size[i], timing[i], s=10, facecolors='red', edgecolors='red')
    # ax.add_patch(circle)
    label = ax.annotate(names[i], xy=(sys_size[i], timing[i],), fontsize=15, ha="center")

    # ax.axis('off')
    # ax.set_aspect('equal')
    # ax.autoscale_view()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("system size")
    plt.ylabel("simulation time")
    # plt.show()
    plt.savefig("paper_data/paper2/illustrations/hohlraum/sys_size_vs_timing.png", dpi=500)
    plt.clf()

    # d) plot only neural network errors
    sns.set_theme()
    sns.set_style("white")
    fig, ax = plt.subplots()

    indices_g0 = [2, 6, 10, 14]
    indices_g1 = [3, 7, 11, 15]
    indices_g2 = [4, 8, 12, 16]
    indices_g3 = [5, 9, 13, 17]

    ind = np.arange(4)  # the x locations for the groups
    width = 0.1  # the width of the bars

    rects1 = ax.bar(ind, errors[indices_g0], width, color='k')  # g0
    rects4 = ax.bar(ind + width, errors[indices_g3], width, color='b')  # g3
    rects3 = ax.bar(ind + 2 * width, errors[indices_g2], width, color='g')  # g2
    rects2 = ax.bar(ind + 3 * width, errors[indices_g1], width, color='r')  # g1

    # add some
    ax.set_ylabel(r'$|M_N -S_{50}|$, ICNN based')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels((r'$M_2$', r'$M_2$, rot', r'$M_3$', r'$M_4$'))
    ax.legend((rects1[0], rects4[0], rects3[0], rects2[0]),
              (r'$\gamma=0$', r'$\gamma=0.001$', r'$\gamma=0.01$', r'$\gamma=0.1$'))

    plt.savefig("paper_data/paper2/illustrations/hohlraum/neural_error_bars.png", dpi=500, bbox_inches="tight")
    plt.clf()

    return 0


def get_infinum_subsequence(time_series: np.ndarray):
    """
    returns g(x_i) = min_k=1...,i g(x_k)
    """
    curr_min = time_series[0]
    out = np.copy(time_series)
    for i in range(len(time_series)):
        curr_min = np.min([curr_min, time_series[i]])
        out[i] = curr_min
    return out


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
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    return 0


if __name__ == '__main__':
    main()
