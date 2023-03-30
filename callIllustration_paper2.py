"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 5.4.2022
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
import seaborn as sns
import tensorflow as tf
from src.utils import plot_1d, plot_1dv2, scatter_plot_2d, load_data, beautify_img
from src.networks.configmodel import init_neural_closure
from src.math import EntropyTools

from adjustText import adjust_text
from scipy.spatial import ConvexHull


def main():
    print("---------- Start Result Illustration Suite ------------")

    # 0) beautify images
    # print_hohlraum_img_rect()

    # print_hohlraum_img()
    # 1) Training performance
    # print_training_performance()
    # print_training_performance_stats()

    # 2) Tests for realizable set
    # test_on_realizable_set_m2()

    # 3) Print memory-computational cost study
    # print_comp_efficiency_memory()

    # 4) Print cross-sections
    # print_cross_sections()

    # 5) Print method errors
    # print_method_errors()

    # 6) Get regularization errors
    # test_regularization_error()

    # 7) Print moment reconstructions
    # print_realizable_set_new_condition()
    # print_entropies()
    # print_realizable_set_by_gamma()

    # 8) rotated Linesource M1 2D monomial cross-sections
    print_linesource_m1_2d_mono_cross_sections()
    return True


def print_linesource_m1_2d_mono_cross_sections():
    folder_name = "paper_data/paper2/linesource_neural_rotations/structured_grid/monomial_g"
    save_folder = "paper_data/paper2/illustrations/linesource_neural_rotations"

    folder_name_reference = "paper_data/paper2/linesource_neural_rotations/structured_grid/baseline/monomial_g"
    font_size = 20
    data_jump = 15

    y_lims = [(-1.8, 2.3), (-4.5, 7), (-4.5, 6), (-2.5, 3.5)]
    y_lims_alpha = [(-1.8, 2.3), (-4.5, 7), (-4.5, 6), (-2.5, 3.5)]
    sym_size = 2.5
    mark_size = 4
    for i in range(0, 4):
        filename = str(i) + "/diag1.csv"
        filename2 = str(i) + "/diag2.csv"

        ds_diag1 = pd.read_csv(folder_name + filename)
        ds_diag2 = pd.read_csv(folder_name + filename2)
        ds_diag_newton = pd.read_csv(folder_name + ds_diag1)
        x_data = ds_diag1["arc_length"].to_numpy()
        npts = len(x_data)
        x_data_formatted = np.linspace(-1, 1, 1001)
        # plot moments
        plot_1dv2([x_data_formatted.reshape((npts, 1)), x_data_formatted.reshape((npts, 1))[::data_jump]],
                  [ds_diag_newton["u_0^0"].to_numpy().reshape((npts, 1)),
                   ds_diag1["u_0^0"].to_numpy().reshape((npts, 1))[::data_jump]],
                  name='linesource_m1_2d_reference_xs_u_g' + str(i),
                  log=False, loglog=False, folder_name=save_folder, font_size=font_size, symbol_size=sym_size,
                  marker_size=mark_size,
                  labels=[r"Newton $u_0$", r"ICNN $u_{0}$"],
                  linetypes=["-", "--", "o", "^"], show_fig=False, xlim=(-1, 1), xlabel=r"$x$",
                  ylabel=r"$u_0$", ylim=y_lims[i], legend_pos="lower right")

        plot_1dv2([x_data_formatted.reshape((npts, 1)), x_data_formatted.reshape((npts, 1)),
                   x_data_formatted.reshape((npts, 1))[::data_jump], x_data_formatted.reshape((npts, 1))[::data_jump]],
                  [ds_diag1["u_0^0"].to_numpy().reshape((npts, 1)),
                   ds_diag1["u_1^0"].to_numpy().reshape((npts, 1)),
                   ds_diag2["u_0^0"].to_numpy().reshape((npts, 1))[::data_jump],
                   -ds_diag2["u_1^1"].to_numpy().reshape((npts, 1))[::data_jump]],
                  name='linesource_m1_2d_mono_xs_u_g' + str(i),
                  log=False, loglog=False, folder_name=save_folder, font_size=font_size, symbol_size=sym_size,
                  marker_size=mark_size,
                  labels=[r"xs 1 $u_0$", r"diag 2 $u_{1} $", r"xs 1 $u_0$", r"diag 2 $-u_{1}$"],
                  linetypes=["-", "--", "o", "^"], show_fig=False, xlim=(-1, 1), xlabel=r"$x$",
                  ylabel=r"$\mathbf{u}$", ylim=y_lims[i], legend_pos="lower right")

        plot_1dv2([x_data_formatted.reshape((npts, 1)), x_data_formatted.reshape((npts, 1)),
                   x_data_formatted.reshape((npts, 1))[::data_jump], x_data_formatted.reshape((npts, 1))[::data_jump]],
                  [ds_diag1["alpha_0^0"].to_numpy().reshape((npts, 1)),
                   ds_diag1["alpha_1^0"].to_numpy().reshape((npts, 1)),
                   ds_diag2["alpha_0^0"].to_numpy().reshape((npts, 1))[::data_jump],
                   -ds_diag2["alpha_1^1"].to_numpy().reshape((npts, 1))[::data_jump]],
                  name='linesource_m1_2d_mono_xs_alpha_0_g' + str(i),
                  log=False, loglog=False, folder_name=save_folder, font_size=font_size, symbol_size=sym_size,
                  marker_size=mark_size,
                  labels=[r"xs 1 $\alpha^p_{\mathbf{u},0}$", r"xs 2 $\alpha^p_{\mathbf{u},1} $",
                          r"xs 1 $\alpha^p_{\mathbf{u},0}$", r"xs 2 $-\alpha^p_{\mathbf{u},1}$"],
                  linetypes=["-", "--", "o", "^"], show_fig=False, xlim=(-1, 1), xlabel=r"$x$",
                  ylabel=r"$\mathbf{alpha}_{\mathbf{u}}^p$", legend_pos="upper left")  # ylim=y_lims[i],

    exit(1)

    return 0


def print_hohlraum_img_rect():
    fontsize = 20

    base_path = "paper_data/paper2/hohlraum_rectangular/img_files/"
    save_path = "paper_data/paper2/illustrations/hohlraum_rectangular/flow_fields"

    names = ["hohlraum_S61", "hohlraum_S61",
             "hohlraum_mono_ICNN_M2_g1", "hohlraum_mono_ICNN_M2_g1_rot", "hohlraum_mono_ICNN_M3_g1",
             "hohlraum_mono_ICNN_M4_g1", "hohlraum_P9", "hohlraum_S41"]
    names_err = ["d_hohlraum_mono_ICNN_M2_g1", "d_hohlraum_mono_ICNN_M2_g1_rot", "d_hohlraum_mono_ICNN_M3_g1",
                 "d_hohlraum_mono_ICNN_M4_g1", "d_hohlraum_P9", "d_hohlraum_S41"]

    for name in names:
        img_path = base_path + name + ".png"
        beautify_img(load_name=img_path, folder_name=save_path, name=name, c_map="RdGy",
                     xlabel="", ylabel="", cbar_ticks=[0, 0.2, 0.4, 0.6], cbar_log=False,
                     font_size=fontsize, img_size=[0, 1, 0, 1])

    for name in names_err:
        img_path = base_path + name + ".png"
        beautify_img(load_name=img_path, folder_name=save_path, name=name,
                     xlabel="", ylabel="", cbar_ticks=[1e-3, 1e-2, 1e-1], cbar_log=True,
                     font_size=fontsize, img_size=[0, 1, 0, 1])

    # without cbar
    names = ["hohlraum_S11", "hohlraum_S21", "hohlraum_S31",
             "hohlraum_mono_ICNN_M2_g0", "hohlraum_mono_ICNN_M2_g3", "hohlraum_mono_ICNN_M2_g2",
             "hohlraum_mono_ICNN_M2_g0_rot", "hohlraum_mono_ICNN_M2_g3_rot", "hohlraum_mono_ICNN_M2_g2_rot",
             "hohlraum_mono_ICNN_M3_g0", "hohlraum_mono_ICNN_M3_g3", "hohlraum_mono_ICNN_M3_g2",
             "hohlraum_mono_ICNN_M4_g0", "hohlraum_mono_ICNN_M4_g3", "hohlraum_mono_ICNN_M4_g2",
             "hohlraum_P3", "hohlraum_P5", "hohlraum_P7",
             "d_hohlraum_S11", "d_hohlraum_S21", "d_hohlraum_S31",
             "d_hohlraum_mono_ICNN_M2_g0", "d_hohlraum_mono_ICNN_M2_g3", "d_hohlraum_mono_ICNN_M2_g2",
             "d_hohlraum_mono_ICNN_M2_g0_rot", "d_hohlraum_mono_ICNN_M2_g3_rot", "d_hohlraum_mono_ICNN_M2_g2_rot",
             "d_hohlraum_mono_ICNN_M3_g0", "d_hohlraum_mono_ICNN_M3_g3", "d_hohlraum_mono_ICNN_M3_g2",
             "d_hohlraum_mono_ICNN_M4_g0", "d_hohlraum_mono_ICNN_M4_g3", "d_hohlraum_mono_ICNN_M4_g2",
             "d_hohlraum_P3", "d_hohlraum_P5", "d_hohlraum_P7"]
    for name in names:
        img_path = base_path + name + ".png"
        beautify_img(load_name=img_path, folder_name=save_path, name=name, xlabel="", ylabel="", cbar_ticks=None,
                     cbar_log=False, font_size=fontsize, img_size=[0, 1, 0, 1])

    return 0


def print_hohlraum_img():
    fontsize = 20

    base_path = "paper_data/paper2/hohlraum/img_files/"
    save_path = "paper_data/paper2/illustrations/hohlraum/flow_fields"

    names = ["hohlraum_S50", "hohlraum_S50", "hohlraum_N2_g1", "hohlraum_N2_g1_r", "hohlraum_N3_g1", "hohlraum_N4_g1",
             "hohlraum_P9",
             "hohlraum_S40"]
    names_err = ["hohlraum_M3_g3_Newton_diff", "hohlraum_N2_g1_diff", "hohlraum_N2_g1_r_diff", "hohlraum_N3_g1_diff",
                 "hohlraum_N4_g1_diff", "hohlraum_P9_diff", "hohlraum_S40_diff"]

    for name in names:
        img_path = base_path + name + ".png"
        beautify_img(load_name=img_path, folder_name=save_path, name=name, c_map="RdGy",
                     xlabel="", ylabel="", cbar_ticks=[0, 0.2, 0.4, 0.6], cbar_log=False,
                     font_size=fontsize, img_size=[0, 1, 0, 1])

    for name in names_err:
        img_path = base_path + name + ".png"
        beautify_img(load_name=img_path, folder_name=save_path, name=name,
                     xlabel="", ylabel="", cbar_ticks=[1e-4, 1e-3, 1e-2, 1e-1], cbar_log=True,
                     font_size=fontsize, img_size=[0, 1, 0, 1])

    # without cbar
    names = ["hohlraum_M2_g3_neural_vs_Newton", "hohlraum_M2_g3_Newton_diff", "hohlraum_M3_g3_neural_vs_Newton",
             "hohlraum_N2_g0", "hohlraum_N2_g3", "hohlraum_N2_g2",
             "hohlraum_N2_g0_diff", "hohlraum_N2_g3_diff", "hohlraum_N2_g2_diff",
             "hohlraum_N2_g0_r", "hohlraum_N2_g3_r", "hohlraum_N2_g2_r",
             "hohlraum_N2_g0_r_diff", "hohlraum_N2_g3_r_diff", "hohlraum_N2_g2_r_diff",
             "hohlraum_N3_g0", "hohlraum_N3_g3", "hohlraum_N3_g2",
             "hohlraum_N3_g0_diff", "hohlraum_N3_g3_diff", "hohlraum_N3_g2_diff",
             "hohlraum_N4_g0", "hohlraum_N4_g3", "hohlraum_N4_g2",
             "hohlraum_N4_g0_diff", "hohlraum_N4_g3_diff", "hohlraum_N4_g2_diff",
             "hohlraum_P3", "hohlraum_P5", "hohlraum_P7",
             "hohlraum_P3_diff", "hohlraum_P5_diff", "hohlraum_P7_diff",
             "hohlraum_S10", "hohlraum_S20", "hohlraum_S30",
             "hohlraum_S10_diff", "hohlraum_S20_diff", "hohlraum_S30_diff"]
    for name in names:
        img_path = base_path + name + ".png"
        beautify_img(load_name=img_path, folder_name=save_path, name=name, xlabel="", ylabel="", cbar_ticks=None,
                     cbar_log=False, font_size=fontsize, img_size=[0, 1, 0, 1])

    return 0


def print_realizable_set_by_gamma():
    folder_name = "paper_data/paper2/u_sampling_by_gamma/"
    save_folder = "paper_data/paper2/illustrations/u_sampling_by_gamma"
    # --- Realizable set illustrations ---
    for i in range(0, 4):
        [u, alpha, h] = load_data(filename=folder_name + "M2_1D_g" + str(i) + "_reduced_ev5.csv",
                                  data_dim=3, selected_cols=[True, True, True])
        max_h = 3
        min_h = np.min(h)
        alpha_bound = 40
        marker_size = 1
        if i == 1:
            lim_x = (-5.2, 5.2)
            lim_y = (-4.5, 5.2)
        elif i == 2:
            lim_x = (-1.5, 1.5)
            lim_y = (-0.5, 1.5)
        else:
            lim_x = (-1.1, 1.1)
            lim_y = (-0.1, 1.1)

        lim_z = (np.min(h), np.max(h))
        xticks_a = [-40, -20, 0, 20, 40]
        xticks_u = [-1, -0.5, 0., 0.5, 1]
        yticks_u = [0, 0.3, 0.6, 1]
        if i == 1:
            yticks_u = [-4, -2, 0, 2, 4]
            xticks_u = [-4, -2, 0, 2, 4]
        if i == 2:
            yticks_u = [-.5, 0, .5, 1, 1.5]
            xticks_u = [-1.5, -0.5, 0.5, 1.5]

        scatter_plot_2d(x_in=u[:, 1:], z_in=h, lim_x=lim_x, lim_y=lim_y, lim_z=lim_z, title=r"$h$ over $\mathcal{R}^r$",
                        label_x=r"$\overline{u}_1$", label_y=r"$\overline{u}_2$",
                        folder_name=save_folder, name="M2_1D_uniform_g" + str(i) + "_u", show_fig=False,
                        log=False, color_map=0, marker_size=marker_size, axis_formatter=True, font_size=28,
                        xticks=xticks_u, yticks=yticks_u)
        scatter_plot_2d(x_in=alpha[:, 1:], z_in=h, lim_x=(-alpha_bound, alpha_bound), lim_y=(-alpha_bound, alpha_bound),
                        lim_z=lim_z, title=r"$h$ over $\alpha^r$",
                        label_x=r"$\alpha_{\overline{\mathbf{u}},1}^\gamma$",
                        label_y=r"$\alpha_{\overline{\mathbf{u}},2}^\gamma$",
                        folder_name=save_folder, name="M2_1D_uniform_g" + str(i) + "_alpha", show_fig=False,
                        log=False, color_map=0, marker_size=marker_size, font_size=28, xticks=xticks_a, yticks=xticks_a)

    return 0


def print_entropies():
    # 1) gamma 0
    [u_g0, alpha_g0, h_g0] = load_data(filename="paper_data/paper2/realizable_set/M1_1D_g0.csv", data_dim=2,
                                       selected_cols=[True, True, True])
    [u_g3, alpha_g3, h_g3] = load_data(filename="paper_data/paper2/realizable_set/M1_1D_g3.csv", data_dim=2,
                                       selected_cols=[True, True, True])
    [u_g2, alpha_g2, h_g2] = load_data(filename="paper_data/paper2/realizable_set/M1_1D_g2.csv", data_dim=2,
                                       selected_cols=[True, True, True])
    [u_g1, alpha_g1, h_g1] = load_data(filename="paper_data/paper2/realizable_set/M1_1D_g1.csv", data_dim=2,
                                       selected_cols=[True, True, True])
    t0 = u_g0[:, 1].argsort()
    t1 = u_g1[:, 1].argsort()
    t2 = u_g2[:, 1].argsort()
    t3 = u_g3[:, 1].argsort()
    y_ticks = [-1.5, -1, -0.5, 0, 0.5, 1.0]
    x_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

    sns.set_theme()
    sns.set_style("ticks")
    colors = ['k-', 'r--', 'g-.', 'b:']
    symbol_size = 2
    font_size = 15

    plt.figure(figsize=(5.8, 4.7), dpi=400)

    plt.plot(u_g0[t0, 1:], h_g0[t0], colors[0], linewidth=symbol_size)
    plt.plot(u_g3[t3, 1:], h_g3[t3], colors[1], linewidth=symbol_size)
    plt.plot(u_g2[t2, 1:], h_g2[t2], colors[2], linewidth=symbol_size)
    plt.plot(u_g1[t1, 1:], h_g1[t1], colors[3], linewidth=symbol_size)
    plt.xlim((-1.3, 1.3))
    plt.ylim((-1.7, 0.5))
    plt.ylabel(r"$\hat{h}^\gamma$", fontsize=font_size)
    plt.xlabel(r"$\overline{u}_1$", fontsize=font_size)
    plt.xticks(fontsize=int(0.7 * font_size))
    plt.yticks(fontsize=int(0.7 * font_size))
    plt.legend([r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"], fontsize=int(0.6 * font_size))
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # draw zoombox
    left = 0.88
    right = 1.03
    bottom = 0.5
    top = 0.7
    plt.plot([left, left], [bottom, top], 'k-', linewidth=0.7)
    plt.plot([right, right], [bottom, top], 'k-', linewidth=0.7)
    plt.plot([left, right], [bottom, bottom], 'k-', linewidth=0.7)
    plt.plot([left, right], [top, top], 'k-', linewidth=0.7)

    # draw "zoom lines"
    plt.plot([left, -0.283], [top, -0.2727], 'k-', linewidth=0.7)
    plt.plot([right, 0.283], [bottom, -1.025], 'k-', linewidth=0.7)

    # 1) gamma 0

    # create zoom in view
    # location for the zoomed portion
    sub_axes = plt.axes([.452, 0.351, 0.22, 0.22])  # [left, bottom, width, height]

    # plot the zoomed portion
    #  gamma 0
    sub_axes.plot(u_g0[t0, 1:], h_g0[t0], colors[0], linewidth=symbol_size)
    sub_axes.plot(u_g3[t3, 1:], h_g3[t3], colors[1], linewidth=symbol_size)
    sub_axes.plot(u_g2[t2, 1:], h_g2[t2], colors[2], linewidth=symbol_size)
    sub_axes.plot(u_g1[t1, 1:], h_g1[t1], colors[3], linewidth=symbol_size)

    sub_axes.set_xlim(left=left, right=right)
    sub_axes.set_ylim(bottom=bottom, top=top)
    # sub_axes.set_yticklabels([])
    # sub_axes.set_xticklabels([])
    # Set aspect ratio
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 1.0
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
    plt.tight_layout()

    plt.savefig("paper_data/paper2/illustrations/realizable_set/entropy_gammas" + ".pdf", dpi=500)
    plt.close()
    plt.clf()


def print_realizable_set_new_condition():
    foldername = "paper_data/paper2/u_sampling_by_gamma/"
    [u_g0, alpha_g0, h_g0] = load_data(filename=foldername + "M2_1D_g0_reduced_ev5.csv",
                                       data_dim=3,
                                       selected_cols=[True, True, True])
    [u_g3, alpha_g3, h_g3] = load_data(filename=foldername + "M2_1D_g3_reduced_ev5.csv",
                                       data_dim=3,
                                       selected_cols=[True, True, True])
    [u_g2, alpha_g2, h_g2] = load_data(filename=foldername + "M2_1D_g2_reduced_ev5.csv",
                                       data_dim=3,
                                       selected_cols=[True, True, True])
    [u_g1, alpha_g1, h_g1] = load_data(filename=foldername + "M2_1D_g1_reduced_ev5.csv",
                                       data_dim=3,
                                       selected_cols=[True, True, True])

    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)

    sns.set_theme()
    sns.set_style("ticks")
    colors = ['k-', 'r--', 'g-.', 'b:']
    symbol_size = 2
    font_size = 15
    # 1) gamma 0
    points_g0 = u_g0[:, 1:]
    hull = ConvexHull(points_g0)
    pts_line0_x = []
    pts_line0_y = []
    for simplex in hull.simplices:
        pts_line0_x.append(points_g0[simplex, 0][0])
        pts_line0_y.append(points_g0[simplex, 1][0])

    pts_line0_x = np.asarray(pts_line0_x)
    pts_line0_y = np.asarray(pts_line0_y)
    mask = pts_line0_x.argsort()
    pts_line0_x = pts_line0_x[mask]
    pts_line0_y = pts_line0_y[mask]
    line1 = plt.plot(pts_line0_x, pts_line0_y, colors[0], linewidth=symbol_size)  # plot underbelly
    plt.plot([pts_line0_x[0], pts_line0_x[-1]], [pts_line0_y[0], pts_line0_y[-1]], colors[0],
             linewidth=symbol_size)  # plot top

    # 2) gamma 0.001
    points_g3 = u_g3[:, 1:]
    hull = ConvexHull(points_g3)
    pts_line3_x = []
    pts_line3_y = []
    for simplex in hull.simplices:
        pts_line3_x.append(points_g3[simplex, 0][0])
        pts_line3_y.append(points_g3[simplex, 1][0])

    pts_line3_x_p = []
    pts_line3_y_p = []
    pts_line3_x_m = []
    pts_line3_y_m = []

    for i in range(len(pts_line3_x)):
        if pts_line3_y[i] > 0.98:
            pts_line3_x_p.append(pts_line3_x[i])
            pts_line3_y_p.append(pts_line3_y[i])
        else:
            pts_line3_x_m.append(pts_line3_x[i])
            pts_line3_y_m.append(pts_line3_y[i])

    pts_line3_x_p = np.asarray(pts_line3_x_p)
    pts_line3_y_p = np.asarray(pts_line3_y_p)
    pts_line3_x_m = np.asarray(pts_line3_x_m)
    pts_line3_y_m = np.asarray(pts_line3_y_m)

    mask_p = pts_line3_x_p.argsort()
    mask_m = pts_line3_x_m.argsort()

    pts_line3_x_p = pts_line3_x_p[mask_p]
    pts_line3_y_p = pts_line3_y_p[mask_p]
    pts_line3_x_m = pts_line3_x_m[mask_m]
    pts_line3_y_m = pts_line3_y_m[mask_m]

    line2 = plt.plot(pts_line3_x_p, pts_line3_y_p, colors[1], linewidth=symbol_size)  # plot underbelly
    plt.plot(pts_line3_x_m, pts_line3_y_m, colors[1], linewidth=symbol_size)  # plot top
    # plt.plot([pts_line3_x_m[0], pts_line3_x_p[0]], [pts_line3_y_m[0], pts_line3_y_p[0]], colors[1],
    #         linewidth=symbol_size)
    # plt.plot([pts_line3_x_m[-1], pts_line3_x_p[-1]], [pts_line3_y_m[-1], pts_line3_y_p[-1]], colors[1],
    #         linewidth=symbol_size)

    # 3) gamma 0.01
    points_g2 = u_g2[:, 1:]
    hull = ConvexHull(points_g2)
    pts_line2_x = []
    pts_line2_y = []
    for simplex in hull.simplices:
        pts_line2_x.append(points_g2[simplex, 0][1])
        pts_line2_y.append(points_g2[simplex, 1][1])

    pts_line2_x_p = []
    pts_line2_y_p = []
    pts_line2_x_m = []
    pts_line2_y_m = []

    for i in range(len(pts_line2_x)):
        if pts_line2_y[i] > 1.0:
            pts_line2_x_p.append(pts_line2_x[i])
            pts_line2_y_p.append(pts_line2_y[i])
        else:
            pts_line2_x_m.append(pts_line2_x[i])
            pts_line2_y_m.append(pts_line2_y[i])

    pts_line2_x_p = np.asarray(pts_line2_x_p)
    pts_line2_y_p = np.asarray(pts_line2_y_p)
    pts_line2_x_m = np.asarray(pts_line2_x_m)
    pts_line2_y_m = np.asarray(pts_line2_y_m)

    mask_p = pts_line2_x_p.argsort()
    mask_m = pts_line2_x_m.argsort()

    pts_line2_x_p = pts_line2_x_p[mask_p]
    pts_line2_y_p = pts_line2_y_p[mask_p]
    pts_line2_x_m = pts_line2_x_m[mask_m]
    pts_line2_y_m = pts_line2_y_m[mask_m]

    line3 = plt.plot(pts_line2_x_p, pts_line2_y_p, colors[2], linewidth=symbol_size)  # plot underbelly
    plt.plot(pts_line2_x_m, pts_line2_y_m, colors[2], linewidth=symbol_size)  # plot top
    plt.plot([pts_line2_x_m[0], pts_line2_x_p[0]], [pts_line2_y_m[0], pts_line2_y_p[0]], colors[2],
             linewidth=symbol_size)
    plt.plot([pts_line2_x_m[-1], pts_line2_x_p[-1]], [pts_line2_y_m[-1], pts_line2_y_p[-1]], colors[2],
             linewidth=symbol_size)

    # 4) gamma 0.1
    points_g1 = u_g1[:, 1:]
    hull = ConvexHull(points_g1)
    pts_line1_x = []
    pts_line1_y = []
    for simplex in hull.simplices:
        pts_line1_x.append(points_g1[simplex, 0][0])
        pts_line1_y.append(points_g1[simplex, 1][0])

    pts_line1_x_p = []
    pts_line1_y_p = []
    pts_line1_x_m = []
    pts_line1_y_m = []

    for i in range(len(pts_line1_x)):
        if pts_line1_y[i] > 1.0:
            pts_line1_x_p.append(pts_line1_x[i])
            pts_line1_y_p.append(pts_line1_y[i])
        else:
            pts_line1_x_m.append(pts_line1_x[i])
            pts_line1_y_m.append(pts_line1_y[i])

    pts_line1_x_p = np.asarray(pts_line1_x_p)
    pts_line1_y_p = np.asarray(pts_line1_y_p)
    pts_line1_x_m = np.asarray(pts_line1_x_m)
    pts_line1_y_m = np.asarray(pts_line1_y_m)

    mask_p = pts_line1_x_p.argsort()
    mask_m = pts_line1_x_m.argsort()

    pts_line1_x_p = pts_line1_x_p[mask_p]
    pts_line1_y_p = pts_line1_y_p[mask_p]
    pts_line1_x_m = pts_line1_x_m[mask_m]
    pts_line1_y_m = pts_line1_y_m[mask_m]
    line4 = plt.plot(pts_line1_x_p, pts_line1_y_p, colors[3], linewidth=symbol_size)  # plot underbelly
    plt.plot(pts_line1_x_m, pts_line1_y_m, colors[3], linewidth=symbol_size)  # plot top
    plt.plot([pts_line1_x_m[0], pts_line1_x_p[0]], [pts_line1_y_m[0], pts_line1_y_p[0]], colors[3],
             linewidth=symbol_size)
    plt.plot([pts_line1_x_m[-1], pts_line1_x_p[-1]], [pts_line1_y_m[-1], pts_line1_y_p[-1]], colors[3],
             linewidth=symbol_size)
    # draw zoom box
    left = 0.9
    right = 1.05
    bottom = 0.9
    top = 1.05
    plt.plot([left, bottom], [left, top], 'k-', linewidth=0.7)
    plt.plot([right, bottom], [right, top], 'k-', linewidth=0.7)
    plt.plot([right, bottom], [left, bottom], 'k-', linewidth=0.7)
    plt.plot([right, top], [left, top], 'k-', linewidth=0.7)

    # draw "zoom lines"
    plt.plot([left, 1.5], [top, 3.8], 'k-', linewidth=0.7)
    plt.plot([right, 3.44], [bottom, 1.63], 'k-', linewidth=0.7)

    # 1) gamma 0
    plt.legend(
        [line1[0], line2[0], line3[0], line4[0]], [r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
        loc="upper left", fontsize=int(0.6 * font_size))

    plt.xlabel(r"$\overline{u}_1$", fontsize=font_size)
    plt.ylabel(r"$\overline{u}_2$", fontsize=font_size)
    plt.xticks(fontsize=int(0.7 * font_size))
    plt.yticks(fontsize=int(0.7 * font_size))

    # create zoom in view
    # location for the zoomed portion
    sub_axes = plt.axes([.65, .65, 0.18, 0.18])  # [left, bottom, width, height]

    # plot the zoomed portion
    #  gamma 0
    sub_axes.plot(pts_line0_x, pts_line0_y, colors[0], linewidth=symbol_size)  # plot underbelly
    sub_axes.plot([pts_line0_x[0], pts_line0_x[-1]], [pts_line0_y[0], pts_line0_y[-1]], colors[0],
                  linewidth=symbol_size)  # plot top

    #  gamma 0.001
    sub_axes.plot(pts_line3_x_p, pts_line3_y_p, colors[1], linewidth=symbol_size)  # plot underbelly
    sub_axes.plot(pts_line3_x_m, pts_line3_y_m, colors[1], linewidth=symbol_size)  # plot top

    sub_axes.set_xlim(left=left, right=right)
    sub_axes.set_ylim(bottom=bottom, top=top)
    # sub_axes.set_yticklabels([])
    # sub_axes.set_xticklabels([])
    # Set aspect ratio
    ax = plt.gca()  # you first need to get the axis handle
    x_left, x_right = ax.get_xlim()
    y_low, y_high = ax.get_ylim()
    ratio = 1.0
    ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

    # save figure

    plt.tight_layout()
    # plt.show()
    plt.savefig("paper_data/paper2/illustrations/realizable_set/nc_realizable_set_gammas" + ".pdf", dpi=500)
    print("Figure successfully saved to file: " + str(
        "paper_data/paper2/illustrations/realizable_set/nc_realizable_set_gammas" + ".pdf"))
    plt.close()
    plt.clf()

    return 0


def test_regularization_error():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    errors_m2 = test_regularization_error_m2()
    errors_m3 = test_regularization_error_m3()
    errors_m4 = test_regularization_error_m4()

    print("ICNN-M2")
    print(errors_m2[0])
    print(errors_m2[2])
    print(errors_m2[4])
    print(errors_m2[6])
    print("ICNN-M3")
    print(errors_m3[0])
    print(errors_m3[2])
    print(errors_m3[4])
    print(errors_m3[6])
    print("ICNN-M4")
    print(errors_m4[0])
    print(errors_m4[2])
    print(errors_m4[4])
    print(errors_m4[6])
    print("ResNet-M2")
    print(errors_m2[1])
    print(errors_m2[3])
    print(errors_m2[5])
    print(errors_m2[7])
    print("ResNet-M3")
    print(errors_m3[1])
    print(errors_m3[3])
    print(errors_m3[5])
    print(errors_m3[7])
    print("ResNet-M4")
    print(errors_m4[1])
    print(errors_m4[3])
    print(errors_m4[5])
    print(errors_m4[7])
    return 0


def test_regularization_error_m2() -> list:
    # M2 2D
    data = load_data(filename="paper_data/paper2/regularization_error/Monomial_M2_2D_normal_gaussian.csv", data_dim=6,
                     selected_cols=[True, True, True])
    u_test = tf.constant(data[0][:, 1:], dtype=tf.float32)

    # Input Convex
    mk11_m2_2d_g0 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g0/best_model")
    mk11_m2_2d_g1 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g1_v2/best_model")
    mk11_m2_2d_g2 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g2/best_model")
    mk11_m2_2d_g3 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk11_m2_2d_g3/best_model")
    # Get model predictions
    pred_g0 = mk11_m2_2d_g0(u_test, training=False)
    errors_M2_g0 = get_mean_error(pred_g0, data)
    pred_g1 = mk11_m2_2d_g1(u_test, training=False)
    errors_M2_g1 = get_mean_error(pred_g1, data)
    pred_g2 = mk11_m2_2d_g2(u_test, training=False)
    errors_M2_g2 = get_mean_error(pred_g2, data)
    pred_g3 = mk11_m2_2d_g3(u_test, training=False)
    errors_M2_g3 = get_mean_error(pred_g3, data)

    # Resnet
    mk12_m2_2d_g0 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk12_m2_2d_g0/best_model")
    mk12_m2_2d_g1 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk12_m2_2d_g1/best_model")
    mk12_m2_2d_g2 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk12_m2_2d_g2/best_model")
    mk12_m2_2d_g3 = tf.keras.models.load_model("paper_data/paper2/2D_M2/mk12_m2_2d_g3/best_model")
    # Get model predictions
    pred_g0 = mk12_m2_2d_g0(u_test, training=False)
    res_errors_M2_g0 = get_mean_error(pred_g0, data)
    pred_g1 = mk12_m2_2d_g1(u_test, training=False)
    res_errors_M2_g1 = get_mean_error(pred_g1, data)
    pred_g2 = mk12_m2_2d_g2(u_test, training=False)
    res_errors_M2_g2 = get_mean_error(pred_g2, data)
    pred_g3 = mk12_m2_2d_g3(u_test, training=False)
    res_errors_M2_g3 = get_mean_error(pred_g3, data)

    return [errors_M2_g0, errors_M2_g3, errors_M2_g2, errors_M2_g1, res_errors_M2_g0, res_errors_M2_g3,
            res_errors_M2_g2, res_errors_M2_g1]


def test_regularization_error_m3() -> list:
    # ------------------------- M3 2D ------------------------------
    data = load_data(filename="paper_data/paper2/regularization_error/Monomial_M3_2D_normal_gaussian.csv", data_dim=10,
                     selected_cols=[True, True, True])
    u_test = tf.constant(data[0][:, 1:], dtype=tf.float32)

    # ICNN
    mk11_m3_2d_g0 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk11_M3_2D_g0/best_model")
    mk11_m3_2d_g1 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk11_M3_2D_g1/best_model")
    mk11_m3_2d_g2 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk11_M3_2D_g2/best_model")
    mk11_m3_2d_g3 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk11_M3_2D_g3/best_model")
    # Get model predictions
    pred_g0 = mk11_m3_2d_g0(u_test, training=False)
    errors_M3_g0 = get_mean_error(pred_g0, data)
    pred_g1 = mk11_m3_2d_g1(u_test, training=False)
    errors_M3_g1 = get_mean_error(pred_g1, data)
    pred_g2 = mk11_m3_2d_g2(u_test, training=False)
    errors_M3_g2 = get_mean_error(pred_g2, data)
    pred_g3 = mk11_m3_2d_g3(u_test, training=False)
    errors_M3_g3 = get_mean_error(pred_g3, data)

    # ResNet
    mk12_m3_2d_g0 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk12_M3_2D_g0/best_model")
    mk12_m3_2d_g1 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk12_M3_2D_g1/best_model")
    mk12_m3_2d_g2 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk12_M3_2D_g2/best_model")
    mk12_m3_2d_g3 = tf.keras.models.load_model("paper_data/paper2/2D_M3/mk12_M3_2D_g3/best_model")
    # Get model predictions
    pred_g0 = mk12_m3_2d_g0(u_test, training=False)
    res_errors_M3_g0 = get_mean_error(pred_g0, data)
    pred_g1 = mk12_m3_2d_g1(u_test, training=False)
    res_errors_M3_g1 = get_mean_error(pred_g1, data)
    pred_g2 = mk12_m3_2d_g2(u_test, training=False)
    res_errors_M3_g2 = get_mean_error(pred_g2, data)
    pred_g3 = mk12_m3_2d_g3(u_test, training=False)
    res_errors_M3_g3 = get_mean_error(pred_g3, data)

    return [errors_M3_g0, errors_M3_g3, errors_M3_g2, errors_M3_g1, res_errors_M3_g0, res_errors_M3_g3,
            res_errors_M3_g2, res_errors_M3_g1]


def test_regularization_error_m4() -> list:
    # M4 2D
    data = load_data(filename="paper_data/paper2/regularization_error/Monomial_M4_2D_normal_gaussian.csv", data_dim=15,
                     selected_cols=[True, True, True])
    u_test = tf.constant(data[0][:, 1:], dtype=tf.float32)

    # ICNN
    mk11_m4_2d_g0 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk11_m4_2d_g0/best_model")
    mk11_m4_2d_g1 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk11_m4_2d_g1/best_model")
    mk11_m4_2d_g2 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk11_m4_2d_g2/best_model")
    mk11_m4_2d_g3 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk11_m4_2d_g3/best_model")
    # Get model predictions
    pred_g0 = mk11_m4_2d_g0(u_test, training=False)
    errors_M4_g0 = get_mean_error(pred_g0, data)
    pred_g1 = mk11_m4_2d_g1(u_test, training=False)
    errors_M4_g1 = get_mean_error(pred_g1, data)
    pred_g2 = mk11_m4_2d_g2(u_test, training=False)
    errors_M4_g2 = get_mean_error(pred_g2, data)
    pred_g3 = mk11_m4_2d_g3(u_test, training=False)
    errors_M4_g3 = get_mean_error(pred_g3, data)

    # REsNet
    mk12_m4_2d_g0 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk12_m4_2d_g0/best_model")
    mk12_m4_2d_g1 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk12_m4_2d_g1/best_model")
    mk12_m4_2d_g2 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk12_m4_2d_g2/best_model")
    mk12_m4_2d_g3 = tf.keras.models.load_model("paper_data/paper2/2D_M4/mk12_m4_2d_g3/best_model")
    # Get model predictions
    pred_g0 = mk12_m4_2d_g0(u_test, training=False)
    res_errors_M4_g0 = get_mean_error(pred_g0, data)
    pred_g1 = mk12_m4_2d_g1(u_test, training=False)
    res_errors_M4_g1 = get_mean_error(pred_g1, data)
    pred_g2 = mk12_m4_2d_g2(u_test, training=False)
    res_errors_M4_g2 = get_mean_error(pred_g2, data)
    pred_g3 = mk12_m4_2d_g3(u_test, training=False)
    res_errors_M4_g3 = get_mean_error(pred_g3, data)

    return [errors_M4_g0, errors_M4_g3, errors_M4_g2, errors_M4_g1, res_errors_M4_g0, res_errors_M4_g3,
            res_errors_M4_g2, res_errors_M4_g1]


def get_mean_error(predictions: list, test_data: list) -> list:
    u_test = tf.constant(test_data[0][:, 1:], dtype=tf.float32)
    alpha_test = tf.constant(test_data[1][:, 1:], dtype=tf.float32)
    h_test = tf.constant(test_data[2], dtype=tf.float32)
    h_pred_g0 = tf.cast(predictions[0], dtype=tf.float32)
    alpha_pred_g0 = tf.cast(predictions[1], dtype=tf.float32)
    u_pred_g0 = tf.cast(predictions[2], dtype=tf.float32)

    # err_u = tf.reduce_mean(tf.square(tf.norm(u_test - u_pred_g0, axis=1, keepdims=False)))
    # err_alpha = tf.reduce_mean(tf.square(tf.norm(alpha_test - alpha_pred_g0, axis=1, keepdims=False)))
    # err_h = tf.reduce_mean(tf.square(tf.norm(h_test - h_pred_g0, axis=1, keepdims=False)))

    mse = tf.keras.losses.MeanSquaredError()

    err_u = mse(u_test, u_pred_g0)
    err_alpha = mse(alpha_test, alpha_pred_g0)
    err_h = mse(h_test, h_pred_g0)

    return [err_u.numpy(), err_alpha.numpy(), err_h.numpy()]


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
                    name="moment_path", log=False,
                    folder_name="paper_data/paper2/illustrations/model_testing")
    ys = [h_recons[:, 0], out_g0[0][:, 0], out_g1[0][:, 0], out_g2[0][:, 0], out_g3[0][:, 0]]
    plot_1dv2(xs=[t_param], ys=ys,
              labels=["h", "g0", "g1", "g2", "g3"],
              name="entropyplot2", log=False, loglog=False, folder_name="paper_data/paper2/illustrations/model_testing",
              linetypes=["-", "--", "--", "--", "--"], show_fig=False, xlim=None, ylim=None, xlabel="u curve",
              ylabel="entropy")

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
              ylabel="entropy")
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
              show_fig=False, xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk11_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m2_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m2_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk11_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m2_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m2_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m2_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk13_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m2_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m2_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk13_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m2_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m2_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m2_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk12_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m2_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m2_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk12_m2_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m2_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m2_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m2_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk11_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m3_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m3_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk11_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m3_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m3_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m3_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk13_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m3_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m3_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk13_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m3_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m3_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m3_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk12_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m3_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m3_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk12_m3_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m3_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m3_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m3_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk11_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m4_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m4_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk11_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m4_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m4_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m4_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk13_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m4_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m4_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk13_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m4_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m4_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m4_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk12_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m4_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m4_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk12_m4_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m4_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m4_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m4_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk11_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m5_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m5_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk11_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk11_m5_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk11_m5_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk11_m5_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk13_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m5_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m5_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk13_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk13_m5_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk13_m5_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk13_m5_u_gammas", log=True, folder_name="paper_data/paper2/illustrations/training",
              show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")

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
              xlabel="epochs", ylabel="loss h", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="lower left")

    plot_1dv2([mk12_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m5_2d_g0["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g3["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g2["val_output_2_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g1["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m5_alpha_gammas", log=True, folder_name="paper_data/paper2/illustrations", show_fig=False,
              xlabel="epochs", ylabel=r"loss $\alpha_u$", xlim=[0, 2000], ylim=[1e-5, 1e-0],
              legend_pos="lower left")

    plot_1dv2([mk12_m5_2d_g0["epoch"].to_numpy()],
              [get_infinum_subsequence(mk12_m5_2d_g0["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g3["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g2["val_output_3_loss"].to_numpy().reshape(n_epochs, 1)),
               get_infinum_subsequence(mk12_m5_2d_g1["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk12_m5_u_gammas", log=True, folder_name="paper_data/paper2/illustrations", show_fig=False,
              xlabel="epochs", ylabel="loss u", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="lower left")
    return 0


def print_stats_run(g_0_folder: str, g_1_folder: str, g_2_folder: str, g_3_folder: str, mk: str, order: str):
    n_epochs = 2000
    epochs = np.linspace(1, n_epochs, n_epochs)

    g0_runs = []
    g1_runs = []
    g2_runs = []
    g3_runs = []

    # load losses
    for i in range(1, 7):
        if mk == "11":
            j = 2 * i
        else:
            j = i
        df = load_history_file(g_0_folder + "history_" + str(j).zfill(3) + "_.csv")
        t0 = get_infinum_subsequence(df["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))
        t1 = get_infinum_subsequence(df["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))
        t2 = get_infinum_subsequence(df["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))
        t_arr = np.concatenate((t0, t1, t2), axis=1)
        g0_runs.append(t_arr)

        df = load_history_file(g_1_folder + "history_" + str(j).zfill(3) + "_.csv")
        t0 = get_infinum_subsequence(df["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))
        t1 = get_infinum_subsequence(df["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))
        t2 = get_infinum_subsequence(df["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))
        t_arr = np.concatenate((t0, t1, t2), axis=1)
        g1_runs.append(t_arr)

        df = load_history_file(g_2_folder + "history_" + str(j).zfill(3) + "_.csv")
        t0 = get_infinum_subsequence(df["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))
        t1 = get_infinum_subsequence(df["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))
        t2 = get_infinum_subsequence(df["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))
        t_arr = np.concatenate((t0, t1, t2), axis=1)
        g2_runs.append(t_arr)

        df = load_history_file(g_3_folder + "history_" + str(j).zfill(3) + "_.csv")
        t0 = get_infinum_subsequence(df["val_output_1_loss"].to_numpy().reshape(n_epochs, 1))
        t1 = get_infinum_subsequence(df["val_output_2_loss"].to_numpy().reshape(n_epochs, 1))
        t2 = get_infinum_subsequence(df["val_output_3_loss"].to_numpy().reshape(n_epochs, 1))
        t_arr = np.concatenate((t0, t1, t2), axis=1)
        g3_runs.append(t_arr)

    # Compute mean of the losses
    g0_mean_runs = np.zeros((n_epochs, 3))
    g1_mean_runs = np.zeros((n_epochs, 3))
    g2_mean_runs = np.zeros((n_epochs, 3))
    g3_mean_runs = np.zeros((n_epochs, 3))

    for i in range(0, 6):
        g0_mean_runs += g0_runs[i]
        g1_mean_runs += g1_runs[i]
        g2_mean_runs += g2_runs[i]
        g3_mean_runs += g3_runs[i]

    g0_mean_runs /= 6
    g1_mean_runs /= 6
    g2_mean_runs /= 6
    g3_mean_runs /= 6

    # compute variance of the losses
    g0_var_runs = np.zeros((n_epochs, 3))
    g1_var_runs = np.zeros((n_epochs, 3))
    g2_var_runs = np.zeros((n_epochs, 3))
    g3_var_runs = np.zeros((n_epochs, 3))

    for i in range(0, 6):
        g0_var_runs += (g0_runs[i] - g0_mean_runs) ** 2
        g1_var_runs += (g1_runs[i] - g1_mean_runs) ** 2
        g2_var_runs += (g2_runs[i] - g2_mean_runs) ** 2
        g3_var_runs += (g3_runs[i] - g3_mean_runs) ** 2

    g0_var_runs /= 6
    g1_var_runs /= 6
    g2_var_runs /= 6
    g3_var_runs /= 6

    plot_1dv2([epochs],
              [g0_mean_runs[:, 0], g3_mean_runs[:, 0], g2_mean_runs[:, 0], g1_mean_runs[:, 0]],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk" + mk + "_m" + order + "_h_gammas", log=True,
              folder_name="paper_data/paper2/illustrations/training/stats_runs/",
              show_fig=False, xlabel="epochs", ylabel=r"${e}_{\hat{h}^\gamma}$", xlim=[0, 2000], ylim=[1e-6, 1e-2],
              legend_pos="upper right", font_size=20, xticks=[0, 500, 1000, 1500, 2000])

    plot_1dv2([epochs],
              [g0_mean_runs[:, 1], g3_mean_runs[:, 1], g2_mean_runs[:, 1], g1_mean_runs[:, 1]],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk" + mk + "_m" + order + "_alpha_gammas", log=True,
              folder_name="paper_data/paper2/illustrations/training/stats_runs/",
              show_fig=False,
              xlabel="epochs", ylabel=r"$e_{(\mathbf{\alpha}^\gamma_{\overline{\mathbf{u}}})_\#}$",
              xlim=[0, 2000],
              ylim=[1e-5, 1e-0],
              legend_pos="upper right", font_size=20, xticks=[0, 500, 1000, 1500, 2000])

    plot_1dv2([epochs],
              [g0_mean_runs[:, 2], g3_mean_runs[:, 2], g2_mean_runs[:, 2], g1_mean_runs[:, 2]],
              labels=[r"$\gamma=0$", r"$\gamma=0.001$", r"$\gamma=0.01$", r"$\gamma=0.1$"],
              name="loss_mk" + mk + "_m" + order + "_u_gammas", log=True,
              folder_name="paper_data/paper2/illustrations/training/stats_runs/",
              show_fig=False,
              xlabel="epochs", ylabel=r"$e_{\overline{\mathbf{u}}}$", xlim=[0, 2000], ylim=[1e-7, 1e-2],
              legend_pos="upper right", font_size=20, xticks=[0, 500, 1000, 1500, 2000])

    with open("paper_data/paper2/illustrations/training/stats_runs/loss_mk" + mk + "_m" + order + ".txt", "w") as f:
        f.write("gamma, h, alpha, u\n")
        f.write(
            "0," + str(g0_mean_runs[-1, 0]) + "," + str(g0_mean_runs[-1, 1]) + "," + str(g0_mean_runs[-1, 2]) + "\n")
        f.write(
            "0.001," + str(g3_mean_runs[-1, 0]) + "," + str(g3_mean_runs[-1, 1]) + "," + str(
                g3_mean_runs[-1, 2]) + "\n")
        f.write(
            "0.01," + str(g2_mean_runs[-1, 0]) + "," + str(g2_mean_runs[-1, 1]) + "," + str(g2_mean_runs[-1, 2]) + "\n")
        f.write(
            "0.1," + str(g1_mean_runs[-1, 0]) + "," + str(g1_mean_runs[-1, 1]) + "," + str(g1_mean_runs[-1, 2]) + "\n")
        f.write("gamma, h-std, alpha-std, u-std\n")
        f.write("0," + str(np.sqrt(g0_var_runs[-1, 0])) + "," + str(np.sqrt(g0_var_runs[-1, 1])) + "," + str(
            np.sqrt(g0_var_runs[-1, 2])) + "\n")
        f.write("0.001," + str(np.sqrt(g3_var_runs[-1, 0])) + "," + str(np.sqrt(g3_var_runs[-1, 1])) + "," + str(
            np.sqrt(g3_var_runs[-1, 2])) + "\n")
        f.write("0.01," + str(np.sqrt(g2_var_runs[-1, 0])) + "," + str(np.sqrt(g2_var_runs[-1, 1])) + "," + str(
            np.sqrt(g2_var_runs[-1, 2])) + "\n")
        f.write("0.1," + str(np.sqrt(g1_var_runs[-1, 0])) + "," + str(np.sqrt(g1_var_runs[-1, 1])) + "," + str(
            np.sqrt(g1_var_runs[-1, 2])) + "\n")
    return 0


def print_training_performance_stats():
    # --------------- M2 2D -----------------------
    # mk11 ---------------
    g0_folder = "paper_data/paper2/2D_M2/stats_runs/mk11_m2_2d_g0/historyLogs/"
    g1_folder = "paper_data/paper2/2D_M2/stats_runs/mk11_m2_2d_g1/historyLogs/"
    g2_folder = "paper_data/paper2/2D_M2/stats_runs/mk11_m2_2d_g2/historyLogs/"
    g3_folder = "paper_data/paper2/2D_M2/stats_runs/mk11_m2_2d_g3/historyLogs/"

    print_stats_run(g_0_folder=g0_folder, g_1_folder=g1_folder, g_2_folder=g2_folder, g_3_folder=g3_folder, mk="11",
                    order="2")

    # mk12 ---------------
    mk12_m2_2d_g0_folder = "paper_data/paper2/2D_M2/stats_runs/mk12_m2_2d_g0/historyLogs/"
    mk12_m2_2d_g1_folder = "paper_data/paper2/2D_M2/stats_runs/mk12_m2_2d_g1/historyLogs/"
    mk12_m2_2d_g2_folder = "paper_data/paper2/2D_M2/stats_runs/mk12_m2_2d_g2/historyLogs/"
    mk12_m2_2d_g3_folder = "paper_data/paper2/2D_M2/stats_runs/mk12_m2_2d_g3/historyLogs/"

    print_stats_run(g_0_folder=mk12_m2_2d_g0_folder, g_1_folder=mk12_m2_2d_g1_folder, g_2_folder=mk12_m2_2d_g2_folder,
                    g_3_folder=mk12_m2_2d_g3_folder, mk="12", order="2")

    # ---------------------M3 mk11 ---------------
    g0_folder = "paper_data/paper2/2D_M3/stats_runs/mk11_M3_2D_g0/historyLogs/"
    g1_folder = "paper_data/paper2/2D_M3/stats_runs/mk11_M3_2D_g1/historyLogs/"
    g2_folder = "paper_data/paper2/2D_M3/stats_runs/mk11_M3_2D_g2/historyLogs/"
    g3_folder = "paper_data/paper2/2D_M3/stats_runs/mk11_M3_2D_g3/historyLogs/"

    print_stats_run(g_0_folder=g0_folder, g_1_folder=g1_folder, g_2_folder=g2_folder, g_3_folder=g3_folder, mk="11",
                    order="3")

    # mk12
    g0_folder = "paper_data/paper2/2D_M3/stats_runs/mk12_M3_2D_g0/historyLogs/"
    g1_folder = "paper_data/paper2/2D_M3/stats_runs/mk12_M3_2D_g1/historyLogs/"
    g2_folder = "paper_data/paper2/2D_M3/stats_runs/mk12_M3_2D_g2/historyLogs/"
    g3_folder = "paper_data/paper2/2D_M3/stats_runs/mk12_M3_2D_g3/historyLogs/"

    print_stats_run(g_0_folder=g0_folder, g_1_folder=g1_folder, g_2_folder=g2_folder, g_3_folder=g3_folder, mk="12",
                    order="3")

    # ----------------M4 2D -------------------
    g0_folder = "paper_data/paper2/2D_M4/stats_runs/mk11_m4_2d_g0/historyLogs/"
    g1_folder = "paper_data/paper2/2D_M4/stats_runs/mk11_m4_2d_g1/historyLogs/"
    g2_folder = "paper_data/paper2/2D_M4/stats_runs/mk11_m4_2d_g2/historyLogs/"
    g3_folder = "paper_data/paper2/2D_M4/stats_runs/mk11_m4_2d_g3/historyLogs/"

    print_stats_run(g_0_folder=g0_folder, g_1_folder=g1_folder, g_2_folder=g2_folder, g_3_folder=g3_folder, mk="11",
                    order="4")

    # mk12
    g0_folder = "paper_data/paper2/2D_M4/stats_runs/mk12_m4_2d_g0/historyLogs/"
    g1_folder = "paper_data/paper2/2D_M4/stats_runs/mk12_m4_2d_g1/historyLogs/"
    g2_folder = "paper_data/paper2/2D_M4/stats_runs/mk12_m4_2d_g2/historyLogs/"
    g3_folder = "paper_data/paper2/2D_M4/stats_runs/mk12_m4_2d_g3/historyLogs/"

    print_stats_run(g_0_folder=g0_folder, g_1_folder=g1_folder, g_2_folder=g2_folder, g_3_folder=g3_folder, mk="12",
                    order="4")


def print_cross_sections():
    # names = ["linesource_M2_g0", "linesource_M2_g1", "linesource_M2_g2", "linesource_M2_g3", "linesource_M4_g1",
    #         "linesource_N2_g0", "linesource_N2_g1", "linesource_N2_g2", "linesource_N2_g3",
    #         "linesource_N2_g0_r", "linesource_N2_g1_r", "linesource_N2_g2_r", "linesource_N2_g3_r",
    #         "linesource_N2_g0_resnet", "linesource_N2_g1_resnet", "linesource_N2_g2_resnet", "linesource_N2_g3_resnet",
    #         "linesource_N2_g0_r_resnet", "linesource_N2_g1_r_resnet", "linesource_N2_g2_r_resnet",
    #         "linesource_N2_g3_r_resnet", "linesource_M4_g1",
    #         "linesource_M3_g0", "linesource_M3_g1", "linesource_M3_g3",
    #         "linesource_N3_g0", "linesource_N3_g1", "linesource_N3_g2", "linesource_N3_g3", ]

    names = ["linesource_N2_g0", "linesource_N2_g1", "linesource_N2_g2", "linesource_N2_g3",
             "linesource_N2_g0_r", "linesource_N2_g1_r", "linesource_N2_g2_r", "linesource_N2_g3_r",
             "linesource_N2_g0_resnet", "linesource_N2_g1_resnet", "linesource_N2_g2_resnet", "linesource_N2_g3_resnet",
             "linesource_N2_g0_r_resnet", "linesource_N2_g1_r_resnet", "linesource_N2_g2_r_resnet",
             "linesource_N2_g3_r_resnet"]

    newton_names = ["linesource_M2_g0", "linesource_M2_g1", "linesource_M2_g2", "linesource_M2_g3",
                    "linesource_M2_g0", "linesource_M2_g1", "linesource_M2_g2", "linesource_M2_g3",
                    "linesource_M2_g0", "linesource_M2_g1", "linesource_M2_g2", "linesource_M2_g3",
                    "linesource_M2_g0", "linesource_M2_g1", "linesource_M2_g2", "linesource_M2_g3", ]

    yticks = [[0, 2, 4, 6, 8], [0, 4, 8, 12, 16], [0, 4, 8, 12, 16], [0, 2, 4, 6, 8],
              [0, 2, 4, 6, 8], [0, 4, 8, 12, 16], [0, 4, 8, 12, 16], [0, 2, 4, 6, 8],
              [0, 2, 4, 6, 8], [0, 4, 8, 12, 16], [0, 4, 8, 12, 16], [0, 2, 4, 6, 8],
              [0, 2, 4, 6, 8], [0, 4, 8, 12, 16], [0, 4, 8, 12, 16], [0, 2, 4, 6, 8]]
    legend_posis = ["upper left", "upper left", "upper left", "upper left",
                    "upper left", "upper left", "upper left", "upper left",
                    "upper left", "upper left", "upper left", "upper left",
                    "upper left", "upper left", "upper left", "upper left"]
    y_lims = [[-0.4, 8], [-0.4, 20], [-0.4, 18], [-0.4, 9],
              [-0.4, 8], [-0.4, 20], [-0.4, 18], [-0.4, 9],
              [-0.4, 8], [-0.4, 20], [-0.4, 18], [-0.4, 9],
              [-0.4, 8], [-0.4, 20], [-0.4, 18], [-0.4, 9], ]
    for (name, newton_name, tick, pos, y_lim) in zip(names, newton_names, yticks, legend_posis, y_lims):
        # if name != "linesource_N2_g1_r":
        #    newton_name += "_vert"
        print_single_xs(name, newton_name, ticks=tick, legend_pos=pos, y_lim=y_lim)
    return 0


def print_single_xs(name, newton_name, ticks, legend_pos, y_lim):
    load_name = "paper_data/paper2/linesource/cross_sections/"
    save_name = "paper_data/paper2/illustrations/cross_sections/"
    df_analytic = pd.read_csv(load_name + "linesource_analytic.csv")
    x_analytic = df_analytic["Points:0"].to_numpy()
    y_analytic = df_analytic["Points:1"].to_numpy()
    xy = np.vstack([x_analytic, y_analytic]).T
    radius_analytic = np.linalg.norm(xy, axis=1)
    radius_analytic[:int(len(radius_analytic) / 2)] = -radius_analytic[:int(len(radius_analytic) / 2)]

    df_newton_vert = pd.read_csv(load_name + newton_name + "_vert.csv")
    x_analytic = df_newton_vert["Points:0"].to_numpy()
    y_analytic = df_newton_vert["Points:1"].to_numpy()
    xy = np.vstack([x_analytic, y_analytic]).T
    radius_vert = np.linalg.norm(xy, axis=1)
    radius_vert[:int(len(radius_vert) / 2)] = -radius_vert[:int(len(radius_vert) / 2)]

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
    plt.figure(figsize=(5.8, 4.7), dpi=400)

    sns.set_theme()
    sns.set_style("ticks")
    colors = ["", 'k-', 'r--', 'g-.']
    symbol_size = 2
    marker_size = 6
    marker_width = 0.5
    data_jump = 1  # 18
    font_size = 28

    # line1 = plt.plot(radius_analytic[::data_jump], df_analytic["analytic radiation flux density"][::data_jump] / 2,
    #                 colors[0],
    #                 linewidth=symbol_size,
    #                 markersize=marker_size,
    #                 markeredgewidth=marker_width, markeredgecolor='k')
    line4 = plt.plot(radius_vert[::data_jump], df_newton_vert["radiation flux density"][::data_jump], colors[1],
                     linewidth=symbol_size,
                     markersize=marker_size,
                     markeredgewidth=marker_width, markeredgecolor='k')
    line2 = plt.plot(radius_vert[::data_jump], df_name_vert["radiation flux density"][::data_jump], colors[2],
                     linewidth=symbol_size,
                     markersize=marker_size,
                     markeredgewidth=marker_width, markeredgecolor='k')
    line3 = plt.plot(radius_45[::data_jump], df_name_45["radiation flux density"][::data_jump], colors[3],
                     linewidth=symbol_size, markersize=marker_size,
                     markeredgewidth=marker_width, markeredgecolor='k')

    plt.xlim([-1, 1])
    plt.ylim(y_lim)
    plt.legend([line4[0], line2[0], line3[0]], ["Newton", "vertical", "diagonal"], loc=legend_pos,
               fontsize=int(0.6 * font_size))
    plt.xlabel(r"$x$", fontsize=font_size)
    plt.ylabel(r"$u_0$", fontsize=font_size)
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], fontsize=int(0.7 * font_size))
    plt.yticks(ticks, fontsize=int(0.7 * font_size))
    plt.tight_layout()
    plt.savefig(save_name + "xs_" + name + ".pdf", dpi=500)
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
    sns.set_style("ticks")
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
    plt.savefig("paper_data/paper2/illustrations/hohlraum/methods.pdf", dpi=500)

    return 0


def print_method_errors():
    df = pd.read_csv("paper_data/paper2/hohlraum/errors/method_errors_relative.csv", delimiter=";")

    names = [r"M$_2^{\gamma_3}$",
             r"M$_3^{\gamma_3}$",
             r"NM$_2$",
             r"NM$_2^{\gamma_1}$",
             r"NM$_2^{\gamma_2}$",
             r"NM$_2^{\gamma_3}$",
             r"NM$_2^{r}$",
             r"NM$_2^{\gamma_1,r}$",
             r"NM$_2^{\gamma_2,r}$",
             r"NM$_2^{\gamma_3,r}$",
             r"NM$_3$",
             r"NM$_3^{\gamma_1}$",
             r"NM$_3^{\gamma_2}$",
             r"NM$_3$",  # ^{\gamma_3}$",
             r"NM$_4$",
             r"NM$_4^{\gamma_1}$",
             r"NM$_4$",  # ^{\gamma_2}$",
             r"NM$_4^{\gamma_3}$",
             r"S$_{10}$",
             r"S$_{20}$",
             r"S$_{30}$",
             r"S$_{40}$",
             r"S$_{50}$",
             r"P$_{2}$",
             r"P$_{3}$",
             r"P$_{5}$",
             r"P$_{7}$",
             r"P$_{9}$"]
    errors = df["rel_spatial_err"].to_numpy(dtype=float)
    sys_size = df["sys_size"].to_numpy()
    timing = df["timing"].to_numpy()

    indices_ref = [0, 1, 2, 13, 16, 18, 19, 20, 21, 23, 24, 25, 26, 27]

    # a) sysSize over errror
    font_size = 26
    plt.clf()
    sns.set_theme()
    sns.set_style("ticks")
    texts = []
    for i in indices_ref:
        if i in [2, 13, 16]:
            label1 = plt.scatter(sys_size[i], errors[i], s=40, facecolors='green', edgecolors='green')
        else:
            label2 = plt.scatter(sys_size[i], errors[i], s=40, facecolors='red', edgecolors='red')
        texts.append(plt.text(sys_size[i], errors[i], names[i], fontsize=int(font_size * 0.7)))

    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=int(font_size * 0.7))
    plt.yticks(fontsize=int(font_size * 0.7))
    plt.xlabel("system size", fontsize=font_size)
    plt.ylabel(r'$e_{\rm{rel},u_0}$', fontsize=font_size)
    plt.legend([label1, label2], ["ours", "reference"], loc="lower left", fontsize=int(font_size * 0.7))

    adjust_text(texts, only_move={'texts': 'y'})
    plt.savefig("paper_data/paper2/illustrations/hohlraum/sys_size_vs_error.pdf", dpi=500, bbox_inches="tight")
    plt.clf()

    # b) timing over errror
    sns.set_theme()
    sns.set_style("ticks")
    for i in indices_ref:
        if i in [2, 13, 16]:
            label1 = plt.scatter(timing[i], errors[i], s=40, facecolors='green', edgecolors='green')
        else:
            label2 = plt.scatter(timing[i], errors[i], s=40, facecolors='red', edgecolors='red')
        texts.append(plt.text(timing[i], errors[i], names[i], fontsize=int(font_size * 0.7)))

    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=int(font_size * 0.7))
    plt.yticks(fontsize=int(font_size * 0.7))
    plt.xlabel("time [s]", fontsize=font_size)
    plt.ylabel(r'$e_{\rm{rel},u_0}$', fontsize=font_size)
    plt.legend([label1, label2], ["ours", "reference"], loc="lower left", fontsize=int(font_size * 0.7))
    adjust_text(texts, only_move={'texts': 'y'})
    plt.savefig("paper_data/paper2/illustrations/hohlraum/timing_vs_error.pdf", dpi=500, bbox_inches="tight")
    plt.clf()

    # c) sys size vs timing
    plt.clf()
    sns.set_theme()
    sns.set_style("ticks")

    for i in indices_ref:
        if i in [2, 13, 16]:
            label1 = plt.scatter(sys_size[i], timing[i], s=40, facecolors='green', edgecolors='green')
        else:
            label2 = plt.scatter(sys_size[i], timing[i], s=40, facecolors='red', edgecolors='red')
        texts.append(plt.text(sys_size[i], timing[i], names[i], fontsize=int(font_size * 0.7)))

    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=int(font_size * 0.7))
    plt.yticks(fontsize=int(font_size * 0.7))
    plt.xlabel("system size", fontsize=font_size)
    plt.ylabel("time [s]", fontsize=font_size)
    plt.legend([label1, label2], ["ours", "reference"], fontsize=int(font_size * 0.7))

    adjust_text(texts, only_move={'texts': 'y'})

    plt.savefig("paper_data/paper2/illustrations/hohlraum/sys_size_vs_timing.pdf", dpi=500, bbox_inches="tight")
    plt.clf()

    # d) plot only neural network errors
    sns.set_theme()
    sns.set_style("ticks")
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
    ax.set_ylabel(r'$|NM_N -S_{50}|$')
    ax.set_xticks(ind + width * 2)
    ax.set_xticklabels((r'$NM_2$', r'$NM_2$, rot', r'$NM_3$', r'$NM_4$'))
    ax.legend((rects1[0], rects4[0], rects3[0], rects2[0]),
              (r'$\gamma=0$', r'$\gamma=0.001$', r'$\gamma=0.01$', r'$\gamma=0.1$'))

    plt.savefig("paper_data/paper2/illustrations/hohlraum/neural_error_bars.pdf", dpi=500, bbox_inches="tight")
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


if __name__ == '__main__':
    main()
