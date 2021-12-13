"""
Script to call different plots and illustrative methods
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import pandas as pd
from src.utils import load_density_function, load_solution, plot_1d, load_data, scatter_plot_2d_N2, scatter_plot_2d


def main():
    print("---------- Start Result Illustration Suite ------------")

    # --- inflow M2 1D --- illustration
    [u_neural15, u_ref15] = load_solution("paper_data/1D_M2/1D_M2_MK15_inflow.csv")
    [u_neural11, u_ref11] = load_solution("paper_data/1D_M2/1D_M2_MK11_inflow.csv")
    x = np.linspace(0, 1, 100)

    res_list = [u_ref11, u_neural11, u_neural15]
    plot_1d([x], res_list,
            labels=[r"$u_0$", r"$u_1$", r"$u_2$", r"Convex - $u_{0,\theta}$", r"Convex - $u_{1,\theta}$",
                    r"Convex - $u_{2,\theta}$", r"Mono - $u_{0,\theta}$", r"Mono - $u_{1,\theta}$",
                    r"Mono - $u_{2,\theta}$"],
            name="inflow_1D_M2",
            folder_name="paper_data/1D_M2", linetypes=['-', 'o', '^'], xlim=[0, 1],
            xlabel='x', ylabel='u', log=False, title=r"$u$ and $u_\theta$  over $x$")

    err_mk11 = np.linalg.norm(u_ref11 - u_neural11, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk11 = err_mk11 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))
    err_mk15 = np.linalg.norm(u_ref15 - u_neural15, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk15 = err_mk15 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))

    err_res_list = [err_mk11, err_mk15]
    plot_1d([x], err_res_list, labels=["Convex", "Mono"], name="err_inflow_1D_M2", folder_name="paper_data/1D_M2",
            linetypes=['o', '^'], xlim=[0, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2$", log=True,
            title=r"$||u-u_\theta||_2$ over $x$")
    rel_err_res_list = [rel_errmk11, rel_errmk15]
    plot_1d([x], rel_err_res_list, labels=["Convex", "Mono"], name="rel_err_inflow_1D_M2",
            folder_name="paper_data/1D_M2",
            linetypes=['o', '^'], xlim=[0, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $x$")

    # --- inflow M2 1D Data Dynamics
    """
    [x, _, kinetic_f] = load_density_function("test_a10_ev5.csv")

    for i in range(kinetic_f.shape[0]):
        plt.plot(x[0], kinetic_f[i, :])

    plt.ylim(0, 3)
    plt.savefig("test_a10_ev5")
    # for i in range(int(kinetic_f.shape[0] / 5)):
    #    kinetic_list = [kinetic_f[i + 0], kinetic_f[i + 1], kinetic_f[i + 2], kinetic_f[i + 3], kinetic_f[i + 4]]
    #    plot_1d(x, kinetic_list, show_fig=False, log=False, name='kinetics_kond3_' + str(i).zfill(3), ylim=[0, 3],
    #            xlim=[x[0, 0], x[0, -1]])
    """

    # --- inflow M1 1D --- illustration
    [u_neural15, u_ref15] = load_solution("paper_data/1D_M1/1D_M1_MK15_inflow.csv")
    [u_neural11, u_ref11] = load_solution("paper_data/1D_M1/1D_M1_MK11_inflow.csv")
    x = np.linspace(0, 1, 100)

    res_list = [u_ref11, u_neural11, u_neural15]
    plot_1d([x], res_list,
            labels=[r"$u_0$", r"$u_1$", r"Convex - $u_{0,\theta}$", r"Convex - $u_{1,\theta}$",
                    r"Mono - $u_{0,\theta}$",
                    r"Mono - $u_{1,\theta}$"], name="inflow_1D_M1", folder_name="paper_data/1D_M1",
            linetypes=['-', 'o', '^'], xlim=[0, 1], xlabel='x', ylabel='u', log=False,
            title=r"$u$ and $u_\theta$  over $x$")

    err_mk11 = np.linalg.norm(u_ref11 - u_neural11, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk11 = err_mk11 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))
    err_mk15 = np.linalg.norm(u_ref15 - u_neural15, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk15 = err_mk15 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))

    err_res_list = [err_mk11, err_mk15]
    plot_1d([x], err_res_list, labels=["Convex", "Mono"], name="err_inflow_1D_M1", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[0, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2$", log=True,
            title=r"$||u-u_\theta||_2$ over $x$")
    rel_err_res_list = [rel_errmk11, rel_errmk15]
    plot_1d([x], rel_err_res_list, labels=["Convex", "Mono"], name="rel_err_inflow_1D_M1",
            folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[0, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $x$")

    # --- synthetic test M1 mk11
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK11_synthetic.csv")
    data_mk11 = df.to_numpy()
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK15_synthetic.csv")
    data_mk15 = df.to_numpy()
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 2].reshape((data_mk11.shape[0], 1)), data_mk15[:, 2].reshape((data_mk15.shape[0], 1))],
            labels=["Convex", "Mono"], name="rel_err_u_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1e-1], xlabel=r'$u^n_1$',
            ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $u^n_1$")
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 4].reshape((data_mk11.shape[0], 1)), data_mk15[:, 4].reshape((data_mk15.shape[0], 1))],
            labels=["Convex", "Mono"], name="rel_err_alpha_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1], xlabel=r'$u^n_1$',
            ylabel=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$",
            log=True, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $u^n_1$")
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 6].reshape((data_mk11.shape[0], 1)), data_mk15[:, 6].reshape((data_mk15.shape[0], 1))],
            labels=["Convex", "Mono"], name="rel_err_h_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1e-1], xlabel=r'$u_1$',
            ylabel=r"$||h-h_\theta||_2/||h||_2$",
            log=True, title=r"$||h-h_\theta||_2/||h||_2$ over $u^n_1$")

    df = pd.read_csv("paper_data/1D_M1/1D_M1_normal_vs_alpha_synthetic.csv")
    data_sampling_compare = df.to_numpy()
    plot_1d([data_sampling_compare[:, 0]],
            [data_sampling_compare[:, 1].reshape((data_sampling_compare.shape[0], 1)),
             data_sampling_compare[:, 4].reshape((data_sampling_compare.shape[0], 1))],
            labels=[r"uniform $u^n$", r"uniform $\alpha_u^n$"], name="rel_err_u_compare_sampling_synthetic",
            folder_name="paper_data/1D_M1", linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1e-1], xlabel=r'$u^n_1$',
            ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $u^n_1$")
    plot_1d([data_sampling_compare[:, 0]],
            [data_sampling_compare[:, 2].reshape((data_sampling_compare.shape[0], 1)),
             data_sampling_compare[:, 5].reshape((data_sampling_compare.shape[0], 1))],
            labels=[r"uniform $u^n$", r"uniform $\alpha_u^n$"], name="rel_err_alpha_compare_sampling_synthetic",
            folder_name="paper_data/1D_M1", linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1], xlabel=r'$u^n_1$',
            ylabel=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$",
            log=True, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $u^n_1$")
    plot_1d([data_sampling_compare[:, 0]],
            [data_sampling_compare[:, 3].reshape((data_sampling_compare.shape[0], 1)),
             data_sampling_compare[:, 6].reshape((data_sampling_compare.shape[0], 1))],
            labels=[r"uniform $u^n$", r"uniform $\alpha_u^n$"], name="rel_err_h_compare_sampling_synthetic",
            folder_name="paper_data/1D_M1", linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1e-1], xlabel=r'$u^n_1$',
            ylabel=r"$||h-h_\theta||_2/||h||_2$",
            log=True, title=r"$||h-h_\theta||_2/||h||_2$ over $u^n_1$")

    # --- perodic M1 2D test case.
    df = pd.read_csv("paper_data/2D_M1/err_2D_M1_MK11_periodic.csv")
    data_mk11 = df.to_numpy()
    df = pd.read_csv("paper_data/2D_M1/err_2D_M1_MK15_periodic.csv")
    data_mk15 = df.to_numpy()
    n = 20
    time = data_mk11[::n, 0]
    err_u_mk11 = data_mk11[::n, 1]
    err_alpha_mk11 = data_mk11[::n, 2]
    h_mk11 = data_mk11[::n, 3]
    h_ref = data_mk11[::n, 4]
    time2 = data_mk15[::n, 0]
    err_u_mk15 = data_mk15[::n, 1]
    err_alpha_mk15 = data_mk15[::n, 2]
    h_mk15 = data_mk15[::n, 3]
    h_ref2 = data_mk15[::n, 4]
    plot_1d([time], [err_u_mk11.reshape((err_u_mk11.shape[0], 1)), err_u_mk15.reshape((err_u_mk11.shape[0], 1))],
            labels=["Convex", "Mono"], name="rel_err_u_2D_M1_over_time", folder_name="paper_data/2D_M1",
            linetypes=['o', '^'], xlim=[0, time[-1]], ylim=[1e-6, 1e-1], xlabel=r'$t$',
            ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $t$")
    plot_1d([time],
            [err_alpha_mk11.reshape((err_u_mk11.shape[0], 1)), err_alpha_mk15.reshape((err_u_mk11.shape[0], 1))],
            labels=["Convex", "Mono"], name="rel_err_alpha_2D_M1_over_time", folder_name="paper_data/2D_M1",
            linetypes=['o', '^'], xlim=[0, time[-1]], ylim=[1e-3, 1e-1], xlabel=r'$t$',
            ylabel=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$",
            log=True, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $t$")
    plot_1d([time], [h_ref.reshape((err_u_mk11.shape[0], 1)), h_mk11.reshape((err_u_mk11.shape[0], 1)),
                     h_mk15.reshape((err_u_mk11.shape[0], 1))], labels=["Newton", "Convex", "Mono"],
            name="entropy_2D_M1_over_time", folder_name="paper_data/2D_M1",
            linetypes=['-', 'o', '^'], xlim=[0, time[-1]], xlabel=r'$t$',
            ylabel=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$",
            log=False, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $t$")

    # --- Realizable set illustrations ---
    [u, alpha, h] = load_data(filename="paper_data/1D_M2/Monomial_M2_1D_normal.csv", input_dim=3,
                              selected_cols=[True, True, True])
    max_h = 3
    min_h = np.min(h)
    alpha_bound = 40
    scatter_plot_2d_N2(x_in=u[:, 1:], z_in=h, lim_x=(-1, 1), lim_y=(0, 1), lim_z=(min_h, max_h),
                       title=r"$h$ over $\mathcal{R}^r$",
                       folder_name="paper_data/1D_M2", name="normal_u_Monomial_M2_1D_u", show_fig=False, log=False,
                       color_map=0)
    scatter_plot_2d(x_in=alpha[:, 1:], z_in=h, lim_x=(-alpha_bound, alpha_bound), lim_y=(-alpha_bound, alpha_bound),
                    lim_z=(min_h, max_h),
                    title=r"$h$ over $\alpha^r$", label_x=r"$\alpha_1^r$", label_y=r"$\alpha_2^r$",
                    folder_name="paper_data/1D_M2", name="normal_u_Monomial_M2_1D_alpha", show_fig=False, log=False,
                    color_map=0)

    [u, alpha, h] = load_data(filename="paper_data/1D_M2/Monomial_M2_1D_normal_alpha_grid.csv", input_dim=3,
                              selected_cols=[True, True, True])
    # max_h = np.max(h)
    # min_h = np.min(h)
    scatter_plot_2d_N2(x_in=u[:, 1:], z_in=h, lim_x=(-1, 1), lim_y=(0, 1), lim_z=(min_h, max_h),
                       title=r"$h$ over $\mathcal{R}^r$",
                       folder_name="paper_data/1D_M2", name="normal_alpha_grid_Monomial_M2_1D_u", show_fig=False,
                       log=False,
                       color_map=0)
    scatter_plot_2d(x_in=alpha[:, 1:], z_in=h, lim_x=(-alpha_bound, alpha_bound), lim_y=(-alpha_bound, alpha_bound),
                    lim_z=(min_h, max_h),
                    title=r"$h$ over $\alpha^r$", label_x=r"$\alpha_1^r$", label_y=r"$\alpha_2^r$",
                    folder_name="paper_data/1D_M2", name="normal_alpha_grid_Monomial_M2_1D_alpha", show_fig=False,
                    log=False,
                    color_map=0)

    [u, alpha, h] = load_data(filename="paper_data/1D_M2/Monomial_M2_1D_normal_gaussian.csv", input_dim=3,
                              selected_cols=[True, True, True])
    # max_h = np.max(h)
    # min_h = np.min(h)
    scatter_plot_2d_N2(x_in=u[:, 1:], z_in=h, lim_x=(-1, 1), lim_y=(0, 1), lim_z=(min_h, max_h),
                       title=r"$h$ over $\mathcal{R}^r$",
                       folder_name="paper_data/1D_M2", name="alpha_gauss_Monomial_M2_1D_normal_u", show_fig=False,
                       log=False,
                       color_map=0)
    scatter_plot_2d(x_in=alpha[:, 1:], z_in=h, lim_x=(-alpha_bound, alpha_bound), lim_y=(-alpha_bound, alpha_bound),
                    lim_z=(min_h, max_h),
                    title=r"$h$ over $\alpha^r$", label_x=r"$\alpha_1^r$", label_y=r"$\alpha_2^r$",
                    folder_name="paper_data/1D_M2", name="alpha_gauss_Monomial_M2_1D_normal_alpha", show_fig=False,
                    log=False,
                    color_map=0)

    # --- print template for periodic 2D results
    scatter_plot_2d_N2(x_in=u[:, 1:], z_in=h, lim_x=(-1.5, 1.5), lim_y=(-1.5, 1.5), lim_z=(0.5, 2.5),
                       title=r"$u_0$ over $(x,y)$", label_x=r"$x$", label_y=r"$y$",
                       folder_name="paper_data/2D_M1", name="periodic_u0_60", show_fig=False,
                       log=False,
                       color_map=0)
    scatter_plot_2d_N2(x_in=u[:, 1:], z_in=h, lim_x=(-1.5, 1.5), lim_y=(-1.5, 1.5), lim_z=(1e-4, 1),
                       title=r"$||u_\theta-u||_2/||u||_2$ over $(x,y)$", label_x=r"$x$", label_y=r"$y$",
                       folder_name="paper_data/2D_M1", name="periodic_u0_60_err", show_fig=False,
                       log=True,
                       color_map=0)
    scatter_plot_2d(x_in=alpha[:, 1:], z_in=h, lim_x=(-alpha_bound, alpha_bound), lim_y=(-alpha_bound, alpha_bound),
                    lim_z=(min_h, max_h),
                    title=r"$h$ over $\alpha^r$", label_x=r"$\alpha_1^r$", label_y=r"$\alpha_2^r$",
                    folder_name="paper_data/1D_M2", name="alpha_gauss_Monomial_M2_1D_normal_alpha", show_fig=False,
                    log=False,
                    color_map=0)

    # --- illustrate training history for alpha sampling vs u sampling --- 

    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK11_history_normal.csv")
    data_normal = df.to_numpy()
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK11_history_normal_alpha.csv")
    data_alpha = df.to_numpy()
    df = pd.read_csv("paper_data/1D_M1/Monomial_M1_1D_normal.csv")
    df = df.iloc[:, 1:]  # drop timestamp
    data_train_u = df.to_numpy()
    df = pd.read_csv("paper_data/1D_M1/Monomial_M1_1D_normal_alpha.csv")
    df = df.iloc[:, 1:]  # drop timestamp
    data_train_alpha = df.to_numpy()

    # compute average value of training data for each sampling method
    u_sample_u = data_train_u[:, 1]  # u_1S
    alpha_sample_u = data_train_u[:, 3]  # u_1S
    h_sample_u = data_train_u[:, 4]
    # shift according to data scaling
    h_min_alpha = min(h_sample_u)
    h_max_alpha = max(h_sample_u)
    h_sample_alpha = (h_sample_u - h_min_alpha)  # / (h_max_alpha - h_min_alpha)
    alpha_sample_u = alpha_sample_u  # / (h_max_alpha - h_min_alpha)
    u_sample_alpha = data_train_alpha[:, 1]  # u_1S
    alpha_sample_alpha = data_train_alpha[:, 3]  # u_1S
    h_sample_alpha = data_train_alpha[:, 4]
    h_min_alpha = min(h_sample_alpha)
    h_max_alpha = max(h_sample_alpha)
    h_sample_alpha = (h_sample_alpha - h_min_alpha)  # / (h_max_alpha - h_min_alpha)
    alpha_sample_alpha = alpha_sample_alpha  # / (h_max_alpha - h_min_alpha)

    u_sample_u_denominator = np.mean(np.square(u_sample_u))
    u_sample_alpha_denominator = np.mean(np.square(u_sample_alpha))
    alpha_sample_u_denominator = np.mean(np.square(alpha_sample_u))
    alpha_sample_alpha_denominator = np.mean(np.square(alpha_sample_alpha))
    h_sample_u_denominator = np.mean(np.square(h_sample_u))
    h_sample_alpha_denominator = np.mean(np.square(h_sample_alpha))

    epoch = data_normal[:, 0]
    normal_loss_h = data_normal[:, 2].reshape(epoch.shape[0], 1) / h_sample_u_denominator
    normal_loss_alpha = data_normal[:, 5].reshape(epoch.shape[0], 1) / alpha_sample_u_denominator
    normal_loss_alpha2 = data_normal[:, 5].reshape(epoch.shape[0], 1)
    normal_loss_u = data_normal[:, 8].reshape(epoch.shape[0], 1) / u_sample_u_denominator
    alpha_loss_h = data_alpha[:, 2].reshape(epoch.shape[0], 1) / h_sample_alpha_denominator
    alpha_loss_alpha = data_alpha[:, 5].reshape(epoch.shape[0], 1) / alpha_sample_alpha_denominator
    alpha_loss_u = data_alpha[:, 8].reshape(epoch.shape[0], 1) / u_sample_alpha_denominator

    plot_1d([epoch.reshape(epoch.shape[0], 1)],
            [np.concatenate([normal_loss_h, alpha_loss_h], axis=1)],
            labels=[r"$\frac{||h_\theta-h||_2^2}{||h||_2^2}$, uniform sampling u",
                    r"$\frac{||h_\theta-h||_2^2}{||h||_2^2}$, uniform sampling $\alpha^r_u$"],
            name="training_loss_h_over_epochs", folder_name="paper_data/1D_M1",
            linetypes=['-'], xlim=[0, epoch[-1]], ylim=[1e-7, 1e-1], xlabel='epoch',
            ylabel=r"loss",
            log=True, title=r"relative training loss $h$ over epochs")

    plot_1d([epoch.reshape(epoch.shape[0], 1)],
            [np.concatenate([normal_loss_alpha, alpha_loss_alpha], axis=1)],
            labels=[r"$\frac{||\alpha^r_\theta-\alpha^r_u||_2^2}{||\alpha^r_u||_2^2}$, uniform sampling u",
                    r"$\frac{||\alpha^r_\theta-\alpha^r_u||_2^2}{||\alpha^r_u||_2^2}$, uniform sampling $\alpha^r_u$"],
            name="training_loss_alpha_over_epochs", folder_name="paper_data/1D_M1",
            linetypes=['-', ], xlim=[0, epoch[-1]], ylim=[1e-5, 1e0], xlabel='epoch',
            ylabel=r"loss",
            log=True, title=r"relative training loss $\alpha_u^r$ over epochs")

    plot_1d([epoch.reshape(epoch.shape[0], 1)],
            [np.concatenate([normal_loss_u, alpha_loss_u], axis=1)],
            labels=[r"$\frac{||u^n_\theta-u^n||_2^2}{||u^n||_2^2}$, uniform sampling u",
                    r"$\frac{||u^n_\theta-u^n||_2^2}{||u^n||_2^2}$, uniform sampling $\alpha^r_u$"],
            name="training_loss_u_over_epochs", folder_name="paper_data/1D_M1",
            linetypes=['-'], xlim=[0, epoch[-1]], ylim=[1e-5, 1e-1], xlabel='epoch',
            ylabel=r"loss",
            log=True, title=r"relative training loss $u$ over epochs")

    # Plot banach fixed point for 1D
    df = pd.read_csv("paper_data/banach/solution_10.csv")
    data_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_20.csv")
    data_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_40.csv")
    data_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_80.csv")
    data_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_160.csv")
    data_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_320.csv")
    data_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_640.csv")
    data_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_1280.csv")
    data_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_2560.csv")
    data_2560 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_5120.csv")
    data_5120 = df.to_numpy()

    df = pd.read_csv("paper_data/banach/solution_ml_mk11_10.csv")
    data_ml_mk11_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_20.csv")
    data_ml_mk11_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_40.csv")
    data_ml_mk11_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_80.csv")
    data_ml_mk11_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_160.csv")
    data_ml_mk11_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_320.csv")
    data_ml_mk11_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_640.csv")
    data_ml_mk11_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_1280.csv")
    data_ml_mk11_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/solution_ml_mk11_2560.csv")
    data_ml_mk11_2560 = df.to_numpy()

    # ---- compute errors ----
    # newton
    # 1. iter
    t1 = data_20[0::2]
    t2 = data_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_10[:, 1:], axis=1)
    theta_mean_1 = np.mean(theta_1)
    # 2. iter
    t1 = data_40[0::2]
    t2 = data_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_20[:, 1:], axis=1)
    theta_mean_2 = np.mean(theta_2)
    # 3. iter
    t1 = data_80[0::2]
    t2 = data_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_40[:, 1:], axis=1)
    theta_mean_3 = np.mean(theta_3)
    # 4. iter
    t1 = data_160[0::2]
    t2 = data_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_80[:, 1:], axis=1)
    theta_mean_4 = np.mean(theta_4)
    # 5. iter
    t1 = data_320[0::2]
    t2 = data_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_160[:, 1:], axis=1)
    theta_mean_5 = np.mean(theta_5)
    # 6. iter
    t1 = data_640[0::2]
    t2 = data_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_320[:, 1:], axis=1)
    theta_mean_6 = np.mean(theta_6)
    # 7. iter
    t1 = data_1280[0::2]
    t2 = data_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_640[:, 1:], axis=1)
    theta_mean_7 = np.mean(theta_7)
    # 8. iter
    t1 = data_2560[0::2]
    t2 = data_5120[0::4]
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_1280[:, 1:], axis=1)
    theta_mean_8 = np.mean(theta_8)

    # input convex mk11
    # 1. iter
    t1 = data_ml_mk11_20[0::2]
    t2 = data_ml_mk11_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    theta_mean_1 = np.mean(theta_1)
    # 2. iter
    t1 = data_ml_mk11_40[0::2]
    t2 = data_ml_mk11_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    theta_mean_2 = np.mean(theta_2)
    # 3. iter
    t1 = data_ml_mk11_80[0::2]
    t2 = data_ml_mk11_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    theta_mean_3 = np.mean(theta_3)
    # 4. iter
    t1 = data_ml_mk11_160[0::2]
    t2 = data_ml_mk11_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    theta_mean_4 = np.mean(theta_4)
    # 5. iter
    t1 = data_ml_mk11_320[0::2]
    t2 = data_ml_mk11_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_160[:, 1:],
                                                                             axis=1)
    theta_mean_5 = np.mean(theta_5)
    # 6. iter
    t1 = data_ml_mk11_640[0::2]
    t2 = data_ml_mk11_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_320[:, 1:],
                                                                             axis=1)
    theta_mean_6 = np.mean(theta_6)
    # 7. iter
    t1 = data_ml_mk11_1280[0::2]
    t2 = data_ml_mk11_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_640[:, 1:],
                                                                             axis=1)
    theta_mean_7 = np.mean(theta_7)
    # 8. iter
    t1 = data_ml_mk11_2560[0::2]
    t2 = data_5120[0::4]  # use original data, since ml data is not yet finished
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_1280[:, 1:],
                                                                             axis=1)
    theta_mean_8 = np.mean(theta_8)

    # Compute the mean over theta_mean to approximate the real theta
    theta_approx = np.mean(
        np.asarray([theta_mean_1, theta_mean_2, theta_mean_3, theta_mean_4, theta_mean_5, theta_mean_6, theta_mean_7,
                    theta_mean_8]))
    # Compute the spatial discretization error approximation:
    # newton
    lvl_1_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_20[::2, 1:] - data_10[:, 1:], axis=1)
    lvl_2_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_40[::2, 1:] - data_20[:, 1:], axis=1)
    lvl_3_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_80[::2, 1:] - data_40[:, 1:], axis=1)
    lvl_4_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_160[::2, 1:] - data_80[:, 1:], axis=1)
    lvl_5_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_320[::2, 1:] - data_160[:, 1:], axis=1)
    lvl_6_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_640[::2, 1:] - data_320[:, 1:], axis=1)
    lvl_7_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_1280[::2, 1:] - data_640[:, 1:], axis=1)
    lvl_8_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_2560[::2, 1:] - data_1280[:, 1:], axis=1)
    lvl_9_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_5120[::2, 1:] - data_2560[:, 1:], axis=1)
    errors = np.asarray([np.mean(lvl_1_error), np.mean(lvl_2_error), np.mean(lvl_3_error), np.mean(lvl_4_error),
                         np.mean(lvl_5_error), np.mean(lvl_6_error), np.mean(lvl_7_error), np.mean(lvl_8_error),
                         np.mean(lvl_9_error)]).reshape((9, 1))
    lvl_1_error_direct = np.linalg.norm(data_5120[::512, 1:] - data_10[:, 1:], axis=1)
    lvl_2_error_direct = np.linalg.norm(data_5120[::256, 1:] - data_20[:, 1:], axis=1)
    lvl_3_error_direct = np.linalg.norm(data_5120[::128, 1:] - data_40[:, 1:], axis=1)
    lvl_4_error_direct = np.linalg.norm(data_5120[::64, 1:] - data_80[:, 1:], axis=1)
    lvl_5_error_direct = np.linalg.norm(data_5120[::32, 1:] - data_160[:, 1:], axis=1)
    lvl_6_error_direct = np.linalg.norm(data_5120[::16, 1:] - data_320[:, 1:], axis=1)
    lvl_7_error_direct = np.linalg.norm(data_5120[::8, 1:] - data_640[:, 1:], axis=1)
    lvl_8_error_direct = np.linalg.norm(data_5120[::4, 1:] - data_1280[:, 1:], axis=1)
    lvl_9_error_direct = np.linalg.norm(data_5120[::2, 1:] - data_2560[:, 1:], axis=1)
    errors_direct = np.asarray([np.mean(lvl_1_error_direct), np.mean(lvl_2_error_direct), np.mean(lvl_3_error_direct),
                                np.mean(lvl_4_error_direct), np.mean(lvl_5_error_direct), np.mean(lvl_6_error_direct),
                                np.mean(lvl_7_error_direct), np.mean(lvl_8_error_direct),
                                np.mean(lvl_9_error_direct)]).reshape((9, 1))
    # ml mk 11 (input convex)
    # newton
    lvl_1_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_20[::2, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    lvl_2_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_40[::2, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    lvl_3_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_80[::2, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    lvl_4_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_160[::2, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    lvl_5_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_320[::2, 1:] - data_ml_mk11_160[:, 1:], axis=1)
    lvl_6_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_640[::2, 1:] - data_ml_mk11_320[:, 1:], axis=1)
    lvl_7_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_1280[::2, 1:] - data_ml_mk11_640[:, 1:], axis=1)
    lvl_8_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_2560[::2, 1:] - data_ml_mk11_1280[:, 1:], axis=1)
    lvl_9_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(data_5120[::2, 1:] - data_ml_mk11_2560[:, 1:],
                                                                          axis=1)
    errors_mk11 = np.asarray(
        [np.mean(lvl_1_error_mk11), np.mean(lvl_2_error_mk11), np.mean(lvl_3_error_mk11), np.mean(lvl_4_error_mk11),
         np.mean(lvl_5_error_mk11), np.mean(lvl_6_error_mk11), np.mean(lvl_7_error_mk11), np.mean(lvl_8_error_mk11),
         np.mean(lvl_9_error_mk11)]).reshape((9, 1))
    lvl_1_error_direct_mk11 = np.linalg.norm(data_5120[::512, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    lvl_2_error_direct_mk11 = np.linalg.norm(data_5120[::256, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    lvl_3_error_direct_mk11 = np.linalg.norm(data_5120[::128, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    lvl_4_error_direct_mk11 = np.linalg.norm(data_5120[::64, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    lvl_5_error_direct_mk11 = np.linalg.norm(data_5120[::32, 1:] - data_ml_mk11_160[:, 1:], axis=1)
    lvl_6_error_direct_mk11 = np.linalg.norm(data_5120[::16, 1:] - data_ml_mk11_320[:, 1:], axis=1)
    lvl_7_error_direct_mk11 = np.linalg.norm(data_5120[::8, 1:] - data_ml_mk11_640[:, 1:], axis=1)
    lvl_8_error_direct_mk11 = np.linalg.norm(data_5120[::4, 1:] - data_ml_mk11_1280[:, 1:], axis=1)
    lvl_9_error_direct_mk11 = np.linalg.norm(data_5120[::2, 1:] - data_ml_mk11_2560[:, 1:], axis=1)
    errors_direct_mk11 = np.asarray(
        [np.mean(lvl_1_error_direct_mk11), np.mean(lvl_2_error_direct_mk11), np.mean(lvl_3_error_direct_mk11),
         np.mean(lvl_4_error_direct_mk11), np.mean(lvl_5_error_direct_mk11), np.mean(lvl_6_error_direct_mk11),
         np.mean(lvl_7_error_direct_mk11), np.mean(lvl_8_error_direct_mk11),
         np.mean(lvl_9_error_direct_mk11)]).reshape((9, 1))
    slope_1x = np.asarray(
        [1. / 10., 1. / 20., 1. / 40., 1. / 80., 1. / 160., 1. / 320., 1. / 640., 1. / 1280., 1. / 2560.]).reshape(
        (9, 1))
    plot_1d(xs=[np.asarray([10, 20, 40, 80, 160, 320, 640, 1280, 2560])],
            ys=[errors, errors_direct, errors_mk11, errors_direct_mk11, slope_1x],
            labels=['banach error estimate', 'direct estimate', 'mk11 error banach estimate',
                    'mk11 error direct estimate', 'slope'],
            name="discretization_error",
            folder_name="paper_data/banach",
            linetypes=['o', 'o', 'v', 'v', '-'], xlim=[10, 2560], ylim=[1e-4, 1], xlabel='$n_x$',
            ylabel=r"$||u-u^*||_2^2$",
            loglog=True, title=r"discretization error")

    return True


if __name__ == '__main__':
    main()
