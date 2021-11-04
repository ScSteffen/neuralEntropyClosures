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
            labels=[r"$u_0$", r"$u_1$", r"$u_2$", r"ICNN - $u_{0,\theta}$", r"ICNN - $u_{1,\theta}$",
                    r"ICNN - $u_{2,\theta}$", r"Mono - $u_{0,\theta}$", r"Mono - $u_{1,\theta}$",
                    r"Mono - $u_{2,\theta}$"],
            name="inflow_1D_M2",
            folder_name="paper_data/1D_M2", linetypes=['-', 'o', '^'], xlim=[0, 1],
            xlabel='x', ylabel='u', log=False, title=r"$u$ and $u_\theta$  over $x$")

    err_mk11 = np.linalg.norm(u_ref11 - u_neural11, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk11 = err_mk11 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))
    err_mk15 = np.linalg.norm(u_ref15 - u_neural15, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk15 = err_mk15 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))

    err_res_list = [err_mk11, err_mk15]
    plot_1d([x], err_res_list, labels=["ICNN", "Mono"], name="err_inflow_1D_M2", folder_name="paper_data/1D_M2",
            linetypes=['o', '^'], xlim=[0, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2$", log=True,
            title=r"$||u-u_\theta||_2$ over $x$")
    rel_err_res_list = [rel_errmk11, rel_errmk15]
    plot_1d([x], rel_err_res_list, labels=["ICNN", "Mono"], name="rel_err_inflow_1D_M2", folder_name="paper_data/1D_M2",
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
            labels=[r"$u_0$", r"$u_1$", r"ICNN - $u_{0,\theta}$", r"ICNN - $u_{1,\theta}$", r"Mono - $u_{0,\theta}$",
                    r"Mono - $u_{1,\theta}$"], name="inflow_1D_M1", folder_name="paper_data/1D_M1",
            linetypes=['-', 'o', '^'], xlim=[0, 1], xlabel='x', ylabel='u', log=False,
            title=r"$u$ and $u_\theta$  over $x$")

    err_mk11 = np.linalg.norm(u_ref11 - u_neural11, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk11 = err_mk11 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))
    err_mk15 = np.linalg.norm(u_ref15 - u_neural15, axis=1).reshape((u_ref15.shape[0], 1))
    rel_errmk15 = err_mk15 / np.linalg.norm(u_ref11, axis=1).reshape((u_ref15.shape[0], 1))

    err_res_list = [err_mk11, err_mk15]
    plot_1d([x], err_res_list, labels=["ICNN", "Mono"], name="err_inflow_1D_M1", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[0, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2$", log=True,
            title=r"$||u-u_\theta||_2$ over $x$")
    rel_err_res_list = [rel_errmk11, rel_errmk15]
    plot_1d([x], rel_err_res_list, labels=["ICNN", "Mono"], name="rel_err_inflow_1D_M1", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[0, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $x$")

    # --- synthetic test M1 mk11
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK11_synthetic.csv")
    data_mk11 = df.to_numpy()
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK15_synthetic.csv")
    data_mk15 = df.to_numpy()
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 2].reshape((data_mk11.shape[0], 1)), data_mk15[:, 2].reshape((data_mk15.shape[0], 1))],
            labels=["ICNN", "Mono"], name="rel_err_u_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1e-1], xlabel=r'$u^n_1$',
            ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $u^n_1$")
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 4].reshape((data_mk11.shape[0], 1)), data_mk15[:, 4].reshape((data_mk15.shape[0], 1))],
            labels=["ICNN", "Mono"], name="rel_err_alpha_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1], xlabel=r'$u^n_1$',
            ylabel=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$",
            log=True, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $u^n_1$")
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 6].reshape((data_mk11.shape[0], 1)), data_mk15[:, 6].reshape((data_mk15.shape[0], 1))],
            labels=["ICNN", "Mono"], name="rel_err_h_1D_M1_synthetic", folder_name="paper_data/1D_M1",
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
    df = pd.read_csv("paper_data/2D_M1/err_1D_M1_MK11_periodic.csv")
    data_mk11 = df.to_numpy()
    df = pd.read_csv("paper_data/2D_M1/err_1D_M1_MK15_periodic.csv")
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
            labels=["ICNN", "Mono"], name="rel_err_u_2D_M1_over_time", folder_name="paper_data/2D_M1",
            linetypes=['-', '--'], xlim=[0, time[-1]], ylim=[1e-6, 1e-1], xlabel=r'$t$',
            ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $t$")
    plot_1d([time],
            [err_alpha_mk11.reshape((err_u_mk11.shape[0], 1)), err_alpha_mk15.reshape((err_u_mk11.shape[0], 1))],
            labels=["ICNN", "Mono"], name="rel_err_alpha_2D_M1_over_time", folder_name="paper_data/2D_M1",
            linetypes=['o', '^'], xlim=[0, time[-1]], ylim=[1e-3, 1e-1], xlabel=r'$t$',
            ylabel=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$",
            log=True, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $t$")
    plot_1d([time], [h_ref.reshape((err_u_mk11.shape[0], 1)), h_mk11.reshape((err_u_mk11.shape[0], 1)),
                     h_mk15.reshape((err_u_mk11.shape[0], 1))], labels=["Newton", "ICNN", "Mono"],
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

    # --- illustrate training history for alpha sampling vs u sampling --- 

    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK11_history_normal.csv")
    data_normal = df.to_numpy()
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK11_history_normal_alpha.csv")
    data_alpha = df.to_numpy()

    epoch = data_normal[:, 0]
    normal_loss_h = data_normal[:, 2].reshape(epoch.shape[0], 1)
    normal_loss_alpha = data_normal[:, 5].reshape(epoch.shape[0], 1)
    normal_loss_u = data_normal[:, 8].reshape(epoch.shape[0], 1)
    alpha_loss_h = data_alpha[:, 2].reshape(epoch.shape[0], 1)
    alpha_loss_alpha = data_alpha[:, 5].reshape(epoch.shape[0], 1)
    alpha_loss_u = data_alpha[:, 8].reshape(epoch.shape[0], 1)

    plot_1d([epoch.reshape(epoch.shape[0], 1)],
            [np.concatenate([normal_loss_h, alpha_loss_h], axis=1)],
            labels=[r"$||h_\theta-h||_2^2$, uniform sampling u",
                    r"$||h_\theta-h||_2^2$, uniform sampling $\alpha^r_u$"],
            name="training_loss_h_over_epochs ", folder_name="paper_data/1D_M1",
            linetypes=['-'], xlim=[0, epoch[-1]], ylim=[1e-6, 1e4], xlabel='epoch',
            ylabel=r"loss",
            log=True, title=r"loss $h$ over epochs")

    plot_1d([epoch.reshape(epoch.shape[0], 1)],
            [np.concatenate([normal_loss_alpha, alpha_loss_alpha], axis=1)],
            labels=[r"$||\alpha^r_\theta-\alpha^r_u||_2^2$, uniform sampling u",
                    r"$||\alpha^r_\theta-\alpha^r_u||_2^2$, uniform sampling $\alpha^r_u$"],
            name="training_loss_alpha_over_epochs ", folder_name="paper_data/1D_M1",
            linetypes=['-', ], xlim=[0, epoch[-1]], ylim=[1e-4, 1e4], xlabel='epoch',
            ylabel=r"loss",
            log=True, title=r"loss $\alpha_u^r$ over epochs")

    plot_1d([epoch.reshape(epoch.shape[0], 1)],
            [np.concatenate([normal_loss_u, alpha_loss_u], axis=1)],
            labels=[r"$||u^n_\theta-u^n||_2^2$, uniform sampling u",
                    r"$||u^n_\theta-u^n||_2^2$, uniform sampling $\alpha^r_u$"],
            name="training_loss_u_over_epochs ", folder_name="paper_data/1D_M1",
            linetypes=['-'], xlim=[0, epoch[-1]], ylim=[1e-6, 1e4], xlabel='epoch',
            ylabel=r"loss",
            log=True, title=r"loss $u$ over epochs")

    return True


if __name__ == '__main__':
    main()
