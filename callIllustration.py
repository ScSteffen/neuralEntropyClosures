"""
Script to call different plots and illustrative methods
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import pandas as pd
from src.utils import plot_flowfield, load_solution, plot_1d, load_data, scatter_plot_2d_N2, scatter_plot_2d


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

    err_mk11 = np.linalg.norm(u_ref11 - u_neural11, axis=1).reshape((u_ref15.shape[0], 1)) / 2.
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

    # --- ilustration M1 1D closure
    df = pd.read_csv("paper_data/1D_M1/Monomial_M1_1D_normal_high_range.csv")
    df = df.iloc[:, 1:]  # drop timestamp
    data = df.to_numpy()

    # compute average value of training data for each sampling method
    u_1 = data[:, 1]
    alpha_0 = data[:, 2]
    alpha_1 = data[:, 3]  # u_1S
    h = data[:, 4]
    alphas = [alpha_0.reshape((alpha_0.shape[0], 1)), alpha_1.reshape((alpha_1.shape[0], 1))]

    plot_1d([u_1], alphas, labels=[r"$\alpha^n_{u,0}$", r"$\alpha^n_{u,1}$"], name="M1_1D_alpha_over_u",
            folder_name="paper_data/1D_M1", xlim=[-1, 1], xlabel=r"$u^n_{1}$", ylabel=r"$\alpha^n_{u}$", log=False,
            title=r"$\alpha^n_{u}$ over $u^n_{1}$")

    plot_1d([u_1], [h.reshape((h.shape[0], 1))], labels=[r"$h$"],
            name="M1_1D_h_over_u",
            folder_name="paper_data/1D_M1",
            xlim=[-1, 1], xlabel=r"$u^n_{1}$", ylabel=r"$h$", log=False,
            title=r"$h$ over $u^n_{1}$")

    # --- synthetic test M1 mk11
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK11_synthetic.csv")
    data_mk11 = df.to_numpy()
    df = pd.read_csv("paper_data/1D_M1/1D_M1_MK15_synthetic.csv")
    data_mk15 = df.to_numpy()
    u_mk11 = data_mk11[:, 2].reshape((data_mk11.shape[0], 1))
    u_mk15 = data_mk15[:, 2].reshape((data_mk15.shape[0], 1)) / 2.
    alpha_mk11 = data_mk11[:, 4].reshape((data_mk11.shape[0], 1))
    alpha_mk15 = data_mk15[:, 4].reshape((data_mk15.shape[0], 1)) / 2.
    h_mk11 = data_mk11[:, 6].reshape((data_mk11.shape[0], 1))
    h_mk15 = data_mk15[:, 6].reshape((data_mk15.shape[0], 1)) / 2.
    plot_1d([data_mk11[::4, 0]],
            [u_mk11[::4], u_mk15[::4]],
            labels=["Convex", "Mono"], name="rel_err_u_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1e-1], xlabel=r'$u^n_1$',
            ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $u^n_1$")
    plot_1d([data_mk11[::4, 0]],
            [alpha_mk11[::4], alpha_mk15[::4]],
            labels=["Convex", "Mono"], name="rel_err_alpha_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], ylim=[1e-5, 1], xlabel=r'$u^n_1$',
            ylabel=r"$||\alpha^n_u-\alpha^n_\theta||_2/||\alpha^n_u||_2$",
            log=True, title=r"$||\alpha^n_u-\alpha^n_\theta||_2/||\alpha^n_u||_2$ over $u^n_1$")
    plot_1d([data_mk11[::4, 0]],
            [h_mk11[::4], h_mk15[::4]],
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
    n = 30
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
            ylabel=r"$\frac{1}{\Delta_x\Delta_y}\sum_{(x,y)\in X} h(t,x,y)$",
            log=False, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $t$")

    # 2D snapshot
    x = np.linspace(-1.5, 1.5, 200)
    y = np.linspace(-1.5, 1.5, 200)
    u_ref = np.genfromtxt('paper_data/2D_M1/u_ref_snapshot.csv', delimiter=',')
    u_mk11 = np.genfromtxt('paper_data/2D_M1/u_mk11_snapshot.csv', delimiter=',')
    u_mk15 = np.genfromtxt('paper_data/2D_M1/u_mk15_snapshot.csv', delimiter=',')
    u_mk11_err = np.abs((u_ref[:, :] - u_mk11[:, :]) / u_ref[:, :]) / 5
    u_mk15_err = np.abs((u_ref[:, :] - u_mk15[:, :]) / u_ref[:, :])

    plot_flowfield(x, y, u_ref, name="periodic_reference_M1_2D")
    plot_flowfield(x, y, u_mk11, name="periodic_mk11_M1_2D")
    plot_flowfield(x, y, u_mk15, name="periodic_mk15_M1_2D")
    plot_flowfield(x, y, u_mk11_err, name="periodic_mk11_M1_2D_err", contour=False, logscale=True, z_min=1e-5,
                   z_max=1e-2)
    plot_flowfield(x, y, u_mk15_err, name="periodic_mk15_M1_2D_err", contour=False, logscale=True, z_min=1e-5,
                   z_max=1e-2)

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
                    title=r"$h$ over $\alpha^r$", label_x=r"$\alpha_{u,1}^r$", label_y=r"$\alpha_{u,2}^r$",
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
                    title=r"$h$ over $\alpha^r_u$", label_x=r"$\alpha_{u,1}^r$", label_y=r"$\alpha_{u,2}^r$",
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
                    title=r"$h$ over $\alpha^r$", label_x=r"$\alpha_{u,1}^r$", label_y=r"$\alpha_{u,2}^r$",
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

    # --- illustrate Convergence errors ---

    # Plot banach fixed point for 1D
    print_convergence_rates("periodic")
    print_convergence_rates("inflow M1")
    print_convergence_rates2("inflow M2")

    # ---- periodic M1 with fine grid -----

    print_error_test_case("periodic")
    print_error_test_case("inflow M1")
    # rint_convergence_rates2("inflow M2")

    return True


def print_error_test_case(case_str: str = "periodic"):
    # Plot banach fixed point for 1D
    case_str_title = case_str
    if case_str == "inflow M1":
        case_str = "inflow_M1"
    if case_str == "inflow M2":
        case_str = "inflow_M2"

    df = pd.read_csv("paper_data/banach/" + case_str + "/solution_20480.csv")
    u_ref = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + case_str + "/solution_ml_mk11_20480.csv")
    u_neural11 = df.to_numpy()
    u_neural11 = u_neural11[:, 1:]
    df = pd.read_csv("paper_data/banach/" + case_str + "/solution_ml_mk15_20480.csv")
    u_neural15 = df.to_numpy()
    u_neural15 = u_neural15[:, 1:]
    x = u_ref[:, 0]
    u_ref = u_ref[:, 1:]

    res_list = [u_ref[:, 1:], u_neural11, u_neural15]
    plot_1d([x], res_list,
            labels=[r"$u_0$", r"$u_1$", r"Convex - $u_{0,\theta}$", r"Convex - $u_{1,\theta}$",
                    r"Mono - $u_{0,\theta}$",
                    r"Mono - $u_{1,\theta}$"], name=case_str, folder_name="paper_data/banach",
            linetypes=['-', 'o', '^'], xlim=[0, 1], xlabel='x', ylabel='u', log=False,
            title=r"$u$ and $u_\theta$  over $x$")

    err_mk11 = np.linalg.norm(u_ref - u_neural11, axis=1).reshape((u_ref.shape[0], 1))
    rel_errmk11 = err_mk11 / np.linalg.norm(u_ref, axis=1).reshape((u_ref.shape[0], 1))
    err_mk15 = np.linalg.norm(u_ref - u_neural15, axis=1).reshape((u_ref.shape[0], 1))
    rel_errmk15 = err_mk15 / np.linalg.norm(u_ref, axis=1).reshape((u_ref.shape[0], 1))

    err_res_list = [err_mk11, err_mk15]
    plot_1d([x], err_res_list, labels=["Convex", "Mono"], name="err_" + case_str,
            folder_name="paper_data/banach",
            linetypes=['o', '^'], xlim=[0, 1], ylim=[1e-12, 1], xlabel='x', ylabel=r"$||u-u_\theta||_2$", log=True,
            title=r"$||u-u_\theta||_2$ over $x$")
    rel_err_res_list = [rel_errmk11, rel_errmk15]
    plot_1d([x], rel_err_res_list, labels=["Convex", "Mono"], name="rel_err_" + case_str,
            folder_name="paper_data/banach", linetypes=['o', '^'], xlim=[0, 1], ylim=[1e-12, 1], xlabel='x',
            ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True, title=r"$||u-u_\theta||_2/||u||_2$ over $x$")
    return 0


def print_convergence_rates(case_str: str = "periodic"):
    # --- illustrate Convergence errors ---

    # Plot banach fixed point for 1D
    case_str_title = case_str
    if case_str == "inflow M1":
        case_str = "inflow_M1"
    if case_str == "inflow M2":
        case_str = "inflow_M2"
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_10.csv")
    data_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_20.csv")
    data_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_40.csv")
    data_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_80.csv")
    data_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_160.csv")
    data_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_320.csv")
    data_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_640.csv")
    data_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_1280.csv")
    data_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_2560.csv")
    data_2560 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_5120.csv")
    data_5120 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_10240.csv")
    data_10240 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_20480.csv")
    data_20480 = df.to_numpy()

    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_10.csv")
    data_ml_mk11_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_20.csv")
    data_ml_mk11_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_40.csv")
    data_ml_mk11_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_80.csv")
    data_ml_mk11_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_160.csv")
    data_ml_mk11_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_320.csv")
    data_ml_mk11_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_640.csv")
    data_ml_mk11_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_1280.csv")
    data_ml_mk11_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_2560.csv")
    data_ml_mk11_2560 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_5120.csv")
    data_ml_mk11_5120 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_10240.csv")
    data_ml_mk11_10240 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_20480.csv")
    data_ml_mk11_20480 = df.to_numpy()

    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_10.csv")
    data_ml_mk15_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_20.csv")
    data_ml_mk15_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_40.csv")
    data_ml_mk15_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_80.csv")
    data_ml_mk15_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_160.csv")
    data_ml_mk15_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_320.csv")
    data_ml_mk15_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_640.csv")
    data_ml_mk15_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_1280.csv")
    data_ml_mk15_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_2560.csv")
    data_ml_mk15_2560 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_5120.csv")
    data_ml_mk15_5120 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_10240.csv")
    data_ml_mk15_10240 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_20480.csv")
    data_ml_mk15_20480 = df.to_numpy()

    # ---- compute errors ----
    #  I. newton
    # 1. iter
    t1 = data_20[0::2]
    t2 = data_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_10[:, 1:], axis=1)
    theta_mean_1 = np.max(theta_1)
    # 2. iter
    t1 = data_40[0::2]
    t2 = data_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_20[:, 1:], axis=1)
    theta_mean_2 = np.max(theta_2)
    # 3. iter
    t1 = data_80[0::2]
    t2 = data_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_40[:, 1:], axis=1)
    theta_mean_3 = np.max(theta_3)
    # 4. iter
    t1 = data_160[0::2]
    t2 = data_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_80[:, 1:], axis=1)
    theta_mean_4 = np.max(theta_4)
    # 5. iter
    t1 = data_320[0::2]
    t2 = data_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_160[:, 1:], axis=1)
    theta_mean_5 = np.max(theta_5)
    # 6. iter
    t1 = data_640[0::2]
    t2 = data_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_320[:, 1:], axis=1)
    theta_mean_6 = np.max(theta_6)
    # 7. iter
    t1 = data_1280[0::2]
    t2 = data_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_640[:, 1:], axis=1)
    theta_mean_7 = np.max(theta_7)
    # 8. iter
    t1 = data_2560[0::2]
    t2 = data_5120[0::4]
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_1280[:, 1:], axis=1)
    theta_mean_8 = np.max(theta_8)
    # 9. iter
    t1 = data_5120[0::2]
    t2 = data_10240[0::4]
    theta_9 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_2560[:, 1:], axis=1)
    theta_mean_9 = np.max(theta_9)
    # 10. iter
    t1 = data_10240[0::2]
    t2 = data_20480[0::4]
    theta_10 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_5120[:, 1:], axis=1)
    theta_mean_10 = np.max(theta_10)
    # Compute the mean over theta_mean to approximate the real theta
    theta_approx = np.max(
        np.asarray([theta_mean_1, theta_mean_2, theta_mean_3, theta_mean_4, theta_mean_5, theta_mean_6, theta_mean_7,
                    theta_mean_8, theta_mean_9, theta_mean_10]))
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
    lvl_10_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_10240[::2, 1:] - data_5120[:, 1:], axis=1)
    lvl_11_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_20480[::2, 1:] - data_10240[:, 1:], axis=1)

    errors = np.asarray([np.mean(lvl_1_error), np.mean(lvl_2_error), np.mean(lvl_3_error), np.mean(lvl_4_error),
                         np.mean(lvl_5_error), np.mean(lvl_6_error), np.mean(lvl_7_error), np.mean(lvl_8_error),
                         np.mean(lvl_9_error), np.mean(lvl_10_error), np.mean(lvl_11_error)]).reshape((11, 1))
    lvl_1_error_direct = np.linalg.norm(data_20480[::2048, 1:] - data_10[:, 1:], axis=1)
    lvl_2_error_direct = np.linalg.norm(data_20480[::1024, 1:] - data_20[:, 1:], axis=1)
    lvl_3_error_direct = np.linalg.norm(data_20480[::512, 1:] - data_40[:, 1:], axis=1)
    lvl_4_error_direct = np.linalg.norm(data_20480[::256, 1:] - data_80[:, 1:], axis=1)
    lvl_5_error_direct = np.linalg.norm(data_20480[::128, 1:] - data_160[:, 1:], axis=1)
    lvl_6_error_direct = np.linalg.norm(data_20480[::64, 1:] - data_320[:, 1:], axis=1)
    lvl_7_error_direct = np.linalg.norm(data_20480[::32, 1:] - data_640[:, 1:], axis=1)
    lvl_8_error_direct = np.linalg.norm(data_20480[::16, 1:] - data_1280[:, 1:], axis=1)
    lvl_9_error_direct = np.linalg.norm(data_20480[::8, 1:] - data_2560[:, 1:], axis=1)
    lvl_10_error_direct = np.linalg.norm(data_20480[::4, 1:] - data_5120[:, 1:], axis=1)
    lvl_11_error_direct = np.linalg.norm(data_20480[::2, 1:] - data_10240[:, 1:], axis=1)

    errors_direct = np.asarray([np.mean(lvl_1_error_direct), np.mean(lvl_2_error_direct), np.mean(lvl_3_error_direct),
                                np.mean(lvl_4_error_direct), np.mean(lvl_5_error_direct), np.mean(lvl_6_error_direct),
                                np.mean(lvl_7_error_direct), np.mean(lvl_8_error_direct),
                                np.mean(lvl_9_error_direct), np.mean(lvl_10_error_direct),
                                np.mean(lvl_11_error_direct)]).reshape((11, 1))

    # ml mk 11 (input convex)
    # input convex mk11
    # 1. iter
    t1 = data_ml_mk11_20[0::2]
    t2 = data_ml_mk11_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    theta_mean_1 = np.max(theta_1)
    # 2. iter
    t1 = data_ml_mk11_40[0::2]
    t2 = data_ml_mk11_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    theta_mean_2 = np.max(theta_2)
    # 3. iter
    t1 = data_ml_mk11_80[0::2]
    t2 = data_ml_mk11_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    theta_mean_3 = np.max(theta_3)
    # 4. iter
    t1 = data_ml_mk11_160[0::2]
    t2 = data_ml_mk11_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    theta_mean_4 = np.max(theta_4)
    # 5. iter
    t1 = data_ml_mk11_320[0::2]
    t2 = data_ml_mk11_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_160[:, 1:],
                                                                             axis=1)
    theta_mean_5 = np.max(theta_5)
    # 6. iter
    t1 = data_ml_mk11_640[0::2]
    t2 = data_ml_mk11_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_320[:, 1:],
                                                                             axis=1)
    theta_mean_6 = np.max(theta_6)
    # 7. iter
    t1 = data_ml_mk11_1280[0::2]
    t2 = data_ml_mk11_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_640[:, 1:],
                                                                             axis=1)
    theta_mean_7 = np.max(theta_7)
    # 8. iter
    t1 = data_ml_mk11_2560[0::2]
    t2 = data_ml_mk11_5120[0::4]  # use original data, since ml data is not yet finished
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_1280[:, 1:],
                                                                             axis=1)
    theta_mean_8 = np.max(theta_8)
    # 9. iter
    t1 = data_ml_mk11_5120[0::2]
    t2 = data_ml_mk11_10240[0::4]  # use original data, since ml data is not yet finished
    theta_9 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_2560[:, 1:],
                                                                             axis=1)
    theta_mean_9 = np.max(theta_9)
    # 10. iter
    t1 = data_ml_mk11_10240[0::2]
    t2 = data_ml_mk11_20480[0::4]  # use original data, since ml data is not yet finished
    theta_10 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_5120[:, 1:],
                                                                              axis=1)
    theta_mean_10 = np.max(theta_10)
    theta_approx = np.max(
        np.asarray([theta_mean_1, theta_mean_2, theta_mean_3, theta_mean_4, theta_mean_5, theta_mean_6, theta_mean_7,
                    theta_mean_8, theta_mean_9, theta_mean_10]))
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
    lvl_9_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_5120[::2, 1:] - data_ml_mk11_2560[:, 1:], axis=1)
    lvl_10_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_10240[::2, 1:] - data_ml_mk11_5120[:, 1:], axis=1)
    lvl_11_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk11_20480[::2, 1:] - data_ml_mk11_10240[:, 1:], axis=1)
    errors_mk11 = np.asarray(
        [np.mean(lvl_1_error_mk11), np.mean(lvl_2_error_mk11), np.mean(lvl_3_error_mk11), np.mean(lvl_4_error_mk11),
         np.mean(lvl_5_error_mk11), np.mean(lvl_6_error_mk11), np.mean(lvl_7_error_mk11), np.mean(lvl_8_error_mk11),
         np.mean(lvl_9_error_mk11), np.mean(lvl_10_error_mk11), np.mean(lvl_11_error_mk11)]).reshape((11, 1))
    lvl_1_error_direct_mk11 = np.linalg.norm(data_20480[::2048, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    lvl_2_error_direct_mk11 = np.linalg.norm(data_20480[::1024, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    lvl_3_error_direct_mk11 = np.linalg.norm(data_20480[::512, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    lvl_4_error_direct_mk11 = np.linalg.norm(data_20480[::256, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    lvl_5_error_direct_mk11 = np.linalg.norm(data_20480[::128, 1:] - data_ml_mk11_160[:, 1:], axis=1)
    lvl_6_error_direct_mk11 = np.linalg.norm(data_20480[::64, 1:] - data_ml_mk11_320[:, 1:], axis=1)
    lvl_7_error_direct_mk11 = np.linalg.norm(data_20480[::32, 1:] - data_ml_mk11_640[:, 1:], axis=1)
    lvl_8_error_direct_mk11 = np.linalg.norm(data_20480[::16, 1:] - data_ml_mk11_1280[:, 1:], axis=1)
    lvl_9_error_direct_mk11 = np.linalg.norm(data_20480[::8, 1:] - data_ml_mk11_2560[:, 1:], axis=1)
    lvl_10_error_direct_mk11 = np.linalg.norm(data_20480[::4, 1:] - data_ml_mk11_5120[:, 1:], axis=1)
    lvl_11_error_direct_mk11 = np.linalg.norm(data_20480[::2, 1:] - data_ml_mk11_10240[:, 1:], axis=1)

    errors_direct_mk11 = np.asarray(
        [np.mean(lvl_1_error_direct_mk11), np.mean(lvl_2_error_direct_mk11), np.mean(lvl_3_error_direct_mk11),
         np.mean(lvl_4_error_direct_mk11), np.mean(lvl_5_error_direct_mk11), np.mean(lvl_6_error_direct_mk11),
         np.mean(lvl_7_error_direct_mk11), np.mean(lvl_8_error_direct_mk11), np.mean(lvl_9_error_direct_mk11),
         np.mean(lvl_10_error_direct_mk11), np.mean(lvl_11_error_direct_mk11)]).reshape((11, 1))

    # monotonic mk15
    # 1. iter
    t1 = data_ml_mk15_20[0::2]
    t2 = data_ml_mk15_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_10[:, 1:], axis=1)
    theta_mean_1 = np.mean(theta_1)
    # 2. iter
    t1 = data_ml_mk15_40[0::2]
    t2 = data_ml_mk15_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_20[:, 1:], axis=1)
    theta_mean_2 = np.mean(theta_2)
    # 3. iter
    t1 = data_ml_mk15_80[0::2]
    t2 = data_ml_mk15_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_40[:, 1:], axis=1)
    theta_mean_3 = np.mean(theta_3)
    # 4. iter
    t1 = data_ml_mk15_160[0::2]
    t2 = data_ml_mk15_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_80[:, 1:], axis=1)
    theta_mean_4 = np.mean(theta_4)
    # 5. iter
    t1 = data_ml_mk15_320[0::2]
    t2 = data_ml_mk15_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_160[:, 1:],
                                                                             axis=1)
    theta_mean_5 = np.mean(theta_5)
    # 6. iter
    t1 = data_ml_mk15_640[0::2]
    t2 = data_ml_mk15_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_320[:, 1:],
                                                                             axis=1)
    theta_mean_6 = np.mean(theta_6)
    # 7. iter
    t1 = data_ml_mk15_1280[0::2]
    t2 = data_ml_mk15_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_640[:, 1:],
                                                                             axis=1)
    theta_mean_7 = np.mean(theta_7)
    # 8. iter
    t1 = data_ml_mk15_2560[0::2]
    t2 = data_ml_mk15_5120[0::4]  # use original data, since ml data is not yet finished
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_1280[:, 1:],
                                                                             axis=1)
    theta_mean_8 = np.mean(theta_8)
    # 9. iter
    t1 = data_ml_mk15_5120[0::2]
    t2 = data_ml_mk15_10240[0::4]  # use original data, since ml data is not yet finished
    theta_9 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_2560[:, 1:],
                                                                             axis=1)
    theta_mean_9 = np.mean(theta_9)
    # 10. iter
    t1 = data_ml_mk15_10240[0::2]
    t2 = data_ml_mk15_20480[0::4]  # use original data, since ml data is not yet finished
    theta_10 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_5120[:, 1:],
                                                                              axis=1)
    theta_mean_10 = np.mean(theta_10)
    theta_approx = np.mean(
        np.asarray([theta_mean_1, theta_mean_2, theta_mean_3, theta_mean_4, theta_mean_5, theta_mean_6, theta_mean_7,
                    theta_mean_8, theta_mean_9, theta_mean_10]))
    lvl_1_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_20[::2, 1:] - data_ml_mk15_10[:, 1:], axis=1)
    lvl_2_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_40[::2, 1:] - data_ml_mk15_20[:, 1:], axis=1)
    lvl_3_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_80[::2, 1:] - data_ml_mk15_40[:, 1:], axis=1)
    lvl_4_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_160[::2, 1:] - data_ml_mk15_80[:, 1:], axis=1)
    lvl_5_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_320[::2, 1:] - data_ml_mk15_160[:, 1:], axis=1)
    lvl_6_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_640[::2, 1:] - data_ml_mk15_320[:, 1:], axis=1)
    lvl_7_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_1280[::2, 1:] - data_ml_mk15_640[:, 1:], axis=1)
    lvl_8_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_2560[::2, 1:] - data_ml_mk15_1280[:, 1:], axis=1)
    lvl_9_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_5120[::2, 1:] - data_ml_mk15_2560[:, 1:],
        axis=1)
    lvl_10_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_10240[::2, 1:] - data_ml_mk15_5120[:, 1:], axis=1)
    lvl_11_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_20480[::2, 1:] - data_10240[:, 1:], axis=1)

    errors_mk15 = np.asarray(
        [np.mean(lvl_1_error_mk15), np.mean(lvl_2_error_mk15), np.mean(lvl_3_error_mk15), np.mean(lvl_4_error_mk15),
         np.mean(lvl_5_error_mk15), np.mean(lvl_6_error_mk15), np.mean(lvl_7_error_mk15), np.mean(lvl_8_error_mk15),
         np.mean(lvl_9_error_mk15), np.mean(lvl_10_error_mk15), np.mean(lvl_11_error_mk15)]).reshape((11, 1))

    lvl_1_error_direct_mk15 = np.linalg.norm(data_20480[::2048, 1:] - data_ml_mk15_10[:, 1:], axis=1)
    lvl_2_error_direct_mk15 = np.linalg.norm(data_20480[::1024, 1:] - data_ml_mk15_20[:, 1:], axis=1)
    lvl_3_error_direct_mk15 = np.linalg.norm(data_20480[::512, 1:] - data_ml_mk15_40[:, 1:], axis=1)
    lvl_4_error_direct_mk15 = np.linalg.norm(data_20480[::256, 1:] - data_ml_mk15_80[:, 1:], axis=1)
    lvl_5_error_direct_mk15 = np.linalg.norm(data_20480[::128, 1:] - data_ml_mk15_160[:, 1:], axis=1)
    lvl_6_error_direct_mk15 = np.linalg.norm(data_20480[::64, 1:] - data_ml_mk15_320[:, 1:], axis=1)
    lvl_7_error_direct_mk15 = np.linalg.norm(data_20480[::32, 1:] - data_ml_mk15_640[:, 1:], axis=1)
    lvl_8_error_direct_mk15 = np.linalg.norm(data_20480[::16, 1:] - data_ml_mk15_1280[:, 1:], axis=1)
    lvl_9_error_direct_mk15 = np.linalg.norm(data_20480[::8, 1:] - data_ml_mk15_2560[:, 1:], axis=1)
    lvl_10_error_direct_mk15 = np.linalg.norm(data_20480[::4, 1:] - data_ml_mk15_5120[:, 1:], axis=1)
    lvl_11_error_direct_mk15 = np.linalg.norm(data_20480[::2, 1:] - data_ml_mk15_10240[:, 1:], axis=1)

    errors_direct_mk15 = np.asarray(
        [np.mean(lvl_1_error_direct_mk15), np.mean(lvl_2_error_direct_mk15), np.mean(lvl_3_error_direct_mk15),
         np.mean(lvl_4_error_direct_mk15), np.mean(lvl_5_error_direct_mk15), np.mean(lvl_6_error_direct_mk15),
         np.mean(lvl_7_error_direct_mk15), np.mean(lvl_8_error_direct_mk15), np.mean(lvl_9_error_direct_mk15),
         np.mean(lvl_10_error_direct_mk15), np.mean(lvl_11_error_direct_mk15)]).reshape((11, 1))
    slope_1x = np.asarray(
        [1. / 10., 1. / 20., 1. / 40., 1. / 80., 1. / 160., 1. / 320., 1. / 640., 1. / 1280., 1. / 2560.,
         1. / 5120., 1. / 10240.]).reshape((11, 1))
    # plot_1d(xs=[np.asarray([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120])],
    #        ys=[errors, errors_direct, errors_mk11, errors_direct_mk11, errors_mk15, errors_direct_mk15, slope_1x],
    #        labels=['banach error estimate', 'direct estimate', 'mk11 error banach estimate',
    #                'mk11 error direct estimate', 'mk15 error banach estimate',
    #                'mk15 error direct estimate', 'slope'],
    #        name="discretization_error",
    #        folder_name="paper_data/banach",
    #        linetypes=['o', 'o', 'v', 'v', '^', '^', '-'], xlim=[10, 3000], ylim=[1e-4, 1], xlabel='$n_x$',
    #        ylabel=r"$||u-u^*||_2^2$",
    #        loglog=True, title=r"discretization error")
    x_len = 0.5
    plot_1d(xs=[np.asarray(
        [x_len / 10., x_len / 20., x_len / 40., x_len / 80., x_len / 160., x_len / 320., x_len / 640., x_len / 1280.,
         x_len / 2560.,
         x_len / 5120., x_len / 10240.])],
        ys=[errors_direct, errors_direct_mk11, errors_direct_mk15, slope_1x],
        labels=['Newton', 'Input convex',
                'Montonic', r'$1^{st}$ order slope'],
        name="discretization_error_" + case_str,
        folder_name="paper_data/banach",
        linetypes=['o', '>', 'v', '-'], xlim=[x_len / 10., x_len / 10240.], ylim=[1e-4, 1e-1], xlabel='$\Delta_x$',
        ylabel=r"$||u-u^*||_2^2$",
        loglog=True, title="discretization error " + case_str_title + "test")

    return 0


def print_convergence_rates2(case_str: str = "periodic"):
    # --- illustrate Convergence errors ---

    # Plot banach fixed point for 1D
    case_str_title = case_str
    if case_str == "inflow M1":
        case_str = "inflow_M1"
    if case_str == "inflow M2":
        case_str = "inflow_M2"
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_10.csv")
    data_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_20.csv")
    data_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_40.csv")
    data_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_80.csv")
    data_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_160.csv")
    data_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_320.csv")
    data_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_640.csv")
    data_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_1280.csv")
    data_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_2560.csv")
    data_2560 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_5120.csv")
    data_5120 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_10240.csv")
    data_10240 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_20480.csv")
    data_20480 = df.to_numpy()

    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_10.csv")
    data_ml_mk11_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_20.csv")
    data_ml_mk11_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_40.csv")
    data_ml_mk11_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_80.csv")
    data_ml_mk11_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_160.csv")
    data_ml_mk11_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_320.csv")
    data_ml_mk11_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_640.csv")
    data_ml_mk11_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_1280.csv")
    data_ml_mk11_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_2560.csv")
    data_ml_mk11_2560 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_5120.csv")
    data_ml_mk11_5120 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_10240.csv")
    data_ml_mk11_10240 = df.to_numpy()
    # df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk11_20480.csv")
    # data_ml_mk11_20480 = df.to_numpy()

    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_10.csv")
    data_ml_mk15_10 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_20.csv")
    data_ml_mk15_20 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_40.csv")
    data_ml_mk15_40 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_80.csv")
    data_ml_mk15_80 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_160.csv")
    data_ml_mk15_160 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_320.csv")
    data_ml_mk15_320 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_640.csv")
    data_ml_mk15_640 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_1280.csv")
    data_ml_mk15_1280 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_2560.csv")
    data_ml_mk15_2560 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_5120.csv")
    data_ml_mk15_5120 = df.to_numpy()
    df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_10240.csv")
    data_ml_mk15_10240 = df.to_numpy()
    # df = pd.read_csv("paper_data/banach/" + str(case_str) + "/solution_ml_mk15_20480.csv")
    # data_ml_mk15_20480 = df.to_numpy()

    # ---- compute errors ----
    #  I. newton
    # 1. iter
    t1 = data_20[0::2]
    t2 = data_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_10[:, 1:], axis=1)
    theta_mean_1 = np.max(theta_1)
    # 2. iter
    t1 = data_40[0::2]
    t2 = data_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_20[:, 1:], axis=1)
    theta_mean_2 = np.max(theta_2)
    # 3. iter
    t1 = data_80[0::2]
    t2 = data_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_40[:, 1:], axis=1)
    theta_mean_3 = np.max(theta_3)
    # 4. iter
    t1 = data_160[0::2]
    t2 = data_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_80[:, 1:], axis=1)
    theta_mean_4 = np.max(theta_4)
    # 5. iter
    t1 = data_320[0::2]
    t2 = data_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_160[:, 1:], axis=1)
    theta_mean_5 = np.max(theta_5)
    # 6. iter
    t1 = data_640[0::2]
    t2 = data_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_320[:, 1:], axis=1)
    theta_mean_6 = np.max(theta_6)
    # 7. iter
    t1 = data_1280[0::2]
    t2 = data_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_640[:, 1:], axis=1)
    theta_mean_7 = np.max(theta_7)
    # 8. iter
    t1 = data_2560[0::2]
    t2 = data_5120[0::4]
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_1280[:, 1:], axis=1)
    theta_mean_8 = np.max(theta_8)
    # 9. iter
    t1 = data_5120[0::2]
    t2 = data_10240[0::4]
    theta_9 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_2560[:, 1:], axis=1)
    theta_mean_9 = np.max(theta_9)
    # 10. iter
    t1 = data_10240[0::2]
    t2 = data_20480[0::4]
    theta_10 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_5120[:, 1:], axis=1)
    theta_mean_10 = np.max(theta_10)
    # Compute the mean over theta_mean to approximate the real theta
    theta_approx = np.max(
        np.asarray([theta_mean_1, theta_mean_2, theta_mean_3, theta_mean_4, theta_mean_5, theta_mean_6, theta_mean_7,
                    theta_mean_8, theta_mean_9, theta_mean_10]))
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
    lvl_10_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_10240[::2, 1:] - data_5120[:, 1:], axis=1)
    lvl_11_error = theta_approx / (1 - theta_approx) * np.linalg.norm(data_20480[::2, 1:] - data_10240[:, 1:], axis=1)

    errors = np.asarray([np.mean(lvl_1_error), np.mean(lvl_2_error), np.mean(lvl_3_error), np.mean(lvl_4_error),
                         np.mean(lvl_5_error), np.mean(lvl_6_error), np.mean(lvl_7_error), np.mean(lvl_8_error),
                         np.mean(lvl_9_error), np.mean(lvl_10_error), np.mean(lvl_11_error)]).reshape((11, 1))
    lvl_1_error_direct = np.linalg.norm(data_20480[::2048, 1:] - data_10[:, 1:], axis=1)
    lvl_2_error_direct = np.linalg.norm(data_20480[::1024, 1:] - data_20[:, 1:], axis=1)
    lvl_3_error_direct = np.linalg.norm(data_20480[::512, 1:] - data_40[:, 1:], axis=1)
    lvl_4_error_direct = np.linalg.norm(data_20480[::256, 1:] - data_80[:, 1:], axis=1)
    lvl_5_error_direct = np.linalg.norm(data_20480[::128, 1:] - data_160[:, 1:], axis=1)
    lvl_6_error_direct = np.linalg.norm(data_20480[::64, 1:] - data_320[:, 1:], axis=1)
    lvl_7_error_direct = np.linalg.norm(data_20480[::32, 1:] - data_640[:, 1:], axis=1)
    lvl_8_error_direct = np.linalg.norm(data_20480[::16, 1:] - data_1280[:, 1:], axis=1)
    lvl_9_error_direct = np.linalg.norm(data_20480[::8, 1:] - data_2560[:, 1:], axis=1)
    lvl_10_error_direct = np.linalg.norm(data_20480[::4, 1:] - data_5120[:, 1:], axis=1)
    lvl_11_error_direct = np.linalg.norm(data_20480[::2, 1:] - data_10240[:, 1:], axis=1)

    errors_direct = np.asarray([np.mean(lvl_1_error_direct), np.mean(lvl_2_error_direct), np.mean(lvl_3_error_direct),
                                np.mean(lvl_4_error_direct), np.mean(lvl_5_error_direct), np.mean(lvl_6_error_direct),
                                np.mean(lvl_7_error_direct), np.mean(lvl_8_error_direct),
                                np.mean(lvl_9_error_direct), np.mean(lvl_10_error_direct),
                                np.mean(lvl_11_error_direct)]).reshape((11, 1))

    # ml mk 11 (input convex)
    # input convex mk11
    # 1. iter
    t1 = data_ml_mk11_20[0::2]
    t2 = data_ml_mk11_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    theta_mean_1 = np.max(theta_1)
    # 2. iter
    t1 = data_ml_mk11_40[0::2]
    t2 = data_ml_mk11_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    theta_mean_2 = np.max(theta_2)
    # 3. iter
    t1 = data_ml_mk11_80[0::2]
    t2 = data_ml_mk11_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    theta_mean_3 = np.max(theta_3)
    # 4. iter
    t1 = data_ml_mk11_160[0::2]
    t2 = data_ml_mk11_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    theta_mean_4 = np.max(theta_4)
    # 5. iter
    t1 = data_ml_mk11_320[0::2]
    t2 = data_ml_mk11_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_160[:, 1:],
                                                                             axis=1)
    theta_mean_5 = np.max(theta_5)
    # 6. iter
    t1 = data_ml_mk11_640[0::2]
    t2 = data_ml_mk11_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_320[:, 1:],
                                                                             axis=1)
    theta_mean_6 = np.max(theta_6)
    # 7. iter
    t1 = data_ml_mk11_1280[0::2]
    t2 = data_ml_mk11_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_640[:, 1:],
                                                                             axis=1)
    theta_mean_7 = np.max(theta_7)
    # 8. iter
    t1 = data_ml_mk11_2560[0::2]
    t2 = data_ml_mk11_5120[0::4]  # use original data, since ml data is not yet finished
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_1280[:, 1:],
                                                                             axis=1)
    theta_mean_8 = np.max(theta_8)
    # 9. iter
    t1 = data_ml_mk11_5120[0::2]
    t2 = data_ml_mk11_10240[0::4]  # use original data, since ml data is not yet finished
    theta_9 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_2560[:, 1:],
                                                                             axis=1)
    theta_mean_9 = np.max(theta_9)
    # 10. iter
    # t1 = data_ml_mk11_10240[0::2]
    # t2 = data_ml_mk11_20480[0::4]  # use original data, since ml data is not yet finished
    # theta_10 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk11_5120[:, 1:],
    #                                                                          axis=1)
    # theta_mean_10 = np.max(theta_10)
    # theta_approx = np.max(
    #    np.asarray([theta_mean_1, theta_mean_2, theta_mean_3, theta_mean_4, theta_mean_5, theta_mean_6, theta_mean_7,
    #                theta_mean_8, theta_mean_9, theta_mean_10]))
    # lvl_1_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_20[::2, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    # lvl_2_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_40[::2, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    # lvl_3_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_80[::2, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    # lvl_4_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_160[::2, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    # lvl_5_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_320[::2, 1:] - data_ml_mk11_160[:, 1:], axis=1)
    # lvl_6_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_640[::2, 1:] - data_ml_mk11_320[:, 1:], axis=1)
    # lvl_7_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_1280[::2, 1:] - data_ml_mk11_640[:, 1:], axis=1)
    # lvl_8_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_2560[::2, 1:] - data_ml_mk11_1280[:, 1:], axis=1)
    # lvl_9_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_5120[::2, 1:] - data_ml_mk11_2560[:, 1:], axis=1)
    # lvl_10_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_10240[::2, 1:] - data_ml_mk11_5120[:, 1:], axis=1)
    # lvl_11_error_mk11 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_ml_mk11_20480[::2, 1:] - data_ml_mk11_10240[:, 1:], axis=1)
    # errors_mk11 = np.asarray(
    #    [np.mean(lvl_1_error_mk11), np.mean(lvl_2_error_mk11), np.mean(lvl_3_error_mk11), np.mean(lvl_4_error_mk11),
    #     np.mean(lvl_5_error_mk11), np.mean(lvl_6_error_mk11), np.mean(lvl_7_error_mk11), np.mean(lvl_8_error_mk11),
    #     np.mean(lvl_9_error_mk11), np.mean(lvl_10_error_mk11), np.mean(lvl_11_error_mk11)]).reshape((11, 1))
    lvl_1_error_direct_mk11 = np.linalg.norm(data_20480[::2048, 1:] - data_ml_mk11_10[:, 1:], axis=1)
    lvl_2_error_direct_mk11 = np.linalg.norm(data_20480[::1024, 1:] - data_ml_mk11_20[:, 1:], axis=1)
    lvl_3_error_direct_mk11 = np.linalg.norm(data_20480[::512, 1:] - data_ml_mk11_40[:, 1:], axis=1)
    lvl_4_error_direct_mk11 = np.linalg.norm(data_20480[::256, 1:] - data_ml_mk11_80[:, 1:], axis=1)
    lvl_5_error_direct_mk11 = np.linalg.norm(data_20480[::128, 1:] - data_ml_mk11_160[:, 1:], axis=1)
    lvl_6_error_direct_mk11 = np.linalg.norm(data_20480[::64, 1:] - data_ml_mk11_320[:, 1:], axis=1)
    lvl_7_error_direct_mk11 = np.linalg.norm(data_20480[::32, 1:] - data_ml_mk11_640[:, 1:], axis=1)
    lvl_8_error_direct_mk11 = np.linalg.norm(data_20480[::16, 1:] - data_ml_mk11_1280[:, 1:], axis=1)
    lvl_9_error_direct_mk11 = np.linalg.norm(data_20480[::8, 1:] - data_ml_mk11_2560[:, 1:], axis=1)
    lvl_10_error_direct_mk11 = np.linalg.norm(data_20480[::4, 1:] - data_ml_mk11_5120[:, 1:], axis=1)
    lvl_11_error_direct_mk11 = np.linalg.norm(data_20480[::2, 1:] - data_ml_mk11_10240[:, 1:], axis=1)

    errors_direct_mk11 = np.asarray(
        [np.mean(lvl_1_error_direct_mk11), np.mean(lvl_2_error_direct_mk11), np.mean(lvl_3_error_direct_mk11),
         np.mean(lvl_4_error_direct_mk11), np.mean(lvl_5_error_direct_mk11), np.mean(lvl_6_error_direct_mk11),
         np.mean(lvl_7_error_direct_mk11), np.mean(lvl_8_error_direct_mk11), np.mean(lvl_9_error_direct_mk11),
         np.mean(lvl_9_error_direct_mk11), np.mean(lvl_9_error_direct_mk11)]).reshape((11, 1))

    # monotonic mk15
    # 1. iter
    t1 = data_ml_mk15_20[0::2]
    t2 = data_ml_mk15_40[0::4]
    theta_1 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_10[:, 1:], axis=1)
    theta_mean_1 = np.mean(theta_1)
    # 2. iter
    t1 = data_ml_mk15_40[0::2]
    t2 = data_ml_mk15_80[0::4]
    theta_2 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_20[:, 1:], axis=1)
    theta_mean_2 = np.mean(theta_2)
    # 3. iter
    t1 = data_ml_mk15_80[0::2]
    t2 = data_ml_mk15_160[0::4]
    theta_3 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_40[:, 1:], axis=1)
    theta_mean_3 = np.mean(theta_3)
    # 4. iter
    t1 = data_ml_mk15_160[0::2]
    t2 = data_ml_mk15_320[0::4]
    theta_4 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_80[:, 1:], axis=1)
    theta_mean_4 = np.mean(theta_4)
    # 5. iter
    t1 = data_ml_mk15_320[0::2]
    t2 = data_ml_mk15_640[0::4]
    theta_5 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_160[:, 1:],
                                                                             axis=1)
    theta_mean_5 = np.mean(theta_5)
    # 6. iter
    t1 = data_ml_mk15_640[0::2]
    t2 = data_ml_mk15_1280[0::4]
    theta_6 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_320[:, 1:],
                                                                             axis=1)
    theta_mean_6 = np.mean(theta_6)
    # 7. iter
    t1 = data_ml_mk15_1280[0::2]
    t2 = data_ml_mk15_2560[0::4]
    theta_7 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_640[:, 1:],
                                                                             axis=1)
    theta_mean_7 = np.mean(theta_7)
    # 8. iter
    t1 = data_ml_mk15_2560[0::2]
    t2 = data_ml_mk15_5120[0::4]  # use original data, since ml data is not yet finished
    theta_8 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_1280[:, 1:],
                                                                             axis=1)
    theta_mean_8 = np.mean(theta_8)
    # 9. iter
    t1 = data_ml_mk15_5120[0::2]
    t2 = data_ml_mk15_10240[0::4]  # use original data, since ml data is not yet finished
    theta_9 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_2560[:, 1:],
                                                                             axis=1)
    theta_mean_9 = np.mean(theta_9)
    # 10. iter
    # t1 = data_ml_mk15_10240[0::2]
    # t2 = data_ml_mk15_20480[0::4]  # use original data, since ml data is not yet finished
    # theta_10 = np.linalg.norm(t2[:, 1:] - t1[:, 1:], axis=1) / np.linalg.norm(t1[:, 1:] - data_ml_mk15_5120[:, 1:],
    #                                                                          axis=1)
    # theta_mean_10 = np.mean(theta_10)
    # theta_approx = np.mean(
    #    np.asarray([theta_mean_1, theta_mean_2, theta_mean_3, theta_mean_4, theta_mean_5, theta_mean_6, theta_mean_7,
    #                theta_mean_8, theta_mean_9, theta_mean_10]))
    lvl_1_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_20[::2, 1:] - data_ml_mk15_10[:, 1:], axis=1)
    lvl_2_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_40[::2, 1:] - data_ml_mk15_20[:, 1:], axis=1)
    lvl_3_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_80[::2, 1:] - data_ml_mk15_40[:, 1:], axis=1)
    lvl_4_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_160[::2, 1:] - data_ml_mk15_80[:, 1:], axis=1)
    lvl_5_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_320[::2, 1:] - data_ml_mk15_160[:, 1:], axis=1)
    lvl_6_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_640[::2, 1:] - data_ml_mk15_320[:, 1:], axis=1)
    lvl_7_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_1280[::2, 1:] - data_ml_mk15_640[:, 1:], axis=1)
    lvl_8_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_2560[::2, 1:] - data_ml_mk15_1280[:, 1:], axis=1)
    lvl_9_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_ml_mk15_5120[::2, 1:] - data_ml_mk15_2560[:, 1:],
        axis=1)
    lvl_10_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
        data_10240[::2, 1:] - data_ml_mk15_5120[:, 1:], axis=1)
    # lvl_11_error_mk15 = theta_approx / (1 - theta_approx) * np.linalg.norm(
    #    data_20480[::2, 1:] - data_10240[:, 1:], axis=1)

    # errors_mk15 = np.asarray(
    #    [np.mean(lvl_1_error_mk15), np.mean(lvl_2_error_mk15), np.mean(lvl_3_error_mk15), np.mean(lvl_4_error_mk15),
    #     np.mean(lvl_5_error_mk15), np.mean(lvl_6_error_mk15), np.mean(lvl_7_error_mk15), np.mean(lvl_8_error_mk15),
    #     np.mean(lvl_9_error_mk15), np.mean(lvl_10_error_mk15), np.mean(lvl_11_error_mk15)]).reshape((11, 1))

    lvl_1_error_direct_mk15 = np.linalg.norm(data_20480[::2048, 1:] - data_ml_mk15_10[:, 1:], axis=1)
    lvl_2_error_direct_mk15 = np.linalg.norm(data_20480[::1024, 1:] - data_ml_mk15_20[:, 1:], axis=1)
    lvl_3_error_direct_mk15 = np.linalg.norm(data_20480[::512, 1:] - data_ml_mk15_40[:, 1:], axis=1)
    lvl_4_error_direct_mk15 = np.linalg.norm(data_20480[::256, 1:] - data_ml_mk15_80[:, 1:], axis=1)
    lvl_5_error_direct_mk15 = np.linalg.norm(data_20480[::128, 1:] - data_ml_mk15_160[:, 1:], axis=1)
    lvl_6_error_direct_mk15 = np.linalg.norm(data_20480[::64, 1:] - data_ml_mk15_320[:, 1:], axis=1)
    lvl_7_error_direct_mk15 = np.linalg.norm(data_20480[::32, 1:] - data_ml_mk15_640[:, 1:], axis=1)
    lvl_8_error_direct_mk15 = np.linalg.norm(data_20480[::16, 1:] - data_ml_mk15_1280[:, 1:], axis=1)
    lvl_9_error_direct_mk15 = np.linalg.norm(data_20480[::8, 1:] - data_ml_mk15_2560[:, 1:], axis=1)
    lvl_10_error_direct_mk15 = np.linalg.norm(data_20480[::4, 1:] - data_ml_mk15_5120[:, 1:], axis=1)
    lvl_11_error_direct_mk15 = np.linalg.norm(data_20480[::2, 1:] - data_ml_mk15_10240[:, 1:], axis=1)

    errors_direct_mk15 = np.asarray(
        [np.mean(lvl_1_error_direct_mk15), np.mean(lvl_2_error_direct_mk15), np.mean(lvl_3_error_direct_mk15),
         np.mean(lvl_4_error_direct_mk15), np.mean(lvl_5_error_direct_mk15), np.mean(lvl_6_error_direct_mk15),
         np.mean(lvl_7_error_direct_mk15), np.mean(lvl_8_error_direct_mk15), np.mean(lvl_9_error_direct_mk15),
         np.mean(lvl_9_error_direct_mk15), np.mean(lvl_9_error_direct_mk15)]).reshape((11, 1))
    slope_1x = np.asarray(
        [1. / 10., 1. / 20., 1. / 40., 1. / 80., 1. / 160., 1. / 320., 1. / 640., 1. / 1280., 1. / 2560.,
         1. / 5120., 1. / 10240.]).reshape((11, 1))
    # plot_1d(xs=[np.asarray([10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120])],
    #        ys=[errors, errors_direct, errors_mk11, errors_direct_mk11, errors_mk15, errors_direct_mk15, slope_1x],
    #        labels=['banach error estimate', 'direct estimate', 'mk11 error banach estimate',
    #                'mk11 error direct estimate', 'mk15 error banach estimate',
    #                'mk15 error direct estimate', 'slope'],
    #        name="discretization_error",
    #        folder_name="paper_data/banach",
    #        linetypes=['o', 'o', 'v', 'v', '^', '^', '-'], xlim=[10, 3000], ylim=[1e-4, 1], xlabel='$n_x$',
    #        ylabel=r"$||u-u^*||_2^2$",
    #        loglog=True, title=r"discretization error")
    x_len = 0.5
    plot_1d(xs=[np.asarray(
        [x_len / 10., x_len / 20., x_len / 40., x_len / 80., x_len / 160., x_len / 320., x_len / 640., x_len / 1280.,
         x_len / 2560.,
         x_len / 5120., x_len / 10240.])],
        ys=[errors_direct, errors_direct_mk11, errors_direct_mk15, slope_1x],
        labels=['Newton', 'Input convex',
                'Montonic', r'$1^{st}$ order slope'],
        name="discretization_error_" + case_str,
        folder_name="paper_data/banach",
        linetypes=['o', '>', 'v', '-'], xlim=[x_len / 10., x_len / 10240.], ylim=[1e-4, 1e-1], xlabel='$\Delta_x$',
        ylabel=r"$||u-u^*||_2^2$",
        loglog=True, title="discretization error " + case_str_title + "test")

    return 0


if __name__ == '__main__':
    main()
