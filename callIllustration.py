"""
Script to call different plots and illustrative methods
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import pandas as pd
from src.utils import load_density_function, load_solution, plot_1d


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
            linetypes=['o', '^'], xlim=[-1, 1], xlabel=r'$u_1$', ylabel=r"$||u-u_\theta||_2/||u||_2$", log=True,
            title=r"$||u-u_\theta||_2/||u||_2$ over $u_1$")
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 4].reshape((data_mk11.shape[0], 1)), data_mk15[:, 4].reshape((data_mk15.shape[0], 1))],
            labels=["ICNN", "Mono"], name="rel_err_alpha_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], xlabel=r'$u_1$', ylabel=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$",
            log=True, title=r"$||\alpha-\alpha_\theta||_2/||\alpha||_2$ over $u_1$")
    plot_1d([data_mk11[:, 0]],
            [data_mk11[:, 6].reshape((data_mk11.shape[0], 1)), data_mk15[:, 6].reshape((data_mk15.shape[0], 1))],
            labels=["ICNN", "Mono"], name="rel_err_h_1D_M1_synthetic", folder_name="paper_data/1D_M1",
            linetypes=['o', '^'], xlim=[-1, 1], xlabel=r'$u_1$', ylabel=r"$||h-h_\theta||_2/||h||_2$",
            log=True, title=r"$||h-h_\theta||_2/||h||_2$ over $u_1$")
    return True


if __name__ == '__main__':
    main()
