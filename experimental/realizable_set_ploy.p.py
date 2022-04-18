import pandas as pd
import numpy as np
from src.utils import scatter_plot_2d_N2, scatter_plot_2d, load_data


def main():
    training_data = load_data(filename="data/1D/test2.csv", data_dim=3)

    scatter_plot_2d(x_in=training_data[0][:, 1:], z_in=training_data[2], name="test2", log=False,
                    lim_x=(-1.7, 1.7), lim_y=(-1.5, 2.2), lim_z=(0, 12))
    scatter_plot_2d(x_in=training_data[1][:, 1:], z_in=training_data[2], name="test2_lag",
                    title=r"$\alpha_u$ over $\mathcal{R}$", log=False,
                    lim_x=(-100, 100), lim_y=(-100, 100), lim_z=(0, 12))
    exit()
    training_data = load_data(filename="data/1D/Harmonics_M2_1D_normal_gaussian.csv", data_dim=3)

    scatter_plot_2d(x_in=training_data[0][:, 1:], z_in=training_data[2], name="realizable_set_cond", log=False,
                    lim_x=(-1.7, 1.7), lim_y=(-1.5, 2.2), lim_z=(0, 12))
    scatter_plot_2d(x_in=training_data[1][:, 1:], z_in=training_data[2], name="lagrange_mults_cond",
                    title=r"$\alpha_u$ over $\mathcal{R}$", log=False,
                    lim_x=(-20, 20), lim_y=(-20, 20), lim_z=(0, 12))

    scatter_plot_2d(x_in=training_data[1][:, 1:], z_in=training_data[2], name="lagrange_mults_cond_large",
                    title=r"$\alpha_u$ over $\mathcal{R}$", log=False,
                    lim_x=(-40, 40), lim_y=(-40, 40), lim_z=(0, 12))

    training_data = load_data(filename="data/1D/Harmonics_M2_1D_normal_alpha.csv", data_dim=3)

    scatter_plot_2d(x_in=training_data[0][:, 1:], z_in=training_data[2], name="realizable_set_ball_infty", log=False,
                    lim_x=(-1.7, 1.7), lim_y=(-1.5, 2.2), lim_z=(0, 12))
    scatter_plot_2d(x_in=training_data[1][:, 1:], z_in=training_data[2], name="lagrange_mults_ball_infty",
                    title=r"$\alpha_u$ over $\mathcal{R}$", log=False,
                    lim_x=(-20, 20), lim_y=(-20, 20), lim_z=(0, 12))

    training_data = load_data(filename="data/1D/Harmonics_M2_1D_normal_alpha_ball.csv", data_dim=3)

    scatter_plot_2d(x_in=training_data[0][:, 1:], z_in=training_data[2], name="realizable_set_ball_2", log=False,
                    lim_x=(-1.7, 1.7), lim_y=(-1.5, 2.2), lim_z=(0, 12))
    scatter_plot_2d(x_in=training_data[1][:, 1:], z_in=training_data[2], name="lagrange_mults_ball_2",
                    title=r"$\alpha_u$ over $\mathcal{R}$", log=False,
                    lim_x=(-20, 20), lim_y=(-20, 20), lim_z=(0, 12))

    return 0


if __name__ == '__main__':
    main()
