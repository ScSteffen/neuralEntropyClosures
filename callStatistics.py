"""
brief: Script to perform data statistics
author: Steffen Schotthöfer
date: 26.08.2021
"""

from src.statistics import DataStatistics
from src.utils import load_data


def main():
    file = "data/1D/Monomial_M3_1D_normal_alpha_big.csv"
    [u, alpha, h] = load_data(filename=file, input_dim=4, selected_cols=[True, True, True])
    data_stat = DataStatistics(u)
    print(data_stat.get_mean())
    t = data_stat.get_cov()
    print(t)

    return True


if __name__ == '__main__':
    main()
