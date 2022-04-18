"""
Script to compute maxwellians
Author: Steffen Schotthoefer
Version: 0.1
Date 2.12.2021
"""

import numpy as np
import matplotlib.pyplot as plt


def maxwellian2D(x, y, rho, T):
    return rho / (2 * np.pi * T) * np.exp(-rho / (2 * T) * (x ** 2 + y ** 2))


def maxwellian1D(x, rho, T):
    return (rho / (2 * np.pi * T)) ** (0.5) * np.exp(-rho / (2 * T) * (x ** 2))


def custom_maxwellian(x, y, alpha_0, alpha_2):
    return np.exp(alpha_0 + alpha_2 * (x ** 2 + y ** 2))


def main():
    print("---------- Start Maxwellian Suite ------------")

    nx, ny = (500, 500)
    x_range = 10
    y_range = 10
    x = np.linspace(-x_range, x_range, nx)
    y = np.linspace(-y_range, y_range, ny)
    # xv, yv = np.meshgrid(x, y)

    rho = 1
    T = 2

    dx = 2 * x_range / nx
    dy = 2 * y_range / ny

    # 1D
    rec_rho = 0.0
    rec_U = 0.0
    rec_T = 0.0

    for x_i in x:
        rec_rho += maxwellian1D(x_i, rho, T) * dx
        rec_U += x_i * maxwellian1D(x_i, rho, T) * dx
        rec_T += x_i ** 2 * maxwellian1D(x_i, rho, T) * dx
    print(rec_rho)
    print(rec_U)
    print(rec_T)

    # 2D
    print("2D")
    rec_rho = 0.0
    rec_U = 0.0
    rec_T = 0.0

    pts = []
    kin = []
    for x_i in x:
        for y_i in y:
            pts.append([x_i, y_i])
            kin.append(maxwellian2D(x_i, y_i, rho, T))
            rec_rho += maxwellian2D(x_i, y_i, rho, T) * dx * dy
            rec_U += x_i * maxwellian2D(x_i, y_i, rho, T) * dx * dy
            rec_T += (x_i ** 2 + y_i ** 2) / 2 * maxwellian2D(x_i, y_i, rho, T) * dx * dy
    print(rec_rho)
    print(rec_U)
    print(rec_T)

    pts = np.asarray(pts)
    kin = np.asarray(kin)
    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    out = plt.scatter(pts[:, 0], pts[:, 1], c=kin)
    cbar = fig.colorbar(out, ax=ax, extend='both')

    plt.show()

    # 2D
    print("2D- custom")
    alpha_0 = -1.837877
    alpha_2 = -0.50000
    rec_rho = 0.0
    rec_U = 0.0
    rec_T = 0.0

    for x_i in x:
        for y_i in y:
            rec_rho += custom_maxwellian(x_i, y_i, alpha_0, alpha_2) * dx * dy
            rec_U += x_i * custom_maxwellian(x_i, y_i, alpha_0, alpha_2) * dx * dy
            rec_T += (x_i ** 2 + y_i ** 2) / 2 * custom_maxwellian(x_i, y_i, alpha_0, alpha_2) * dx * dy
    print(rec_rho)
    print(rec_U)
    print(rec_T)

    return True


if __name__ == '__main__':
    main()
