import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import cm
import seaborn as sns


def main():
    x = np.linspace(-1.5, 1.5, 5)
    y = np.linspace(-1.5, 1.5, 5)

    z = 1.5 + np.cos(2 * np.pi * y[:, np.newaxis]) * np.cos(2 * np.pi * x)

    # --- plot ---

    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=400)

    c_map = cm.hot

    # filled contours
    im = ax.contourf(x, y, z, levels=10, cmap=c_map)

    # contour lines
    im2 = ax.contour(x, y, z, colors='k')

    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(im, ax=ax, pad=0.02)
    plt.savefig("periodic.png", dpi=150)

    return 0


def plot_flowfield(x, y, u_ref, u_neural):
    # --- plot ---
    z1 = np.abs(u_ref[:, :, 0] - u_neural[:, :, 0]) / u_ref[:, :, 0]
    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=400)

    c_map = cm.hot

    # filled contours
    im = ax.contourf(x, y, z1, 100, cmap=c_map)

    # contour lines
    im2 = ax.contour(x, y, z1, colors='k')

    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(im, ax=ax, pad=0.02)
    plt.savefig("periodic_err.png", dpi=150)

    ######
    z1 = u_ref[:, :, 0]
    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=400)

    c_map = cm.hot

    # filled contours
    im = ax.contourf(x, y, z1, levels=10, cmap=c_map)

    # contour lines
    im2 = ax.contour(x, y, z1, colors='k')

    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(im, ax=ax, pad=0.02)
    plt.savefig("periodic_ref.png", dpi=150)

    #####

    z1 = u_neural[:, :, 0]
    fig, ax = plt.subplots(figsize=(5.8, 4.7), dpi=400)

    c_map = cm.hot

    # filled contours
    im = ax.contourf(x, y, z1, levels=10, cmap=c_map)

    # contour lines
    im2 = ax.contour(x, y, z1, colors='k')

    plt.xlabel("x")
    plt.ylabel("y")
    fig.colorbar(im, ax=ax, pad=0.02)
    plt.savefig("periodic_ref.png", dpi=150)

    return 0


if __name__ == '__main__':
    main()
