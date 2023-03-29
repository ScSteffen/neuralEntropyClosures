# brief script to illustrate spherical harmonics
import scipy

from matplotlib.ticker import MultipleLocator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# The following import configures Matplotlib for 3D plotting.
from mpl_toolkits.mplot3d import Axes3D

# from matplotlib import rc

from src import math


def main():
    plot_family_Ys(max_degree_l=4, res=100)
    # max_l = 2
    # for l in range(0, max_l + 1):
    # for k in range(-l, l + 1):
    #        plot_sh(80, l, k)
    return 0


def plot_sh(res: int, order_l: int, degree_k: int):
    # resolution - increase this to get a smoother plot
    #  at the cost of slower processing
    N = res

    theta = np.linspace(0, 2 * np.pi, N)
    phi = np.linspace(0, np.pi, N)
    theta, phi = np.meshgrid(theta, phi)

    m = degree_k
    n = order_l

    Yvals = scipy.special.sph_harm(abs(m), n, theta, phi, out=None)

    # Linear combination of Y_l,m and Y_l,-m to create the real form.
    if m < 0:
        Yvals = np.sqrt(2) * (-1) ** m * Yvals.imag
    elif m > 0:
        Yvals = np.sqrt(2) * (-1) ** m * Yvals.real
    elif m == 0:
        Yvals = Yvals.real

    # Make some adjustments for nice plots
    Ymax, Ymin = Yvals.max(), Yvals.min()
    if (Ymax != Ymin):
        # normalize the values to [1, -1]
        Yvals = 2 * (Yvals - Ymin) / (Ymax - Ymin) - 1
        # Use the absolute value of Y(l,m) as the radius
        radii = np.abs(Yvals)
        # put the colors in the range [1, 0]
        Ycolors = 0.5 * (Yvals + 1)
    else:
        # can't normalize b/c Y(0,0) is single-valued
        radii = np.ones(Yvals.shape)
        Ycolors = np.ones(Yvals.shape)

    # Compute Cartesian coordinates of the surface
    x = radii * np.sin(theta) * np.cos(phi)
    y = radii * np.sin(theta) * np.sin(phi)
    z = radii * np.cos(theta)

    # Do the actual plotting
    # negative values will be blue, positive red
    fig = plt.figure(figsize=plt.figaspect(1.))

    ax = fig.add_subplot(111, projection='3d')
    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(x, y, z, facecolors=cmap.to_rgba(Yvals.real),
                    rstride=1, cstride=1)
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.coolwarm(Ycolors))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)

    # Get rid of colored axes planes
    # First remove fill
    ax.grid(False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4
    ax.zaxis._axinfo['tick']['outward_factor'] = 0.4

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.zaxis.set_major_locator(MultipleLocator(0.5))

    ax.view_init(elev=15, azim=45)
    # Now set color to white (or whatever is "invisible")
    # ax.xaxis.pane.set_edgecolor('w')
    # ax.yaxis.pane.set_edgecolor('w')
    # ax.zaxis.pane.set_edgecolor('w')

    # Bonus: To get rid of the grid as well:
    # ax.grid(True)
    # plt.axis('off')
    plt.tight_layout()
    plt.savefig("spherical_harmonics_" + str(n) + "_" + str(m) + ".png", dpi=400)
    print("Figure successfully saved to file: spherical_harmonics_" + str(n) + "_" + str(m) + ".png")
    # plt.show()
    return 0


def plot_family_Ys(max_degree_l=3, res=100):
    figsize_px, DPI = 2000, 1000
    figsize_in = 9
    fig = plt.figure(figsize=(figsize_in, figsize_in), dpi=DPI)

    max_cols = int(math.getCurrDegreeSize(max_degree_l, 3))
    spec = gridspec.GridSpec(ncols=max_cols, nrows=max_degree_l + 1, figure=fig)
    for idx_degree in range(0, max_degree_l + 1):
        count = 0
        print("----")
        print(idx_degree)
        for a in range(0, idx_degree + 1):
            for b in range(0, idx_degree - a + 1):
                c = idx_degree - a - b
                ax = fig.add_subplot(spec[idx_degree, count], projection='3d')
                plot_monomial(ax, a, b, c, res)
                print("a:" + str(a) + ", b:" + str(b) + ", c:" + str(c))
                count += 1

    # plt.tight_layout()
    # plt.subplots_adjust(left=-5, right=-5, bottom=-5, top=-5)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    plt.savefig('monomials.png')
    # plt.show()
    return 0


def plot_monomial(ax, a, b, c, res):
    # Create grid
    # Grids of polar and azimuthal angles
    theta = np.linspace(0, np.pi, res)
    phi = np.linspace(0, 2 * np.pi, res)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([np.sin(theta) * np.sin(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta)])

    monomial_basis = np.power(xyz[0], a) * np.power(xyz[1], b) * np.power(xyz[2], c)
    monomial_basis_x, monomial_basis_y, monomial_basis_z = abs(monomial_basis) * xyz

    """Plot the spherical harmonic of degree el and order m on Axes ax."""

    # NB In SciPy's sph_harm function the azimuthal coordinate, theta,
    # comes before the polar coordinate, phi.
    # Y = scipy.special.sph_harm(abs(m), el, phi, theta)

    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('seismic'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(monomial_basis_x, monomial_basis_y, monomial_basis_z,
                    facecolors=cmap.to_rgba(monomial_basis),
                    rstride=1, cstride=1)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    # ax.plot([-ax_lim, ax_lim], [0, 0], [0, 0], c='0.5', lw=1, zorder=10)
    # ax.plot([0, 0], [-ax_lim, ax_lim], [0, 0], c='0.5', lw=1, zorder=10)
    # ax.plot([0, 0], [0, 0], [-ax_lim, ax_lim], c='0.5', lw=1, zorder=10)
    # Set the Axes limits and title, turn off the Axes frame.
    # ax.set_title(r"$Y_{" + str(el) + "}^{" + str(m) + "}$", y=-0.1)
    ax_lim = 0.75
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')
    # ax.tight_layout()

    # plt.tight_layout()
    # plt.savefig("spherical_harmonics_" + str(el) + "_" + str(m) + ".png", dpi=400)
    # print("Figure successfully saved to file: spherical_harmonics_" + str(el) + "_" + str(m) + ".png")

    return 0


if __name__ == '__main__':
    main()