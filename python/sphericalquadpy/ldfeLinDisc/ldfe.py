from time import time

from numpy import sqrt, linspace, array, ones, \
    zeros, eye, transpose
from numpy.linalg import solve, norm
from numpy.random import rand
from scipy.integrate import dblquad
from scipy.optimize import least_squares, dual_annealing
from numba import jit
from sphericalquadpy.ldfeLinDisc.helper.miscellaneous import project, EPSILON
from sphericalquadpy.ldfeLinDisc.helper.sphericaltrigonometry import s2area
from sphericalquadpy.ldfeLinDisc.vistools import scatterplot


def ldfe(n=3):
    """ Return the quadrature points `p`
    and the corresponding quadrature weights `w`
    for the LDFE quadrature of order `n`.
    Introduced by Lau & Adams in 2016. """

    # We will use the following coordinate system.
    #
    #       | z, top
    #       |
    #       |
    #       |
    #       o-------     x, right
    #      /
    #     /
    #    /
    #   / y, front

    # Cube inside the octant that touches the sphere at
    a = 1 / sqrt(3)

    # We have three important faces of the cube.
    # Start with the front face and refine it in N segments.
    x = linspace(0, a, n + 1)
    z = linspace(0, a, n + 1)

    # Then delta Omega_ij =  [x_i,x_i+1] x [z_j,z_j+1]
    # Now go through every cell.
    points = zeros((1 * 1 * 4 * n * n, 3))  # 1/3 of the octants
    weights = zeros(1 * 1 * 4 * n * n)
    square = zeros(1 * 1 * 4 * n * n)
    counter = 0
    rhos0 = 0.1 * ones(4)
    for i in range(n):
        for j in range(n):
            x0, x1, z0, z1 = x[i], x[i + 1], z[j], z[j + 1]

            omegas = computeomegas(x0, x1, z0, z1)
            areas = computeareas(omegas, x0, x1, z0, z1)
            print("\n\nOptimiztation for:")
            print("Domain:")
            print([x0, x1, z0, z1])

            rhos = optimizeposition_leastsquares(areas, omegas, x0, x1, z0, z1,
                                                 rhos0)
            rhos0 = rhos  # take the optimal parameter of this cell as the starting value for the optimizer in the next cell
            dummy = rand()
            for k in range(4):
                points[counter, :] = project(omegas[k](rhos[k]))
                weights[counter] = areas[k]
                square[counter] = dummy
                counter += 1
    scatterplot(points, weights, square)
    return points, weights


def optimizeposition_leastsquares(areas, omegas, x0, x1, z0, z1, rhos0):
    def tomin(r):
        y = f(r, omegas, 1 / sqrt(3), x0, x1, z0, z1, areas)
        # print(norm(y))
        return norm(y)

    res = least_squares(tomin, rhos0, bounds=((0, 1)))
    print("Optimal rho:")
    print(res.x)
    print("areas:")
    print(areas)
    print("weights:")
    print(computeweights(res.x, omegas, 1 / sqrt(3), x0, x1, z0, z1))

    return res.x


def optimizeposition(areas, omegas, x0, x1, z0, z1):
    """ Solves for the optimal positions of the four
    quadrature points inside one cell, such that
    the associated weights equal the areas of the
    associated quadrilaterals.
    The associated weight is given by integrating
    the associated (bi)linear basis function over
    the domain [x0,x1]x[z0,z1].
    The position of the quadrature points are defined via
    `rhos = [rho0,rho1,rho2,rho3]` and the relation
    that quadpoint_i = (1-rho_i) * midpoint + rho_i * corner_i.
    omegai are lambdas that linearly interpolate between the midpoint
    and the corresponding corner point with omega[i](0) = corneri
    and omega[i](1) = midpoint.
    """

    # initial position of each quadpoint is at the center
    # of the edge connecting the midpoint and a corner point
    rhos = 0.5 * ones(4)
    a = 1 / sqrt(3)
    deltarhos = 0.25 * ones(4)  # delta for finite differences

    while True:  # while method has not converged
        # print("################## new iteration #############")
        rhs = f(rhos, omegas, a, x0, x1, z0, z1, areas)
        print("##")
        print(rhs)
        print(rhos)
        if norm(rhs) < 1e-5:
            break
        mat = df(rhos, omegas, a, x0, x1, z0, z1, areas, deltarhos)
        update = solve(mat, rhs)

        rhos += update
        # for i in range(4):
        #    rhos[i] = max(0,min(1,rhos[i]))
        """
        print("the norm of the rhs is ")
        print(norm(rhs))
        print(mat)
        print("rhs")
        print(rhs)
        print(update)
        print("rhos")
        print(rhos)
        """
        # print(alpha)
    return rhos


def df(rhos, omegas, a, x0, x1, z0, z1, areas, deltarhos):
    weights = f(rhos, omegas, a, x0, x1, z0, z1, areas)
    dweights = zeros((4, 4))
    # print("the weights are")
    # print(computeweights(rhos, omegas, a, x0, x1, z0, z1))
    for j in range(4):
        rhos[j] += deltarhos[j]
        x = f(rhos, omegas, a, x0, x1, z0, z1, areas)
        y = (x - weights) / deltarhos[j]
        dweights[:, j] = y
        rhos[j] -= deltarhos[j]
    return dweights


# @jit
def f(rhos, omegas, a, x0, x1, z0, z1, areas):
    y = areas - computeweights(rhos, omegas, a, x0, x1, z0, z1)
    return y


#    return 1 / 2 * sum(y ** 2)

# @jit
def computeweights(rhos, omegas, a, x0, x1, z0, z1):
    """ Computes the corresponding basis functions given rho.
    From that, we compute the integral of every basis function
    over the domain [x0,x1] x [z0,z1]. The weights are returned
    as a vector. """
    # print(rhos)
    mat = ones((4, 4))
    for i in range(4):
        mat[i, 1:] = project(omegas[i](rhos[i]))
    # print(mat)
    c = solve(mat, eye(4))
    weights = zeros(4)

    # compute the weights for integrating every basis function over the domain
    for i in range(4):
        # define the i-th basis function
        def bi(x, z):
            r = sqrt(x ** 2 + a ** 2 + z ** 2)
            return (c[0, i]
                    + c[1, i] * x / r
                    + c[2, i] * a / r
                    + c[3, i] * z / r
                    ) * abs(a ** 1 / r ** 3)

        val, err = dblquad(bi, z0, z1, x0, x1)
        weights[i] = val
    return weights


# @jit
def computeomegas(x0, x1, z0, z1):
    a = 1 / sqrt(3)
    # compute cell midpoint
    mid = array([x1 + x0, 2 * a, z1 + z0]) / 2

    #
    #         3 ---------------  0
    #           |\           / |        /\ z
    #           |  \       /   |        |
    #           |    \   /     |        |
    #           |      x       |        |
    #           |    /  \      |        |
    #           |  /      \    |        |
    #         2 |/          \  | 1      -------------->x
    #           ----------------
    # See Eq. 33, we define the directions on the diagonal lines
    # in dependency of rho = d/L

    def omega0(rho): return rho * array([x1, a, z1]) + (1 - rho) * mid

    def omega1(rho): return rho * array([x1, a, z0]) + (1 - rho) * mid

    def omega2(rho): return rho * array([x0, a, z0]) + (1 - rho) * mid

    def omega3(rho): return rho * array([x0, a, z1]) + (1 - rho) * mid

    return [omega0, omega1, omega2, omega3]


# @jit
def computeareas(omegas, x0, x1, z0, z1):
    a = 1 / sqrt(3)
    # compute cell midpoint
    mid = array([x1 + x0, 2 * a, z1 + z0]) / 2

    # compute area on sphere
    #           ---------------
    #           |      |       |
    #           |  a3  |   a0  |
    #           |      |       |
    #           |--------------|
    #           |      |       |
    #           |  a2  |   a1  |
    #           |      |       |
    #           ----------------
    # compute area ai as sum of two triangles
    omega0, omega1, omega2, omega3 = omegas

    a0 = (s2area(mid, omega0(1), (array([(x1 + x0) / 2, a, z1])))
          + s2area(mid, omega0(1), (array([x1, a, (z0 + z1) / 2]))))

    a1 = (s2area(mid, omega1(1), (array([(x1 + x0) / 2, a, z0])))
          + s2area(mid, omega1(1), (array([x1, a, (z0 + z1) / 2]))))

    a2 = (s2area(mid, omega2(1), (array([(x1 + x0) / 2, a, z0])))
          + s2area(mid, omega2(1), (array([x0, a, (z0 + z1) / 2]))))

    a3 = (s2area(mid, omega3(1), (array([(x1 + x0) / 2, a, z1])))
          + s2area(mid, omega3(1), (array([x0, a, (z0 + z1) / 2]))))

    # check if the four cells at up to the full cell
    areacell = (s2area(omega0(1), omega1(1), omega2(1))
                + s2area(omega0(1), omega2(1), omega3(1)))
    assert abs(areacell - a0 - a1 - a2 - a3) < EPSILON

    return [a0, a1, a2, a3]


if __name__ == '__main__':
    t = time()
    ldfe()
    print("Finished in {:2.4f} seconds.".format(time() - t))
