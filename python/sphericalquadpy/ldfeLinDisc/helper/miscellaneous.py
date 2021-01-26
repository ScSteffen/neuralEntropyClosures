from numba import jit
from numpy import linspace, outer, dot, arctan, sin, double, zeros
from numpy.linalg import norm


EPSILON = 1e-12


@jit
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    # taken from https://gist.github.com/ufechner7/98bcd6d9915ff4660a10
    result = zeros(3)
    return cross_(vec1, vec2, result)


@jit(nopython=True)
def cross_(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. """
    # taken from https://gist.github.com/ufechner7/98bcd6d9915ff4660a10
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


@jit(nopython=True)
def project(x):
    """Projects the point x onto the unit sphere. """
    return x / norm(x)


@jit(nopython=True)
def lerp(pointa, pointb, n):
    """ Linear interpolation between two points. """
    t = linspace(0, 1, n)
    return outer(pointa, t) + outer(pointb, 1 - t)


@jit(nopython=True)
def slerp(pointa, pointb, n):
    """ Spherical linear interpolation between
    two points. """

    omega = arctan(norm(cross(pointa, pointb))
                   / dot(pointa, pointb))
    t = linspace(0, 1, n)

    return (outer(pointa, sin(1 - t) * omega / sin(omega))
            + outer(pointb, sin(t) * omega / sin(omega)))
