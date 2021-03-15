from numba import jit
from numpy import zeros, double


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
