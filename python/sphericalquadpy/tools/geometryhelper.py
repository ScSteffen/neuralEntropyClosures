from numba import jit
from numpy import cross, pi, cos, arccos
from numpy import linspace, outer, dot, arctan, sin, double, zeros
from numpy.linalg import norm

EPSILON = 1e-12


# @jit
def cross(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    # taken from https://gist.github.com/ufechner7/98bcd6d9915ff4660a10
    result = zeros(3)
    return cross_(vec1, vec2, result)


# @jit(nopython=True)
def cross_(vec1, vec2, result):
    """ Calculate the cross product of two 3d vectors. """
    # taken from https://gist.github.com/ufechner7/98bcd6d9915ff4660a10
    a1, a2, a3 = double(vec1[0]), double(vec1[1]), double(vec1[2])
    b1, b2, b3 = double(vec2[0]), double(vec2[1]), double(vec2[2])
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


# @jit(nopython=True)
def project(x):
    """Projects the point x onto the unit sphere. """
    return x / norm(x)


# @jit(nopython=True)
def lerp(pointa, pointb, n):
    """ Linear interpolation between two points. """
    t = linspace(0, 1, n)
    return outer(pointa, t) + outer(pointb, 1 - t)


# @jit(nopython=True)
def slerp(pointa, pointb, n):
    """ Spherical linear interpolation between
    two points. """

    omega = arctan(norm(cross(pointa, pointb)) / dot(pointa, pointb))
    t = linspace(0, 1, n)

    return outer(pointa, sin((1 - t) * omega) / sin(omega)) + outer(
        pointb, sin((t) * omega) / sin(omega)
    )


# @jit(nopython=True)
def distance(pointa, pointb):
    """ Returns the spherical distance between
    two points. """

    ra, rb = norm(pointa), norm(pointb)
    return ra * arccos(dot(pointa, pointb) / ra ** 2)


# @jit(nopython=True)
def angle(pointb, pointa, pointc):
    """ Returns the spherical angle between the lines
     pointb<->pointa and pointa<->pointc
     https://en.wikipedia.org/wiki/Spherical_trigonometry#Cosine_rules_and_sine_rules ."""

    c = distance(pointb, pointa)
    b = distance(pointa, pointc)
    a = distance(pointc, pointb)
    cosangle = (cos(a) - cos(b) * cos(c)) / (sin(b) * sin(c))
    return arccos(cosangle)


def s2area(pointa, pointb, pointc):
    """ Projects the three points onto the S2 unit sphere and then computes
    the area via the function below"""
    ra, rb, rc = norm(pointa), norm(pointb), norm(pointc)
    return area(pointa / ra, pointb / rb, pointc / rc)


# @jit(nopython=True)
def area(pointa, pointb, pointc):
    """" Returns the spherical area of the triangle
    spanned by the three points, see:
    https://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess . """

    ra, rb, rc = norm(pointa), norm(pointb), norm(pointc)

    pointa /= ra
    pointb /= rb
    pointc /= rc
    # project onto unit sphere

    alpha = angle(pointb, pointa, pointc)
    beta = angle(pointc, pointb, pointa)
    gamma = angle(pointa, pointc, pointb)
    return (alpha + beta + gamma - pi) * ra ** 2


def eightfold(pts):
    """Takes points that live on one octant and duplicates them"""
    _, n = pts.shape
    allpts = zeros((3, 8 * n))

    allpts[0, 0 * n : 1 * n] = +pts[0, :]
    allpts[1, 0 * n : 1 * n] = +pts[1, :]
    allpts[2, 0 * n : 1 * n] = +pts[2, :]

    allpts[0, 1 * n : 2 * n] = +pts[0, :]
    allpts[1, 1 * n : 2 * n] = -pts[1, :]
    allpts[2, 1 * n : 2 * n] = +pts[2, :]

    allpts[0, 2 * n : 3 * n] = +pts[0, :]
    allpts[1, 2 * n : 3 * n] = +pts[1, :]
    allpts[2, 2 * n : 3 * n] = -pts[2, :]

    allpts[0, 3 * n : 4 * n] = +pts[0, :]
    allpts[1, 3 * n : 4 * n] = -pts[1, :]
    allpts[2, 3 * n : 4 * n] = -pts[2, :]

    # duplicate upper to lower
    allpts[0, 4 * n :] = -allpts[0, : 4 * n]
    allpts[1, 4 * n :] = +allpts[1, : 4 * n]
    allpts[2, 4 * n :] = +allpts[2, : 4 * n]

    return allpts
