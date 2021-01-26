from numba import jit
from numpy import dot, cos, \
    sin, arccos, pi
from numpy.linalg import norm


@jit(nopython=True)
def distance(pointa, pointb):
    """ Returns the spherical distance between
    two points. """

    ra, rb = norm(pointa), norm(pointb)
    return ra * arccos(dot(pointa, pointb) / ra ** 2)


@jit(nopython=True)
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


@jit(nopython=True)
def area(pointa, pointb, pointc):
    """" Returns the spherical area of the triangle
    spanned by the three points, see:
    https://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess . """

    ra, rb, rc = norm(pointa), norm(pointb), norm(pointc)

    # project onto unit sphere

    alpha = angle(pointb, pointa, pointc)
    beta = angle(pointc, pointb, pointa)
    gamma = angle(pointa, pointc, pointb)
    return (alpha + beta + gamma - pi) * ra ** 2
