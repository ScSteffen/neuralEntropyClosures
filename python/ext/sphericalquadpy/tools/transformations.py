"""Transformations for points on the sphere.
We can transform from cartesian to spherical
or from spherical to cartesian.
Points are assumed to live on the unit sphere.
The notation is consistent with the one found here:
http://mathworld.wolfram.com/SphericalCoordinates.html
"""
# pylint: disable=C0103
# pylint: disable=E1111
import numpy
from numpy import arctan2, arccos, cos, sin, zeros, reshape, sqrt
from numpy.linalg import norm


def cast2matrix(x, dim):
    """Handles different cases of the input x that might happen to
    be given when xyz2thetaphi or thetaphi2xyz is called.
    These functions require the input to be an (n,dim) matrix (with dim=2 or 3
    depending on which direction).
    Sometimes the user calls it with a vector of (dim,) which will fail.
    Some other weird formats have occured before and we will handle them
    and convert everything (when possible) into a (n,dim) matrix.
    """
    if type(x) == list:  # cast list to numpy array and then call cas2matrix
        return cast2matrix(numpy.array(x), dim)

    if type(x) == numpy.ndarray:
        if len(x.shape) == 1:  # vector to 1x3 matrix
            return reshape(x, (1, dim))
        if len(x.shape) == 2:
            if x.shape[1] == dim:  # is n x 3
                return x
            if x.shape[0] == dim:  # has to be transposed since it is 3 x n
                return x.T
        if len(x.shape) > 2:
            raise ValueError("Numpy ndarrays as input have to be vectors or matrices.")


def xyz2thetaphi(xyz):
    """Transformation from points on the unit sphere given by their
    cartesian representation as (x,y,z) to their
    spherical representation as (theta,phi),
    with the azimuthal angle theta in [0,2pi) and
    the polar angle phi in [0,pi].

    Args:
        xyz: An (n,3) numpy.ndarray of cartesian points living on the unit
        sphere.
    Raises:
        ValueError: If any point does not live on the unit sphere.
        ValueError: If not exactly three points per row are given.
    Returns:
        thetaphi: An (n,2) numpy.ndarray of spherical points living on the unit
        sphere.
    """
    xyz = cast2matrix(xyz, 3)

    n, dim = xyz.shape
    thetaphi = zeros((n, 2))
    for i in range(n):
        x, y, z = xyz[i, :]
        r = x ** 2 + y ** 2 + z ** 2
        if not abs(r - 1.0) < 1.0e-6:
            raise ValueError(
                "Point %i does not live on the unit sphere. "
                "The coordinates are (%d,%d,%d) wit norm %d." % (i, x, y, z, sqrt(r))
            )
        thetaphi[i, 0] = arctan2(xyz[i, 1], xyz[i, 0])
        thetaphi[i, 1] = arccos(xyz[i, 2])
    return thetaphi


def thetaphi2xyz(thetaphi):
    """Transformation from points on the unit sphere given by their
    spherical representation as (theta,phi) to their
    cartesian representation as (x,y,z),
    with the azimuthal angle theta in [0,2pi) and
    the polar angle phi in [0,pi].

    Args:
        thetaphi: An (n,2) numpy.ndarray of spherical points living on the unit
        sphere.
    Raises:
        ValueError: If not exactly two points per row are given.
    Returns:
        xyz: An (n,3) numpy.ndarray of cartesian points living on the unit
        sphere.
    """
    thetaphi = cast2matrix(thetaphi, 2)
    n, dim = thetaphi.shape

    xyz = zeros((n, 3))
    for i in range(n):
        theta, phi = thetaphi[i, :]
        xyz[i, 0] = cos(theta) * sin(phi)
        xyz[i, 1] = sin(theta) * sin(phi)
        xyz[i, 2] = cos(phi)
    return xyz
