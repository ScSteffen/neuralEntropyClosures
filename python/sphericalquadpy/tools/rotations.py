"""Rotating cartesian points that live on the unit sphere
around a random or specified axis by a random or specified amount"""
from numpy.random import rand, randn
from numpy import zeros, cos, sin, pi, multiply, matmul
from numpy.linalg import norm


# pylint: disable=C0103


def rotate(axis, angle, xyz):
    """Rotates xyz around axis by angle."""
    rot = rotationmatrix(axis, angle)
    if len(xyz.shape) == 1:  # just one point
        if not len(xyz) == 3:
            raise ValueError("Points have to either be of shape (3,) or (n,3). ")
        return multiply(rot, xyz)
    if len(xyz.shape) == 2:
        # we assume xyz to be of shape n x 3
        if not xyz.shape[1] == 3:
            raise ValueError("Points have to either be of shape (3,) or (n,3). ")

        return matmul(rot, xyz.T).T

    raise ValueError("Points have to either be of shape (3,) or (n,3). ")


def randomaxisrotate(angle, xyz):
    """Rotates xyz around a random axis by angle."""

    # get random point on unit sphere
    axis = randn(3)
    axis = axis / norm(axis)
    return rotate(axis, angle, xyz)


def randomanglerotate(axis, xyz):
    """Rotates xyz around axis by a random angle."""
    angle = 2 * pi * rand()
    return rotate(axis, angle, xyz)


def randomrotate(xyz):
    """Rotates xyz around a random axis by a random angle."""
    # get random point on unit sphere
    axis = randn(3)
    axis = axis / norm(axis)
    angle = 2 * pi * rand()
    return rotate(axis, angle, xyz)


def rotationmatrix(axis, angle):
    """Returns the rotation matrix of size 3x3 that rotates a point
    around axis by angle.
    See: https://en.wikipedia.org/wiki/Rotation_matrix
    """
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    costheta = cos(angle)
    sintheta = sin(angle)
    rot = zeros((3, 3))

    rot[0, 0] = ux * ux * (1 - costheta) + costheta
    rot[0, 1] = ux * uy * (1 - costheta) - uz * sintheta
    rot[0, 2] = ux * uz * (1 - costheta) + uy * sintheta

    rot[1, 0] = uy * ux * (1 - costheta) + uz * sintheta
    rot[1, 1] = uy * uy * (1 - costheta) + costheta
    rot[1, 2] = uy * uz * (1 - costheta) - ux * sintheta

    rot[2, 0] = uz * ux * (1 - costheta) - uy * sintheta
    rot[2, 1] = uz * uy * (1 - costheta) + ux * sintheta
    rot[2, 2] = uz * uz * (1 - costheta) + costheta

    return rot
