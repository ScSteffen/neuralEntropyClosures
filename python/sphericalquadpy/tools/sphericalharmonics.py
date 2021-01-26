"""Redefines the spherical harmonics interface from scipy."""
# pylint: disable=C0103
# pylint: disable=E0611

from numpy import stack
from scipy.special import sph_harm
from sphericalquadpy.tools.transformations import xyz2thetaphi


def ylm(l, m, *args):
    """Spherical harmonics can either be evaluated by specifying
    the input as cartesian or spherical."""
    if len(args) == 2:  # args = [theta,phi]
        return sph_harm(l, m, args[1], args[0])
    if len(args) == 3:  # args = [x,y,z]
        xyz = stack([args[0], args[1], args[2]], axis=1)
        thetaphi = xyz2thetaphi(xyz)
        return sph_harm(l, m, thetaphi[:, 1], thetaphi[:, 0])

    raise ValueError(
        "Spherical Harmonics need either (theta,phi)"
        " or (x,y,z) but not a vector of length %i." % len(args)
    )
