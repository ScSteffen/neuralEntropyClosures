"""Dummy quadrature."""
from numpy import pi, inf, zeros, sqrt, cos, sin
from sphericalquadpy.quadrature.quadrature import Quadrature


class Dummy(Quadrature):
    """Dummy Quadrature"""

    def name(self):
        return "Dummy Quadrature"

    def getmaximalorder(self):
        return inf

    def computequadpoints(self, order):
        """Quadrature points for Dummy quadrature. Read from file."""
        xyz = 1
        return xyz

    def computequadweights(self, order):
        """Quadrature weights for Dummy quadrature. Read from file."""
        w = 1

        return w

    def nqbyorder(self, order):
        """Scales in the following way"""
        nq = order
        return order, nq
