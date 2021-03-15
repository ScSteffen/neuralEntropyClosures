"""Monte Carlo Quadrature uses random quadrature points and equal
weights to integrate a function."""
from numpy import pi, ones, inf
from numpy.random import randn
from numpy.linalg import norm
from sphericalquadpy.quadrature.quadrature import Quadrature


class MonteCarlo(Quadrature):
    """Monte Carlo Quadrature"""

    def name(self):
        return "MonteCarlo Quadrature"

    def getmaximalorder(self):
        return inf

    def nqbyorder(self, order):
        """For MonteCarlo, order = nq"""
        return order, order

    def computequadpoints(self, order):
        """Random points on the sphere can be generated
        by normalizing normally distributed points on the sphere."""
        xyz = randn(order, 3)
        xyz /= norm(xyz, axis=1)[:, None]
        return xyz

    def computequadweights(self, order):
        """Equal weights"""
        weights = 4 * pi / order * ones(order)
        return weights
