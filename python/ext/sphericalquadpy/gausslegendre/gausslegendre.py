"""GaussLegendre quadrature."""
from numpy import pi, inf, zeros, sqrt, cos, sin
from numpy.polynomial.legendre import leggauss
from sphericalquadpy.quadrature.quadrature import Quadrature


class GaussLegendre(Quadrature):
    """GaussLegendre Quadrature"""

    def name(self):
        return "GaussLegendre Quadrature"

    def getmaximalorder(self):
        return inf

    def computequadpoints(self, order):
        """Quadrature points for GaussLegendre quadrature. Read from file."""
        mu, _ = leggauss(order)
        phi = [pi * (k + 1 / 2) / order for k in range(2 * order)]
        xyz = zeros((2 * order * order, 3))
        count = 0
        for i in range(order):
            for j in range(2 * order):
                mui = mu[i]
                phij = phi[j]
                xyz[count, 0] = sqrt(1 - mui ** 2) * cos(phij)
                xyz[count, 1] = sqrt(1 - mui ** 2) * sin(phij)
                xyz[count, 2] = mui
                count += 1

        return xyz

    def computequadweights(self, order):
        """Quadrature weights for GaussLegendre quadrature. Read from file."""
        _, leggaussweights = leggauss(order)
        w = zeros(2 * order * order)
        count = 0
        for i in range(order):
            for j in range(2 * order):
                w[count] = 2 * pi / order * leggaussweights[i]
                count += 1

        w /= sum(w)
        w *= 4 * pi
        return w

    def nqbyorder(self, order):
        """Scales quadratically"""
        return order, 2 * order ** 2
