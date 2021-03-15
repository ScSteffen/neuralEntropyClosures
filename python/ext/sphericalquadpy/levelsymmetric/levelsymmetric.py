"""Levelsymmetric quadrature"""
from numpy import pi, loadtxt
from sphericalquadpy.quadrature.quadrature import Quadrature
from sphericalquadpy.tools.findnearest import find_nearest
from sphericalquadpy.levelsymmetric.writtendict import levelsymmetricdictionary
import os

AVAILABLEORDERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
NUMBERQUADPOINTS = [8, 24, 48, 80, 120, 168, 224, 288, 360, 432]


# NUMBERQUADPOINTS = zeros(len(AVAILABLEORDERS), dtype=int)
# for i, order in enumerate(AVAILABLEORDERS):
#    tmp = loadtxt("data/" + str(order) + "_levelsym.txt", delimiter=",")
#    NUMBERQUADPOINTS[i] = tmp.shape[0]


class Levelsymmetric(Quadrature):
    """Levelsymmetric Quadrature"""

    def name(self):
        return "Levelsymmetric Quadrature"

    def getmaximalorder(self):
        return 20

    def computequadpoints(self, order):
        """Quadrature points for Levelsymmetric quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be" "%i.",
                AVAILABLEORDERS[neighbor],
            )
        filename = "data/" + str(order) + "_levelsym.txt"
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        path = os.path.join(__location__, filename)
        xyzw = loadtxt(path, delimiter=",")
        # d = levelsymmetricdictionary()
        # xyzw = d[order]
        return xyzw[:, 0:3]

    def computequadweights(self, order):
        """Quadrature weights for Levelsymmetric quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be" "%i.",
                AVAILABLEORDERS[neighbor],
            )
        filename = "data/" + str(order) + "_levelsym.txt"
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        path = os.path.join(__location__, filename)
        xyzw = loadtxt(path, delimiter=",")
        # d = levelsymmetricdictionary()
        # xyzw = d[order]
        w = xyzw[:, 3]
        w /= sum(w)
        w *= 4 * pi
        return w

    def nqbyorder(self, order):
        """Scaling was derived from files in data/"""
        idx = find_nearest(AVAILABLEORDERS, order)
        return AVAILABLEORDERS[idx], NUMBERQUADPOINTS[idx]
