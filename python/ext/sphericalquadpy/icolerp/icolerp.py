"""Icolerp quadrature."""
from numpy import pi, loadtxt
from sphericalquadpy.quadrature.quadrature import Quadrature
from sphericalquadpy.tools.findnearest import find_nearest
import os

AVAILABLEORDERS = [2 * (i + 1) for i in range(20)]

NUMBERQUADPOINTS = [10 * n ** 2 - 20 * n + 12 for _, n in enumerate(AVAILABLEORDERS)]


class Icolerp(Quadrature):
    """Icolerp Quadrature"""

    def name(self):
        return "Icolerp Quadrature"

    def getmaximalorder(self):
        return 40

    def computequadpoints(self, order):
        """Quadrature points for icolerp quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be" "%i.",
                AVAILABLEORDERS[neighbor],
            )
        filename = "data/" + str(order) + "_l_ico.txt"
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        path = os.path.join(__location__, filename)
        xyzw = loadtxt(path, delimiter="\t")
        return xyzw[:, 0:3]

    def computequadweights(self, order):
        """Quadrature weights for icolerp quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be" "%i.",
                AVAILABLEORDERS[neighbor],
            )

        filename = "data/" + str(order) + "_l_ico.txt"
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        path = os.path.join(__location__, filename)
        xyzw = loadtxt(path, delimiter="\t")
        w = xyzw[:, 3]
        w /= sum(w)
        w *= 4 * pi
        return w

    def nqbyorder(self, order):
        """Scaling was derived from files in data/"""
        idx = find_nearest(AVAILABLEORDERS, order)
        return AVAILABLEORDERS[idx], NUMBERQUADPOINTS[idx]
