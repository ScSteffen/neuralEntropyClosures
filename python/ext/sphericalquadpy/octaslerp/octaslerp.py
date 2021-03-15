"""octaslerp quadrature."""
from numpy import pi, loadtxt
from sphericalquadpy.quadrature.quadrature import Quadrature
from sphericalquadpy.tools.findnearest import find_nearest
import os

AVAILABLEORDERS = [2 * (i + 1) for i in range(40)]

NUMBERQUADPOINTS = [4 * n ** 2 - 8 * n + 6 for _, n in enumerate(AVAILABLEORDERS)]


class Octaslerp(Quadrature):
    """Octaslerp Quadrature"""

    def name(self):
        return "Octaslerp Quadrature"

    def getmaximalorder(self):
        return 80

    def computequadpoints(self, order):
        """Quadrature points for octaslerp quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be" "%i.",
                AVAILABLEORDERS[neighbor],
            )
        filename = "data/" + str(order) + "_s_octa.txt"
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__))
        )
        path = os.path.join(__location__, filename)
        xyzw = loadtxt(path, delimiter="\t")
        return xyzw[:, 0:3]

    def computequadweights(self, order):
        """Quadrature weights for octaslerp quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be" "%i.",
                AVAILABLEORDERS[neighbor],
            )

        filename = "data/" + str(order) + "_s_octa.txt"
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
