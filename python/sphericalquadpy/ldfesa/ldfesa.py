"""LDFESA quadrature."""
from numpy import pi
from sphericalquadpy.quadrature.quadrature import Quadrature
from sphericalquadpy.tools.findnearest import find_nearest
from sphericalquadpy.ldfesa.writtendict import ldfesadictionary

AVAILABLEORDERS = [1, 2, 3]

NUMBERQUADPOINTS = [32, 128, 512]


# NUMBERQUADPOINTS = zeros(len(AVAILABLEORDERS), dtype=int)
# for i, order in enumerate(AVAILABLEORDERS):
#    tmp = loadtxt("data/"+str(order) + "_ldfesa.txt", delimiter=",")
#    NUMBERQUADPOINTS[i] = tmp.shape[0]


class LDFESA(Quadrature):
    """LDFESA Quadrature"""

    def name(self):
        return "LDFESA Quadrature"

    def getmaximalorder(self):
        return 3

    def computequadpoints(self, order):
        """Quadrature points for LDFESA quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be" "%i.",
                AVAILABLEORDERS[neighbor],
            )

        d = ldfesadictionary()
        xyzw = d[order]
        return xyzw[:, 0:3]

    def computequadweights(self, order):
        """Quadrature weights for LDFESA quadrature. Read from file."""
        if order not in AVAILABLEORDERS:
            neighbor = find_nearest(AVAILABLEORDERS, order)
            raise ValueError(
                "Order not available. Next closest would be %i. You chose %i",
                AVAILABLEORDERS[neighbor],
                order,
            )

        d = ldfesadictionary()
        xyzw = d[order]
        w = xyzw[:, 3]
        w /= sum(w)
        w *= 4 * pi
        return w

    def nqbyorder(self, order):
        """Scaling was derived from files in data/"""
        idx = find_nearest(AVAILABLEORDERS, order)
        return AVAILABLEORDERS[idx], NUMBERQUADPOINTS[idx]
