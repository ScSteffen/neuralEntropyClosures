"""This file can be used to create a new python file that will return
the dictionary of Levelsymmetric quadratures."""
import numpy as np
import sys


def createdict():
    """Create a dictionary based on the quadrature files stored in data/"""
    orders = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    D = dict()
    for order in orders:
        xyzw = np.loadtxt("data/" + str(order) + "_levelsym.txt", delimiter=",")
        xyzw[:, 3] = xyzw[:, 3] / sum(xyzw[:, 3]) * 4 * np.pi
        D[order] = xyzw

    return D


def writedict():
    """Dump a dictionary to a python file and add a function definition
    first. That way, we can read the dictionary later from that file without
    using the files in data/"""
    d = createdict()
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(precision=15)
    with open("writtendict.py", "w") as f:
        mystring = (
            "from numpy import array\n"
            "def levelsymmetricdictionary():\n"
            "\treturn (" + str(d) + ")"
        )

        print(mystring, file=f)
