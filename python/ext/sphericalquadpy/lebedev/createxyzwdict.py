"""This file can be used to create a new python file that will return
the dictionary of Lebedev quadratures."""
import numpy as np
import sys


def createdict():
    """Create a dictionary based on the quadrature files stored in data/"""
    orders = [
        3,
        5,
        7,
        9,
        11,
        13,
        15,
        17,
        19,
        21,
        23,
        25,
        27,
        29,
        31,
        35,
        41,
        47,
        53,
        59,
        65,
        71,
        77,
        83,
        89,
        95,
        101,
        107,
        113,
        119,
        125,
        131,
    ]

    D = dict()
    for order in orders:
        xyzw = np.loadtxt("data/" + str(order) + "_lebedev.txt", delimiter=",")
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
            "def lebedevdictionary():\n"
            "\treturn (" + str(d) + ")"
        )

        print(mystring, file=f)
