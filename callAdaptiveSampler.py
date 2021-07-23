"""
brief: driver for adaptive samples
author: Steffen Schotthöfer
"""

from src.sampler.adaptiveSampler import adaptiveSampler

import numpy as np
import matplotlib.pyplot as plt


def testFunc(x):
    # quadratic function
    return 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)


def gradFunc(x):
    # gard of testFunc
    return x


def main():
    # sample some test points
    x = np.asarray([[0.1, 0.1], [0.13, 0.1], [0.16, 0.1], [0.2, 0.1],
                    [0.1, 0.13], [0.13, 0.13], [0.16, 0.13], [0.2, 0.13],
                    [0.1, 0.16], [0.13, 0.16], [0.16, 0.16], [0.2, 0.16],
                    [0.1, 0.2], [0.13, 0.2], [0.16, 0.2], [0.2, 0.2]])
    """
    x = np.asarray([[0.1, 0.1], [0.3, 0.3], [0.3, 0.1], [0.2, 0.1],
                    [0.1, 0.3]])
    """
    y = testFunc(x)
    grads = gradFunc(x)

    poi = np.asarray([0.124, 0.11])
    poiGrad = gradFunc(poi)

    # plt.plot(x[:3, 0], x[:3, 1], '*')
    # plt.plot(poi[0], poi[1], '+')
    # plt.show()
    samplerTest = adaptiveSampler(len(x), 2, x, grads)
    res = samplerTest.compute_a(poi, poiGrad)
    resVert = samplerTest.vertices[res]
    allVertices = samplerTest.get_vertices()
    plt.plot(allVertices[:, 0], allVertices[:, 1], '*')
    plt.plot(resVert[:, 0], resVert[:, 1], 'o')
    plt.plot(poiGrad[0], poiGrad[1], '+')
    plt.show()

    # a = samplerTest.vertices_used
    # print(a)
    return 0


if __name__ == '__main__':
    main()
