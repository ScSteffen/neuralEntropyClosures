"""
brief: driver for adaptive samples
author: Steffen Schotthöfer
"""

from src.sampler.adaptiveSampler import adaptiveSampler

import numpy as np
import matplotlib.pyplot as plt
from src.utils import loadData


def testFunc(x):
    # quadratic function
    return 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)


def gradFunc(x):
    # gard of testFunc
    return x


def main():
    """
    # sample some test points
    x = np.asarray([[0.1, 0.1], [0.13, 0.1], [0.16, 0.1], [0.2, 0.1],
                    [0.1, 0.13], [0.13, 0.13], [0.16, 0.13], [0.2, 0.13],
                    [0.1, 0.16], [0.13, 0.16], [0.16, 0.16], [0.2, 0.16],
                    [0.1, 0.2], [0.13, 0.2], [0.16, 0.2], [0.2, 0.2]])
    y = testFunc(x)
    grads = gradFunc(x)

    poi = np.asarray([0.101, 0.1401])
    poiGrad = gradFunc(poi)

    # plt.plot(x[:3, 0], x[:3, 1], '*')
    # plt.plot(poi[0], poi[1], '+')
    # plt.show()
    samplerTest = adaptiveSampler(len(x), 2, x, grads)
    res = samplerTest.compute_a(poi)
    resVert = samplerTest.vertices[res]
    t0 = samplerTest.vertices[:, :, 0]
    t1 = samplerTest.vertices[:, :, 1]
    diam = samplerTest.compute_diam_a()
    print(diam)

    allVertices = samplerTest.get_vertices()
    plt.plot(allVertices[:, 0], allVertices[:, 1], '*')
    plt.plot(resVert[:, 0], resVert[:, 1], 'o')
    plt.plot(poiGrad[0], poiGrad[1], '+')
    plt.show()

    # a = samplerTest.vertices_used
    # print(a)

    """

    # load sampled data
    [u, alpha, h] = loadData("data/1D/Monomial_M2_1D_normal_alpha.csv", 3, [True, True, True])
    u_normal = u[:, 1:]
    alpha_normal = alpha[:, 1:]
    poi = np.asarray([0.9, 0.9])
    samplerTest = adaptiveSampler(u_normal, alpha_normal, knn_param=3)
    res = samplerTest.compute_a(poi)
    resVert = samplerTest.vertices[res]
    diam = samplerTest.compute_diam_a()
    print(diam)
    allVertices = samplerTest.get_vertices()
    plt.plot(allVertices[:, 0], allVertices[:, 1], '*')
    plt.plot(resVert[:, 0], resVert[:, 1], 'o')
    # plt.plot(poiGrad[0], poiGrad[1], '+')
    plt.show()

    return 0


if __name__ == '__main__':
    main()
