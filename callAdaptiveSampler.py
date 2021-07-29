"""
brief: driver for adaptive samples
author: Steffen Schotthöfer
"""

from src.sampler.adaptiveSampler import adaptiveSampler

import numpy as np
import matplotlib.pyplot as plt
from src.utils import loadData, scatterPlot2D
from src.math import EntropyTools


def test_func(x):
    # quadratic function
    return 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)


def grad_func(x):
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
    """
    # generate poi
    entropy_tools = EntropyTools(N=2)
    poi_grad = np.asarray([10, 10])
    alpha_part = np.asarray([poi_grad])
    alpha_recons = entropy_tools.reconstruct_alpha(entropy_tools.convert_to_tensorf(alpha_part))
    u = entropy_tools.reconstruct_u(alpha_recons)
    poi = u[0, 1:].numpy()
    print("Point of interest: " + str(poi))
    print("With gradient: " + str(poi_grad))
    # load sampled data
    [u, alpha, h] = loadData("data/1D/Monomial_M2_1D_normal_alpha_big.csv", 3, [True, True, True])
    u_normal = u[:, 1:]
    alpha_normal = alpha[:, 1:]
    scatterPlot2D(x_in=u_normal, y_in=h, folder_name="delete", show_fig=False, log=False)
    # Query max error bound
    sampler_test = adaptiveSampler(u_normal, alpha_normal, knn_param=10)
    res = sampler_test.compute_a(poi)
    res_vert = sampler_test.vertices[res]
    diam = sampler_test.compute_diam_a()
    print(diam)
    allVertices = sampler_test.get_vertices()
    plt.plot(allVertices[:, 0], allVertices[:, 1], '*')
    plt.plot(res_vert[:, 0], res_vert[:, 1], 'o')
    plt.plot(poi_grad[0], poi_grad[1], '+')
    plt.show()
    """
    ### ---- Start here
    # reference data
    [u, alpha, h] = loadData("data/1D/Monomial_M2_1D_normal_alpha_big.csv", 3, [True, True, True])
    u_normal = u[:, 1:]
    alpha_normal = alpha[:, 1:]
    sampler_test = adaptiveSampler(u_normal, alpha_normal, knn_param=10)

    # Load query data
    [u_query, alpha_query, h] = loadData("data/1D/Monomial_M2_1D_normal.csv", 3, [True, True, True])
    u_query_normal = u_query[:, 1:]
    diams = np.zeros(h.shape)
    for i in range(1000, 1001):  # len(u_query_normal)):
        poi = u_query_normal[i]
        success = sampler_test.compute_a(poi)
        if success:
            diam = sampler_test.compute_diam_a()
            sampler_test.sample_adative(poi, max_diam=0.05, max_iter=10)
        else:
            diam = np.nan
        print(str(i) + str("/") + str(len(u_query_normal)) + ". Diam = " + str(diam))
        diams[i] = diam

    scatterPlot2D(x_in=u_query_normal, y_in=diams, name="test_grid", folder_name="delete", show_fig=False,
                  log=True)
    return 0


if __name__ == '__main__':
    main()
