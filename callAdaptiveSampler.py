"""
brief: driver for adaptive samples
author: Steffen Schotthöfer
"""

import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import csv

from src.utils import load_data, scatterPlot2D
from src.sampler.adaptiveSampler import AdaptiveSampler


def test_func(x):
    # quadratic function
    return 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)


def grad_func(x):
    # gard of testFunc
    return x


def compute_diams(pois, points, grads, knn_param):
    adap_sampler = AdaptiveSampler(points, grads, knn_param)
    # print(len(pois))
    diam_list: list = []
    count = 0
    count_good = 0
    count_bad = 0
    for poi in pois:
        # for poi in pois:
        vertices, success = adap_sampler.compute_a_wrapper(poi)
        # --- Preprocess Vertices ---
        if success:
            diam_list.append(adap_sampler.compute_diam_a(vertices))
            count_good += 1
        else:
            count_bad += 1
            diam_list.append(np.nan)
        count += 1
        if count % 100 == 0:
            print("Poi count: " + str(count) + "/" + str(len(pois)) + ". Diam: " + str(diam_list[count - 1]))
    print("count Bad: " + str(count_bad))
    print("count Good: " + str(count_good))
    diams: np.ndarray = np.asarray(diam_list)
    # print(diams)
    return diams


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    """
    # sample some test points
    # x = np.asarray([[0.1, 0.1], [0.13, 0.1], [0.1, 0.13], [0.13, 0.13]])

    x = np.asarray([[0.1, 0.1], [0.13, 0.1], [0.16, 0.1], [0.2, 0.1],
                    [0.1, 0.13], [0.13, 0.13], [0.16, 0.13], [0.2, 0.13],
                    [0.1, 0.16], [0.13, 0.16], [0.16, 0.16], [0.2, 0.16],
                    [0.1, 0.2], [0.13, 0.2], [0.16, 0.2], [0.2, 0.2]])

    y = test_func(x)
    grads = grad_func(x)
    cloud_knn = 10

    poi = np.asarray([0.111, 0.1101])
    poiGrad = grad_func(poi)

    # plt.plot(x[:, 0], x[:, 1], '*')
    # plt.plot(poi[0], poi[1], '+')
    # plt.show()
    s_t = adaptiveSampler(points=x, grads=grads, local_cloud_size=cloud_knn)
    s_t.compute_a(poi)
    resVert = s_t.get_used_vertices()

    t0 = s_t.local_vertices[:, :, 0]
    t1 = s_t.local_vertices[:, :, 1]
    diam = s_t.compute_diam_a()
    print(diam)

    for i in range(0, cloud_knn):
        l_0 = s_t.compute_boundary(i)
    plt.plot(l_0[:, 0], l_0[:, 1], '--')

    allVertices = s_t.get_all_vertices()
    plt.plot(allVertices[:, 0], allVertices[:, 1], '*')
    plt.plot(resVert[:, 0], resVert[:, 1], 'o')
    plt.plot(poiGrad[0], poiGrad[1], '+')
    plt.show()
    """
    """
    # a = samplerTest.vertices_used
    # print(a)
    # generate poi

    entropy_tools = EntropyTools(N=2)
    poi_grad = np.asarray([0.2, 0.3])
    alpha_part = np.asarray([poi_grad])
    alpha_recons = entropy_tools.reconstruct_alpha(entropy_tools.convert_to_tensor_float(alpha_part))
    u = entropy_tools.reconstruct_u(alpha_recons)
    poi = u[0, 1:].numpy()
    print("Point of interest: " + str(poi))
    print("With gradient: " + str(poi_grad))
    # load sampled data
    """

    [u, alpha, h] = load_data("data/1D/Monomial_M2_1D_normal_alpha_big.csv", 3, [True, True, True])
    u_normal = u[:, 1:]
    alpha_normal = alpha[:, 1:]

    # scatterPlot2D(x_in=u_normal, y_in=h, folder_name="delete", show_fig=False, log=False)
    # Query max error bound
    """
    sampler_test.compute_a(poi)
    res_vert = sampler_test.get_used_vertices()
    diam = sampler_test.compute_diam_a(sampler_test.get_used_vertices())
    print(diam)
    allVertices = sampler_test.get_vertices()
    # plt.plot(allVertices[:, 0], allVertices[:, 1], '*')
    plt.plot(res_vert[:, 0], res_vert[:, 1], 'o')
    plt.plot(poi_grad[0], poi_grad[1], '+')
    plt.show()
    """

    ada_sampler = AdaptiveSampler(points=u_normal, grads=alpha_normal, knn_param=20)
    pois = ada_sampler.compute_pois()

    process_count = 24
    # split pois across processes
    chunk: int = int(len(pois) / process_count)
    pois_chunk = []
    for i in range(process_count - 1):
        pois_chunk.append(pois[i * chunk:(i + 1) * chunk])
    pois_chunk.append(pois[(process_count - 1) * chunk:])
    with Pool(process_count) as p:
        diams_chunk = p.map(partial(compute_diams, points=u_normal, grads=alpha_normal, knn_param=15), pois_chunk)

    # merge the computed chunks
    diams: np.ndarray = np.zeros(len(pois))
    # print(diams.shape)
    count = 0
    for diam in diams_chunk:
        for d in diam:
            diams[count] = d
            count += 1
    print(len(diams))
    # print(diams)

    with open('diameters.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(pois[:, 0])
        writer.writerow(pois[:, 0])
        writer.writerow(diams)

    scatterPlot2D(x_in=pois, y_in=diams, folder_name="delete", name="triangle_diameter_at_query_point", show_fig=False,
                  log=True,
                  z_lim=100)

    # sampler_test.sample_adative(poi, max_diam=1e-5, max_iter=100, poi_grad=poi_grad)

    ### ---- Start here

    """
    # reference data
    [u, alpha, h] = load_data("data/1D/Monomial_M2_1D_normal_alpha_big.csv", 3, [True, True, True])
    u_normal = u[:, 1:]
    alpha_normal = alpha[:, 1:]
    sampler_test = adaptiveSampler(u_normal, alpha_normal, knn_param=10)

    # Load query data
    [u_query, alpha_query, h] = load_data("data/1D/Monomial_M2_1D_normal.csv", 3, [True, True, True])
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

    """

    return 0


if __name__ == '__main__':
    main()
