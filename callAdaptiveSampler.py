"""
brief: driver for adaptive samples
author: Steffen Schotthöfer
"""

import os
import numpy as np
from multiprocessing import Pool
from functools import partial
import csv
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils import load_data, scatter_plot_2d
from src.sampler.adaptiveSampler import AdaptiveSampler
from src.math import EntropyTools


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
        # if count % 100 == 0:
        print("Poi count: " + str(count) + "/" + str(len(pois)) + ". Diam: " + str(diam_list[count - 1]))
    print("count Bad: " + str(count_bad))
    print("count Good: " + str(count_good))
    diams: np.ndarray = np.asarray(diam_list)
    # print(diams)
    return diams


def compute_diams_relative(pois_n_pois_grads, points, grads, knn_param):
    pois = pois_n_pois_grads[0]
    pois_grads = pois_n_pois_grads[1]

    adap_sampler = AdaptiveSampler(points, grads, knn_param)
    # print(len(pois))
    diam_list: list = []
    count = 0
    count_good = 0
    count_bad = 0
    for poi_idx in range(len(pois)):
        # for poi in pois:
        vertices, success = adap_sampler.compute_a_wrapper(pois[poi_idx])
        # --- Preprocess Vertices ---
        if success:
            diam_list.append(adap_sampler.compute_diam_a(vertices) / (np.linalg.norm(pois_grads[poi_idx]) + 0.00001))
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


def main2():
    et = EntropyTools(2)
    alpha = [[1.0, 1.1], [1.0, 1.0], [1.1, 1.1]]
    alpha_c = et.reconstruct_alpha(tf.constant(alpha))
    u = et.reconstruct_u(alpha_c)
    u_poi = (u[0] + u[1] + u[2]) / 3
    u_poi = u_poi[1:].numpy()
    alpha_normal = np.asarray(alpha)
    u_normal = u[:, 1:].numpy()
    ada_sampler = AdaptiveSampler(points=u_normal, grads=alpha_normal, knn_param=3)
    vertices, success = ada_sampler.compute_a_wrapper(u_poi)

    plt.plot(vertices[:, 0], vertices[:, 1], '--')
    plt.plot(alpha_normal[:, 0], alpha_normal[:, 1], '+')
    plt.show()

    return 0


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    [u, alpha, h] = load_data("data/1D/Monomial_M2_1D_normal_alpha.csv", 3, [True, True, True])
    lim_x_in = (-1, 1)  # (-0.5, -0.35)
    lim_y_in = (0, 1)  # (0.2, 0.3)

    def preprocess(u_p, a_p, h_p, lim_x, lim_y):
        keep_idx = []
        for idx in range(len(u_p)):
            if lim_x[0] <= u_p[idx, 1] <= lim_x[1] and lim_y[0] <= u_p[idx, 2] <= lim_y[1]:
                keep_idx.append(idx)
        return u_p[keep_idx, :], a_p[keep_idx, :], h_p[keep_idx, :]

    u, alpha, h = preprocess(u, alpha, h, lim_x=lim_x_in, lim_y=lim_y_in)
    t = u[:, 1:]
    scatter_plot_2d(x_in=t, z_in=h, lim_x=lim_x_in, lim_y=lim_y_in, lim_z=(-1, 5),
                    title=r"diam($h$) over ${\mathcal{R}^r}$",
                    folder_name="delete", name="test", show_fig=False, log=False)

    [u_pois, alpha_pois, h_pois] = load_data("data/1D/pois_M2.csv", 3, [True, True, True])
    u_pois, alpha_pois, h_pois = preprocess(u_pois, alpha_pois, h_pois, lim_x=lim_x_in, lim_y=lim_y_in)
    # u_pois = u_pois[:1000, :]
    u_normal = u[:, 1:]
    alpha_normal = alpha[:, 1:]
    pois = u_pois[:, 1:]
    pois_grads = alpha_pois[:, 1:]

    # ada_sampler = AdaptiveSampler(points=u_normal, grads=alpha_normal, knn_param=20)
    # pois = ada_sampler.compute_pois()

    process_count = 24
    # split pois across processes
    chunk: int = int(len(pois) / process_count)
    pois_chunk = []
    pois_n_grads_chunk = []
    for i in range(process_count - 1):
        pois_chunk.append(pois[i * chunk:(i + 1) * chunk])
        pois_n_grads_chunk.append([pois[i * chunk:(i + 1) * chunk], pois_grads[i * chunk:(i + 1) * chunk]])
    pois_chunk.append(pois[(process_count - 1) * chunk:])
    pois_n_grads_chunk.append([pois[(process_count - 1) * chunk:], pois_grads[(process_count - 1) * chunk:]])
    with Pool(process_count) as p:
        # diams_chunk = p.map(partial(compute_diams, points=u_normal, grads=alpha_normal, knn_param=20),
        #                    pois_chunk)
        diams_chunk = p.map(partial(compute_diams_relative, points=u_normal, grads=alpha_normal, knn_param=20),
                            pois_n_grads_chunk)

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

    # (x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
    #
    #                     name: str = 'defaultName', log: bool = True, folder_name: str = "figures",
    #                     show_fig: bool = False, ):
    scatter_plot_2d(x_in=pois, z_in=diams, lim_x=lim_x_in, lim_y=lim_y_in, lim_z=(0.01, 10),
                    title=r"diam($A_{x^*}$)$/|\alpha|$ over ${\mathcal{R}^r}$",
                    folder_name="delete", name="diam_A_relative_alpha", show_fig=False, log=True)

    return 0


if __name__ == '__main__':
    # main2()
    main()
