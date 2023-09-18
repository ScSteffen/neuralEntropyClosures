"""
author: Steffen Schotthöfer
date: 12.04.22
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from src.math_utils import EntropyTools
from src.networks.configmodel import init_neural_closure


def main():
    err_list = []

    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.001, basis="spherical_harmonics")
    # 1)  load model

    if degree == 1:
        model = init_neural_closure(network_mk=11, poly_degree=degree, spatial_dim=2, folder_name="tmp",
                                    loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                    input_decorrelation=True, scale_active=False, gamma_lvl=0)
        model.load_model("models/monomial/mk11_M1_2D_g0/")
        alpha_size = 2
    else:
        model = init_neural_closure(network_mk=11, poly_degree=degree, spatial_dim=2, folder_name="tmp",
                                    loss_combination=2, nw_width=350, nw_depth=3, normalized=True,
                                    input_decorrelation=True, scale_active=False, gamma_lvl=2,
                                    basis="spherical_harmonics")
        model.load_model("models/Harmonic_models/Harmonic_Mk11_M2_2D_gamma3/")
        alpha_size = 5

    max_alpha = 2
    for i in range(1000):
        alpha1 = np.random.uniform(low=-max_alpha, high=max_alpha, size=(1, alpha_size))
        alpha2 = np.random.uniform(low=-max_alpha, high=max_alpha, size=(1, alpha_size))
        if degree == 1:
            err_list.append(tester(alpha1, alpha2, et, model, i))
        else:
            err_list.append(tester2D(alpha1, alpha2, et, model, i))

        # print(i)

    c = 0
    for err in err_list:
        for elem in err:
            if elem > 0:
                c += 1
    print("number of convex violations:" + str(c))
    return 0


def tester(alpha1: np.ndarray, alpha2: np.ndarray, et2, mk11_m2_2d_g0, count):
    n_iterpolators = 10
    # Construct two moments
    alpha_orig = tf.constant(alpha1, dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig1 = et2.reconstruct_u(alpha_orig).numpy()
    alpha_orig = tf.constant(alpha2, dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig2 = et2.reconstruct_u(alpha_orig).numpy()

    # Sampling using a grid

    # sample convex interpolation between the moments
    lambdas = np.linspace(0, 1, n_iterpolators)
    u_batch = np.zeros(shape=(n_iterpolators, 3))
    for i in range(n_iterpolators):
        l = lambdas[i]
        u_batch[i, :] = l * u_orig1 + (1 - l) * u_orig2

        plt.plot(u_batch[i, 1], u_batch[i, 2], 'ro')

        # plt.plot(lambdas, h_r)
        # plt.plot(lambdas, h_l)
        # plt.legend(["average", "right", "left"])
        # plt.show()

    # u_batch[0, :] =u_orig1
    # u_batch[-1, :] =u_orig2

    # Rotate+evaluate
    # h_res, alpha_res, u_rot_batch = rotate_evaluate_M1_network(u_batch, mk11_m2_2d_g0)
    h_res, alpha_res, u_rot_batch = rotate_compute_M1_network(u_batch, mk11_m2_2d_g0, et=et2)

    errs = []
    for i in range(n_iterpolators):
        if h_res[i] > lambdas[i] * h_res[n_iterpolators - 1] + (1 - lambdas[i]) * h_res[0]:  # and h_res[i]< 20.0:
            errs.append(1)
            print("error in img " + str(i) + "|" + str(count))
            print("current function value " + str(h_res[i]))
            print("left function value " + str(h_res[0]))
            print("right function value " + str(h_res[n_iterpolators - 1]))
            print("lambda value " + str(lambdas[i]))
            print("rhs value " + str(lambdas[i] * h_res[n_iterpolators - 1] + (1 - lambdas[i]) * h_res[0]))
            print("----")
            u_temp = u_batch[:, 1:]
            for i in range(n_iterpolators):
                plt.plot(u_temp[i, 0], u_temp[i, 1], 'ro', label="pre rot")
                plt.plot(u_rot_batch[i, 0], u_rot_batch[i, 1], 'k*', label="post rot")
            # plt.xlim([-2, 2])
            # plt.ylim([-2, 2])
            plt.xlabel("$u_1$")
            plt.ylabel("$u_2$")
            plt.legend(["pre rotation", "post rotation"])
            plt.savefig("test_imgs_M1/test_" + str(count) + "_u_2.png")

            plt.clf()

            plt.plot(lambdas, h_res, 'o-')
            # plt.plot(lambdas, h_r)
            # plt.plot(lambdas, h_l)
            # plt.legend(["average", "right", "left"])
            # plt.show()

            plt.xlabel("$\lambda$")
            plt.ylabel("$h(u(\lambda))$")

            plt.savefig("test_imgs_M1/test_" + str(count) + ".png")
            plt.clf()
        else:
            errs.append(0)
    return errs


def tester2D(alpha1: np.ndarray, alpha2: np.ndarray, et2, mk11_m2_2d_g0, count):
    n_iterpolators = 20
    # Construct two moments
    # et2 = EntropyTools(polynomial_degree=2, spatial_dimension=2)
    alpha_orig = tf.constant(alpha1, dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig1 = et2.reconstruct_u(alpha_orig).numpy()
    # u_orig1  =np.asarray([1,-1,-1,0.5,0.5,0.25])

    # alpha_orig1 = alpha_orig.numpy()
    alpha_orig = tf.constant(alpha2, dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig2 = et2.reconstruct_u(alpha_orig).numpy()
    # alpha_orig2 = alpha_orig.numpy()
    # u_orig2  =np.asarray([1,1,1,0.5,0.5,0.25])

    # sample convex interpolation between the moments
    lambdas = np.linspace(0, 1, n_iterpolators)
    u_batch = np.zeros(shape=(n_iterpolators, 6))
    for i in range(n_iterpolators):
        l = lambdas[i]
        u_batch[i, :] = l * u_orig1 + (1 - l) * u_orig2

    # h_res, alpha_res, u_rot_batch = rotate_evaluate_M2_network(u_batch, mk11_m2_2d_g0, et=et2)
    h_res, alpha_res, u_rot_batch, h_res_nr = rotate_compute_M2_network(u_batch, et=et2)
    # h_res, alpha_res, u_rot_batch = rotate_evaluate_M2_network_2(u_batch, mk11_m2_2d_g0,et=et2)

    # reconstruct u.... to check prediction
    # alpha = tf.constant(alpha_res, dtype=tf.float64)
    # alpha = et2.reconstruct_alpha(alpha)
    # u_res = et2.reconstruct_u(alpha).numpy()
    #
    # recons_errs = np.linalg.norm(u_res-u_batch,axis =0)
    # for i in range(len(recons_errs)):
    #    if recons_errs[i] > 1e-1:
    #        print("error in reconstruction")
    #        print(u_res[i])
    #        print(u_batch[i])

    errs = []
    for i in range(n_iterpolators):
        if h_res[i] > lambdas[i] * h_res[- 1] + (1 - lambdas[i]) * h_res[0]:  # and h_res[i]< 20.0:
            errs.append(1)
            print("error in img " + str(i) + "|" + str(count))
            print("current function value " + str(h_res[i]))
            print("left function value " + str(h_res[0]))
            print("right function value " + str(h_res[n_iterpolators - 1]))
            print("lambda value " + str(lambdas[i]))
            print("rhs value " + str(lambdas[i] * h_res[n_iterpolators - 1] + (1 - lambdas[i]) * h_res[0]))
            print("----")
            u_temp = u_batch[:, 1:]
            for i in range(n_iterpolators):
                plt.plot(u_temp[i, 0], u_temp[i, 1], 'ro', label="pre rot")
                plt.plot(u_rot_batch[i, 0], u_rot_batch[i, 1], 'k*', label="post rot")
            # plt.xlim([-2, 2])
            # plt.ylim([-2, 2])
            plt.xlabel("$u_1$")
            plt.ylabel("$u_2$")
            plt.legend(["pre rotation", "post rotation"])
            plt.savefig("test_imgs_M2/test_" + str(count) + "_u_2.png")

            plt.clf()

            for i in range(n_iterpolators):
                plt.plot(u_temp[i, 0], u_temp[i, 2], 'ro', label="pre rot")
                plt.plot(u_rot_batch[i, 0], u_rot_batch[i, 2], 'k*')
            # plt.xlim([-2, 2])
            # plt.ylim([-2, 2])
            plt.xlabel("$u_1$")
            plt.ylabel("$u_3$")
            plt.legend(["pre rotation", "post rotation"])

            plt.savefig("test_imgs_M2/test_" + str(count) + "_u_3.png")

            plt.clf()

            for i in range(n_iterpolators):
                plt.plot(u_temp[i, 0], u_temp[i, 3], 'ro')
                plt.plot(u_rot_batch[i, 0], u_rot_batch[i, 3], 'k*')
            # plt.xlim([-2, 2])
            # plt.ylim([-2, 2])
            plt.xlabel("$u_1$")
            plt.ylabel("$u_3$")
            plt.legend(["pre rotation", "post rotation"])

            plt.savefig("test_imgs_M2/test_" + str(count) + "_u_4.png")

            plt.clf()

            for i in range(n_iterpolators):
                plt.plot(u_temp[i, 0], u_temp[i, 4], 'ro')
                plt.plot(u_rot_batch[i, 0], u_rot_batch[i, 4], 'k*')
            # plt.xlim([-2, 2])
            # plt.ylim([-2, 2])
            plt.xlabel("$u_1$")
            plt.ylabel("$u_5$")
            plt.legend(["pre rotation", "post rotation"])

            plt.savefig("test_imgs_M2/test_" + str(count) + "_u_5.png")

            plt.clf()

            plt.plot(lambdas, h_res, 'o-')
            # plt.plot(lambdas, h_r)
            # plt.plot(lambdas, h_l)
            # plt.legend(["average", "right", "left"])
            # plt.show()

            plt.xlabel("$\lambda$")
            plt.ylabel("$h(u(\lambda))$")

            plt.savefig("test_imgs_M2/test_" + str(count) + ".png")
            plt.clf()
        else:
            errs.append(0)

    # err = np.abs(u_res[:, 1:] - u_not_rotated)
    return errs


def rotate_evaluate_M1_network(u_batch: np.ndarray, mk11_m2_2d_g0):
    """
    brief: Rotates to x-line, evaluates, rotates back
    """

    u_rot_batch = np.zeros(shape=(u_batch.shape[0], u_batch.shape[1] - 1))
    u_mirr_batch = np.zeros(shape=(u_batch.shape[0], u_batch.shape[1] - 1))

    G_list = []
    G_mirror = -np.eye(N=2)

    # 2) Rotate all tensors to v_x line in u_1
    for i in range(u_batch.shape[0]):
        u_orig_1 = u_batch[i, 1:3]
        # Rotate to x-axis
        G = create_rotator(u_orig_1)
        G_list.append(G)
        u_rot_1 = rotate_m1(u_orig_1, G)
        u_rot_batch[i, :] = u_rot_1
        # Rotate by 180 degrees to mirror on origin
        u_mirr_1 = rotate_m1(u_rot_1, G_mirror)
        u_mirr_batch[i, :] = u_mirr_1

    # 3) Evaluatate model
    [h_pred, alpha_pred, _] = mk11_m2_2d_g0.model(tf.constant(u_rot_batch))
    [h_pred_mir, alpha_pred_mir, _] = mk11_m2_2d_g0.model(tf.constant(u_mirr_batch))
    alpha_pred_mir = alpha_pred_mir.numpy()
    alpha_pred = alpha_pred.numpy()
    # 4) rotate alpha of odd moments (back) by 180 degrees (even moments ignore the -eye rotation matrix)
    alpha_pred_mir[:, :2] = - alpha_pred_mir[:, :2]
    # 5)  Average
    h_res = (h_pred_mir + h_pred) / 2
    h_res = h_res.numpy()
    alpha_res = (alpha_pred + alpha_pred_mir) / 2

    # 4) Rotate alpha_res back to original position
    for i in range(u_batch.shape[0]):
        alpha_res_1 = alpha_res[i, :2]
        alpha_res_rot_1 = G_list[i].transpose() @ alpha_res_1
        alpha_res[i, :] = alpha_res_rot_1

    return h_res, alpha_res, u_rot_batch


def rotate_compute_M1_network(u_batch: np.ndarray, mk11_m2_2d_g0, et):
    """
    brief: Rotates to x-line, evaluates, rotates back
    """

    u_rot_batch = np.ones(shape=(u_batch.shape[0], u_batch.shape[1]))
    alpha_res = np.zeros(shape=(u_batch.shape[0], u_batch.shape[1]))
    G_list = []

    # 2) Rotate all tensors to v_x line in u_1
    for i in range(u_batch.shape[0]):
        u_orig_1 = u_batch[i, 1:3]
        # Rotate to x-axis
        G = create_rotator(u_orig_1)
        G_list.append(G)
        u_rot_1 = G @ u_batch[i, :]
        u_rot_batch[i, :] = u_rot_1

    # 3) Evaluatate model
    for i in range(u_batch.shape[0]):
        alpha_res[i] = et.minimize_entropy(u_rot_batch[i], alpha_res[i])
    h_res = et.compute_h(u_rot_batch, alpha_res).numpy()

    return h_res, alpha_res, u_rot_batch


def rotate_evaluate_M2_network_2(u_batch: np.ndarray, mk11_m2_2d_g0, et):
    # alpha_res = np.zeros(u_batch.shape)
    #
    # for i in range(u_batch.shape[0]):
    #    alpha_res[i] = et.minimize_entropy(u_batch[i],u_batch[i])
    # h_res = et.compute_h(u_batch,alpha_res).numpy()

    [h_pred, alpha_pred, _] = mk11_m2_2d_g0.model(tf.constant(u_batch[:, 1:], dtype=tf.float32))

    h_res = h_pred.numpy()
    alpha_res = alpha_pred.numpy()

    return h_res, alpha_res, u_batch[:, 1:]


def rotate_evaluate_M2_network(u_batch: np.ndarray, mk11_m2_2d_g0, et):
    """
    brief: Rotates to x-line, evaluates, rotates back
    """

    u_rot_batch = np.zeros(shape=(u_batch.shape[0], u_batch.shape[1] - 1))
    u_mirr_batch = np.zeros(shape=(u_batch.shape[0], u_batch.shape[1] - 1))

    G_list = []
    G_mirror, _ = create_rotator_2D(np.asarray([-1, 0]))
    # 2) Rotate all tensors to v_x line in u_1
    for i in range(u_batch.shape[0]):
        u_orig_1 = u_batch[i, 1:3]
        G, _ = create_rotator_2D(u_orig_1)
        G_list.append(G)

        u_rot_batch[i, :] = G @ u_batch[i][1:]
        # print(u_rot_batch[i,:])
        # print(u_batch[i,1:])
        # print("---")
        # if np.linalg.norm(u_batch[i][1:2]) - np.linalg.norm(u_rot_batch[i][1:2])>1e-5:
        #    print("error")
        #    print(u_rot_batch[i,:])
        #    print(u_batch[i][1:])
        #    print( np.linalg.norm(u_batch[i][1:]) - np.linalg.norm(u_rot_batch[i][1:]))
        #    print(G)
        #    print(np.linalg.norm(G, "fro"))
        #    print(G[:2,:2])
        #    print(np.linalg.norm(G[:2,:2], "fro"))
        #    for i in range(5):
        #        print(G[i,:])
        #
        #        print(np.linalg.norm(G[i,:]))
        #    exit()
        u_mirr_batch[i, :] = G_mirror @ u_rot_batch[i, :]

        # 3) Evaluatate model

    [h_pred, alpha_pred, _] = mk11_m2_2d_g0.model(tf.constant(u_rot_batch, dtype=tf.float32))
    [h_pred_mir, alpha_pred_mir, _] = mk11_m2_2d_g0.model(tf.constant(u_mirr_batch, dtype=tf.float32))
    alpha_pred_mir = alpha_pred_mir.numpy()
    alpha_pred = alpha_pred.numpy()
    # 4) rotate alpha of odd moments (back) by 180 degrees (even moments ignore the -eye rotation matrix)
    # for i in range(u_batch.shape[0]):
    #    alpha_pred_mir[i, :] = G_mirror @ alpha_pred_mir[i, :]
    # 5)  Average
    # h_res = (h_pred_mir + h_pred) / 2
    # h_res = h_res.numpy()
    alpha_res = alpha_pred  # (alpha_pred + alpha_pred_mir) / 2
    h_res = (h_pred_mir + h_pred) / 2
    # 4) Rotate alpha_res back to original position
    # for i in range(u_batch.shape[0]):
    #     alpha_res[i, :] = G_list[i].transpose() @ alpha_res[i, :]
    # alpha_res_full = et.reconstruct_alpha(tf.constant(alpha_res,dtype=tf.float64))
    # h_res = et.compute_h(tf.constant(u_batch,dtype=tf.float64),alpha_res_full).numpy()
    return h_res, alpha_res, u_rot_batch


def rotate_compute_M2_network(u_batch: np.ndarray, et):
    """
    brief: Rotates to x-line, evaluates, rotates back
    """

    u_rot_batch = np.ones(shape=(u_batch.shape[0], u_batch.shape[1]))
    u_mirr_batch = np.ones(shape=(u_batch.shape[0], u_batch.shape[1]))

    G_list = []
    _, G_mirror = create_rotator_2D(np.asarray([-1, 0]))
    # 2) Rotate all tensors to v_x line in u_1
    for i in range(u_batch.shape[0]):
        u_orig_1 = u_batch[i, 1:3]
        _, G = create_rotator_2D(u_orig_1)
        G_list.append(G)

        u_rot_batch[i, :] = G @ u_batch[i]

        u_mirr_batch[i, :] = G_mirror @ u_rot_batch[i, :]

        # with np.printoptions(precision=3, suppress=True):
    #    print(u_batch)
    #    print(u_rot_batch)
    # 3) Evaluatate model

    # [h_pred, alpha_pred, _] = mk11_m2_2d_g0.model(tf.constant(u_rot_batch, dtype=tf.float32))
    # [h_pred_mir, alpha_pred_mir, _] = mk11_m2_2d_g0.model(tf.constant(u_mirr_batch, dtype=tf.float32))

    alpha_res = np.zeros(u_rot_batch.shape)
    alpha_res_nr = np.zeros(u_rot_batch.shape)
    h_res = np.zeros(u_batch.shape[0])
    h_res_nr = np.zeros(u_batch.shape[0])

    for i in range(u_batch.shape[0]):
        et.reset_basis()

        alpha_res_nr[i] = et.minimize_entropy(u_batch[i], alpha_res_nr[i])
        et.rotate_basis(G_list[i])
        alpha_res[i] = et.minimize_entropy(u_rot_batch[i], alpha_res[i])
        h_res[i] = et.compute_h(u_rot_batch[i].reshape(1, 6), alpha_res[i].reshape(1, 6)).numpy()
        et.reset_basis()
        # compute non rotated u, too

    # h_res = et.compute_h(u_rot_batch, alpha_res).numpy()
    h_res_nr = et.compute_h(u_batch, alpha_res_nr).numpy()

    if np.linalg.norm(h_res - h_res_nr) > 0.0:
        print(np.linalg.norm(h_res - h_res_nr))

    return h_res, alpha_res, u_rot_batch[:, 1:], h_res_nr


def create_rotator(u_1_in) -> np.ndarray:
    theta = np.arctan2(u_1_in[0], u_1_in[1]) - np.pi / 2.0;
    c = np.cos(theta)
    s = np.sin(theta)

    G = np.zeros((3, 3))
    G[0, 0] = 1
    G[1, 1] = c
    G[2, 2] = c
    G[1, 2] = -s
    G[2, 1] = s
    return G


def create_rotator_2D(u_1_in) -> np.ndarray:
    theta = np.arctan2(u_1_in[0], u_1_in[1]) - np.pi / 2.0;
    c = np.cos(theta)
    s = np.sin(theta)
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)

    G = np.zeros((6, 6))
    G[0, 0] = 1
    G[1, 1] = c
    G[2, 2] = c
    G[1, 2] = -s
    G[2, 1] = s

    G[3, 3] = c2
    G[4, 4] = 1.0
    G[5, 5] = c2
    G[3, 5] = s2
    G[5, 3] = -s2
    # print(G)
    return G[1:, 1:], G


def rotate_m1(vec_orig_1, G) -> np.ndarray:
    """
    params: vec_orig_1: dims = (2,)
            G: dims = (2,2)
    """
    return np.matmul(G, vec_orig_1)


if __name__ == '__main__':
    main()
