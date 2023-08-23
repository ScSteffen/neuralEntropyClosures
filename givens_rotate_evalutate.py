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

from src.math import EntropyTools
from src.networks.configmodel import init_neural_closure


def main():
    err_list = []
    
    et = EntropyTools(polynomial_degree=1, spatial_dimension=2)
        # 1)  load model
    model = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=2, folder_name="tmp",
                                        loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                        input_decorrelation=True, scale_active=False, gamma_lvl=0)
    model.load_model("models/monomial/mk11_M1_2D_g0/")
    max_alpha = 20
    for i in range(1000):
        alpha1 = np.random.uniform(low=-max_alpha, high=max_alpha, size=(1, 2))
        alpha2 = np.random.uniform(low=-max_alpha, high=max_alpha, size=(1, 2))
        err_list.append(tester(alpha1, alpha2,et, model, i))
        #print(i)
    
    c = 0
    for err in err_list:
        for elem in err:
            if elem >0:
                c+=1
    print("number of convex violations:" +str(c))
    return 0


def tester(alpha1: np.ndarray, alpha2: np.ndarray, et2,mk11_m2_2d_g0,count):
    n_iterpolators = 10
    # Construct two moments
    #et2 = EntropyTools(polynomial_degree=2, spatial_dimension=2)
    alpha_orig = tf.constant(alpha1, dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig1 = et2.reconstruct_u(alpha_orig).numpy()
    #alpha_orig1 = alpha_orig.numpy()
    alpha_orig = tf.constant(alpha2, dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig2 = et2.reconstruct_u(alpha_orig).numpy()
    #alpha_orig2 = alpha_orig.numpy()

    # sample convex interpolation between the moments
    lambdas = np.linspace(0, 1, n_iterpolators)
    u_batch = np.zeros(shape=(n_iterpolators, 3))
    for i in range(n_iterpolators):
        l = lambdas[i]
        u_batch[i, :] = l * u_orig1 + (1 - l) * u_orig2
    
        plt.plot(u_batch[i, 1], u_batch[i, 2], 'ro')
       

        # plt.plot(lambdas, h_r)
        # plt.plot(lambdas, h_l)
        #plt.legend(["average", "right", "left"])
        #plt.show()
    
    
    #u_batch[0, :] =u_orig1
    #u_batch[-1, :] =u_orig2
    
    # Rotate+evaluate
    h_res, alpha_res, u_rot_batch = rotate_evaluate_M1_network(u_batch, mk11_m2_2d_g0)

    for i in range(n_iterpolators):
        plt.plot(u_rot_batch[i, 0], u_rot_batch[i, 1], 'k*')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.savefig("test_imgs/test_" +str(count) + "_u.png")
    plt.clf()
    
    # reconstruct u.... to check prediction
    alpha = tf.constant(alpha_res, dtype=tf.float64)
    alpha = et2.reconstruct_alpha(alpha)
    u_res = et2.reconstruct_u(alpha).numpy()


    errs = []
    for i in range(n_iterpolators):
        if h_res[i] > lambdas[i]*h_res[n_iterpolators-1] + (1-lambdas[i])*h_res[0]:
            errs.append(1)
            print("error in img "  +str(i)+ "|" + str(count))
            print("current function value " + str(h_res[i]))
            print("left function value " + str(h_res[0]))
            print("right function value " + str(h_res[n_iterpolators-1]))
            print("lambda value " + str(lambdas[i]))
            print("rhs value " +str(lambdas[i]*h_res[n_iterpolators-1] + (1-lambdas[i])*h_res[0]))
            print("----")
        else:
            errs.append(0)
    plt.plot(lambdas, h_res, 'o-')
    # plt.plot(lambdas, h_r)
    # plt.plot(lambdas, h_l)
    #plt.legend(["average", "right", "left"])
    plt.show()
    plt.savefig("test_imgs/test_" +str(count) + ".png")
    plt.clf()
    #err = np.abs(u_res[:, 1:] - u_not_rotated)
    return errs


def rotate_evaluate_M1_network(u_batch: np.ndarray, mk11_m2_2d_g0 ):
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
        alpha_res_rot_1 = back_rotate_m1(alpha_res_1, G_list[i])
        alpha_res[i, :] = alpha_res_rot_1

    return h_res, alpha_res, u_rot_batch

def rotate_evaluate_M2_network(u_batch: np.ndarray,mk11_m2_2d_g0 ):
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
        u_orig_2 = np.asarray([[u_batch[i, 3], u_batch[i, 4]], [u_batch[i, 4], u_batch[i, 5]]])
        # Rotate to x-axis
        G = create_rotator(u_orig_1)
        G_list.append(G)
        u_rot_1 = rotate_m1(u_orig_1, G)
        u_rot_2 = rotate_m2(u_orig_2, G)
        u_rot_batch[i, :] = np.asarray([u_rot_1[0], u_rot_1[1], u_rot_2[0, 0], u_rot_2[1, 0], u_rot_2[1, 1]])
        # Rotate by 180 degrees to mirror on origin
        u_mirr_1 = rotate_m1(u_rot_1, G_mirror)
        u_mirr_2 = rotate_m2(u_rot_2, G_mirror)
        u_mirr_batch[i, :] = np.asarray([u_mirr_1[0], u_mirr_1[1], u_mirr_2[0, 0], u_mirr_2[1, 0], u_mirr_2[1, 1]])
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
        alpha_res_2 = np.asarray([[alpha_res[i, 2], alpha_res[i, 3] * 0.5], [alpha_res[i, 3] * 0.5, alpha_res[i, 4]]])
        alpha_res_rot_1 = back_rotate_m1(alpha_res_1, G_list[i])
        alpha_res_rot_2 = back_rotate_m2(alpha_res_2, G_list[i])
        alpha_res[i, :] = np.asarray(
            [alpha_res_rot_1[0], alpha_res_rot_1[1], alpha_res_rot_2[0, 0], 2 * alpha_res_rot_2[1, 0],
             alpha_res_rot_2[1, 1]])

    return h_res, alpha_res


def rotate_evaluate_M2_entropy(u_batch: np.ndarray) -> tf.Tensor:
    """
    brief: Rotates to x-line, evaluates, rotates back
    """
    # 1)  load model
    mk11_m2_2d_g0 = init_neural_closure(network_mk=11, poly_degree=2, spatial_dim=2, folder_name="tmp",
                                        loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                        input_decorrelation=True, scale_active=False, gamma_lvl=0)
    mk11_m2_2d_g0.load_model("models/mk11_m2_2d_g0/")

    u_rot_batch = np.zeros(shape=(u_batch.shape[0], u_batch.shape[1] - 1))
    u_mirr_batch = np.zeros(shape=(u_batch.shape[0], u_batch.shape[1] - 1))

    G_list = []
    # 2) Rotate all tensors to v_x line in u_1
    for i in range(u_batch.shape[0]):
        u_orig_1 = u_batch[i, 1:3]
        u_orig_2 = np.asarray([[u_batch[i, 3], u_batch[i, 4]], [u_batch[i, 4], u_batch[i, 5]]])
        # Rotate to x-axis
        G = create_rotator(u_orig_1)
        G_list.append(G)
        u_rot_1 = rotate_m1(u_orig_1, G)
        u_rot_2 = rotate_m2(u_orig_2, G)
        u_rot_batch[i, :] = np.asarray([u_rot_1[0], u_rot_1[1], u_rot_2[0, 0], u_rot_2[1, 0], u_rot_2[1, 1]])
        # Rotate by 180 degrees to mirror on origin
        G_mirror = -G
        u_mirr_1 = rotate_m1(u_orig_1, G_mirror)
        u_mirr_2 = rotate_m2(u_orig_2, G_mirror)
        u_mirr_batch[i, :] = np.asarray([u_mirr_1[0], u_mirr_1[1], u_mirr_2[0, 0], u_mirr_2[1, 0], u_mirr_2[1, 1]])
    # 3) Evaluatate model
    [h_pred, alpha_pred, _] = mk11_m2_2d_g0.model(tf.constant(u_rot_batch))
    [h_pred_mir, alpha_pred_mir, _] = mk11_m2_2d_g0.model(tf.constant(u_mirr_batch))
    # 3)  Average
    h_res = (h_pred_mir + h_pred) / 2
    alpha_res = (alpha_pred + alpha_pred_mir) / 2

    return h_res, alpha_res


def create_rotator(u_1_in) -> np.ndarray:
    r = np.linalg.norm(u_1_in)
    c = u_1_in[0] / r
    s = -u_1_in[1] / r
    G = np.asarray([[c, -s], [s, c]])
    return G


def rotate_m1(vec_orig_1, G) -> np.ndarray:
    """
    params: vec_orig_1: dims = (2,)
            G: dims = (2,2)
    """
    return np.matmul(G, vec_orig_1)


def back_rotate_m1(vec_rot_1, G) -> np.ndarray:
    """
    params: vec_rot_1: dims = (2,)
            G: dims = (2,2)
    """
    return np.matmul(np.transpose(G), vec_rot_1)


def rotate_m2(vec_orig_2, G) -> np.ndarray:
    """
        params: vec_orig_2: dims = (2,2)
                G: dims = (2,2)
    """
    return np.matmul(G, np.matmul(vec_orig_2, np.transpose(G)))


def back_rotate_m2(vec_rot_2, G) -> np.ndarray:
    """
        params: vec_rot_2: dims = (2,2)
                G: dims = (2,2)
    """
    return np.matmul(np.transpose(G), np.matmul(vec_rot_2, G))


def m2_inner_prod(u_0, u_1, u_2, alpha_0, alpha_1, alpha_2):
    """
        brief: returns <alpha,u> for M2-2D closure in tensor formulation
        params: u_0 zero moment  0-tensor
                u_1 first moment 1-tensor
                u_2 second moment 2-tensor
                alpha_0 zeroth multiplier 0-tensor
                alpha_1 first multiplier  1-tensor
                alpha_2 second multiplier 2-tensor
    """
    t0 = u_0 * alpha_0
    t1 = np.inner(u_1, alpha_1)
    t2 = np.sum(np.multiply(alpha_2, u_2))
    return t0 + t1 + t2


if __name__ == '__main__':
    main()
