import numpy as np
import tensorflow as tf
from src.math import EntropyTools


def test_M2():
    et2 = EntropyTools(polynomial_degree=2, spatial_dimension=2, gamma=0.0)
    gamma = 0.0
    for theta in np.linspace(0, np.pi, 12):
        tr = rot_mat(theta)
        r_inv = rot_m2_mat_inv(tr)
        r_inv_top = r_inv.transpose()
        r_ = rot_m2_mat(tr)
        print(r_)
        print(r_inv)
        print(r_inv_top)
        print("----####")
        ie = r_inv @ r_
        alpha1 = np.random.uniform(low=-10, high=10, size=(1, 5))
        alpha_orig = tf.constant(alpha1, dtype=tf.float64)
        alpha_orig = et2.reconstruct_alpha(alpha_orig)
        u_tens = et2.reconstruct_u(alpha_orig)

        u = et2.reconstruct_u(alpha_orig).numpy()[0, :]
        alpha_u = alpha_orig.numpy()[0, :]
        R = big_rot_mat(theta)
        e = R.transpose() @ R
        Ru = R @ u
        Ru_tens = tf.constant([Ru], dtype=tf.float64)
        alpha_Ru = tf.constant(alpha_u, dtype=tf.float64)

        alpha_test = et2.minimize_entropy(u_tens, alpha_orig).numpy()[0, :]

        alpha_Ru = et2.minimize_entropy(Ru_tens, alpha_Ru).numpy()[0, :]
        R_mT_alpha_U = np.linalg.inv(R).transpose() @ alpha_u
        rtest_alpha = R @ alpha_u

        # print(R_mT_alpha_U)
        # print(alpha_Ru)
        # print(R_mT_alpha_U - rtest_alpha)
        # print(R_mT_alpha_U - alpha_Ru)
        # print(rtest_alpha - alpha_Ru)
        print("--------------------")
    return 0


def test_M1():
    gamma = 0.1
    et2 = EntropyTools(polynomial_degree=1, spatial_dimension=2, gamma=gamma)

    for theta in np.linspace(0, np.pi, 12):
        alpha1 = np.random.uniform(low=-10, high=10, size=(1, 2))
        alpha_orig = tf.constant(alpha1, dtype=tf.float64)
        alpha_orig = et2.reconstruct_alpha(alpha_orig)
        u_tens = et2.reconstruct_u(alpha_orig)
        u = et2.reconstruct_u(alpha_orig).numpy()[0, :]
        alpha_u = alpha_orig.numpy()[0, :]
        R = big_rot_mat_M1(theta)
        e = R.transpose() @ R
        Ru = R @ u
        Ru_tens = tf.constant([Ru], dtype=tf.float64)
        alpha_Ru = tf.constant(alpha_u, dtype=tf.float64)

        alpha_test = et2.minimize_entropy(u_tens, alpha_orig).numpy()[0, :]

        alpha_Ru = et2.minimize_entropy(Ru_tens, alpha_Ru).numpy()[0, :]
        Ralpha_U = R @ alpha_u
        # print(R_mT_alpha_U)
        # print(alpha_Ru)
        print(Ralpha_U - alpha_Ru)
        print("...")
    return 0


def main():
    # test_M1()
    test_M2()
    return 0


def rot_mat(theta):
    return np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rot_m2_mat(rot_mat):
    r_m2 = np.zeros((3, 3))
    r_m2[0, 0] = rot_mat[0, 0] ** 2
    r_m2[0, 1] = rot_mat[0, 0] * rot_mat[0, 1] * 2
    r_m2[0, 2] = rot_mat[0, 1] ** 2
    r_m2[1, 0] = rot_mat[0, 0] * rot_mat[1, 0]
    r_m2[1, 1] = rot_mat[0, 0] * rot_mat[1, 1] + rot_mat[0, 1] * rot_mat[1, 0]
    r_m2[1, 2] = rot_mat[0, 1] * rot_mat[1, 1]
    r_m2[2, 0] = rot_mat[1, 0] ** 2
    r_m2[2, 1] = rot_mat[1, 1] * rot_mat[1, 0] * 2
    r_m2[2, 2] = rot_mat[1, 1] ** 2
    return r_m2


def rot_m2_mat_inv(rot_mat):
    r_m2 = np.zeros((3, 3))
    r_m2[0, 0] = rot_mat[0, 0] ** 2
    r_m2[0, 1] = rot_mat[0, 0] * rot_mat[0, 1]
    r_m2[0, 2] = rot_mat[0, 1] ** 2
    r_m2[1, 0] = rot_mat[0, 0] * rot_mat[1, 0] * 2
    r_m2[1, 1] = rot_mat[0, 0] * rot_mat[1, 1] + rot_mat[0, 1] * rot_mat[1, 0]
    r_m2[1, 2] = rot_mat[0, 1] * rot_mat[1, 1] * 2
    r_m2[2, 0] = rot_mat[1, 0] ** 2
    r_m2[2, 1] = rot_mat[1, 1] * rot_mat[1, 0]
    r_m2[2, 2] = rot_mat[1, 1] ** 2

    return r_m2.transpose()


def big_rot_mat(theta):
    R = np.zeros((6, 6))
    R[0, 0] = 1
    r = rot_mat(theta)
    R[1:3, 1:3] = r
    R[3:6, 3:6] = rot_m2_mat(r)
    return R


def big_rot_mat_M1(theta):
    R = np.zeros((3, 3))
    R[0, 0] = 1
    r = rot_mat(theta)
    R[1:3, 1:3] = r
    return R


if __name__ == '__main__':
    main()
