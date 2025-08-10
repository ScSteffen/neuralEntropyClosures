import numpy as np
import tensorflow as tf

from src.math import EntropyTools


def test_M2():
    gamma = 0.0
    err_cases_u = 0
    err_cases_u2 = 0
    err_cases_u3 = 0
    err_cases_alpha = 0
    n_test = 100
    et2 = EntropyTools(polynomial_degree=2, spatial_dimension=2, gamma=gamma)
    alpha1 = np.random.uniform(low=-10, high=10, size=(1, 5))

    for theta in np.linspace(0, np.pi, n_test):
        tr = rot_mat(theta)
        r_inv = rot_m2_mat_inv(tr)
        r_inv_top = r_inv.transpose()
        r_ = rot_m2_mat(tr)
        # print(r_)
        # print(r_inv)
        # print(r_inv_top)
        # print("----####")
        ie = r_inv @ r_
        alpha_orig = tf.constant(alpha1, dtype=tf.float64)
        alpha_orig = et2.reconstruct_alpha(alpha_orig)
        u_tens = et2.reconstruct_u(alpha_orig)

        u = et2.reconstruct_u(alpha_orig).numpy()[0, :]
        alpha_u = alpha_orig.numpy()[0, :]
        M_RT = big_rot_mat(theta)
        # MT_R = np.linalg.inv(M_RT).transpose()
        MT_R = big_rot_mat(-theta).transpose()
        e = M_RT.transpose() @ M_RT
        M_RTu = M_RT @ u
        M_RTu_tens = tf.constant([M_RTu], dtype=tf.float64)
        alpha_M_RTu = tf.constant(alpha_u, dtype=tf.float64)

        # alpha_test = et2.minimize_entropy(u_tens, alpha_orig).numpy()[0, :]  # just to sanity check the minimizer
        # print(np.linalg.norm(alpha_test - alpha_u))

        alpha_M_RTu = et2.minimize_entropy(M_RTu_tens, alpha_M_RTu).numpy()[0, :]

        # Check if alpha_M_RTu = MT_R_alpha_u
        MT_R_alpha_U = MT_R @ alpha_u
        M_RT_alpha_u = M_RT @ alpha_u

        u_recons_MT_R_alpha_U = et2.reconstruct_u(tf.constant(MT_R_alpha_U, shape=(1, 6)))
        u_recons_M_RT_alpha_u = et2.reconstruct_u(tf.constant(M_RT_alpha_u, shape=(1, 6)))
        u_recons_alpha_M_RTu = et2.reconstruct_u(tf.constant(alpha_M_RTu, shape=(1, 6), dtype=tf.float64))

        # print("M_RTu - u_recons_MT_R_alpha_U")
        err_u = np.linalg.norm(M_RTu - u_recons_MT_R_alpha_U)
        # print(err_u)
        # print("M_RTu - u_recons_M_RT_alpha_u")
        err_u2 = np.linalg.norm(M_RTu - u_recons_M_RT_alpha_u)
        # print(err_u2)
        # print("M_RTu - u_recons_alpha_M_RTu")
        err_u3 = np.linalg.norm(M_RTu - u_recons_alpha_M_RTu)
        # print(err_u3)
        # print("alpha_M_RTu - MT_R_alpha_U")
        err_alpha = np.linalg.norm(MT_R_alpha_U - alpha_M_RTu)
        # print(err_alpha)
        if err_u > 1e-5:
            err_cases_u += 1
        if err_u2 > 1e-5:
            err_cases_u2 += 1
        if err_u3 > 1e-5:
            err_cases_u3 += 1
        if err_alpha > 1e-3:
            err_cases_alpha += 1
        # print("--------------------")
    print("overall error cases u: " + str(err_cases_u))
    print("overall error cases u: " + str(err_cases_u2))
    print("overall error cases u: " + str(err_cases_u3))
    print("overall error cases alpha: " + str(err_cases_alpha))
    print("overall tested: " + str(n_test))
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
