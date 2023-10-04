from src.math_utils import EntropyTools
from src.math_utils import create_sh_rotator_2D, create_sh_rotator_1D

from src.networks.configmodel import init_neural_closure

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    np.set_printoptions(precision=4)

    # test_ml_closure()
    # test_numerical_closure()
    # check_single_moment()
    # test_gradient_inequality()
    test_gradient_inequality2()
    # test_rotation_idempotence()
    return 0


def test_gradient_inequality2():
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.0, basis="spherical_harmonics")

    # sample normalized moment on the line
    w_b = np.asarray([1, 0.5, 0.2, 0.5, 0.0, 0.5])
    _, M_b = create_sh_rotator_2D(w_b[1:3])
    z = M_b @ w_b
    w_a = -0.2 * np.ones(6)
    w_a[0] = 1

    w_ab = M_b @ w_a

    alpha_w_b = et.minimize_entropy(w_b, w_b)
    h_w_b = et.compute_h_dual(w_b, alpha_w_b)
    alpha_z = et.minimize_entropy(z, z)
    h_z = et.compute_h_dual(z, alpha_z)
    err_1 = np.linalg.norm(h_w_b - h_z)

    alpha_w_a = et.minimize_entropy(w_a, w_a)
    h_w_a = et.compute_h_dual(w_a, alpha_w_a)
    alpha_ab = et.minimize_entropy(w_ab, w_ab)
    h_w_ab = et.compute_h_dual(w_ab, alpha_ab)
    err_2 = np.linalg.norm(h_w_ab - h_w_a)

    # test convexity
    rhs1 = h_w_b + np.inner(alpha_w_b, w_a - w_b)
    rhs2 = h_z + np.inner(alpha_z, w_ab - z)
    return 0


def test_gradient_inequality():
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.0, basis="spherical_harmonics")

    # sample normalized moment on the line
    u = np.asarray([1, 0.5, 0, 0.5, 0.0, 0.5])
    # compute entropy gradient, aka lagrange mutliplier alpha
    alpha = et.minimize_entropy(u=u, alpha_start=u)
    h_u = et.compute_h_dual(u, alpha)
    # sample arbitrary moment
    w_ab = -0.2 * np.ones(6)
    w_ab[0] = 1
    alpha_ab = et.minimize_entropy(u=w_ab, alpha_start=u)
    h_w_ab = et.compute_h_dual(u, alpha)
    # rotate it
    _, M = create_sh_rotator_2D(w_ab[1:3])
    m_w_ab = M @ w_ab

    residual_ab = w_ab - m_w_ab
    # test gradient inequality
    res = np.inner(alpha, residual_ab)
    # test convexity at u
    res2 = np.inner(u[1:], alpha[1:])
    # second test for convexity
    rhs = h_u + np.inner(alpha, u - w_ab)
    lhs = h_w_ab
    return 0


def test_rotation_idempotence():
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.0, basis="spherical_harmonics")

    # sample normalized moment on the line
    u = np.asarray([1, 0.5, 0.5, 0.5, 0.0, 0.5])
    _, M_u = create_sh_rotator_2D(u[1:3])

    u_a = np.asarray([1, -0.5, 0.1, 0.2, 0.0, 0.5])  # sample second moment to create a arbitrary rotation
    _, M_ua = create_sh_rotator_2D(u_a[1:3])

    m_ua_u = M_ua @ u  # rotate original moment
    _, M_tilde = create_sh_rotator_2D(m_ua_u[1:3])

    # Test image of both rotations
    u_res1 = M_u @ u
    u_res2 = M_tilde @ m_ua_u

    residual = np.linalg.norm(u_res1 - u_res2)
    u_res_test = M_ua @ u_a
    return 0


def check_single_moment():
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.0, basis="spherical_harmonics")

    u = np.asarray([1, 0.5, 0, 0.5, 0.0, 0.5])
    res = et.minimize_entropy(u=u, alpha_start=u)

    u_res = et.reconstruct_u(res)
    u_res2 = et.reconstruct_u(np.asarray([1, 1., 0, 0.5, 0, 0.5]))
    return 0


def test_ml_closure():
    degree = 1
    n_iterpolators = 10
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.1, basis="spherical_harmonics")

    #   load model
    if degree == 1:
        closure = init_neural_closure(network_mk=11, poly_degree=degree, spatial_dim=2, folder_name="tmp",
                                      loss_combination=2, nw_width=100, nw_depth=3, normalized=True,
                                      input_decorrelation=True, scale_active=False, gamma_lvl=0)
        closure.load_model("models/monomial/mk11_M1_2D_g0/")
        # minimizer_u = find_minimum_m1(model)

        alpha_size = 2
    else:
        closure = init_neural_closure(network_mk=11, poly_degree=degree, spatial_dim=2, folder_name="tmp",
                                      loss_combination=2, nw_width=350, nw_depth=3, normalized=True,
                                      input_decorrelation=True, scale_active=False, gamma_lvl=1,
                                      basis="spherical_harmonics")
        closure.load_model("models/Harmonic_Mk11_M2_2D_gamma1")

        u_0 = find_minimum_m2(closure)  # * 0
        alpha_size = 5

    # 1) create input moment

    # u_a = np.asarray([0.32363081, 0.82139511, -0.26551047, -1.06051513, -0.53799217])
    # u_b = np.asarray([1.79668719e-04, 2.24985473e-02, 1.09392859e-01, -1.30130912e+00, -1.78054693e+00])
    # u_a = np.asarray([1, 1, -0.26551047, -1.06051513, -0.53799217])
    # u_b = np.asarray([0, 0, 1.09392859e-01, -1.30130912e+00, -1.78054693e+00])
    # u_a = np.asarray([1.16138925, -0.85133795, -1.14083069, 1.00895157, -0.50821521])
    # u_b = np.asarray([0.18788227, 0.18168944, -0.87451784, 1.48927047, 0.32683049])

    u_a = np.asarray([1.10002299, 0.50879753, 1.56387997, -0.79977685, -1.25200321])
    u_b = np.asarray([0.31362133, -0.36796771, 1.50736318, -1.11147807, -1.51821143])

    # 1.5) Create interpolants
    lambdas = np.linspace(0, 1, n_iterpolators)
    u_batch = np.zeros(shape=(n_iterpolators, alpha_size))
    u_batch_r = np.zeros(shape=(n_iterpolators, alpha_size))
    u_batch_r_m = np.zeros(shape=(n_iterpolators, alpha_size))
    M_Rs = []
    M_R_fulls = []
    M_R_pm, _ = create_sh_rotator_2D(np.asarray([-1, 0]))

    for i in range(n_iterpolators):
        l = lambdas[i]
        u_batch[i, :] = l * u_b + (1 - l) * u_a
        # u_batch[i, :] -= u_0
        M_R, M_R_full = create_sh_rotator_2D(u_batch[i, 0:2])
        M_R_fulls.append(M_R_full)
        M_Rs.append(M_R)
        u_batch_r[i, :] = M_R @ u_batch[i, :]
        print(u_batch_r[i, :])
        u_batch_r_m[i, :] = M_R_pm @ u_batch_r[i, :]  # + u_0
        # u_batch_r[i, :]  # += u_0

    u_tf = tf.constant(u_batch, dtype=tf.float32)
    u_r_tf = tf.constant(u_batch_r, dtype=tf.float32)
    u_r_m_tf = tf.constant(u_batch_r_m, dtype=tf.float32)

    # 3) evaluate network
    [h_pred, _, _] = closure.model(u_tf)
    [h_pred_r, alpha_pred_r, _] = closure.model(u_r_tf)
    [h_pred_r_m, _, _] = closure.model(u_r_m_tf)
    # h_pred = np.linalg.norm(u_batch) ** 2
    # h_pred_r = np.linalg.norm(u_batch_r) ** 2
    # h_pred_r_m = np.linalg.norm(u_batch_r_m) ** 2

    h_num_r = np.zeros(n_iterpolators)
    h_num = np.zeros(n_iterpolators)
    alpha_num_r = np.zeros((n_iterpolators, alpha_size + 1))
    alpha_num = np.zeros((n_iterpolators, alpha_size + 1))
    for i in range(n_iterpolators):
        alpha_num_r[i] = et.minimize_entropy(u=np.append(1, u_batch_r[i]), alpha_start=np.append(0, u_batch_r[i]))
        h_num_r[i] = et.compute_h_dual(u=np.append(1, u_batch_r[i]), alpha=alpha_num_r[i])
        # evaluate at  without rotation
        et.rotate_basis(M_R_fulls[i].T)
        alpha_num[i] = et.minimize_entropy(u=np.append(1, u_batch[i]), alpha_start=np.append(0, u_batch[i]))
        h_num[i] = et.compute_h_dual(u=np.append(1, u_batch[i]), alpha=alpha_num[i])
        et.rotate_basis(M_R_fulls[i])

    # print(alpha_pred_r.numpy())
    # print(alpha_num_r[:, 1:])
    print("-----")
    print(u_batch)
    print(u_batch_r)
    print(u_batch_r_m)

    plt.plot(lambdas, h_pred.numpy(), 'o-')
    plt.xlabel("$\lambda$")
    plt.ylabel("$h(u(\lambda))$")
    plt.savefig("test_imgs/test_h" + ".png")
    plt.clf()

    plt.plot(lambdas, h_pred_r.numpy(), 'o-')
    plt.xlabel("$\lambda$")
    plt.ylabel("$h(u(\lambda))$")
    plt.savefig("test_imgs/test_h_r" + ".png")
    plt.clf()

    plt.plot(lambdas, h_pred_r_m.numpy(), 'o-')
    plt.xlabel("$\lambda$")
    plt.ylabel("$h(u(\lambda))$")
    plt.savefig("test_imgs/test_h_r_m" + ".png")
    plt.clf()

    plt.plot(lambdas, (h_pred_r_m.numpy() + h_pred_r.numpy()) / 2., 'o-')
    plt.xlabel("$\lambda$")
    plt.ylabel("$h(u(\lambda))$")
    plt.savefig("test_imgs/test_h_s" + ".png")
    plt.clf()

    plt.plot(lambdas, h_num, 'o-')
    plt.xlabel("$\lambda$")
    plt.ylabel("$h(u(\lambda))$")
    plt.savefig("test_imgs/test_h_num" + ".png")
    plt.clf()

    plt.plot(lambdas, h_num_r, 'o-')
    plt.xlabel("$\lambda$")
    plt.ylabel("$h(u(\lambda))$")
    plt.savefig("test_imgs/test_h_num_r" + ".png")
    plt.clf()

    for i in range(n_iterpolators):
        plt.plot(u_batch[i, 0], u_batch[i, 1], 'ro')
        plt.plot(u_batch_r[i, 0], u_batch_r[i, 1], 'k*')
        plt.plot(u_batch_r_m[i, 0], u_batch_r_m[i, 1], 'g+')
    plt.xlabel("$u_1$")
    plt.ylabel("$u_2$")
    plt.legend(["pre rotation", "post rotation", "rot and mirr"])
    plt.savefig("test_imgs/test_" + "_u_2.png")
    plt.clf()

    for i in range(n_iterpolators):
        plt.plot(u_batch[i, 0], u_batch[i, 2], 'ro')
        plt.plot(u_batch_r[i, 0], u_batch_r[i, 2], 'k*')
        plt.plot(u_batch_r_m[i, 0], u_batch_r_m[i, 2], 'g+')
    plt.xlabel("$u_1$")
    plt.ylabel("$u_3$")
    plt.legend(["pre rotation", "post rotation", "rot and mirr"])
    plt.savefig("test_imgs/test_" + "_u_3.png")
    plt.clf()

    for i in range(n_iterpolators):
        plt.plot(u_batch[i, 0], u_batch[i, 3], 'ro')
        plt.plot(u_batch_r[i, 0], u_batch_r[i, 3], 'k*')
        plt.plot(u_batch_r_m[i, 0], u_batch_r_m[i, 3], 'g+')
    plt.xlabel("$u_1$")
    plt.ylabel("$u_3$")
    plt.legend(["pre rotation", "post rotation", "rot and mirr"])
    plt.savefig("test_imgs/test_" + "_u_4.png")
    plt.clf()

    for i in range(n_iterpolators):
        plt.plot(u_batch[i, 0], u_batch[i, 4], 'ro')
        plt.plot(u_batch_r[i, 0], u_batch_r[i, 4], 'k*')
        plt.plot(u_batch_r_m[i, 0], u_batch_r_m[i, 4], 'g+')
    plt.xlabel("$u_1$")
    plt.ylabel("$u_5$")
    plt.legend(["pre rotation", "post rotation", "rot and mirr"])
    plt.savefig("test_imgs/test_" + "_u_5.png")
    plt.clf()

    return 0


def find_minimum_m2(closure):
    u_mini = np.zeros(shape=(1, 5))
    u = tf.constant(u_mini, dtype=tf.float32)
    h, alpha, _ = closure.model(u)
    # alpha = alpha * tf.constant([1.0, 0.0, 1.0, 1.0, 1.0], dtype=tf.float32)
    delta = 0.5
    while np.linalg.norm(alpha.numpy()) > 1e-4:
        u = u - delta * alpha
        h, alpha, _ = closure.model(u)

        print(h.numpy())
        print(np.linalg.norm(alpha.numpy()))
        print("---")

    print(u.numpy()[0])
    # alpha = alpha * tf.constant([1.0, 0.0, 1.0, 1.0, 1.0], dtype=tf.float32)
    return u.numpy()[0]  # np.append(0, u.numpy())


def test_numerical_closure():
    np.set_printoptions(precision=2)

    max_alpha = 2
    alpha_size = 5
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.0, basis="spherical_harmonics")

    alpha1 = 0.5 * np.ones(alpha_size)
    alpha = et.reconstruct_alpha(alpha1)

    u_mom = et.reconstruct_u(alpha)
    _, M_R = create_sh_rotator_2D(u_mom[1:3])
    # _, M_R = create_sh_rotator(u_mom[1:3])
    # Test if moment basis is orthonormal
    m_basis = et.moment_basis_np
    w_q = np.reshape(et.quadWeights.numpy(), newshape=(et.nq,))
    sum = np.einsum("j->", w_q)
    res = np.einsum("kj,lj, j->kl", m_basis, m_basis, w_q)

    # Test if the rotation matrices are orthonormal
    print(M_R)
    print(M_R.T @ M_R)
    res2 = M_R.T @ M_R

    u_mom_rot = M_R @ u_mom

    et.rotate_basis(M_R.T)
    alpha_res = et.minimize_entropy(u=u_mom, alpha_start=alpha)
    h = et.compute_h_dual(u=u_mom, alpha=alpha_res)

    et.rotate_basis(M_R)
    alpha_res_rot = et.minimize_entropy(u=u_mom_rot, alpha_start=M_R @ alpha)
    h_rot = et.compute_h_dual(u=u_mom_rot, alpha=alpha_res_rot)

    if np.linalg.norm(h - h_rot) > 1e-5:
        print("entropy does not match")

    det_M_R = np.linalg.det(M_R)
    print(det_M_R)
    # test if rotation of multipliers correspond
    u_br = M_R.T @ u_mom_rot

    alpha_br = M_R.T @ alpha_res_rot

    # check what happens to mirroring
    _, M_R_pm = create_sh_rotator_2D(np.asarray([-1, 0]))
    u_rot_pm = M_R_pm @ u_mom_rot
    alpha_res_rot_mirror = et.minimize_entropy(u=u_rot_pm, alpha_start=M_R_pm @ M_R @ alpha)
    h_rot_mirror = et.compute_h_dual(u=u_rot_pm, alpha=alpha_res_rot_mirror)

    if np.linalg.norm(h_rot_mirror - h_rot) > 1e-5:
        print("entropy does not match")
    return 0


if __name__ == '__main__':
    main()
