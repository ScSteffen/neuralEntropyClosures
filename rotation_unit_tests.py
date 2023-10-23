from src.math_utils import EntropyTools
from src.math_utils import create_sh_rotator_2D, create_sh_rotator_1D, create_roation_grad_M2, \
    create_sh_rotator_2D_red, create_nabla_w_z_M2

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
    # test_gradient_inequality2()
    # test_rotation_idempotence()
    # test rotation convexity
    # test_multiple_reduced_convexity()
    # test_rotation_reflection_3d()
    # test_multiple_legendre_transform()
    # test_non_standard_rotation()
    # test_gradient_offsets()
    test_level_sets()
    return 0


def test_level_sets():
    # create et
    et = EntropyTools(polynomial_degree=2, spatial_dimension=2, gamma=0.001, basis='spherical_harmonics')
    # sample moments
    n_samples = 10
    sys_size = 6
    u = np.ones(shape=(2 * n_samples, sys_size))
    alpha = np.zeros(shape=(2 * n_samples, sys_size))
    h = np.zeros(shape=(2 * n_samples,))
    r = 1.0
    ts = np.linspace(-1, 1, n_samples)
    for i in range(n_samples):
        t = ts[i]  # np.random.uniform(low=-1, high=1)
        u[i, :] = np.asarray([1, 0.5, 0, t, 0, np.sqrt(r ** 2 - t ** 2)])
        u[i + n_samples, :] = np.asarray([1, 1, 0, t, 0, -np.sqrt(r ** 2 - t ** 2)])
        print(np.linalg.norm(u[i, :]))
        print(np.linalg.norm(u[i + n_samples, :]))
        alpha[i, :] = et.minimize_entropy(u=u[i, :], alpha_start=np.zeros(sys_size))
        alpha[i + n_samples, :] = et.minimize_entropy(u=u[i + n_samples, :], alpha_start=np.zeros(sys_size))
        h[i] = et.compute_h_dual(u=u[i, :], alpha=alpha[i, :])
        h[i + n_samples] = et.compute_h_dual(u=u[i + n_samples, :], alpha=alpha[i + n_samples, :])
    print(h)
    plt.figure()
    plt.plot(u[:, 3], u[:, 5], '*')
    plt.gca().set_aspect("equal", 'box')
    plt.tight_layout()
    plt.show()
    print('==> No radial symmetry in higher dimensions')
    return 0


def test_rotation_reflection_3d():
    w_a = np.asarray([1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])
    w_b = np.asarray([-1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5])
    # create rotation and rotation gradient
    nabla_w_zb, M_b = create_nabla_w_z_M2(w_b)
    nabla_w_za, M_a = create_nabla_w_z_M2(w_a)

    t1 = approximate_M2_3d_entropy(M_b @ w_b)
    t2 = approximate_M2_3d_entropy(M_a @ w_a)
    g1 = approximate_M2_3d_entropy_grad(M_b @ w_b) @ nabla_w_zb
    g2 = approximate_M2_3d_entropy_grad(M_a @ w_a) @ nabla_w_za
    # check for convexity of the function itself (sanity check)
    return 0


def test_legendre_transform(z_c, z_b):
    h_s_b = approximate_M2_entropy(z_b)
    delta_z_b = approximate_M2_entropy_grad(z_b)
    h_s_b_conj = approximate_M2_entropy_conjugate(delta_z_b)

    rhs = delta_z_b @ z_c - h_s_b_conj
    if h_s_b < rhs:
        print('something is wrong')
        print('z_b: ' + str(z_b))
        print('z_c: ' + str(z_c))
        print('h_s_b: ' + str(h_s_b))
        print('rhs: ' + str(rhs))
    return 0


def test_multiple_legendre_transform():
    n_test_a = 1
    n_test_b = 1
    for i in range(n_test_b):
        z_b = np.random.uniform(low=-1, high=1, size=(4,))
        for j in range(n_test_a):
            z_c = np.random.uniform(low=-1, high=1, size=(4,))
            test_legendre_transform(z_c, z_b)

        print('i =' + str(i) + ' of ' + str(n_test_b))

    return 0


def test_reduced_convexity(w_a, w_b):
    # create rotation and rotation gradient
    nabla_w_zb, M_b = create_nabla_w_z_M2(w_b)
    _, M_a = create_nabla_w_z_M2(w_a)

    lhs = approximate_M2_entropy(M_a @ w_a)

    h_s_b = approximate_M2_entropy(M_b @ w_b)
    h_s_c = approximate_M2_entropy(nabla_w_zb @ w_a)

    t1 = approximate_M2_entropy_grad(M_b @ w_b)
    t2 = nabla_w_zb @ w_a
    t3 = nabla_w_zb @ w_b
    z_b = M_b @ w_b
    z_a = M_a @ w_a
    rhs = approximate_M2_entropy(M_b @ w_b) + approximate_M2_entropy_grad(M_b @ w_b) @ nabla_w_zb @ (w_a - w_b)
    res = lhs - rhs

    # check for convexity of the function itself (sanity check)
    sanity = approximate_M2_entropy(z_b) + approximate_M2_entropy_grad(z_b) @ (z_a - z_b) - approximate_M2_entropy(z_a)

    if sanity >= 1e-6:
        print("something is wrong, subspace function seems to be non convex:")
        print('z_b: ' + str(z_b))
        print('z_a: ' + str(z_a))
        print('h_s_a: ' + str(lhs))
        print('h_s_b: ' + str(h_s_b))
        print('inequality value:' + str(sanity))
        exit(1)
    # if lhs < h_s_c < h_s_b:
    #    print("case: lhs < h_s_c <h_s_b")
    #    print('w_b: ' + str(w_b))
    #    print('w_a: ' + str(w_a))
    #    print('h_s_a: ' + str(lhs))
    #    print('h_s_c: ' + str(h_s_c))
    #    print('h_s_b: ' + str(h_s_b))

    if res < 0:
        print("non convex sample detected:")
        print('w_b: ' + str(w_b))
        print('w_a: ' + str(w_a))
        print('z_b: ' + str(z_b))
        print('z_a: ' + str(z_a))
        print('z_c: ' + str(nabla_w_zb @ w_a))
        print('h_s_a: ' + str(lhs))
        print('h_s_c: ' + str(h_s_c))
        print('h_s_b: ' + str(h_s_b))
        print('lhs: ' + str(lhs))
        print('rhs: ' + str(rhs))
        print('result: ' + str(res))
        print('\n\n\n ==============')
    return 0


def test_multiple_reduced_convexity():
    n_test_a = 100  # 00
    n_test_b = 100  # 00
    for i in range(n_test_b):
        w_b = np.random.uniform(low=-1, high=1, size=(5,))
        # w_b = np.asarray([1., 1., 0., 0., 1.])
        for j in range(n_test_a):
            w_a = np.random.uniform(low=-1, high=1, size=(5,))
            # w_a = np.asarray([0., 1., 0., 0., 1.])

            test_reduced_convexity(w_a, w_b)
            test_jensen(w_a, w_b)

        print('i =' + str(i) + ' of ' + str(n_test_b))

    return 0


def test_jensen(w_a, w_b):
    n_interpolators = 20
    lambdas = np.linspace(0, 1, n_interpolators)

    z_batch = np.zeros(shape=(n_interpolators, 4))
    h_batch = np.zeros(shape=(n_interpolators,))
    for i in range(n_interpolators):
        l = lambdas[i]
        w_lambda = l * w_a + (1 - l) * w_b
        _, M_lambda = create_nabla_w_z_M2(w_lambda)
        z_lambda = M_lambda @ w_lambda
        z_batch[i, :] = z_lambda
        h_batch[i] = approximate_M2_entropy(z_lambda)

    for i in range(n_interpolators):
        if h_batch[i] > lambdas[i] * h_batch[- 1] + (1 - lambdas[i]) * h_batch[0]:
            print("Non convex sample detected" + str(i) + "|" + str(i))
            print("current function value " + str(h_batch[i]))
            print("left function value " + str(h_batch[0]))
            print("right function value " + str(h_batch[- 1]))
            print("lambda value " + str(lambdas[i]))
            print("rhs value " + str(lambdas[i] * h_batch[- 1] + (1 - lambdas[i]) * h_batch[0]))
            print("z_a" + str(z_batch[0]))
            print("z_b" + str(z_batch[-1]))
            print("----")
    return 0


def approximate_M2_entropy_conjugate(delta):
    t = 1 / 2 * delta[0, 0] ** 2 + 3 / 4 * delta[0, 1] ** 4 + 5 / 6 * delta[0, 2] ** 6 + 7 / 8 * delta[0, 3] ** 8
    # t = 1 / 2 * z[0] ** 2 + 1 / 2 * z[1] ** 2 + 1 / 2 * z[2] ** 2 + 1 / 2 * z[3] ** 2
    return t


def approximate_M2_3d_entropy(z):
    # t = 1 / 2 * z[0] ** 2 + 1 / 4 * z[1] ** 4 + 1 / 6 * z[2] ** 6 + 1 / 8 * z[3] ** 8
    # t = 1 / 2 * z[0] ** 2 + 1 / 4 * z[1] ** 4 + 1 / 6 * z[2] ** 6 + 1 / 4 * z[3] ** 4
    t = 1 / 2 * z[0] ** 2 + 1 / 2 * z[1] ** 2 + 1 / 2 * z[2] ** 2 + 1 / 2 * z[3] ** 2 + 1 / 2 * z[1] ** 2 + 1 / 2 * z[
        2] ** 2

    # t = 1 / 2 * z[0] ** 2 + 1 / 4 * z[1] ** 4 + 1 / 4 * z[2] ** 4 + 1 / 4 * z[3] ** 4
    # t = z[0] + z[1] + z[2] + z[3]
    return t


def approximate_M2_3d_entropy_grad(z):
    # t = np.asarray([z[0], z[1] ** 3, z[2] ** 5, z[3] ** 7])
    # t = np.asarray([z[0], z[1] ** 3, z[2] ** 5, z[3] ** 3])
    t = z
    # t = np.asarray([z[0] ** 1, z[1] ** 3, z[2] ** 3, z[3] ** 3])
    # t = np.asarray([1, 1, 1, 1])
    t1 = np.reshape(t, newshape=(1, 6))
    return t1


def approximate_M2_entropy(z):
    # t = 1 / 2 * z[0] ** 2 + 1 / 4 * z[1] ** 4 + 1 / 6 * z[2] ** 6 + 1 / 8 * z[3] ** 8
    # t = 1 / 2 * z[0] ** 2 + 1 / 4 * z[1] ** 4 + 1 / 6 * z[2] ** 6 + 1 / 4 * z[3] ** 4
    t = 1 / 2 * z[0] ** 2 + 1 / 2 * z[1] ** 2 + 1 / 2 * z[2] ** 2 + 1 / 2 * z[3] ** 2

    # t = 1 / 2 * z[0] ** 4 + 1 / 4 * z[1] ** 4 + 1 / 4 * z[2] ** 4 + 1 / 4 * z[3] ** 4
    # t = z[0] + z[1] + z[2] + z[3]
    return t


def approximate_M2_entropy_grad(z):
    # t = np.asarray([z[0], z[1] ** 3, z[2] ** 5, z[3] ** 7])
    # t = np.asarray([z[0], z[1] ** 3, z[2] ** 5, z[3] ** 3])
    t = np.asarray([z[0], z[1], z[2], z[3]])
    # t = np.asarray([z[0] ** 1, z[1] ** 3, z[2] ** 3, z[3] ** 3])
    # t = np.asarray([1, 1, 1, 1])
    t1 = np.reshape(t, newshape=(1, 4))
    return t1


def test_gradient_offsets():
    np.set_printoptions(precision=2)

    max_alpha = 2
    alpha_size = 5
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.1, basis="spherical_harmonics")

    alpha1 = 0.5 * np.ones(alpha_size)
    alpha = et.reconstruct_alpha(alpha1)
    u = np.reconstruct
    u = np.asarray([1, 0.5, 0.5, 0, 0, 0.5])  # [1, 0.5, 0., 0.0, 1., 0.0]
    u_mom_rot_non_std = np.asarray(
        [1, 0.5, 0.5, 0.5, 0, 0.5])  # np.asarray([1, 0.5, 0.5, 0, 0, 0.5])  # M_R_non_std @ u

    _, M_R = create_sh_rotator_2D(u[1:3])

    M_R_non_std = np.copy(M_R)

    theta = np.pi
    M_R_non_std[3:6, 3:6] = np.asarray(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1 * np.sin(theta), 0, np.cos(theta)]])  # np.eye(3)

    # Test if the rotation matrices are orthonormal
    print(M_R)
    print(M_R_non_std)
    print(M_R.T @ M_R)
    print(M_R_non_std.T @ M_R_non_std)

    u_mom_rot = M_R @ u
    u_mom_rot_non_std = M_R_non_std @ u

    print(u)
    print(u_mom_rot)
    print(u_mom_rot_non_std)
    # et.rotate_basis(M_R.T)
    # alpha_res = et.minimize_entropy(u=u, alpha_start=alpha)
    # h = et.compute_h_dual(u=u, alpha=alpha_res)
    # et.rotate_basis(M_R)

    alpha_res = et.minimize_entropy(u=u, alpha_start=0 * u)
    h = et.compute_h_dual(u=u, alpha=alpha_res)

    # et.rotate_basis(M_R)
    alpha_res_rot = et.minimize_entropy(u=u_mom_rot, alpha_start=0 * u_mom_rot)
    h_rot = et.compute_h_dual(u=u_mom_rot, alpha=alpha_res_rot)
    # et.rotate_basis(M_R.T)

    # et.rotate_basis(M_R_non_std)
    alpha_res_rot_nstd = et.minimize_entropy(u=u_mom_rot_non_std, alpha_start=0 * u_mom_rot_non_std)
    h_rot_nstd = et.compute_h_dual(u=u_mom_rot_non_std, alpha=alpha_res_rot_nstd)
    # et.rotate_basis(M_R_non_std.T)

    print(h)
    print(h_rot)
    print(h_rot_nstd)
    return 0


def test_non_standard_rotation():
    np.set_printoptions(precision=2)

    max_alpha = 2
    alpha_size = 5
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.1, basis="spherical_harmonics")

    alpha1 = 0.5 * np.ones(alpha_size)
    alpha = et.reconstruct_alpha(alpha1)

    u = np.asarray([1, 0.5, 0.5, 0, 0, 0.5])  # [1, 0.5, 0., 0.0, 1., 0.0]
    u_mom_rot_non_std = np.asarray(
        [1, 0.5, 0.5, 0.5, 0, 0.5])  # np.asarray([1, 0.5, 0.5, 0, 0, 0.5])  # M_R_non_std @ u

    _, M_R = create_sh_rotator_2D(u[1:3])

    M_R_non_std = np.copy(M_R)

    theta = np.pi
    M_R_non_std[3:6, 3:6] = np.asarray(
        [[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1 * np.sin(theta), 0, np.cos(theta)]])  # np.eye(3)

    # Test if the rotation matrices are orthonormal
    print(M_R)
    print(M_R_non_std)
    print(M_R.T @ M_R)
    print(M_R_non_std.T @ M_R_non_std)

    u_mom_rot = M_R @ u
    u_mom_rot_non_std = M_R_non_std @ u

    print(u)
    print(u_mom_rot)
    print(u_mom_rot_non_std)
    # et.rotate_basis(M_R.T)
    # alpha_res = et.minimize_entropy(u=u, alpha_start=alpha)
    # h = et.compute_h_dual(u=u, alpha=alpha_res)
    # et.rotate_basis(M_R)

    alpha_res = et.minimize_entropy(u=u, alpha_start=0 * u)
    h = et.compute_h_dual(u=u, alpha=alpha_res)

    # et.rotate_basis(M_R)
    alpha_res_rot = et.minimize_entropy(u=u_mom_rot, alpha_start=0 * u_mom_rot)
    h_rot = et.compute_h_dual(u=u_mom_rot, alpha=alpha_res_rot)
    # et.rotate_basis(M_R.T)

    # et.rotate_basis(M_R_non_std)
    alpha_res_rot_nstd = et.minimize_entropy(u=u_mom_rot_non_std, alpha_start=0 * u_mom_rot_non_std)
    h_rot_nstd = et.compute_h_dual(u=u_mom_rot_non_std, alpha=alpha_res_rot_nstd)
    # et.rotate_basis(M_R_non_std.T)

    print(h)
    print(h_rot)
    print(h_rot_nstd)
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
