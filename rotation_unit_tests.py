from src.math_utils import EntropyTools
from src.math_utils import create_sh_rotator_2D
import numpy as np
import tensorflow as tf


def main():
    np.set_printoptions(precision=2)

    max_alpha = 2
    alpha_size = 5
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.001, basis="spherical_harmonics")

    alpha1 = np.reshape(np.asarray([0.5, 0.5, 0.5, 0.5, 0.5]), newshape=(1, alpha_size))
    alpha = et.reconstruct_alpha(tf.constant(alpha1, dtype=tf.float64))

    u_mom = et.reconstruct_u(alpha).numpy()
    _, M_R = create_sh_rotator_2D(u_mom[0, 1:3])

    # Test if moment basis is orthonormal
    m_basis = et.moment_basis_np
    w_q = np.reshape(et.quadWeights.numpy(), newshape=(et.nq,))
    sum = np.einsum("j->", w_q)
    res = np.einsum("kj,lj, j->kl", m_basis, m_basis, w_q)

    u_mom_rot = np.reshape(M_R @ u_mom[0], newshape=(1, alpha_size + 1))
    _, alpha_res = et.minimize_entropy(u=u_mom[0])
    _, alpha_res_rot = et.minimize_entropy(u=u_mom_rot[0])

    # test if entropy values correspond
    h_ref = et.compute_h(u=tf.constant(u_mom, shape=(1, 6)), alpha=alpha).numpy()
    h = et.compute_h(u=tf.constant(u_mom, shape=(1, 6)), alpha=tf.constant(alpha_res, shape=(1, 6))).numpy()
    h_rot = et.compute_h(u=tf.constant(u_mom_rot, shape=(1, 6)), alpha=tf.constant(alpha_res_rot, shape=(1, 6))).numpy()

    if np.linalg.norm(h - h_rot) > 1e-5:
        print("entropy does not match")

    # test if rotation of multipliers correspond
    u_br = M_R.T @ u_mom_rot[0]

    alpha_br = M_R.T @ alpha_res_rot

    return 0


if __name__ == '__main__':
    main()
