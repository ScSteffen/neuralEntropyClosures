"""
author: Steffen Schotthöfer
date: 12.04.22
"""

import numpy as np
import tensorflow as tf

from src.math import EntropyTools
from src.networks.configmodel import init_neural_closure


def main():
    # Script to test the tensor rotations
    # --- sanity check ---
    u = np.asarray([0.5, 0.2])
    G = create_rotator(u)
    u_rot = np.matmul(G, u)
    u_orig = np.matmul(np.linalg.inv(G), u_rot)

    # --- M1 2D test ---
    et2 = EntropyTools(polynomial_degree=1, spatial_dimension=2)
    alpha_orig = tf.constant([[3, 2]], dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig = et2.reconstruct_u(alpha_orig).numpy()
    alpha_orig = alpha_orig.numpy()
    u_orig_1 = u_orig[0, 1:]
    alpha_orig_1 = alpha_orig[0, 1:3]
    # Rotate
    G = create_rotator(u_orig_1)
    u_rot_1 = rotate_m1(u_orig_1, G)
    alpha_rot_1 = rotate_m1(alpha_orig_1, G)
    # assemble u_rot and alpha_rot
    u_rot_vec = np.asarray([u_orig[0, 0], u_rot_1[0], u_rot_1[1]])
    alpha_rot_vec = np.asarray([alpha_orig[0, 0], alpha_rot_1[0], alpha_rot_1[1]])

    # sanity checks
    dot_orig = np.inner(u_orig, alpha_orig)
    dot_rot = np.inner(u_rot_vec, alpha_rot_vec)

    # compute h
    u_tens = tf.constant(u_orig, dtype=tf.float64)
    alpha_tens = tf.constant(alpha_orig, dtype=tf.float64)
    u_rot_tens = tf.constant([u_rot_vec], dtype=tf.float64)
    alpha_rot_tens = tf.constant([alpha_rot_vec], dtype=tf.float64)
    h_orig = et2.compute_h(u_tens, alpha_tens)
    h_rot = et2.compute_h_rot(u_rot_tens, alpha_rot_tens, alpha_tens)

    # --- M2 2D test ---
    et2 = EntropyTools(polynomial_degree=2, spatial_dimension=2)
    alpha_orig = tf.constant([[3, 2, 1, 0.5, 0.2]], dtype=tf.float64)
    alpha_orig = et2.reconstruct_alpha(alpha_orig)
    u_orig = et2.reconstruct_u(alpha_orig).numpy()
    alpha_orig = alpha_orig.numpy()
    u_orig_1 = u_orig[0, 1:3]
    u_orig_2 = np.asarray([[u_orig[0, 3], u_orig[0, 4]], [u_orig[0, 4], u_orig[0, 5]]])
    alpha_orig_1 = alpha_orig[0, 1:3]
    alpha_orig_2 = np.asarray([[alpha_orig[0, 3], alpha_orig[0, 4]], [alpha_orig[0, 4], alpha_orig[0, 5]]])
    # Rotate
    G = create_rotator(u_orig_1)
    u_rot_1 = rotate_m1(u_orig_1, G)
    alpha_rot_1 = rotate_m1(alpha_orig_1, G)
    alpha_rot_2 = rotate_m2(alpha_orig_2, G)
    u_rot_2 = rotate_m2(u_orig_2, G)
    t = np.multiply(alpha_rot_2, u_rot_2)
    dot1 = np.sum(t)
    t2 = np.multiply(alpha_orig_2, u_orig_2)
    dot2 = np.sum(t2)

    dot1_ = m2_inner_prod(u_orig[0, 0], u_orig_1, u_orig_2, alpha_orig[0, 0], alpha_orig_1, alpha_orig_2)
    dot2_ = m2_inner_prod(u_orig[0, 0], u_rot_1, u_rot_2, alpha_orig[0, 0], alpha_rot_1, alpha_rot_2)

    u_rot_vec = np.asarray([u_orig[0, 0], u_rot_1[0], u_rot_1[1], u_rot_2[0, 0], u_rot_2[1, 0], u_rot_2[1, 1]])
    alpha_rot_vec = np.asarray(
        [alpha_orig[0, 0], alpha_rot_1[0], alpha_rot_1[1], alpha_rot_2[0, 0], alpha_rot_2[1, 0], alpha_rot_2[1, 1]])

    dot_1 = np.inner(u_orig[0], alpha_orig[0]) + alpha_orig_2[1, 0] * u_orig_2[1, 0]
    dot_2 = np.inner(u_rot_vec, alpha_rot_vec) + alpha_rot_2[1, 0] * u_rot_2[1, 0]

    # compute h
    u_tens = tf.constant(u_orig, dtype=tf.float64)
    alpha_tens = tf.constant(alpha_orig, dtype=tf.float64)
    u_rot_tens = tf.constant([u_rot_vec], dtype=tf.float64)
    alpha_rot_tens = tf.constant([alpha_rot_vec], dtype=tf.float64)
    h_orig = et2.compute_h(u_tens, alpha_tens)
    h_rot = et2.compute_h_rot(u_rot_tens, alpha_rot_tens, alpha_tens)

    return 0


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
