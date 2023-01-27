from src.math import EntropyTools

import tensorflow as tf
import numpy as np


def main():
    et = EntropyTools(polynomial_degree=2, spatial_dimension=1, gamma=0)

    low = -10
    high = 10

    for i in range(100000):
        alpha_red = tf.constant(
            [np.random.uniform(low=low, high=high, size=(2,)), np.random.uniform(low=low, high=high, size=(2,)),
             np.random.uniform(low=low, high=high, size=(2,))], dtype=tf.float64)
        alpha = et.reconstruct_alpha(alpha_red)
        u_orig = et.reconstruct_u(alpha).numpy()
        if np.linalg.norm(u_orig[0] - u_orig[2]) > np.linalg.norm(u_orig[0] - u_orig[1]) + np.linalg.norm(
                u_orig[1] - u_orig[2]):
            print("triangle inequality broken")
            print(u_orig)
            exit(1)

    print("triangle inequality ok")
    return 0


if __name__ == '__main__':
    main()
