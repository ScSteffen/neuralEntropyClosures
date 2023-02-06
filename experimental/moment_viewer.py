from src.math import EntropyTools
import tensorflow as tf


def main():
    et = EntropyTools(polynomial_degree=3, spatial_dimension=1, gamma=0)

    alpha = tf.constant([[0, 0, 0]], dtype=tf.float64)
    alpha_complete = et.reconstruct_alpha(alpha)

    u = et.reconstruct_u(alpha_complete)
    print(u)
    return 0


if __name__ == '__main__':
    main()
