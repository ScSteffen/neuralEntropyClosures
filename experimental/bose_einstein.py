from src.math import EntropyToolsBoseEinstein
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    print("---------- Start Experiments Suite ------------")

    et_be = EntropyToolsBoseEinstein(polynomial_degree=1, spatial_dimension=1, gamma=0)
    n_samples = 10
    alphas_1 = np.linspace(-10, 10, n_samples)
    alphas_np = np.zeros((n_samples, 2))
    alphas_np[:, 1] = alphas_1

    # alphas_np.reshape((n_samples, 2))
    alphas_tf = tf.constant(alphas_np, shape=(n_samples, 2))

    alphas_n = alphas_tf  # et_be.reconstruct_alpha(alphas_tf)
    u_n = et_be.reconstruct_u(alphas_n)
    f_u = et_be.reconstruct_f(alphas_n)
    f_u_np = f_u.numpy()
    plt.plot(f_u_np)
    plt.show()
    h_n = et_be.compute_h(u_n, alphas_n)
    return True


if __name__ == '__main__':
    main()
