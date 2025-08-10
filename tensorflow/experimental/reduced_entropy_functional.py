import numpy as np
from src.math import EntropyTools
import matplotlib.pyplot as plt


def reduced_entropy(alpha_r):
    u = 0.1
    h = 1 + np.log(1 / alpha_r * (np.exp(alpha_r) - np.exp(-alpha_r))) - u * alpha_r
    return h


def main():
    tree = [1, 2, 2, 2, "null", 2]

    sym = True
    for i in range(len(tree)):
        if tree[i] != tree[-i - 1]:
            sym = False

    print("---------- Start experiment Suite ------------")

    et1 = EntropyTools(polynomial_degree=1, spatial_dimension=1)

    alpha_r = np.linspace(-10, 10, 100)
    h = reduced_entropy(alpha_r)

    plt.plot(alpha_r, h)
    plt.show()
    return True


if __name__ == '__main__':
    main()
