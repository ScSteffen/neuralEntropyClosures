"""
A Script for proof of work of the  ideas of Max and myself
"""
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

# project imports
import src.utils as utils


def main():
    # load N1 sampling
    filenameData = "data/1D/Monomial_M2_1D_normal.csv"
    inputDim = 3

    [u, alpha, h] = utils.load_data(filenameData, inputDim)
    uN = u[:, 1:]
    alphaN = alpha[:, 1:]
    knn = NearestNeighbors(n_neighbors=8)
    knn.fit(uN)
    nnArray = knn.kneighbors(uN, return_distance=False)

    # look at one point:
    pointIdx = 1661  # 962
    idx0 = nnArray[pointIdx, 0]
    idx1 = nnArray[pointIdx, 1]
    idx2 = nnArray[pointIdx, 2]
    idx3 = nnArray[pointIdx, 3]
    idx4 = nnArray[pointIdx, 4]

    alphas = alphaN[nnArray[pointIdx]]
    alpha3 = alphas[:6]
    us = uN[nnArray[pointIdx]]

    # plt.plot(alphas[:, 0], alphas[:, 1], '*')
    # plt.show()

    # consider point of interest in interior of conv(us[0:3])
    u3 = us[:6]
    uI = [0.00679, 0.53381]  # [-0.977409, 0.9558425]
    # [-0.26989, 0.09263]  # [-0.25122, 0.09257]  # [-0.26408, 0.09251]  # [-0.26245, 0.089]
    plt.plot(u3[0, 0], u3[0, 1], '*')
    plt.plot(u3[1, 0], u3[1, 1], '^')
    plt.plot(u3[2, 0], u3[2, 1], '.')
    # plt.plot(u3[3, 0], u3[3, 1], '+')
    # plt.plot(u3[4, 0], u3[4, 1], '*')
    # plt.plot(u3[5, 0], u3[5, 1], '*')

    plt.plot(uI[0], uI[1], 'o')
    plt.show()

    # orient points
    u3I = u3 - uI
    # plt.plot(u3I[:, 0], u3I[:, 1], '*')
    # plt.plot([0], [0], '*')
    plt.show()

    # calculate triangle A
    v0 = np.linspace(-100, 100, 100)
    v1x = funcV2(u3I[0], alpha3[0], v0)
    v2x = funcV2(u3I[1], alpha3[1], v0)
    v3x = funcV2(u3I[2], alpha3[2], v0)
    v4x = funcV2(u3I[3], alpha3[3], v0)
    v5x = funcV2(u3I[4], alpha3[4], v0)
    v6x = funcV2(u3I[5], alpha3[5], v0)

    plt.plot(alpha3[0:3, 0], alpha3[0:3, 1], '*')

    plt.plot(v0, v1x)
    plt.plot(v0, v2x)
    plt.plot(v0, v3x)
    # plt.plot(v0, v4x)
    # plt.plot(v0, v5x)
    # plt.plot(v0, v6x)
    plt.show()
    return 0


def funcV2(x, gradx, v0):
    # calculate boundary for triangle
    rhs = np.dot(x, gradx)
    return (rhs - x[0] * v0) / x[1]


if __name__ == '__main__':
    main()
