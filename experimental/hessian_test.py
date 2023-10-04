import numpy as np


def main():
    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100)
    z1 = np.linspace(-2, 2, 100)
    z2 = np.linspace(-2, 2, 100)

    res = hessian_def(theta, z1, z2)
    return 0


def hessian_def(theta, z1_vec, z2_vec):
    res = np.zeros((100, 100, 100))
    for i in range(100):
        for j in range(100):
            for k in range(100):
                u1 = np.cos(theta[i])
                u2 = - np.sin(theta[i])
                z1 = z1_vec[j]
                z2 = z2_vec[k]

                res1 = -1 * (u1 ** 2 - u2 ** 2) * (-u2 * z1 + u1 * z2) ** 2

                res2 = 4 * u1 * u2 * (z1 * z2 * (u2 ** 2 - u1 ** 2) + u1 * u2 * (z1 ** 2 - z2 ** 2))

                res3 = u1 ** 2 * z2 ** 2 - u2 ** 2 * z1 ** 2

                res4 = 2 * u1 * u2 * (u1 * z2 - u2 * z1)

                res[i, j, k] = res1 + res2 + res3 + res4 ** 2

                if res[i, j, k] < -1e-2:
                    print("here")
                    print(res[i, j, k])
                    print(theta[i] / np.pi)
                    print(z1)
                    print(z2)
                    print(u1)
                    print(u2)
                    print('----')

    return res


if __name__ == '__main__':
    main()
