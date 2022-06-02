import numpy as np


def main():
    theta = np.pi / 4

    c = np.cos(theta)
    s = np.sin(theta)

    R = np.asarray([[c ** 2, 2 * c * (-s), s ** 2],
                    [c * s, c ** 2 - s ** 2, -s * c],
                    [s ** 2, 2 * c * s, c ** 2]])
    # R = np.asarray([[c ** 2, 1 * c * (-s), s ** 2], [c * s, 1 * s ** 2, -s * c], [s ** 2, 1 * c * s, c ** 2]])

    Rt = np.asarray([[c ** 2, c * (-s), s ** 2],
                     [2 * c * s, (c ** 2 - s ** 2), -2 * s * c],
                     [s ** 2, c * s, c ** 2]])

    R_inverse = np.linalg.inv(R)

    print(R)
    print("----")
    print(R_inverse)
    print(Rt.T)
    print("---")
    print(np.matmul(R, Rt.T))
    # print(np.matmul(R, R.T))
    print("----")

    R_tilde = np.asarray([[c, -s], [s, c]])
    temp = [R_tilde[0, 0] * R_tilde[0, 0], R_tilde[0, 1] * R_tilde[0, 0] * 2, R_tilde[0, 1] * R_tilde[0, 1]]
    print(temp)

    ## sanity check
    print("----")
    u = np.asarray([1, 2, 3])
    u_t = np.asarray([[1, 2], [2, 3]])
    print(u)
    print(u_t)
    uR = np.matmul(R, u)
    u_tR = np.matmul(R_tilde, np.matmul(u_t, R_tilde.T))
    print(uR)
    print(u_tR)

    print("--- sanity check for alpha ---")
    a = np.asarray([1, 2, 3])
    a_t = np.asarray([[1, 1], [1, 3]])
    aR = np.matmul(Rt, a)
    a_tR = np.matmul(R_tilde, np.matmul(a_t, R_tilde.T))
    print(aR)
    print(a_tR)
    return 0


if __name__ == '__main__':
    main()
