import numpy as np


def main():
    for theta in np.linspace(0, np.pi, 12):
        r = rot_mat(theta)
        print(r @ r.transpose())
        r_2 = rot_m2_mat(r)
        # i_e = r_2 @ r_2.transpose()

        r_2_inv = rot_m2_mat_inv(r)
        a_alpha = r_2_inv.transpose()
        print(r_2_inv.transpose())
        a = r_2 @ r_2_inv
        print(a)
        print("...")

    return 0


def rot_mat(theta):
    return np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def rot_m2_mat(rot_mat):
    r_m2 = np.zeros((3, 3))
    r_m2[0, 0] = rot_mat[0, 0] ** 2
    r_m2[0, 1] = rot_mat[0, 0] * rot_mat[0, 1] * 2
    r_m2[0, 2] = rot_mat[0, 1] ** 2
    r_m2[1, 0] = rot_mat[0, 0] * rot_mat[1, 0]
    r_m2[1, 1] = rot_mat[0, 0] * rot_mat[1, 1] + rot_mat[0, 1] * rot_mat[1, 0]
    r_m2[1, 2] = rot_mat[0, 1] * rot_mat[1, 1]
    r_m2[2, 0] = rot_mat[1, 0] ** 2
    r_m2[2, 1] = rot_mat[1, 1] * rot_mat[1, 0] * 2
    r_m2[2, 2] = rot_mat[1, 1] ** 2

    return r_m2.transpose()


def rot_m2_mat_inv(rot_mat):
    r_m2 = np.zeros((3, 3))
    r_m2[0, 0] = rot_mat[0, 0] ** 2
    r_m2[0, 1] = rot_mat[0, 0] * rot_mat[0, 1]
    r_m2[0, 2] = rot_mat[0, 1] ** 2
    r_m2[1, 0] = rot_mat[0, 0] * rot_mat[1, 0] * 2
    r_m2[1, 1] = rot_mat[0, 0] * rot_mat[1, 1] + rot_mat[0, 1] * rot_mat[1, 0]
    r_m2[1, 2] = rot_mat[0, 1] * rot_mat[1, 1] * 2
    r_m2[2, 0] = rot_mat[1, 0] ** 2
    r_m2[2, 1] = rot_mat[1, 1] * rot_mat[1, 0]
    r_m2[2, 2] = rot_mat[1, 1] ** 2

    return r_m2


if __name__ == '__main__':
    main()
