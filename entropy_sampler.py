from src.math_utils import EntropyTools
from src.math_utils import create_sh_rotator_2D
import numpy as np
import csv


def main():
    np.set_printoptions(precision=2)

    max_alpha = 1
    alpha_size = 5
    degree = 2
    et = EntropyTools(polynomial_degree=degree, spatial_dimension=2, gamma=0.1, basis="spherical_harmonics")

    _, M_R_pm = create_sh_rotator_2D(np.asarray([-1, 0]))

    res = []
    for i in range(100000):
        alpha1 = np.random.uniform(low=-max_alpha, high=max_alpha, size=(alpha_size,))

        alpha = et.reconstruct_alpha(alpha1)
        u = et.reconstruct_u(alpha)
        _, M_R = create_sh_rotator_2D(u[1:3])
        h = et.compute_h_dual(u, alpha)
        alpha_res = et.minimize_entropy(u, alpha)
        u_p = M_R @ u
        u_m = M_R_pm @ u_p

        alpha_p = M_R @ alpha
        alpha_m = M_R_pm @ alpha_p
        h_p = et.compute_h_dual(u_p, alpha_p)
        h_m = et.compute_h_dual(u_m, alpha_m)

        res_p = np.append(0, u_p)
        res_p = np.append(res_p, alpha_p)
        res_p = np.append(res_p, h_p)
        res_m = np.append(0, u_m)
        res_m = np.append(res_m, alpha_m)
        res_m = np.append(res_m, h_m)
        res.append(res_p)
        res.append(res_m)

        # res_p = np.append(0, u)
        # res_p = np.append(res_p, alpha)
        # res_p = np.append(res_p, h)
        #
        # res.append(res_p)

    csv_file_name = 'data/2D/SphericalHarmonics_M2_2D_normal_gaussian_gamma1_rot.csv'

    # Write the random vectors to the CSV file
    with open(csv_file_name, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write a header row (optional)
        csv_writer.writerow(
            ['blank', 'u_0', 'u_1', 'u_2', 'u_3', 'u_4', 'u_5', 'alpha_0', 'alpha_1', 'alpha_2', 'alpha_3', 'alpha_4',
             'alpha_5', 'h'])

        # Write each random vector to a separate line
        for vector in res:
            csv_writer.writerow(vector)

    return 0


if __name__ == '__main__':
    main()
