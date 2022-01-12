"""
brief: Testing ground for 1D moment solver
Author: Steffen Schotthöfer
Date: 17.05.2021
"""
import sys
import csv

sys.path.append('../..')
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib import animation
import tensorflow as tf
import multiprocessing
import pandas as pd
import os
# inpackage imports
# from networks.configModel import initNeuralClosure
from src import math
from src.networks.configmodel import init_neural_closure

num_cores = multiprocessing.cpu_count()


def main():
    solver = MNSolver1D(traditional=False, polyDegree=2)
    # solver.solve_animation(maxIter=100)
    # solver.solve_animation_iter_error(maxIter=100)
    # solver.solve_iter_error(maxIter=100)
    solver.solve(max_iter=2000)
    return 0


class MNSolver1D:

    def __init__(self, traditional=False, polyDegree=3, model_mk=11, gridsize=80):

        # Prototype for  spatialDim=1, polyDegree=2
        self.model_mk = model_mk
        self.n_system = polyDegree + 1
        self.polyDegree = polyDegree
        self.quadOrder = 200
        self.traditional = traditional
        [self.quadPts, self.quadWeights] = math.qGaussLegendre1D(self.quadOrder)  # dims = nq
        self.nq = self.quadWeights.size
        self.mBasis = math.computeMonomialBasis1D(self.quadPts, self.polyDegree)  # dims = (N x nq)
        self.inputDim = self.mBasis.shape[0]  # = self.nSystem

        # generate geometry
        self.x0 = 0
        self.x1 = 0.5
        self.nx = gridsize  # 1280
        self.dx = (self.x1 - self.x0) / self.nx

        # physics (isotropic, homogenious)
        self.sigmaS = 1.0
        self.scatter_vector = np.zeros(self.n_system)
        for i in range(self.n_system):
            if i % 2 == 0:
                self.scatter_vector[i] = 1.0 / float(i + 1)

        # time
        self.tEnd = 1.0
        self.cfl = 0.5  # 0.3
        self.dt = round(self.cfl * self.dx, 12)

        # boundary
        self.boundary = 1  # 0 = periodic, 1 = neumann with l.h.s. source,
        if self.boundary == 0:
            print("Periodic boundary conditions")
        else:
            print("Dirichlet boundary conditions")
        # self.datafile = "data_file_1D_M" + str(self.polyDegree) + "_MK" + str(self.model_mk) + "_inflow.csv"
        # self.solution_file = "1D_M" + str(self.polyDegree) + "_MK" + str(self.model_mk) + "_inflow.csv"
        # self.errorfile = "err_1D_M" + str(self.polyDegree) + "_MK" + str(self.model_mk) + "_inflow.csv"
        self.datafile = "data_file_1D_M" + str(self.polyDegree) + "_MK" + str(self.model_mk) + "_periodic.csv"
        self.solution_file = "1D_M" + str(self.polyDegree) + "_MK" + str(self.model_mk) + "_periodic.csv"
        self.errorfile = "err_1D_M" + str(self.polyDegree) + "_MK" + str(self.model_mk) + "_periodic.csv"
        # Solver variables Traditional
        self.u = self.ic_zero()  # self.ic_periodic()  # self.ic_zero()  #
        self.alpha = np.zeros((self.n_system, self.nx))
        self.xFlux = np.zeros((self.n_system, self.nx + 1), dtype=float)
        self.h = np.zeros(self.nx)
        self.h2 = np.zeros(self.nx)

        self.u2 = self.ic_zero()  # self.ic_periodic()  # self.ic_zero()  #
        self.alpha2 = np.zeros((self.n_system, self.nx))
        self.xFlux2 = np.zeros((self.n_system, self.nx + 1), dtype=float)
        # Neural closure
        self.neuralClosure = None
        print("Using tensorflow with version:")
        print(tf.__version__)
        self.legacy_model = False
        if not self.traditional:
            if self.model_mk == 11:
                if self.polyDegree == 1:
                    self.neuralClosure = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1,
                                                             folder_name="tmp",
                                                             loss_combination=2,
                                                             nw_width=10, nw_depth=7, normalized=True)
                    self.neuralClosure.create_model()
                    ### Need to load this model as legacy code
                    print("Load model in legacy mode. Model was created using tf 2.2.0")
                    self.legacy_model = True
                    imported = tf.keras.models.load_model("models/_simulation/mk11_M1_1D/best_model")
                    self.neuralClosure.model_legacy = imported
                elif self.polyDegree == 2:
                    self.neuralClosure = init_neural_closure(network_mk=11, poly_degree=2, spatial_dim=1,
                                                             folder_name="tmp",
                                                             loss_combination=2,
                                                             nw_width=15, nw_depth=7, normalized=True)
                    self.neuralClosure.create_model()
                    ### Need to load this model as legacy code
                    print("Load model in legacy mode. Model was created using tf 2.2.0")
                    self.legacy_model = True
                    imported = tf.keras.models.load_model("models/_simulation/mk11_M2_1D/best_model")
                    self.neuralClosure.model_legacy = imported

                elif self.polyDegree == 3:
                    self.neuralClosure = init_neural_closure(network_mk=13, poly_degree=3, spatial_dim=1,
                                                             folder_name="002_sim_M3_1D", loss_combination=2,
                                                             nw_width=20, nw_depth=7, normalized=True)
                    self.neuralClosure.loadModel("../../models/002_sim_M3_1D")
            elif self.model_mk == 15:
                if self.polyDegree == 1:
                    self.neuralClosure = init_neural_closure(network_mk=self.model_mk, poly_degree=1, spatial_dim=1,
                                                             folder_name="_simulation/mk15_M1_1D",
                                                             loss_combination=2, nw_width=30, nw_depth=2,
                                                             normalized=True, input_decorrelation=True,
                                                             scale_active=True)
                    self.neuralClosure.load_model()
                if self.polyDegree == 2:
                    self.neuralClosure = init_neural_closure(network_mk=self.model_mk, poly_degree=2, spatial_dim=1,
                                                             folder_name="_simulation/mk15_M2_1D",
                                                             loss_combination=2, nw_width=50, nw_depth=2,
                                                             normalized=True, input_decorrelation=True,
                                                             scale_active=True)
                    self.neuralClosure.load_model()
                elif self.polyDegree == 3:
                    self.neuralClosure = init_neural_closure(network_mk=13, poly_degree=3, spatial_dim=1,
                                                             folder_name="_simulation/mk15_M3_1D", loss_combination=2,
                                                             nw_width=20, nw_depth=7, normalized=True)
                    self.neuralClosure.loadModel("../../models/002_sim_M3_1D")

        # Analysis variables
        self.errorMap = np.zeros((self.n_system, self.nx))
        self.normErrorMap = np.zeros(self.nx)
        self.realizabilityMap = np.zeros(self.nx)
        columns = ['u0', 'u1', 'u2', 'alpha0', 'alpha1', 'alpha2', 'h']  # , 'realizable']
        self.dfErrPoints = pd.DataFrame(columns=columns)
        with open('figures/solvers/' + self.errorfile, 'w+', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            row = ["t", "err_u", "err_alpha", "int_h", "int_h_ref"]
            writer.writerow(row)

        with open('figures/solvers/' + self.datafile, 'w+', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            row = ["t", "u_0", "u_1", "u_2", "alpha_0", "alpha_1", "alpha_2", "entropy"]
            writer.writerow(row)
        with open('figures/solvers/' + self.solution_file, 'w+', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            if self.polyDegree == 1:
                row = ["u_0", "u_1", "u_0_ref", "u_1_ref"]
            elif self.polyDegree == 2:
                row = ["u_0", "u_1", "u_2", "u_0_ref", "u_1_ref", "u_2_ref"]
            writer.writerow(row)

    def get_realizable_moment(self, value=1.0):
        u_ic0 = value * np.ones((1,))
        u_ic1 = np.zeros((1,))
        erg = np.concatenate([u_ic0, u_ic1], axis=0)
        if self.polyDegree >= 2:
            u_ic2 = 0.5 * value * np.ones((self.n_system - 2,))
            erg = np.concatenate([u_ic0, u_ic1, u_ic2], axis=0)
        return erg

    def boundary_inflow(self):
        """
        Gives the Flux for an inflow condition with from the left
        with f(t>0,0,v>0) = 0.5
        """
        f = 0.5 * np.ones((self.nq))

        for idx_quad in range(self.nq):
            if self.quadPts[idx_quad] <= 0:
                f[idx_quad] *= 0.0
        return f

    def ic_zero(self):
        self.boundary = 1
        f0 = 0.0001 * np.ones(self.nq)
        t = np.multiply(f0, self.mBasis)
        t2 = np.multiply(t, self.quadWeights)
        u_ic_pt = np.sum(t2, axis=1)
        u_ic = np.zeros((self.n_system, self.nx))
        for idx_cell in range(self.nx):
            u_ic[:, idx_cell] = u_ic_pt
        return u_ic

    def ic_periodic(self):
        def sincos(x):
            return 1.5 + np.cos(2 * np.pi * x)

        u_ic = np.zeros((self.n_system, self.nx))

        for i in range(self.nx):
            x_koor = self.x0 + (i - 0.5) * self.dx
            u_ic[0, i] = sincos(x_koor)
            u_ic[1, i] = 0.3 * u_ic[0, i]
            if self.polyDegree > 1:
                u_ic[2, i] = 0.5 * u_ic[0, i]
            if self.polyDegree > 2:
                u_ic[3, i] = 0.5 * u_ic[0, i]

        return u_ic

    def ic_linesource(self):
        """
        brief: linesource test case
        """

        def normal_dist(x, mean, sd):
            prob_density = 1 / (4.0 * np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)

            return max(prob_density, 0.001)

        u_ic = np.zeros((self.n_system, self.nx))

        for i in range(self.nx):
            x_koor = self.x0 + (i - 0.5) * self.dx
            u_ic[0, i] = normal_dist(x_koor, 0.0, 0.01)
            u_ic[1, i] = 0.0
            if self.polyDegree >= 2:
                u_ic[2, i] = 0.3 * u_ic[0, i]
            if self.polyDegree >= 3:
                u_ic[3, i] = 0.005 * u_ic[0, i]

        print("using linesource initial conditions")

        return u_ic

    def ic_soft_linesource(self):
        """
        brief: linesource test case
        """

        def coscos(x):
            if -0.49 < x < 0.49:
                return np.cos(x) * np.cos(x)
            else:
                return 0.01

        u_ic = np.zeros((self.n_system, self.nx))

        for i in range(self.nx):
            x_koor = self.x0 + (i - 0.5) * self.dx
            u_ic[0, i] = coscos(x_koor)
            u_ic[1, i] = 0.0
            if self.polyDegree >= 2:
                u_ic[2, i] = 0.05 * u_ic[0, i]
            if self.polyDegree >= 3:
                u_ic[3, i] = 0.05 * u_ic[0, i]

        print("using soft linesource initial conditions")

        return u_ic

    def ic_bump(self):
        u_ic = np.zeros((self.n_system, self.nx))
        for i in range(self.nx):
            x_koor = self.x0 + (i - 0.5) * self.dx
            if 1 > x_koor > -1:
                u_ic[0, i] = 1.0
                u_ic[1, i] = 0.0
                if self.polyDegree >= 2:
                    u_ic[2, i] = 0.5
                if self.polyDegree == 3:
                    N1 = u_ic[1, i] / u_ic[0, i]
                    N2 = u_ic[2, i] / u_ic[0, i]
                    upper = N2 - (N1 - N2) ** 2 / (1 - N1)
                    lower = - N2 + (N1 + N2) ** 2 / (1 + N1)
                    u_ic[3, i] = (upper + lower / 2) * u_ic[0, i]
            else:
                u_ic[0, i] = 0.5
                u_ic[1, i] = 0.0
                if self.polyDegree >= 2:
                    u_ic[2, i] = 0.25
                # u_ic[2, i] = 0.25
                if self.polyDegree == 3:
                    N1 = u_ic[1, i] / u_ic[0, i]
                    N2 = u_ic[2, i] / u_ic[0, i]
                    upper = N2 - (N1 - N2) ** 2 / (1 - N1)
                    lower = - N2 + (N1 + N2) ** 2 / (1 + N1)
                    u_ic[3, i] = (upper + lower / 2) * u_ic[0, i]

            # uIc[0, i] = sincos(x=xKoor)
            # uIc[1, i] = 0.0  # 0.8 * uIc[0, i]  # 0.5 * uIc[0, i]  # realizable
            # uIc[2, i] = 0.1 * uIc[0, i]  # 1 + (0.8 ** 2 + 0.05) * uIc[
            #    0, i]  # uIc[1, i] ** 2 + 0.1  # uIc[1, i] ** 2 + (1 - uIc[1, i] ** 2) / 2  # realizable

            # if self.polyDegree == 3:
            #    N1 = uIc[1, i] / uIc[0, i]
            #    N2 = uIc[2, i] / uIc[0, i]
            #    uIc[3, i] = -N2 + (N1 + N2) ** 2 / (1 + N1) + 0.002  # error!
        return u_ic

    def solve(self, max_iter=100, t_end=1.0):
        max_iter = int(t_end / self.dt)
        print("The simulation will run for " + str(max_iter) + " iterations.")
        print("The grid consists of " + str(self.nx) + " cells.")
        print("The time step size is " + str(self.dt))
        print("After " + str(max_iter) + " steps. The simulation has reached time " + str(max_iter * self.dt))
        idx_time = 0
        while idx_time <= max_iter and idx_time * self.dt <= t_end:  # max_iter
            if idx_time <= max_iter - 50:
                self.solve_iter_newton(idx_time)
                if idx_time == max_iter - 50:
                    self.u2 = self.u
            else:
                self.solver_iter_ml(idx_time)
            print("Iteration: " + str(idx_time) + " of " + str(max_iter) + ". Time " + str(
                idx_time * self.dt) + " of " + str(t_end))
            # self.error_analysis(idx_time * self.dt)
            # print iteration results
            # self.show_solution(idx_time)
            idx_time += 1
        # self.show_solution(idx_time)
        # self.write_solution()
        # self.write_solution_banach()
        self.write_solution_banach_ml()
        return self.u

    def solve_animation_iter_error(self, maxIter=100):
        fps = 1 / self.dt

        # First set up the figure, the axis, and the plot element we want to animate
        fig, ax = plt.subplots()

        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-0.15, 1.15))
        line1, = ax.plot([], [], "ro", label="u0_ML")
        line2, = ax.plot([], [], "ro", label="u1_ML")
        line3, = ax.plot([], [], "ro", label="u2_ML")

        line4, = ax.plot([], [], "k-", label="u0_trad")
        line5, = ax.plot([], [], "k--", label="u1_trad")
        line6, = ax.plot([], [], "k:", label="u2_trad")

        if self.polyDegree == 3:
            line7, = ax.plot([], [], "ro", label="u3_ML")
            line8, = ax.plot([], [], "k.", label="u3_trad")

        x = np.linspace(self.x0, self.x1, self.nx)

        ax.legend()

        def animate_func(i):
            # entropy closure and
            self.entropy_closure_newton()
            # reconstruction
            self.realizability_reconstruction()
            # entropy closure and
            self.compute_closure_ml()
            self.compare_and_retrain()

            # flux computation
            self.compute_flux_newton()
            # FVM update
            self.fvm_update_newton()

            # flux computation
            self.compute_flux_ml()
            # FVM update
            self.fvm_update_ml()

            # self.solve_iter_newton(i)
            # self.solve_iter_ml(i)

            # step by step execution

            # self.compare_and_retrain()

            print("Iteration: " + str(i))

            # ax.plot(x, self.u2[0, :])
            line1.set_data(x, self.u2[0, :])
            line2.set_data(x, self.u2[1, :])
            line3.set_data(x, self.u2[2, :])
            if self.polyDegree == 3:
                line7.set_data(x, self.u2[3, :])
            line4.set_data(x, self.u[0, :])
            line5.set_data(x, self.u[1, :])
            line6.set_data(x, self.u[2, :])
            if self.polyDegree == 3:
                line8.set_data(x, self.u[3, :])
                return [line1, line2, line3, line4, line5, line6, line7, line8]

            return [line1, line2, line3, line4, line5, line6]

        # anim = animation.FuncAnimation(fig, animate_func, frames=maxIter, interval=10000 * self.dt)
        anim = animation.FuncAnimation(fig, animate_func, frames=maxIter, interval=20000 * self.dt, blit=True)
        if self.traditional:
            filename = "newton_version.gif"
        else:
            filename = "ErrorPerIter.gif"
        # anim.save('ErrorPerIter.gif', writer='imagemagick', fps=60)
        anim.save(filename, writer=animation.PillowWriter(fps=fps))

    def solve_iter_newton(self, t_idx):
        # entropy closure and
        self.entropy_closure_newton()
        # reconstruction
        # self.realizability_reconstruction()
        # flux computation
        self.compute_flux_newton()
        # FVM update
        self.fvm_update_newton()
        return 0

    def solver_iter_ml(self, t_idx):
        # entropy closure and
        self.compute_closure_ml()
        # flux computation
        self.compute_flux_ml()
        # FVM update
        self.fvm_update_ml()
        return 0

    def entropy_closure_newton(self):
        # if (self.traditional): # NEWTON
        for i in range(self.nx):
            self.entropy_closure_single_row(i)
        return 0

    def entropy_closure_single_row(self, i):
        rowRes = 0

        opti_u = self.u[:, i]
        alpha_init = self.alpha[:, i]
        # test objective functions
        # t = self.create_opti_entropy(opti_u)(alpha_init)
        # tp = self.create_opti_entropy_prime(opti_u)(alpha_init)
        # t = self.create_opti_entropy_hessian()(alpha_init)
        # print(t)
        # print(tp)
        normU = np.abs(self.u[1, i])
        u0 = self.u[0, i]
        if u0 == 0:
            print("u0 = 0")
        # elif normU / u0 > 0.95:
        #    print("Warning")
        opt_result = opt.minimize(fun=self.create_opti_entropy(opti_u), x0=alpha_init,
                                  jac=self.create_opti_entropy_prime(opti_u),
                                  tol=1e-6)
        # hess=self.create_opti_entropy_hessian(),
        # method='Newton-CG',
        if not opt_result.success:
            print("Optimization unsuccessfull! u=" + str(opti_u))
            exit(ValueError)
        else:
            self.alpha[:, i] = opt_result.x
            rowRes = opt_result.x
            self.h[i] = opt_result.fun
        return rowRes

    def create_opti_entropy(self, u):
        def opti_entropy(alpha):
            """
            brief: returns the negative entropy functional with fixed u
            nS = batchSize
            N = basisSize
            nq = number of quadPts
            input: alpha, dims = (1 x N)
                   u, dims = (1 x N)
            used members: m    , dims = (N x nq)
                        w    , dims = nq
            returns h = - alpha*u + <eta_*(alpha*m)>
            """
            # Currently only for maxwell Boltzmann entropy
            # compute negative entropy functional
            f_quad = np.exp(np.tensordot(alpha, self.mBasis, axes=([0], [0])))  # alpha*m
            t1 = np.tensordot(f_quad, self.quadWeights, axes=([0], [0]))  # f*w
            t2 = np.inner(alpha, u)
            return t1 - t2

        return opti_entropy

    def create_opti_entropy_prime(self, u):
        def opti_entropy_prime(alpha):
            """
             brief: returns the derivative negative entropy functional with fixed u
             nS = batchSize
             N = basisSize
             nq = number of quadPts

             input: alpha, dims = (1 x N)
                    u, dims = (1 x N)
             used members: m    , dims = (N x nq)
                         w    , dims = nq

             returns h = - alpha + <m eta_*(alpha*m)>
            """
            # Currently only for maxwell Boltzmann entropy
            f_quad = np.exp(np.tensordot(alpha, self.mBasis, axes=([0], [0])))  # alpha*m
            tmp = np.multiply(f_quad, self.quadWeights)  # f*w
            t2 = np.tensordot(tmp, self.mBasis, axes=([0], [1]))  # f * w * momentBasis
            return t2 - u

        return opti_entropy_prime

    def create_opti_entropy_hessian(self):
        def opti_entropy_hessian(alpha):
            """
             brief: returns the derivative negative entropy functional with fixed u
             nS = batchSize
             N = basisSize
             nq = number of quadPts

             input: alpha, dims = (1 x N)
                    u, dims = (1 x N)
             used members: m    , dims = (N x nq)
                         w    , dims = nq

             returns h = <mxm eta_*(alpha*m)>
            """
            # Currently only for maxwell Boltzmann entropy

            f_quad = np.exp(np.tensordot(alpha, self.mBasis, axes=([0], [0])))  # alpha*m
            tmp = np.multiply(f_quad, self.quadWeights)  # f*w
            t2 = 0
            for i in range(self.nq):
                t = np.tensordot(self.mBasis[:, i], self.mBasis[:, i], axes=0)
                t2 += t * tmp[i]
            return t2

        return opti_entropy_hessian

    def realizability_reconstruction(self):
        for i in range(self.nx):
            # self.u2[:, i] = np.copy(self.u[:, i])
            a = np.reshape(self.alpha[:, i], (1, self.n_system))
            self.u[:, i] = math.reconstructU(alpha=a, m=self.mBasis, w=self.quadWeights)
            # print("(" + str(self.u2[:, i]) + " | " + str(self.u[:, i]))
            # h = self.create_opti_entropy(self.u[:, i])(self.alpha[:, i])
            # row = [0, self.u[0, i], self.u[1, i], self.u[2, i], self.alpha[0, i], self.alpha[1, i],
            #       self.alpha[2, i], h]

        return 0

    def compare_and_retrain(self):
        # open the file in the write mode
        with open('figures/solvers/csv_writeout/Monomial_M2_1D.csv', 'a+', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            for i in range(self.nx):
                h = self.create_opti_entropy(self.u[:, i])(self.alpha[:, i])
                row = [0, self.u[0, i], self.u[1, i], self.u[2, i], self.alpha[0, i], self.alpha[1, i],
                       self.alpha[2, i], h]
                writer.writerow(row)

                h = self.create_opti_entropy(self.u2[:, i])(self.alpha2[:, i])
                row = [1, self.u2[0, i], self.u2[1, i], self.u2[2, i], self.alpha2[0, i], self.alpha2[1, i],
                       self.alpha2[2, i], h]
                # write a row to the csv file
                writer.writerow(row)
        return 0

    def compute_flux_newton(self):
        """
        for periodic boundaries and inflow boundaries, upwinding.
        writes to xFlux and yFlux, uses alpha
        """
        # reconstruct all densities
        densities = np.zeros((self.nx, self.nq))
        for i in range(self.nx):
            densities[i, :] = math.entropyDualPrime(np.tensordot(self.alpha[:, i], self.mBasis, axes=([0], [0])))
        # compute fluxes in interior cell faces
        for i in range(1, self.nx):
            flux = np.zeros(self.n_system)
            for q in range(self.nq):  # integrate upwinding result
                upwind = self.upwinding(densities[i - 1, q], densities[i, q], self.quadPts[q])
                flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
            self.xFlux[:, i] = flux
        # compute boundary layer fluxes
        if self.boundary == 0:  # periodic boundary
            flux = np.zeros(self.n_system)
            for q in range(self.nq):  # integrate upwinding result
                upwind = self.upwinding(densities[self.nx - 1, q], densities[0, q], self.quadPts[q])
                flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
            self.xFlux[:, 0] = flux
            self.xFlux[:, self.nx] = flux
        if self.boundary == 1:  # neumann with l.h.s inflow
            flux_lhs = self.boundary_inflow()
            flux = np.zeros(self.n_system)
            for q in range(self.nq):  # integrate upwinding result
                upwind = self.upwinding(flux_lhs[q], densities[0, q], self.quadPts[q])
                flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
            self.xFlux[:, 0] = flux  # inflow conditions
            self.xFlux[:, self.nx] = self.xFlux[:, self.nx - 1]  # do nothing conditions
        return 0

    def upwinding(self, fluxL, fluxR, quadpt):
        # t = np.inner(quadpt, normal)
        if quadpt > 0:
            return quadpt * fluxL
        else:
            return quadpt * fluxR

    def fvm_update_newton(self):
        for i in range(self.nx):
            ip1 = i + 1
            # advection
            self.u[:, i] = self.u[:, i] + ((self.xFlux[:, i] - self.xFlux[:, ip1]) / self.dx) * self.dt
            # Scattering
            self.u[:, i] += self.dt * self.sigmaS * (self.scatter_vector * self.u[0, i] - self.u[:, i])
        return 0

    def compute_closure_ml(self):
        tmp = np.copy(np.transpose(self.u2))
        for i in range(self.nx):
            if tmp[i, 0] < 0.0001:
                tmp[i, 0] = 0.0001
        [u_pred, alpha_pred, h] = self.neuralClosure.call_scaled_64(np.asarray(tmp), legacy_mode=self.legacy_model)

        for i in range(self.nx):
            self.alpha2[:, i] = alpha_pred[i, :]
            self.h2[i] = h[i]

        return 0

    def compute_flux_ml(self):
        """
        for periodic boundaries and inflow boundaries, upwinding.
        writes to xFlux and yFlux, uses alpha
        """
        # reconstruct all densities
        densities = np.zeros((self.nx, self.nq))
        for i in range(self.nx):
            densities[i, :] = math.entropyDualPrime(np.tensordot(self.alpha2[:, i], self.mBasis, axes=([0], [0])))
        # compute fluxes in interior cell faces
        for i in range(1, self.nx):
            flux = np.zeros(self.n_system)
            for q in range(self.nq):  # integrate upwinding result
                upwind = self.upwinding(densities[i - 1, q], densities[i, q], self.quadPts[q])
                flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
            self.xFlux2[:, i] = flux
        # compute boundary layer fluxes
        if self.boundary == 0:  # periodic boundary
            flux = np.zeros(self.n_system)
            for q in range(self.nq):  # integrate upwinding result
                upwind = self.upwinding(densities[self.nx - 1, q], densities[0, q], self.quadPts[q])
                flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
            self.xFlux2[:, 0] = flux
            self.xFlux2[:, self.nx] = flux
        if self.boundary == 1:  # neumann with l.h.s inflow
            flux_lhs = self.boundary_inflow()
            flux = np.zeros(self.n_system)
            for q in range(self.nq):  # integrate upwinding result
                upwind = self.upwinding(flux_lhs[q], densities[0, q], self.quadPts[q])
                flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
            self.xFlux2[:, 0] = flux  # inflow conditions
            self.xFlux2[:, self.nx] = self.xFlux[:, self.nx - 1]  # do nothing conditions
        return 0

    def fvm_update_ml(self):
        for i in range(self.nx):
            ip1 = i + 1
            # advection
            self.u2[:, i] = self.u2[:, i] + ((self.xFlux2[:, i] - self.xFlux2[:, ip1]) / self.dx) * self.dt
            # Scattering
            self.u2[:, i] += self.dt * self.sigmaS * (self.scatter_vector * self.u2[0, i] - self.u2[:, i])
        return 0

    def show_solution(self, idx):
        plt.clf()
        x = np.linspace(self.x0, self.x1, self.nx)

        plt.plot(x, self.u[0, :], "k-", linewidth=1, label="Newton u0")
        plt.plot(x, self.u2[0, :], 'o', markersize=2, markerfacecolor='orange',
                 markeredgewidth=0.5, markeredgecolor='k', label="Neural u0")
        plt.xlim([self.x0, self.x1])
        # plt.ylim([0.0, 1.5])
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        # plt.savefig("figures/solvers/u_0_comparison_" + str(idx) + ".png", dpi=450)
        # plt.clf()

        plt.plot(x, self.u[1, :], "k-", linewidth=1, label="Newton u1")
        plt.plot(x, self.u2[1, :], 'o', markersize=2, markerfacecolor='orange',
                 markeredgewidth=0.5, markeredgecolor='k', label="Neural u1")
        plt.xlim([self.x0, self.x1])
        # plt.ylim([0.0, 1.5])
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        # plt.savefig("figures/solvers/u_1_comparison_" + str(idx) + ".png", dpi=450)
        # plt.clf()

        if self.polyDegree >= 2:
            plt.plot(x, self.u[2, :], "k-", linewidth=1, label="Newton u2")
            plt.plot(x, self.u2[2, :], 'o', markersize=2, markerfacecolor='orange',
                     markeredgewidth=0.5, markeredgecolor='k', label="Neural u2")
            plt.xlim([self.x0, self.x1])
            # plt.ylim([0.0, 1.5])
            plt.xlabel("x")
            plt.ylabel("u")
            plt.legend()
        plt.savefig("figures/solvers/u_comparison_" + str(idx) + "_" + str(self.nx) + "_" + str(self.model_mk) + ".png",
                    dpi=450)
        plt.clf()

        err = np.linalg.norm(self.u - self.u2, axis=0)
        plt.plot(x, err, "k-", linewidth=1, label="Newton closure")
        plt.xlim([self.x0, self.x1])
        # plt.ylim([0.0, 1.5])
        plt.xlabel("x")
        plt.ylabel("norm(u-u-theta)")
        plt.legend()
        plt.savefig("figures/solvers/error" + str(idx) + ".png", dpi=450)
        plt.clf()
        # plt.show()
        return 0

    def error_analysis(self, time):

        # mean absulote error
        with open('figures/solvers/' + self.datafile, 'a+', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            # writer.writerow("u0,u1,u2,u0_ref,u1_ref,u2_ref")
            for i in range(self.nx):
                if self.polyDegree == 1:
                    row = [i, self.u[0, i], self.u[1, i], self.alpha[0, i], self.alpha[1, i],
                           self.h[i]]
                elif self.polyDegree == 2:
                    row = [i, self.u[0, i], self.u[1, i], self.u[2, i], self.alpha[0, i], self.alpha[1, i],
                           self.alpha[2, i], self.h[i]]
                writer.writerow(row)

        err_u = np.linalg.norm(self.u - self.u2, axis=0)
        rel_err_u = err_u / np.linalg.norm(self.u, axis=0)
        avg_rel_err_u = np.sum(rel_err_u) / self.nx
        err_alpha = np.linalg.norm(self.alpha - self.alpha2, axis=0)
        rel_err_alpha = err_alpha / np.linalg.norm(self.alpha, axis=0)
        avg_rel_err_alpha = np.sum(rel_err_alpha) / self.nx
        entropy_orig = - self.h.sum() * self.dx
        entropy_ml = self.h2.sum() * self.dx

        with open('figures/solvers/' + self.errorfile, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time, avg_rel_err_u, avg_rel_err_alpha, entropy_ml, entropy_orig])
        print("Error data written")
        return 0

    def write_solution(self):
        with open('figures/solvers/' + self.solution_file, 'a+', newline='') as f:
            # create the csv writer
            writer = csv.writer(f)
            # writer.writerow("u0,u1,u2,u0_ref,u1_ref,u2_ref")
            for i in range(self.nx):
                if self.polyDegree == 1:
                    row = [self.u2[0, i], self.u2[1, i], self.u[0, i], self.u[1, i]]
                elif self.polyDegree == 2:
                    row = [self.u2[0, i], self.u2[1, i], self.u2[2, i], self.u[0, i], self.u[1, i], self.u[2, i]]
                # row = [iter, self.u[0, i], self.u[1, i], self.u[2, i], self.alpha[0, i], self.alpha[1, i],
                #       self.alpha[2, i], self.h[i]]
                writer.writerow(row)
        return 0

    def write_solution_banach(self):
        boundary_str = "periodic_M" + str(self.polyDegree)
        if self.boundary == 1:
            boundary_str = "inflow_M" + str(self.polyDegree)
        print("Writing solution to")
        print('paper_data/banach/' + boundary_str + '/solution_' + str(self.nx) + '.csv')
        with open(
                'paper_data/banach/' + boundary_str + '/solution_' + str(self.nx) + '.csv',
                'w+', newline='') as f:
            writer = csv.writer(f)
            if self.polyDegree == 1:
                row = ["x", "u0", "u1"]
            elif self.polyDegree == 2:
                row = ["x", "u0", "u1", "u2"]
            writer.writerow(row)
            for i in range(self.nx):
                if self.polyDegree == 1:
                    row = [i * self.dx, self.u[0, i], self.u[1, i]]
                elif self.polyDegree == 2:
                    row = [i * self.dx, self.u[0, i], self.u[1, i], self.u[2, i]]
                writer.writerow(row)
        return 0

    def write_solution_banach_ml(self):
        boundary_str = "periodic_M" + str(self.polyDegree)
        if self.boundary == 1:
            boundary_str = "inflow_M" + str(self.polyDegree)
        print("Writing solution to")
        print('paper_data/banach/' + boundary_str + '/solution_ml_mk' + str(self.model_mk) + '_' + str(
            self.nx) + '.csv')
        with open('paper_data/banach/' + boundary_str + '/solution_ml_mk' + str(self.model_mk) + '_' + str(
                self.nx) + '.csv', 'w+', newline='') as f:
            writer = csv.writer(f)
            if self.polyDegree == 1:
                row = ["x", "u0", "u1"]
            elif self.polyDegree == 2:
                row = ["x", "u0", "u1", "u2"]
            writer.writerow(row)
            for i in range(self.nx):
                if self.polyDegree == 1:
                    row = [i * self.dx, self.u2[0, i], self.u2[1, i]]
                elif self.polyDegree == 2:
                    row = [i * self.dx, self.u2[0, i], self.u2[1, i], self.u2[2, i]]
                writer.writerow(row)
        return 0


if __name__ == '__main__':
    main()
