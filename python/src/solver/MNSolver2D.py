"""
brief: Testing ground for 2D moment solver 
Author: Steffen Schotthöfer
Date: 17.05.2021
"""
from src import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LogNorm
import multiprocessing
import pandas as pd
from joblib import Parallel, delayed

# inpackage imports
from src.neuralClosures.configModel import initNeuralClosure

num_cores = multiprocessing.cpu_count()


def main():
    solver = MNSolver2D(traditional=False)
    # solver.solveAnimation(maxIter=100)
    solver.solveAnimationIterError(maxIter=60)
    # solver.solveIterError(maxIter=100)
    # solver.solve(maxIter=100)
    return 0


class MNSolver2D:
    def __init__(self, traditional=True):

        # Prototype for  spatialDim=2, polyDegree=1
        self.nSystem = 3
        self.polyDegree = 1
        self.quadOrder = 10
        self.traditional = traditional
        [self.quadPts, self.quadWeights] = math.qGaussLegendre2D(self.quadOrder)  # dims = nq
        self.nq = self.quadWeights.size
        self.mBasis = math.computeMonomialBasis2D(self.quadPts, self.polyDegree)  # dims = (N x nq)
        self.inputDim = self.mBasis.shape[0]  # = self.nSystem

        # generate geometry
        self.x0 = -1.5
        self.x1 = 1.5
        self.y0 = -1.5
        self.y1 = 1.5
        self.nx = 100
        self.ny = 100
        self.dx = (self.x1 - self.x0) / self.nx
        self.dy = (self.y1 - self.y0) / self.ny

        # physics (homogeneous)
        self.sigmaS = 1.0
        self.sigmaA = 0.0
        self.sigmaT = self.sigmaS + self.sigmaA

        # time
        self.tEnd = 1.0
        self.cfl = 0.95
        self.dt = self.cfl / 2 * (self.dx * self.dy) / (self.dx + self.dy)

        # Solver variables Traditional
        self.u = self.ICperiodic()  # periodic IC
        self.alpha = np.zeros((self.nSystem, self.nx, self.ny))
        self.xFlux = np.zeros((self.nSystem, self.nx, self.ny), dtype=float)
        self.yFlux = np.zeros((self.nSystem, self.nx, self.ny), dtype=float)

        self.u2 = self.ICperiodic()
        self.alpha2 = np.zeros((self.nSystem, self.nx, self.ny))
        self.xFlux2 = np.zeros((self.nSystem, self.nx, self.ny), dtype=float)
        self.yFlux2 = np.zeros((self.nSystem, self.nx, self.ny), dtype=float)
        # Neural closure
        self.neuralClosure = None
        if not self.traditional:
            self.neuralClosure = initNeuralClosure(modelNumber=13, polyDegree=1, spatialDim=2,
                                                   folderName="002_sim_M1_2D", lossCombi=3,
                                                   width=15, depth=7, normalized=True)
            self.neuralClosure.loadModel("../../models/002_sim_M1_2D")

        # Analysis variables
        self.errorMap = np.zeros((self.nSystem, self.nx, self.ny))
        self.normErrorMap = np.zeros((self.nx, self.ny))
        self.realizabilityMap = np.zeros((self.nx, self.ny))
        columns = ['u0', 'u1', 'u2', 'alpha0', 'alpha1', 'alpha2', 'h']  # , 'realizable']
        self.dfErrPoints = pd.DataFrame(columns=columns)

    def ICperiodic(self):
        def sincos(x, y):
            return 1.5 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

        uIc = np.zeros((self.nSystem, self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                xKoor = self.x0 + (i - 0.5) * self.dx
                yKoor = self.y0 + (j - 0.5) * self.dy
                uIc[0, i, j] = sincos(xKoor, yKoor)  # all other moments are 0 (isotropic)
                uIc[1, i, j] = 0.9 / 3.0 * uIc[0, i, j]
                uIc[2, i, j] = 0.9 / 3.0 * uIc[0, i, j]
        return uIc

    def solve(self, maxIter=100):
        # self.showSolution(0)
        for idx_time in range(maxIter):  # time loop
            self.solveIterNewton(idx_time)
            self.solverIterML(idx_time)
            print("Iteration: " + str(idx_time))
            self.errorAnalysis()
            # print iteration results
            self.showSolution(idx_time)

        return self.u

    def solveIterError(self, maxIter=100):
        # self.showSolution(0)
        for idx_time in range(maxIter):  # time loop
            self.u2 = self.u  # sync solvers
            self.solveIterNewton(idx_time)
            self.entropyClosureML()

            # self.solverIterML(idx_time)
            print("Iteration: " + str(idx_time))
            self.errorAnalysis()
            # print iteration results
            self.showSolution(idx_time)

        return self.u

    def solveAnimationIterError(self, maxIter=100):
        fps = 1 / self.dt

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(10, 10))

        im = plt.imshow(np.zeros((self.nx, self.ny)), cmap='hot', interpolation='nearest', vmin=1e-4, vmax=1.0,
                        norm=LogNorm())

        # im = plt.imshow(np.zeros((self.nx, self.ny)), cmap='hot', interpolation='nearest', vmin=1, vmax=3.0)
        cbar = fig.colorbar(im)

        def animate_func(i):
            # self.u2 = np.copy(self.u)
            self.entropyClosureNewton()
            self.entropyClosureML()  # compare
            self.realizabilityReconstruction()
            self.errorAnalysis()

            # flux computation
            self.computeFluxNewton()
            # FVM update
            self.FVMUpdateNewton()
            # flux computation
            self.computeFluxML()
            # FVM update
            self.FVMUpdateML()

            print("Iteration: " + str(i))
            img = self.normErrorMap[:, :]
            im.set_array(img)
            return [im]

        anim = animation.FuncAnimation(fig, animate_func, frames=maxIter, interval=10000 * self.dt)

        if self.traditional:
            filename = "newton_version.gif"
        else:
            filename = "ErrorPerIter.gif"
        anim.save(filename, writer=animation.PillowWriter(fps=fps))

        print('Done!')
        self.dfErrPoints.to_csv("errorPts.csv", index=True)
        return anim

    def solveAnimation(self, maxIter=100):
        fps = 1 / self.dt

        # First set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(10, 10))

        # im = plt.imshow(np.zeros((self.nx, self.ny)), cmap='hot', interpolation='nearest', vmin=1e-4, vmax=1.0,
        #                norm=LogNorm())

        im = plt.imshow(np.zeros((self.nx, self.ny)), cmap='hot', interpolation='nearest', vmin=0, vmax=3.0)
        cbar = fig.colorbar(im)

        def animate_func(i):
            # self.solveIterNewton(i)
            self.solverIterML(i)
            print("Iteration: " + str(i))
            # self.errorAnalysis()
            img = self.u2[0, :, :]
            im.set_array(img)
            return [im]

        anim = animation.FuncAnimation(fig, animate_func, frames=maxIter, interval=10000 * self.dt)

        if self.traditional:
            filename = "newton_version.gif"
        else:
            filename = "ML_version.gif"
        anim.save(filename, writer=animation.PillowWriter(fps=fps))

        print('Done!')

        return anim

    def solveIterNewton(self, t_idx):
        # entropy closure and
        self.entropyClosureNewton()
        # reconstruction
        self.realizabilityReconstruction()
        # flux computation
        self.computeFluxNewton()
        # FVM update
        self.FVMUpdateNewton()
        return 0

    def solverIterML(self, t_idx):
        # entropy closure and
        self.entropyClosureML()
        # flux computation
        self.computeFluxML()
        # FVM update
        self.FVMUpdateML()
        return 0

    def entropyClosureML(self):
        count = 0
        tmp = np.zeros((self.nx * self.ny, self.nSystem))
        for i in range(self.nx):
            for j in range(self.ny):
                tmp[count, :] = self.u2[:, i, j]
                count = count + 1
        # call neuralEntropy
        [u_pred, alpha, h] = self.neuralClosure.call_scaled_64(np.asarray(tmp))
        count = 0
        for i in range(self.nx):
            for j in range(self.ny):
                # print(str(self.u[:, i, j]) + " | " + str(u_pred[count, :]))
                self.u2[:, i, j] = u_pred[count, :]  # reconstruction
                self.alpha2[:, i, j] = alpha[count, :]
                count = count + 1

        return 0

    def entropyClosureNewton(self):

        # if (self.traditional): # NEWTON
        for i in range(self.nx):
            self.entropyClosureSingleRow(i)

        # resList = Parallel(n_jobs=num_cores)(delayed(self.entropyClosureSingleRow)(i) for i in range(self.nx))
        # for i in range(self.nx):
        #    for j in range(self.ny):
        #        self.alpha[:, i, j] = resList[i][j]

        # else: #TENSORFLOW

        return 0

    def entropyClosureSingleRow(self, i):
        rowRes = []
        for j in range(self.ny):

            opti_u = self.u[:, i, j]
            alpha_init = self.alpha[:, i, j]
            # test objective functions
            t = self.create_opti_entropy(opti_u)(alpha_init)
            tp = self.create_opti_entropy_prime(opti_u)(alpha_init)
            # print(t)
            # print(tp)
            normU = np.sqrt(self.u[1, i, j] * self.u[1, i, j] + self.u[2, i, j] * self.u[2, i, j])
            u0 = self.u[0, i, j]
            if (normU / u0 > 0.95):
                print("Warning")
            opt_result = scipy.optimize.minimize(fun=self.create_opti_entropy(opti_u), x0=alpha_init,
                                                 jac=self.create_opti_entropy_prime(opti_u),
                                                 tol=1e-7)
            if not opt_result.success:
                print("Optimization unsuccessfull!")
            else:
                self.alpha[:, i, j] = opt_result.x
                rowRes.append(opt_result.x)
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

             returns h = - alpha*u + <eta_*(alpha*m)>
            """
            # Currently only for maxwell Boltzmann entropy

            f_quad = np.exp(np.tensordot(alpha, self.mBasis, axes=([0], [0])))  # alpha*m
            tmp = np.multiply(f_quad, self.quadWeights)  # f*w
            t2 = np.tensordot(tmp, self.mBasis, axes=([0], [1]))  # f * w * momentBasis
            return t2 - u

        return opti_entropy_prime

    def realizabilityReconstruction(self):
        for i in range(self.nx):
            for j in range(self.ny):
                # self.u2[:, i, j] = self.u[:, i, j]
                a = np.reshape(self.alpha[:, i, j], (1, self.nSystem))
                self.u[:, i, j] = math.reconstructU(alpha=a, m=self.mBasis, w=self.quadWeights)
                # print("(" + str(self.u2[:, i, j]) + " | " + str(self.u[:, i, j]))
        return 0

    def computeFluxNewton(self):
        """
        for periodic boundaries, upwinding.
        writes to xFlux and yFlux, uses alpha
        """
        for j in range(self.ny):
            for i in range(self.nx):

                # Computation in x direction
                im1 = i - 1
                if i == 0:  # periodic boundaries
                    im1 = self.nx - 1
                left = np.tensordot(self.alpha[:, im1, j], self.mBasis, axes=([0], [0]))
                right = np.tensordot(self.alpha[:, i, j], self.mBasis, axes=([0], [0]))
                fluxL = math.entropyDualPrime(left)
                fluxR = math.entropyDualPrime(right)
                flux = 0
                for q in range(self.nq):  # integrate upwinding result
                    upwind = self.upwinding(fluxL[q], fluxR[q], self.quadPts[q], [1, 0])
                    flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
                self.xFlux[:, i, j] = flux

                # Computation in y direction
                jm1 = j - 1
                if j == 0:  # periodic boundaries
                    jm1 = self.ny - 1
                lower = np.tensordot(self.alpha[:, i, jm1], self.mBasis, axes=([0], [0]))
                upper = np.tensordot(self.alpha[:, i, j], self.mBasis, axes=([0], [0]))
                fluxLow = math.entropyDualPrime(lower)
                fluxUp = math.entropyDualPrime(upper)
                flux = 0
                for q in range(self.nq):  # integrate upwinding result
                    upwind = self.upwinding(fluxLow[q], fluxUp[q], self.quadPts[q], [0, 1])
                    flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
                self.yFlux[:, i, j] = flux
        return 0

    def computeFluxML(self):
        """
        for periodic boundaries, upwinding.
        writes to xFlux and yFlux, uses alpha
        """
        for j in range(self.ny):
            for i in range(self.nx):

                # Computation in x direction
                im1 = i - 1
                if i == 0:  # periodic boundaries
                    im1 = self.nx - 1
                left = np.tensordot(self.alpha2[:, im1, j], self.mBasis, axes=([0], [0]))
                right = np.tensordot(self.alpha2[:, i, j], self.mBasis, axes=([0], [0]))
                fluxL = math.entropyDualPrime(left)
                fluxR = math.entropyDualPrime(right)
                flux = 0
                for q in range(self.nq):  # integrate upwinding result
                    upwind = self.upwinding(fluxL[q], fluxR[q], self.quadPts[q], [1, 0])
                    flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
                self.xFlux2[:, i, j] = flux

                # Computation in y direction
                jm1 = j - 1
                if j == 0:  # periodic boundaries
                    jm1 = self.ny - 1
                lower = np.tensordot(self.alpha2[:, i, jm1], self.mBasis, axes=([0], [0]))
                upper = np.tensordot(self.alpha2[:, i, j], self.mBasis, axes=([0], [0]))
                fluxLow = math.entropyDualPrime(lower)
                fluxUp = math.entropyDualPrime(upper)
                flux = 0
                for q in range(self.nq):  # integrate upwinding result
                    upwind = self.upwinding(fluxLow[q], fluxUp[q], self.quadPts[q], [0, 1])
                    flux = flux + upwind * self.quadWeights[q] * self.mBasis[:, q]
                self.yFlux2[:, i, j] = flux
        return 0

    def upwinding(self, fluxL, fluxR, quadpt, normal):
        t = np.inner(quadpt, normal)
        if t > 0:
            return t * fluxL
        else:
            return t * fluxR

    def FVMUpdateNewton(self):
        for j in range(self.ny):
            for i in range(self.nx):
                ip1 = i + 1
                jp1 = j + 1
                # periodic boundaries
                if i == self.nx - 1:
                    ip1 = 0
                if j == self.ny - 1:
                    jp1 = 0

                # Advection
                self.u[:, i, j] = self.u[:, i, j] + ((self.xFlux[:, i, j] - self.xFlux[:, ip1, j]) / self.dx + (
                        self.yFlux[:, i, j] - self.yFlux[:, i, jp1]) / self.dy) * self.dt
                # Scattering
                # self.u[0, i, j] = self.u[0, i, j] + (
                #        self.sigmaS * self.u[0, i, j] - self.sigmaT * self.u[0, i, j]) * self.dt
                # self.u[1:, i, j] = self.u[0, i, j] + (self.sigmaT * self.u[1:, i, j]) * self.dt
        return 0

    def FVMUpdateML(self):
        for j in range(self.ny):
            for i in range(self.nx):
                ip1 = i + 1
                jp1 = j + 1
                # periodic boundaries
                if i == self.nx - 1:
                    ip1 = 0
                if j == self.ny - 1:
                    jp1 = 0

                # Advection
                self.u2[:, i, j] = self.u2[:, i, j] + ((self.xFlux2[:, i, j] - self.xFlux2[:, ip1, j]) / self.dx + (
                        self.yFlux2[:, i, j] - self.yFlux2[:, i, jp1]) / self.dy) * self.dt
                # Scattering
                # self.u[0, i, j] = self.u[0, i, j] + (
                #        self.sigmaS * self.u[0, i, j] - self.sigmaT * self.u[0, i, j]) * self.dt
                # self.u[1:, i, j] = self.u[0, i, j] + (self.sigmaT * self.u[1:, i, j]) * self.dt
        return 0

    def showSolution(self, idx):
        plt.imshow(self.u[0, :, :], cmap='hot', interpolation='nearest')
        plt.savefig("Periodic_" + str(idx) + ".png", dpi=150)

        return 0

    def errorAnalysis(self):
        # Compare both methods
        count = 0
        count2 = 0
        count3 = 0
        newEntries = []
        for i in range(self.nx):
            for j in range(self.ny):
                for n in range(self.nSystem):
                    self.errorMap[n, i, j] = np.abs(self.u[n, i, j] - self.u2[n, i, j]) / np.max([
                        np.abs(self.u[n, i, j]), 0.0001])

                self.normErrorMap[i, j] = np.linalg.norm(self.u[:, i, j] - self.u2[:, i, j], 2) / np.max([
                    np.linalg.norm(self.u[:, i, j], 2), 0.0001])
                self.realizabilityMap[i, j] = np.linalg.norm(self.u[1:, i, j], 2) / self.u[0, i, j]

                if self.normErrorMap[i, j] > 0.01:  # rel error bigger than 5%
                    count = count + 1
                    h = self.create_opti_entropy(self.u[:, i, j])(self.alpha[:, i, j])
                    entry = np.concatenate(
                        (self.u[:, i, j], self.alpha[:, i, j], [h]))  # , [self.realizabilityMap[i, j]]))
                    newEntries.append(entry)
                if self.normErrorMap[i, j] > 0.02:  # rel error bigger than 5%
                    count2 = count2 + 1

                if self.normErrorMap[i, j] > 0.3:  # rel error bigger than 5%
                    count3 = count3 + 1

        columns = ['u0', 'u1', 'u2', 'alpha0', 'alpha1', 'alpha2', 'h']
        df = pd.DataFrame(data=newEntries, columns=columns)
        self.dfErrPoints = pd.concat([self.dfErrPoints, df])
        print("percentage of points with error >1%: " + str(count / (self.nx * self.ny) * 100))
        print("percentage of points with error >2%: " + str(count2 / (self.nx * self.ny) * 100))
        print("percentage of points with error >3%: " + str(count3 / (self.nx * self.ny) * 100))

        # def printSolutionsToCSV(self):


if __name__ == '__main__':
    main()
