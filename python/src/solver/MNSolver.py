"""
brief: Testing ground for 2D moment solver 
Author: Steffen Schotthöfer
Date: 17.05.2021
"""
from src import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


def main():
    solver = MNSolver()
    solver.solve(maxIter=10)
    return 0


class MNSolver:
    def __init__(self):
        # Prototype for  spatialDim=2, polyDegree=1
        self.nSystem = 3
        self.polyDegree = 1
        self.quadOrder = 2
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
        self.cfl = 0.8
        self.dt = self.cfl / 2 * (self.dx * self.dy) / (self.dx + self.dy)

        # Solver variables
        self.u = self.ICperiodic()  # periodic IC
        self.alpha = np.zeros((self.nSystem, self.nx, self.ny))
        self.opti_u = np.zeros(self.nSystem)

        self.xFlux = np.zeros((self.nSystem, self.nx, self.ny), dtype=float)
        self.yFlux = np.zeros((self.nSystem, self.nx, self.ny), dtype=float)

    def ICperiodic(self):
        def sincos(x, y):
            return 1.5 + np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

        uIc = np.zeros((self.nSystem, self.nx, self.ny))

        for i in range(self.nx):
            for j in range(self.ny):
                xKoor = self.x0 + (i - 0.5) * self.dx
                yKoor = self.y0 + (j - 0.5) * self.dy
                uIc[0, i, j] = sincos(xKoor, yKoor)  # all other moments are 0 (isotropic)
        return uIc

    def solve(self, maxIter=100):
        # time loop
        self.showSolution()

        for idx_time in range(maxIter):
            # entropy closure
            self.entropyClosureNumerical()  # compute alpha
            # entropy reconstruction
            self.realizabilityReconstruction()
            # flux computation
            self.computeFlux()
            # FVM update
            self.FVMUpdate()

            # print iteration results
            self.showSolution()

        return self.u

    def entropyClosureNumerical(self):

        for i in range(self.nx):
            for j in range(self.ny):
                self.opti_u = self.u[:, i, j]
                alpha_init = self.alpha[:, i, j]
                # test objective functions
                t = self.opti_entropy(alpha_init)
                tp = self.opti_entropy_prime(alpha_init)
                opt_result = scipy.optimize.minimize(fun=self.opti_entropy, x0=alpha_init,
                                                     jac=self.opti_entropy_prime,
                                                     tol=1e-4)
                if not opt_result.success:
                    exit("Optimization unsuccessfull!")
                else:
                    self.alpha[:, i, j] = opt_result.x

        return 0

    def opti_entropy(self, alpha):
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
        t2 = np.inner(alpha, self.opti_u)

        return t1 - t2

    def opti_entropy_prime(self, alpha):
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
        return t2 - self.opti_u

    def realizabilityReconstruction(self):

        for i in range(self.nx):
            for j in range(self.ny):
                a = np.reshape(self.alpha[:, i, j], (1, self.nSystem))
                self.u[:, i, j] = math.reconstructU(alpha=a, m=self.mBasis, w=self.quadWeights)

        return 0

    def computeFlux(self):
        """
        for periodic boundaries, upwinding.
        writes to xFlux and yFlux
        """
        for j in range(self.ny):
            for i in range(self.nx):

                # Computation in x direction
                im1 = i - 1
                if i == 1:  # periodic boundaries
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
                if j == 1:  # periodic boundaries
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

    def upwinding(self, fluxL, fluxR, quadpt, normal):
        t = np.inner(quadpt, normal)
        if t > 0:
            return t * fluxL
        else:
            return t * fluxR

    def FVMUpdate(self):

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
                self.u[0, i, j] = self.u[0, i, j] + (
                        self.sigmaS * self.u[0, i, j] - self.sigmaT * self.u[0, i, j]) * self.dt
                self.u[1:, i, j] = self.u[0, i, j] + (self.sigmaT * self.u[1:, i, j]) * self.dt

        return 0

    def showSolution(self):

        plt.imshow(self.u[0, :, :], cmap='hot', interpolation='nearest')
        plt.show()

        return 0


if __name__ == '__main__':
    main()
