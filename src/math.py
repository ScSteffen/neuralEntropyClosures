"""
Script with functions for quadratures, moment basis for 1D-3D spatial dimensions
Author:  Steffen Schotthöfer
Date: 16.03.21
"""

from numpy.polynomial.legendre import leggauss
import numpy as np
import tensorflow as tf
import scipy


class EntropyTools:
    """
    Same functions implemented in the sobolev Network.
    Also uses Tensorflow
    """

    polyDegree: int
    nq: int
    inputDim: int
    quadPts: tf.Tensor  # dims = (1 x nq)
    quadWeights: tf.Tensor  # dims = (1 x nq)
    momentBasis: tf.Tensor  # dims = (batchSIze x N x nq)
    opti_u: np.ndarray
    opti_m: np.ndarray
    opti_w: np.ndarray

    def __init__(self, polynomial_degree=1) -> object:
        """
        Class to compute the 1D entropy closure up to degree N
        input: N  = degree of polynomial basis
        """

        # Create quadrature and momentBasis. Currently only for 1D problems
        self.polyDegree = polynomial_degree
        self.nq = 100
        [quadPts, quadWeights] = qGaussLegendre1D(self.nq)  # dims = nq
        self.quadPts = tf.constant(quadPts, shape=(1, self.nq), dtype=tf.float32)
        self.quadWeights = tf.constant(quadWeights, shape=(1, self.nq),
                                       dtype=tf.float32)
        mBasis = computeMonomialBasis1D(quadPts, self.polyDegree)  # dims = (N x nq)
        self.inputDim = mBasis.shape[0]
        self.momentBasis = tf.constant(mBasis, shape=(self.inputDim, self.nq),
                                       dtype=tf.float32)

    def reconstruct_alpha(self, alpha: tf.Tensor) -> tf.Tensor:
        """
        brief:  Reconstructs alpha_0 and then concats alpha_0 to alpha_1,... , from alpha1,...
                Only works for maxwell Boltzmann entropy so far.
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N-1)
               m    , dims = (N x nq)
               w    , dims = nq
        returns alpha_complete = [alpha_0,alpha], dim = (nS x N), where alpha_0 = - ln(<exp(alpha*m)>)
        """
        tmp = tf.math.exp(tf.tensordot(alpha, self.momentBasis[1:, :], axes=([1], [0])))  # tmp = alpha * m
        alpha_0 = -tf.math.log(tf.tensordot(tmp, self.quadWeights, axes=([1], [1])))  # ln(<tmp>)
        return tf.concat([alpha_0, alpha], axis=1)  # concat [alpha_0,alpha]

    def reconstruct_u(self, alpha: tf.Tensor) -> tf.Tensor:
        """
        brief: reconstructs u from alpha

        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N)
        used members: m    , dims = (N x nq)
                      w    , dims = nq
        returns u = <m*eta_*'(alpha*m)>, dim = (nS x N)
        """
        # Currently only for maxwell Boltzmann entropy
        f_quad = tf.math.exp(tf.tensordot(alpha, self.momentBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.math.multiply(f_quad, self.quadWeights)  # f*w
        return tf.tensordot(tmp, self.momentBasis[:, :], axes=([1], [1]))  # f * w * momentBasis

    def compute_u(self, f: tf.Tensor) -> tf.Tensor:
        """
                brief: reconstructs u from kinetic density f
                nS = batchSize
                N = basisSize
                nq = number of quadPts

                input: f, dims = (nS x nq)
                used members: m    , dims = (N x nq)
                              w    , dims = nq
                returns u = <m*eta_*'(alpha*m)>, dim = (nS x N)
                """
        tmp = tf.math.multiply(f, self.quadWeights)  # f*w
        return tf.tensordot(tmp, self.momentBasis[:, :], axes=([1], [1]))  # f * w * momentBasis

    def compute_h(self, u: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
        """
        brief: computes the entropy functional h on u and alpha

        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N)
               u, dims = (nS x N)
        used members: m    , dims = (N x nq)
                    w    , dims = nq

        returns h = alpha*u - <eta_*(alpha*m)>
        """
        # Currently only for maxwell Boltzmann entropy
        f_quad = tf.math.exp(tf.tensordot(alpha, self.momentBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.tensordot(f_quad, self.quadWeights, axes=([1], [1]))  # f*w
        # tmp2 = tf.tensordot(alpha, u, axes=([1], [1]))
        tmp2 = tf.math.reduce_sum(tf.math.multiply(alpha, u), axis=1, keepdims=True)
        return tmp2 - tmp

    def convert_to_tensor_float(self, vector: np.ndarray) -> tf.Tensor:
        """
        brief: converts to tensor, keeps dimensions
        """
        return tf.constant(vector, shape=vector.shape, dtype=tf.float32)

    def minimize_entropy(self, u: tf.Tensor, start: tf.Tensor) -> tf.Tensor:
        """
        brief: computes the minimal entropy at u
        input: u = dims (1,N)
           start =  start_valu of alpha
        """
        dim = u.numpy().shape[1]
        self.opti_u = np.reshape(u.numpy(), (dim,))
        self.opti_m = self.momentBasis.numpy()
        self.opti_w = self.quadWeights.numpy()

        opti_start = np.reshape(start.numpy(), (dim,))

        opt_result = scipy.optimize.minimize(fun=self.opti_entropy, x0=opti_start, jac=self.opti_entropy_prime,
                                             tol=1e-4)

        if not opt_result.success:
            exit("Optimization unsuccessfull!")
        return tf.constant(opt_result.x, dtype=tf.float32, shape=(1, dim))

    def opti_entropy(self, alpha: np.ndarray) -> np.ndarray:
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
        f_quad = np.exp(np.tensordot(alpha, self.opti_m, axes=([0], [0])))  # alpha*m
        t1 = np.tensordot(f_quad, self.opti_w, axes=([0], [1]))  # f*w
        t2 = np.inner(alpha, self.opti_u)

        return t1 - t2

    def opti_entropy_prime(self, alpha: np.ndarray) -> np.ndarray:
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

        f_quad = np.exp(np.tensordot(alpha, self.opti_m, axes=([0], [0])))  # alpha*m
        tmp = np.multiply(f_quad, self.opti_w)  # f*w
        t2 = np.tensordot(tmp, self.opti_m[:, :], axes=([1], [1]))  # f * w * momentBasis
        dim = t2.shape[1]
        return np.reshape(t2 - self.opti_u, (dim,))

    def KL_divergence(self, alpha_true: tf.Tensor, alpha: tf.Tensor) -> tf.Tensor:
        """
        brief: computes the Kullback-Leibler Divergence of the kinetic density w.r.t alpha given the kinetic density w.r.t
                alpha_true
        input: alpha_true , dim= (1,N+1)
               alpha , dim = (ns, N+1)
        output: pointwise KL Divergence, dim  = ns x 1
        """

        diff = alpha_true - alpha

        t1 = tf.math.exp(tf.tensordot(alpha_true, self.momentBasis, axes=([1], [0])))
        t2 = tf.tensordot(diff, self.momentBasis, axes=([1], [0]))
        integrand = tf.math.multiply(t1, t2)

        res = tf.tensordot(integrand, self.quadWeights, axes=([1], [1]))
        return res

    def compute_kinetic_density(self, alpha: tf.Tensor) -> tf.Tensor:
        """
                brief: computes the kinetic density w.r.t alpha
                input: alpha , dim = (ns, N+1)
                output: kinetic density, dim  = ns x nq
        """
        return tf.math.exp(tf.tensordot(alpha, self.momentBasis, axes=([1], [0])))

    def compute_maxwellian(self):
        """
        returns the maxwellian distribution at quadpts
        return: maxwellian, dims = (1,nq)
        """
        return 0


### Standalone features


### Integration


def qGaussLegendre1D(order: int):
    """
    order: order of quadrature
    returns: [mu, weights] : quadrature points and weights
    """
    return leggauss(order)


def qGaussLegendre2D(Qorder):
    """
       order: order of quadrature, uses all quadpts... inefficient
       returns: [pts, weights] : quadrature points and weights, dim(pts) = nq x 2
    """

    def computequadpoints(order):
        """Quadrature points for GaussLegendre quadrature. Read from file."""
        mu, _ = leggauss(order)
        phi = [np.pi * (k + 1 / 2) / order for k in range(2 * order)]
        xy = np.zeros((2 * order * order, 2))
        count = 0
        for i in range(order):
            for j in range(2 * order):
                mui = mu[i]
                phij = phi[j]
                xy[count, 0] = np.sqrt(1 - mui ** 2) * np.cos(phij)
                xy[count, 1] = np.sqrt(1 - mui ** 2) * np.sin(phij)
                # xyz[count, 2] = mui
                count += 1

        return xy

    def computequadweights(order):
        """Quadrature weights for GaussLegendre quadrature. Read from file."""
        _, leggaussweights = leggauss(order)
        w = np.zeros(2 * order * order)
        count = 0
        for i in range(order):
            for j in range(2 * order):
                w[count] = 2 * np.pi / order * leggaussweights[i]
                count += 1

        w /= sum(w)
        w *= 4 * np.pi
        return w

    pts = computequadpoints(Qorder)
    weights = computequadweights(Qorder)

    return [pts, weights]


def integrate(integrand, weights):
    """
    params: weights = quadweights vector (at quadpoints) (dim = nq)
            integrand = integrand vector, evaluated at quadpts (dim = vectorlen x nq)
    returns: integral <integrand>
    """
    return np.dot(integrand, weights)


### Entropy functions

def negEntropyFunctional(u, alpha, m, w):
    """
    compute entropy functional at one point using
    inputs: u = moment vector, dim = N+1
            alpha = corresponding lagrange multiplier, dim = N+1
            m = moment basis vector, evaluated at quadpts, dim = (N + 1) x nQuad
            quadPts = number of quadpts
    returns: h = alpha*u - <entropyDual(alpha*m)>
    """
    # tmp = integrate(entropyDualPrime(np.matmul(alpha, m)), w)
    return 0  # Todo


def entropy(x):
    return x * np.log(x) - x


def entropyDual(y):
    return np.exp(y)


def entropyPrime(x):
    return np.log(x)


def entropyDualPrime(y):
    return np.exp(y)


def reconstructU(alpha, m, w):
    """
    imput: alpha, dims = (nS x N)
           m    , dims = (N x nq)
           w    , dims = nq
    returns u = <m*eta_*'(alpha*m)>, dim = (nS x N)
    """

    # tensor version
    temp = entropyDualPrime(np.matmul(alpha, m))  # ns x nq
    ## extend to 3D tensor
    mTensor = m.reshape(1, m.shape[0], m.shape[1])  # ns x N  x nq
    tempTensor = temp.reshape(temp.shape[0], 1, temp.shape[1])  # ns x N x nq

    return integrate(mTensor * tempTensor, w)


def reconstructL1F(alpha, m, w):
    """
    imput: alpha, dims = (nS x N)
           m    , dims = (N x nq)
           w    , dims = nq
    returns:  the L1 norm of f, the kinetic density, <|f|>
    """
    return integrate(np.abs(entropyDualPrime(np.matmul(alpha, m))), w)


def reconstructUSingleCell(alpha, m, w):
    """
    imput: alpha, dims = (N)
           m    , dims = (N x nq)
           w    , dims = nq
    returns u = <m*eta_*'(alpha*m)>, dim = (nS x N)
    """

    temp = entropyDualPrime(np.matmul(alpha, m))  # ns x nq
    res = m * temp

    return integrate(res, w)


### Basis Computation
def computeMonomialBasis1D(quadPts, polyDegree):
    """
    params: quadPts = quadrature points to evaluate
            polyDegree = maximum degree of the basis
    return: monomial basis evaluated at quadrature points
    """
    basisLen = getBasisSize(polyDegree, 1)
    nq = quadPts.shape[0]
    monomialBasis = np.zeros((basisLen, nq))

    for idx_quad in range(0, nq):
        for idx_degree in range(0, polyDegree + 1):
            monomialBasis[idx_degree, idx_quad] = np.power(quadPts[idx_quad], idx_degree)
    return monomialBasis


def computeMonomialBasis2D(quadPts, polyDegree):
    """
    brief: hardcoded for polyDegree 1 right now ! # TODO
    params: quadPts = quadrature points to evaluate
            polyDegree = maximum degree of the basis
    return: monomial basis evaluated at quadrature points
    """
    basisLen = getBasisSize(polyDegree, 2)
    nq = quadPts.shape[0]
    monomialBasis = np.zeros((basisLen, nq))

    for idx_quad in range(0, nq):
        # Hardcoded for degree 1
        monomialBasis[0, idx_quad] = 1.0
        monomialBasis[1, idx_quad] = quadPts[idx_quad, 0]
        monomialBasis[2, idx_quad] = quadPts[idx_quad, 1]
    # for idx_degree in range(0, polyDegree + 1):
    #    monomialBasis[idx_degree, idx_quad] = np.power(quadPts[idx_quad], idx_degree)
    return monomialBasis


def getBasisSize(polyDegree, spatialDim):
    """
    params: polyDegree = maximum Degree of the basis
            spatialDIm = spatial dimension of the basis
    returns: basis size
    """

    basisLen = 0

    for idx_degree in range(0, polyDegree + 1):
        basisLen += int(
            getCurrDegreeSize(idx_degree, spatialDim))

    return basisLen


def getCurrDegreeSize(currDegree, spatialDim):
    """
    Computes the number of polynomials of the current spatial dimension
    """
    return np.math.factorial(currDegree + spatialDim - 1) / (
            np.math.factorial(currDegree) * np.math.factorial(spatialDim - 1))
