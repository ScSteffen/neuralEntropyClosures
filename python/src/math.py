"""
Script with functions for quadratures, moment basis for 1D-3D spatial dimensions
Author:  Steffen Schotthöfer
Date: 16.03.21
"""

from numpy.polynomial.legendre import leggauss
import numpy as np


### Integration
def qGaussLegendre1D(order):
    """
    order: order of quadrature
    returns: [mu, weights] : quadrature points and weights
    """
    return leggauss(order)


def integrate(integrand, weights):
    """
    params: weights = quadweights vector (at quadpoints) (dim = nq)
            integrand = integrand vector, evaluated at quadpts (dim = vectorlen x nq)
    returns: integral <integrand>
    """
    return np.dot(integrand, weights)


### Entropy functions
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
    res = np.zeros(alpha.shape)
    for i in range(alpha.shape[0]):
        res[i, :] = reconstructUSingleCell(alpha[i, :], m, w)

    # tensor version
    temp = entropyDualPrime(np.matmul(alpha, m))  # ns x nq
    ## extend to 3D tensor
    mTensor = m.reshape(1, m.shape[0], m.shape[1])  # ns x N  x nq
    tempTensor = temp.reshape(temp.shape[0], 1, temp.shape[1])  # ns x N x nq

    return integrate(mTensor * tempTensor, w)


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
