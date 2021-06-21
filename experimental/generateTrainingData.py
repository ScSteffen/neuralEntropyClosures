### This is a script for generation of the training data
'''Author:  Steffen Schotthöfer'''

### imports
import numpy as np

#in-project imports
import sphericalquadpy as sqp
import csv
from legacyCode import nnUtils

#parallelism
from joblib import Parallel, delayed
import multiprocessing


def main():

    ### 0) Set variables #######################################################
    # Set up Quadrature
    quadOrder = 10 #QuadOrder for training
    BasisDegree = 0  # Determines the number of moments used
    SetSize = 10000000#300000 # Determines the size of the training set

    # Set up data generator
    mean = 0  # mean of normal distribution
    sigma = 2  # std deviation of normal distribution

    # Name of modelDirectory
    filenameU = "trainingData_M0_u.csv"
    filenameAlpha = "trainingData_M0_alpha.csv"
    filenameH = "trainingData_M0_h.csv"

    #### 1) Generate and clean Training Data
    print("Generate Training Data")
    (u,alpha,h) = generateTrainingData(BasisDegree,quadOrder,SetSize, mean, sigma) #np.ones((20000, 9))
    print("Clean Training Data")
    (u,alpha,h) = cleanTrainingData(u,alpha,h)


    ### 2) Store Training Data
    storeTrainingData(u,alpha,h,filenameU,filenameAlpha,filenameH)
    return 0

def generateTrainingData(BasisDegree, quadOrder, setSize, mean, sigma):
    # Brief: Generates normal distributed alpha and computes corresponding u and h.
    # Input:   basisDegree = maximum Degree of spherical harmonics
    #          quadOrder   = order of Gauss Legendre quadrature
    #          setSize     = size of the training set
    #          mean        = mean value for normal distributed alpha
    #          sigma       = std. deviation for normal distributed alpha
    #
    # Output:  uTrain.shape = (setSize, lenMomentBasis)
    #          alphas.shape = (setSize, lenMomentBasis)
    #          h.shape      = (setSize, 1)

    # --- Define Helper Functions ---
    def computeBasisHelper(quadorder, basisDegree):
        # Brief: Generates spherical harmonics of degree "basisDegree" evalutated at the quadrature points of
        #        Gauss-Legendre Quadrature of order quadorder
        # Input: quadorder   = order of Gauss-Legendre quadrature
        #        basisDegree = maximal degree of spherical harmonics
        # Output: basisManual.shape =  (lenBasis, numQuadPts)
        #         quadPts.shape     = (numQuadPts,3)
        #         quadWeights.shape = (numQuadPts)

        # Spawn quadrature
        Q = sqp.gausslegendre.GaussLegendre(order=quadorder)

        # a) Get quadrature points.
        quadPts = Q.computequadpoints(quadorder)
        quadWeights = Q.computequadweights(quadorder)
        # b) Compute spherical harmonics basis at quadPts
        mBasisList = list()

        # This is errorous!
        for l in range(0, basisDegree + 1):
            for k in range(-l, l + 1):
                temp = sqp.tools.sphericalharmonics.ylm(k, l, quadPts[:, 0], quadPts[:, 1], quadPts[:, 2])
                if (k < 0):
                    mBasisList.append(temp.imag)
                elif (k == 0):
                    mBasisList.append(temp.real)
                else:  # (k > 0):
                    mBasisList.append(temp.real)

        basis = np.array(mBasisList)

        ### Manual computation of spherical harmonics up to order 2###
        # the first nine harmonics:
        def Y0_0 (mu,phi):
            return np.sqrt(1/(4*np.pi))

        def Y1_m1 (mu,phi):
            return -np.sqrt(3/(4*np.pi))*np.sqrt(1-mu*mu)*np.sin(phi)
        def Y1_0 (mu,phi):
            return np.sqrt(3/(4*np.pi))*mu
        def Y1_1 (mu,phi):
            return -np.sqrt(3/(4*np.pi))*np.sqrt(1-mu*mu)*np.cos(phi)

        def Y2_m2 (mu,phi):
            return np.sqrt(15/(16*np.pi))*(1-mu*mu)*np.sin(2*phi)
        def Y2_m1 (mu,phi):
            return -1*np.sqrt(15/(4*np.pi))*mu*np.sqrt(1-mu*mu)*np.sin(phi)
        def Y2_0 (mu,phi):
            return np.sqrt(5/(16*np.pi))*(3*mu*mu-1)
        def Y2_1 (mu,phi):
            return -1*np.sqrt(15/(4*np.pi))*mu*np.sqrt(1-mu*mu)*np.cos(phi)
        def Y2_2 (mu,phi):
            return np.sqrt(15/(16*np.pi))*(1-mu*mu)*np.cos(2*phi)

        # Transform karth coordinates to shperical coordinates:
        thetaMu =  sqp.tools.xyz2thetaphi(quadPts) #mu in [0,pi]
        phi = thetaMu[:, 0]
        mu = np.cos(thetaMu[:, 1])
        nPts = quadPts.shape[0]
        basisManual = np.zeros(basis.shape)

        for i in range(0,nPts): #manual computation...

            basisManual[0,i] = Y0_0(mu[i],phi[i])
            #basisManual[1,i] = Y1_m1(mu[i],phi[i])
            #basisManual[2,i] = Y1_0(mu[i],phi[i])
            #basisManual[3,i] = Y1_1(mu[i],phi[i])
            #basisManual[4,i] = Y2_m2(mu[i],phi[i])
            #basisManual[5,i] = Y2_m1(mu[i],phi[i])
            #basisManual[6,i] = Y2_0(mu[i],phi[i])
            #basisManual[7,i] = Y2_1(mu[i],phi[i])
            #basisManual[8,i] = Y2_2(mu[i],phi[i])

            '''
            if(np.linalg.norm(basisManual[:,i]-basis[:,i]) > 1e-8):
                print("at quad pt")
                print(i)
                print(basisManual[:,i]-basis[:,i]) # ==> python spherical harmonics have wrong weights
            '''

        '''
        # Unit test orthonormality
        for j in range(0,9):
            integral = np.zeros(basis.shape[0])

            for i in range(0, basis.shape[1]):
                integral += quadWeights[i] * basisManual[j, i] * basisManual[:, i]

            print(integral)
        '''
        return (basisManual, quadPts, quadWeights)

    def integrateMoments(mBasis, psi, quadWeight):
        # input: mBasis.shape = (NmaxMoment, numQuadPts)
        #         alpha.shape = (batchSize, NmaxMoment)
        #         quadWeight.shape = (numQuadPts)
        # output: integral.shape = (batchSize, NmaxMoment)

        (NmaxMoment, numPts) = mBasis.shape

        integral = np.zeros(NmaxMoment)

        for i in range(0, numPts):
            integral += quadWeight[i] * psi[i] * mBasis[:, i]

        return integral

    def integrateEntropy(psi, quadWeight):
        # brief: Computes the value of the entropy for given psi.
        # input:  psi.shape = (numQuadPts)
        #         quadWeight.shape = (numQuadPts)
        # output: integral.shape = (1)

        integral = 0

        for i in range(0, quadWeight.shape[0]):  # loop over quadpoints
            entropy = psi[i] * np.log(psi[i]) - psi[i]  # eta(psi)
            integral += quadWeight[i] * entropy

        return integral

    # --- Initialize Variables ---
    N_moments = nnUtils.getGlobalIdx(BasisDegree,BasisDegree)+1
    uTrain = np.zeros((setSize,N_moments))
    alphas = np.zeros((setSize,N_moments))
    h = np.zeros((setSize,1))
    # generate basis functions
    (basis_v, quadPts, quadWeights) = computeBasisHelper(quadOrder, BasisDegree)
    # need the transpose for convenience
    basis_vT = np.matrix.transpose(basis_v)

    # --- define data generator ---
    def dataGenerator(i):
        # 2) generate random normally distributed alphas
        #alphas[i, :] = np.random.normal(mean, sigma, N_moments)
        alpha = np.random.normal(mean, sigma, N_moments)

        # 3) generate psi_v
        #psi_v = np.exp(np.matmul(basis_vT, alphas[i, :]))
        psi_v = np.exp(np.matmul(basis_vT, alpha)) #evaluates psi at all quad points

        # 4) compute the Moments of psi
        #uTrain[i, :] = integrateMoments(basis_v, psi_v, quadWeights)
        if (i % 100000 == 0):
            print("Status: {:.2f} percent".format(i / setSize * 100))

        return (integrateMoments(basis_v, psi_v, quadWeights),alpha, integrateEntropy(psi_v,quadWeights))

    # --- parallelize data generation ---
    num_cores = multiprocessing.cpu_count()
    print("Starting data generation using " + str(num_cores) + " cores")
    uTrain_Alpha_h_List = Parallel(n_jobs=num_cores)(delayed(dataGenerator)(i) for i in range(0,setSize)) #(u, alpha, h)

    # --- postprocessing
    for i in range(0,setSize):
        uTrain[i, :] = uTrain_Alpha_h_List[i][0]
        alphas[i, :] = uTrain_Alpha_h_List[i][1]
        h[i,:]       = uTrain_Alpha_h_List[i][2]

    return (uTrain, alphas, h)

def cleanTrainingData(alphaTrain,uTrain, hTrain):
    # brief: removes unrealistic values of the training set
    # input: uTrain.shape     = (setSize, basisSize) Moment vector
    #        alphaTrain.shape = (setSize, basisSize) Lagrange  multiplier
    #        hTrain.shape     = (setSize, 1) entropy
    # output: uTrain.shape     = (setSize, basisSize) Moment vector
    #         alphaTrain.shape = (setSize, basisSize) Lagrange multiplier
    #         hTrain.shape     = (setSize, 1) entropy

    setSize = uTrain.shape[0]

    def entryMarker(idx):
        # --- mark entries, that should be deleted ---
        deleteEntry = False
        # check for NaN values
        nanEntry = False
        for jdx in range(0, alphaTrain.shape[1]):
            if (np.isnan(alphaTrain[idx, jdx])):
                nanEntry = True
            if (np.isnan(uTrain[idx, jdx])):
                nanEntry = True
        if (nanEntry):
            deleteEntry = True


        # check if first moment is too big ==> unrealistic value
        if (uTrain[idx, 0] > 250):
            deleteEntry = True

        '''
               if (uTrain[idx, 1] < -60):
                   deleteEntry = True
               if (uTrain[idx, 1] > 60):
                   deleteEntry = True

               if (uTrain[idx, 2] < -0.1):
                   deleteEntry = True
               if (uTrain[idx, 2] > 0.1):
                   deleteEntry = True

               if (uTrain[idx, 3] < -60):
                   deleteEntry = True
               if (uTrain[idx, 3] > 60):
                   deleteEntry = True
               # further conditions go here
        '''
        keepEntry = not deleteEntry

        if (idx % 100000 == 0): #Progress info
            print("Status: {:.2f} percent".format(idx / setSize * 100))

        return keepEntry

    # --- parallelize data cleanup ---
    num_cores = multiprocessing.cpu_count()
    print("Starting data cleanup using " + str(num_cores) + " cores")
    deletionList = Parallel(n_jobs=num_cores)(
        delayed(entryMarker)(i) for i in range(0, setSize))  # (u,alpha,  h)

    # --- delete entries ---
    uTrain = uTrain[deletionList]
    hTrain = hTrain[deletionList]
    alphaTrain = alphaTrain[deletionList]

    # --- sort remaining entries ---
    zipped_lists = zip(uTrain,alphaTrain, hTrain)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    listU,listAlpha, listH = [list(tuple) for tuple in tuples]

    return (np.asarray(listU),np.asarray(listAlpha),np.asarray(listH))

def storeTrainingData(u,alpha,h, filenameU, filenameAlpha,filenameH):
    ### 2) Store Training Data
    print("Write Training Data")
    # store u
    f = open(filenameU, 'w')
    with f:
        writer = csv.writer(f)
        for row in u:
            writer.writerow(row)
    # store alphas
    f = open(filenameAlpha, 'w')
    with f:
        writer = csv.writer(f)
        for row in alpha:
            writer.writerow(row)
    # store h
    f = open(filenameH, 'w')
    with f:
        writer = csv.writer(f)
        for row in h:
            writer.writerow(row)

    return 0

if __name__ == '__main__':
    main()
