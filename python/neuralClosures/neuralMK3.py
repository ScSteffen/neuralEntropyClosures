'''
Derived network class "MK3" for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''
from .neuralBase import neuralBase
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor
import csv

'''
import pandas as pd
import sphericalquadpy as sqp
from joblib import Parallel, delayed
import multiprocessing
'''


class neuralMK3(neuralBase):
    '''
    MK1 Model: Train u to alpha
    Training data generation: b) read solver data from file
    Loss function:  MSE between alpha and real_alpha
    '''

    def __init__(self, maxDegree_N=0, folderName= "testFolder"):
        if(folderName is "testFolder"):
            tempString = "MK1_N" + str(maxDegree_N)
        else:
            tempString=folderName
        self.maxDegree_N = maxDegree_N
        self.model = self.createModel()
        self.filename = "models/"+ tempString
        self.trainingData = ([0], [0])

    def createModel(self):
        inputDim = self.getIdxSphericalHarmonics(self.maxDegree_N, self.maxDegree_N) + 1

        # Define Residual block
        def residual_block(x: Tensor) -> Tensor:
            y = keras.layers.Dense(20, activation="relu")(x)
            y = keras.layers.Dense(20, activation="relu")(y)
            y = keras.layers.Dense(20, activation="relu")(y)

            out = keras.layers.Add()([x, y])
            out = keras.layers.ReLU()(out)
            out = keras.layers.BatchNormalization()(out)
            return out

        # Define the input
        # Number of basis functions used:

        input_ = keras.Input(shape=(inputDim,))

        # Hidden layers
        hidden = keras.layers.Dense(20, activation="relu")(input_)

        # Resnet Layers
        hidden = residual_block(hidden)
        hidden = residual_block(hidden)
        hidden = residual_block(hidden)
        hidden = residual_block(hidden)

        hidden = keras.layers.Dense(20, activation="relu")(hidden)

        # Define the ouput
        output_ = keras.layers.Dense(inputDim)(hidden)

        # Create the model
        model = keras.Model(name="MK3closure", inputs=[input_], outputs=[output_])
        model.summary()

        # alternative way of training
        # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', metrics=['mean_absolute_error'])

        return model

    def createTrainingData(self):
        # TODO: Create once, then write to file and reuse.
        filenameU = self.filename + "/trainingDataU.csv"
        filenameAlpha = self.filename + "/trainingDataAlpha.csv"

        # Load Alpha
        f = open(filenameAlpha, 'r')
        alphaList = list()
        uList = list()

        with f:
            reader = csv.reader(f)

            for row in reader:
                numRow = []
                for word in row:
                    numRow.append(float(word))

                alphaList.append(numRow)

        f = open(filenameU, 'r')
        with f:
            reader = csv.reader(f)

            for row in reader:
                numRow = []
                for word in row:
                    numRow.append(float(word))
                uList.append(numRow)

        self.trainingData = (np.asarray(uList), np.asarray(alphaList),)
        '''
        mean = 0  # mean of normal distribution
        sigma = 2  # std deviation of normal distribution
        BasisDegree = self.maxDegree_N
        quadOrder = 10
        setSize = 1000

        (alphas, uTrain) = self.generateTrainingDataMK3(BasisDegree, quadOrder, setSize, mean, sigma)
        self.trainingData = self.cleanTrainingDataMK3(alphas, uTrain)
        '''

    ### Helper functions ###

    '''
    ### Training data generator
    def generateTrainingDataMK3(self, BasisDegree, quadOrder, setSize, mean, sigma):
        # Returns a training set of size setSize of moments corresponding to BasisDegree.

        ### define helper functions
        # helper functions
        def computeBasisHelper(quadorder, basisDegree):
            Q = sqp.gausslegendre.GaussLegendre(order=quadorder)

            # a) compute quadrature points.
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
            def Y0_0(mu, phi):
                return np.sqrt(1 / (4 * np.pi))

            def Y1_m1(mu, phi):
                return -np.sqrt(3 / (4 * np.pi)) * np.sqrt(1 - mu * mu) * np.sin(phi)

            def Y1_0(mu, phi):
                return np.sqrt(3 / (4 * np.pi)) * mu

            def Y1_1(mu, phi):
                return -np.sqrt(3 / (4 * np.pi)) * np.sqrt(1 - mu * mu) * np.cos(phi)

            def Y2_m2(mu, phi):
                return np.sqrt(15 / (16 * np.pi)) * (1 - mu * mu) * np.sin(2 * phi)

            def Y2_m1(mu, phi):
                return -1 * np.sqrt(15 / (4 * np.pi)) * mu * np.sqrt(1 - mu * mu) * np.sin(phi)

            def Y2_0(mu, phi):
                return np.sqrt(5 / (16 * np.pi)) * (3 * mu * mu - 1)

            def Y2_1(mu, phi):
                return -1 * np.sqrt(15 / (4 * np.pi)) * mu * np.sqrt(1 - mu * mu) * np.cos(phi)

            def Y2_2(mu, phi):
                return np.sqrt(15 / (16 * np.pi)) * (1 - mu * mu) * np.cos(2 * phi)

            # Transform karth coordinates to shperical coordinates:
            thetaMu = sqp.tools.xyz2thetaphi(quadPts)  # mu in [0,pi]
            phi = thetaMu[:, 0]
            mu = np.cos(thetaMu[:, 1])
            nPts = quadPts.shape[0]
            basisManual = np.zeros(basis.shape)

            for i in range(0, nPts):  # manual computation...

                basisManual[0, i] = Y0_0(mu[i], phi[i])
                basisManual[1, i] = Y1_m1(mu[i], phi[i])
                basisManual[2, i] = Y1_0(mu[i], phi[i])
                basisManual[3, i] = Y1_1(mu[i], phi[i])
                # basisManual[4,i] = Y2_m2(mu[i],phi[i])
                # basisManual[5,i] = Y2_m1(mu[i],phi[i])
                # basisManual[6,i] = Y2_0(mu[i],phi[i])
                # basisManual[7,i] = Y2_1(mu[i],phi[i])
                # basisManual[8,i] = Y2_2(mu[i],phi[i])

               
                #if(np.linalg.norm(basisManual[:,i]-basis[:,i]) > 1e-8):
                #    print("at quad pt")
                #    print(i)
                #    print(basisManual[:,i]-basis[:,i]) # ==> python spherical harmonics have wrong weights

            # Unit test orthonormality
            #for j in range(0,9):
            #    integral = np.zeros(basis.shape[0])
            #    for i in range(0, basis.shape[1]):
            #        integral += quadWeights[i] * basisManual[j, i] * basisManual[:, i]
            #    print(integral)
            
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

        N_moments = self.getIdxSphericalHarmonics(BasisDegree, BasisDegree) + 1

        uTrain = np.zeros((setSize, N_moments))
        alphas = np.zeros((setSize, N_moments))

        # 1) generate basis functions
        (basis_v, quadPts, quadWeights) = computeBasisHelper(quadOrder, BasisDegree)

        # need the transpose for convenience
        basis_vT = np.matrix.transpose(basis_v)

        # data generator
        def dataGenerator(i):
            # 2) generate random normally distributed alphas

            # alphas[i, :] = np.random.normal(mean, sigma, N_moments)
            alpha = np.random.normal(mean, sigma, N_moments)

            # 3) generate psi_v
            # psi_v = np.exp(np.matmul(basis_vT, alphas[i, :]))
            psi_v = np.exp(np.matmul(basis_vT, alpha))

            # 4) compute the Moments of psi
            # uTrain[i, :] = integrateMoments(basis_v, psi_v, quadWeights)

            # print status every 100 000 steps
            if (i % 100000 == 0):
                print("Status: {:.2f} percent".format(i / setSize * 100))

            return (integrateMoments(basis_v, psi_v, quadWeights), alpha)

        # parallelize data generation
        num_cores = multiprocessing.cpu_count()
        print("Starting data generation using " + str(num_cores) + "cores")
        # (alphas, uTrain)
        uTrain_Alpha_List = Parallel(n_jobs=num_cores)(delayed(dataGenerator)(i) for i in range(0, setSize))

        for i in range(0, setSize):
            # (alphas[i, :], uTrain[i, :]) = dataGenerator(i)
            alphas[i, :] = uTrain_Alpha_List[i][1]
            uTrain[i, :] = uTrain_Alpha_List[i][0]

        return (alphas, uTrain)

    def cleanTrainingDataMK3(self,alphaTrain, uTrain):
        deletionIndices = list()

        for idx in range(0, alphaTrain.shape[0]):
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

            # delete the failing entry
            if (deleteEntry):
                deletionIndices.append(idx)

        uTrain = np.delete(uTrain, deletionIndices, 0)
        alphaTrain = np.delete(alphaTrain, deletionIndices, 0)

        return (alphaTrain, uTrain)
    '''
