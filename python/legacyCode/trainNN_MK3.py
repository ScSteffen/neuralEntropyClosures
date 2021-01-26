### This is a script for the training of the
### Third NN approach

'''
Improvements:
1)  accepts u as a N-vector
2)  Generalized Loss function
3)  Adapted network layout
4)  RESNet Used as Netowork ( TODO )
'''

import csv
import multiprocessing
import pandas as pd
from joblib import Parallel, delayed

### imports
import numpy as np
# in-project imports
import sphericalquadpy as sqp
import nnUtils
import csv
# Tensorflow
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------  Code starts here --------

def main():
    ### 0) Set variables #######################################################
    # Set up Quadrature

    quadOrder = 10 #QuadOrder for training
    BasisDegree = 1  # Determines the number of moments used
    SetSize = 9000000#300000 # Determines the size of the training set

    # Set up data generator
    mean = 0 # mean of normal distribution
    sigma = 2   # std deviation of normal distribution

    # Name of modelDirectory
    filename = "models/MK3_nnM_1"
    filenameAlpha = "trainingData_M1_alpha.csv"

    filenameU = "trainingData_M1_u.csv"

    # Training Parameters
    batchSize = SetSize
    epochCount = 150000

    ### 1)  Generate Training Data #############################################

    print("Create Training Data")
    # build training data!

    #(alphas, uTrain) = load_trainingData(filenameAlpha, filenameU)  # generate_trainingDataMK3(BasisDegree,quadOrder,SetSize, mean, sigma) #np.ones((20000, 9))
    (uTrain, alphas) = read_data("testNN_M1.csv")

    # print(alphas.shape)
    # print(uTrain.shape)

    # clean training data from outliers!
    print("Clean Training Data")
    # (alphas, uTrain) = clean_trainingDataMK3(alphas,uTrain)
    (alphas,uTrain) = load_trainingData(filenameAlpha, filenameU) #generate_trainingDataMK3(BasisDegree,quadOrder,SetSize, mean, sigma) #np.ones((20000, 9))

    # Translate uTrain to a tf tensor.
    uTensor = tf.constant(uTrain, dtype=tf.float32, shape=None, name='Const')

    # nnUtils.plot_trainingData(uTrain)
    print(uTrain.shape)

    ### 2) Create Model ########################################################
    print("Create Model")
    model = create_modelMK3(quadOrder, BasisDegree)
    # Load weights
    model.load_weights(filename + '/best_model.h5')

    ### 3) Setup Training and Train the model ##################################

    # Create Early Stopping callback
    es = EarlyStopping(monitor='loss', mode='min', min_delta=0.000000001, patience=500,
                     verbose=10)  # loss == custom_loss1dMBPrime by model definition
    mc_best = ModelCheckpoint(filename+'/best_model.h5', monitor='loss', mode='min', save_best_only=True)
    mc_500 = ModelCheckpoint(filename+'/model_quicksave.h5', monitor='loss', mode='min', save_best_only=False, save_freq = 500)

    # Train the model
    print("Train Model")
    history = model.fit(uTensor, alphas, validation_split=0.3, epochs=epochCount, batch_size=batchSize, verbose=1,
                        callbacks=[es,mc_best, mc_500]) #batch size = 900000

    # View History
    # nnUtils.print_history(history.history)

    ### 4) Save trained model and history ########################################

    print("Save model and history")
    nnUtils.save_training(filename, model, history)
    print("Training successfully saved")

    # load history
    history1 = nnUtils.load_trainHistory(filename)
    # print history as a check
    nnUtils.print_history(history1)

    print("Training Sequence successfully finished")
    return 0


### Custom loss function
def cLoss_FONC_varD(qOrder, bDegree):
    # 1) Compute basis at quadrature points of GaussLegendreQuadrature
    def computeBasis(quadorder, basisDegree):
        Q = sqp.gausslegendre.GaussLegendre(order=quadorder)

        # a) compute quadrature points.
        quadPts = Q.computequadpoints(quadorder)
        quadWeights = Q.computequadweights(quadorder)
        # b) Compute spherical harmonics basis at quadPts
        mBasisList = list()
        for l in range(0, basisDegree + 1):
            for k in range(-l, l + 1):
                temp = sqp.tools.sphericalharmonics.ylm(k, l, quadPts[:, 0], quadPts[:, 1], quadPts[:, 2])
                if (k < 0):
                    mBasisList.append(temp.imag)
                elif (k == 0):
                    mBasisList.append(temp.real)
                else:  # (k > 0):
                    mBasisList.append(temp.real)

        temp = np.array(mBasisList)

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
        basisManual = np.zeros(temp.shape)

        for i in range(0, nPts):  # manual computation...

            basisManual[0, i] = Y0_0(mu[i], phi[i])
            basisManual[1, i] = Y1_m1(mu[i], phi[i])
            basisManual[2, i] = Y1_0(mu[i], phi[i])
            basisManual[3, i] = Y1_1(mu[i], phi[i])
            basisManual[4, i] = Y2_m2(mu[i], phi[i])
            basisManual[5, i] = Y2_m1(mu[i], phi[i])
            basisManual[6, i] = Y2_0(mu[i], phi[i])
            basisManual[7, i] = Y2_1(mu[i], phi[i])
            basisManual[8, i] = Y2_2(mu[i], phi[i])
        #### End Manual compuation ####

        return (basisManual, quadPts, quadWeights)

    # 2) Helper functions
    def innerProd(alphaX, mBasisY):
        # alphaX.shape = (batchSize,NmaxMoment)
        # mBasisY.shape = (NmaxMoment,1)
        # output.shape = (batchSize,1)
        return K.dot(alphaX, mBasisY)

    def entropyDualPrimeMaxwellBoltzmann(alpha, mBasisPt):
        # alpha.shape = (batchSize,NmaxMoment)
        # mBasisPt.shape = (NmaxMoment,1)
        # output.shape = (batchSize,NmaxMoment)

        mBasisPtTensor = tf.constant(mBasisPt, dtype=tf.float32, shape=(mBasisPt.shape[0], 1),
                                     name='Const')  # shape = (N,1)

        temp = K.exp(innerProd(alpha, mBasisPtTensor))
        return K.dot(temp, K.transpose(mBasisPtTensor))

    def integrate(mBasis, alpha, quadWeight):
        # input: mBasis.shape = (NmaxMoment, numQuadPts)
        #         alpha.shape = (batchSize, NmaxMoment)
        #         quadWeight.shape = (numQuadPts)
        # output: integral.shape = (batchSize, NmaxMoment)

        (NmaxMoment, numPts) = mBasis.shape
        (batchSize, NmaxMoment_redundant) = alpha.shape

        # print(batchSize)
        # print(NmaxMoment)

        if (batchSize == None):
            batchSize = 1

        integral = tf.constant(np.zeros((batchSize, NmaxMoment)), dtype=tf.float32, name='Const')

        for i in range(0, numPts):
            integral += quadWeight[i] * entropyDualPrimeMaxwellBoltzmann(alpha, mBasis[:, i])

        return integral

    (mbasis, quadPts, quadWeights) = computeBasis(qOrder, bDegree)

    # 3) Compute the Loss
    def loss(u_input, alpha_pred):
        # input: u_input.shape = (batchSize, NmaxMoment)
        #         alpha_pred.shape = (batchSize, NmaxMoment)
        # output: out.shape = (batchSize, 1)

        temp = integrate(mbasis, alpha_pred, quadWeights)
        return K.sum(K.square(temp - u_input), axis=1)

    return loss

### Build the network:
def create_modelMK3(quadOrder, BasisDegree):
    # Define Residual block
    def residual_block(x: Tensor) -> Tensor:
        y = layers.Dense(20, activation="relu")(x)
        y = layers.Dense(20, activation="relu")(y)
        y = layers.Dense(20, activation="relu")(y)

        out = layers.Add()([x, y])
        out = layers.ReLU()(out)
        out = layers.BatchNormalization()(out)
        return out

    # Define the input
    # Number of basis functions used:

    NmaxBasis =nnUtils.getGlobalIdx( BasisDegree,BasisDegree)+1

    input_ = keras.Input(shape=(NmaxBasis,))

    # Hidden layers
    # hidden = layers.BatchNormalization()(input_)
    hidden = layers.Dense(20, activation="relu")(input_)

    '''
    hidden = layers.Dense(20, activation="relu")(hidden)
    hidden = layers.Dense(20, activation="relu")(hidden)
    hidden = layers.Dense(20, activation="relu")(hidden)
    hidden = layers.Dense(20, activation="relu")(hidden)
    hidden = layers.Dense(20, activation="relu")(hidden)
    hidden = layers.Dense(20, activation="relu")(hidden)
    hidden = layers.Dense(20, activation="relu")(hidden)
    '''

    # Resnet Layers

    hidden = residual_block(hidden)
    hidden = residual_block(hidden)
    hidden = residual_block(hidden)
    hidden = residual_block(hidden)

    hidden = layers.Dense(20, activation="relu")(hidden)

    # Define the ouput
    output_ = layers.Dense(NmaxBasis)(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    # , metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    return model


### Training data generator
def generate_trainingDataMK3(BasisDegree, quadOrder, setSize, mean, sigma):
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

    N_moments = getGlobalIdx(BasisDegree, BasisDegree) + 1

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

### Load Training Data
def load_trainingData(filenameAlpha, filenameU):

    #Load Alpha
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

    alpha = np.asarray(alphaList)
    uTrain = np.asarray(uList)

    return (alpha, uTrain)

### Delete outliers in training data
def clean_trainingDataMK3(alphaTrain, uTrain):
    deletionIndices = list()

    for idx in range(0, alphaTrain.shape[0]):
        deleteEntry = False

        #check for NaN values

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

### Read Data
def read_data(filename):
    # Note! This is essentially the same code as preprocess_data(filename) of trainNN_MK1.py. Needs to be refactored!

    ### ------- Code Starts Here ----- ####

    # reading csv file
    dataFrameInput = pd.read_csv(filename)  # outputs a dataframe object

    idx = 0
    xDataList = list()
    yDataList = list()
    data = dataFrameInput.values  # numpy array

    ## Somehow t = 0 has an odd number of elements, just cut it out

    for row in data:
        if (row[2] > 0):
            if (idx % 2 == 0):
                xDataList.append(row)
            else:
                yDataList.append(row)

            idx = idx + 1

    # merge the lists
    DataList = list()
    for rowX, rowY in zip(xDataList, yDataList):
        DataList.append([rowX, rowY])

    # Shuffle data
    # random.shuffle(DataList)

    DataArray = np.asarray(DataList)

    # Strip off header information, i.e. the first 3 cols
    DataArraySlim = DataArray[:, :, 3:]

    xData = DataArraySlim[:, 0, :]
    yData = DataArraySlim[:, 1, :]

    return (xData, yData)

if __name__ == '__main__':
    main()
