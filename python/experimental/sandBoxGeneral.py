### imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

import src.utils as utils

# import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from src.utils import finiteDiff, integrate, loadData, evaluateModel

plt.style.use("kitish")


# ------  Code starts here --------

def main():
    # --- Set Parameters ---
    batchSize = 64
    epochCount = 5000
    filename1 = 'models/sandbox/best_model_linear.h5'
    filename2 = 'models/sandbox/best_model_tscheb.h5'

    nwHeight = 8
    nwWidth = 5
    inputDim = 1
    nPts = 5000
    maxIter = 1000

    # test Data
    [xTest, yTest] = createTrainingData(nPts * 100, -5, 5, mode="linear")
    # xTest = xTest[1::2]
    # yTest = yTest[1::2]

    ### linear data
    [xL, yL] = createTrainingData(maxIter * 3, -5, 5, mode="linear")  # samples data between -1 and 1
    [xT, yT] = [xL, yL]  # utils.shuffleTrainData(x, y)

    model1 = createModelRelu(nwWidth, nwHeight, inputDim)
    # model1.load_weights(filenameInit)

    multistepTraining(xL, yL, model1, maxIter, epochCount, batchSize)

    return 0


def multistepTraining(xT, yT, model, maxIter, epochs, batchSize):
    filename1 = 'models/sandbox/best_model_linear.h5'
    trainLen = xT.shape[0]
    mc_best = tf.keras.callbacks.ModelCheckpoint(filename1, monitor='loss', mode='min',
                                                 save_best_only=True,
                                                 verbose=2)
    xTList = list(xT)
    yTList = list(yT)

    yList = []
    xList = []

    ypred = model(xT)
    ypredArray = np.asarray(ypred)
    yDiff = np.linalg.norm(ypredArray - yT, axis=0, ord=2)
    newY = np.amax(yDiff)
    newIdx = np.where(yDiff == newY)[0]

    yList.append([yTList.pop(0)])
    yList.append([yTList.pop(-1)])
    xList.append([xTList.pop(0)])
    xList.append([xTList.pop(-1)])

    for iter in range(0, maxIter):
        xarr = np.asarray(xList)
        yarr = np.asarray(yList)
        history = model.fit(x=xarr, y=yarr,
                            validation_split=0.0,
                            epochs=epochs,
                            batch_size=batchSize,
                            verbose=0)

        print("Trained on iteration: " + str(iter))

        # Get new data an evaluate current data
        ypred = model(np.asarray(xTList))
        ypredArray = np.asarray(ypred)
        tmp = np.asarray(yTList).reshape(ypredArray.shape)
        yDiff = ypredArray - tmp
        yDiff = np.absolute(yDiff)
        newY = np.amax(yDiff)
        newIdxes = np.where(yDiff == newY)
        newIdx = newIdxes[0]

        utils.plot1D(np.asarray(xTList), [np.asarray(yTList), ypredArray, yDiff], ["y", "model", "difference"],

                     '../models/sandbox/prediction' + str(iter),
                     log=False)

        # sort points

        utils.plot1D(xarr, [yarr], ["Interpolation points"],
                     '../models/sandbox/datapts' + str(iter),
                     log=False, linetypes=['*'])

        # print histories
        utils.plot1D(history.epoch, [history.history['loss']],
                     ["model loss"],
                     '../models/sandbox/traininghistory' + str(iter),
                     log=True, linetypes=['-', '--'])

        yList.append([yTList.pop(newIdx[0])])
        xList.append([xTList.pop(newIdx[0])])
    return 0


def createTrainingData(nPts, a=-1, b=1, mode="linear"):
    if (mode == "tscheb"):
        x = np.zeros((nPts,))
        degN = nPts - 1
        for k in range(0, nPts):
            tmp = np.cos((1 + 2 * (degN - k)) / (2 * (degN + 1)) * np.pi)
            x[k] = a + (tmp + 1) / 2 * (b - a)

    else:  # (mode == "linear"):
        x = np.linspace(a, b, nPts)

    y = rungeFunc(x)

    return [x, y]


def rungeFunc(x):
    return 1 / (1 + x * x)


def quadFunc(x):
    return x * x


def createModel(nwWidth, nwHeight, inputDim):  # Build the network:

    # basic dense network
    # Define the input

    # Weight initializer for sofplus  after K Kumar
    input_stddev = np.sqrt((1 / inputDim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
    hidden_stddev = np.sqrt((1 / nwWidth) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))

    hiddenInitializer = initializers.RandomNormal(mean=0., stddev=hidden_stddev)
    inputLayerInitializer = initializers.RandomNormal(mean=0., stddev=input_stddev)
    # hiddenInitializer = initializers.Zeros()
    # inputLayerInitializer = initializers.Zeros()

    biasInitializer = initializers.Zeros()

    #### input layer ####
    input_ = keras.Input(shape=(inputDim,))
    hidden = layers.Dense(nwWidth, activation="softplus", kernel_initializer=inputLayerInitializer,
                          bias_initializer=biasInitializer)(input_)

    # hidden Layer
    for idx in range(0, nwHeight):
        hidden = layers.Dense(nwWidth, activation="softplus", kernel_initializer=hiddenInitializer,
                              bias_initializer=biasInitializer)(hidden)

    output_ = layers.Dense(1, activation=None, kernel_initializer=inputLayerInitializer,
                           bias_initializer=biasInitializer)(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_], name="model1")
    model.summary()

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])

    return model


def createModelRelu(nwWidth, nwHeight, inputDim):  # Build the network:

    # basic dense network
    # Define the input

    # Weight initializer for sofplus  after K Kumar
    input_stddev = np.sqrt((1 / inputDim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
    hidden_stddev = np.sqrt((1 / nwWidth) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))

    hiddenInitializer = initializers.RandomNormal(mean=0., stddev=hidden_stddev)
    inputLayerInitializer = initializers.RandomNormal(mean=0., stddev=input_stddev)
    # hiddenInitializer = initializers.Zeros()
    # inputLayerInitializer = initializers.Zeros()

    biasInitializer = initializers.Zeros()

    #### input layer ####
    input_ = keras.Input(shape=(inputDim,))
    hidden = layers.Dense(nwWidth, activation="softplus", kernel_initializer=inputLayerInitializer,
                          bias_initializer=biasInitializer, kernel_regularizer='l1_l2')(input_)

    # hidden Layer
    for idx in range(0, nwHeight):
        hidden = layers.Dense(nwWidth, activation="softplus", kernel_initializer=hiddenInitializer,
                              bias_initializer=biasInitializer, kernel_regularizer='l1_l2')(hidden)

    output_ = layers.Dense(1, activation=None, kernel_initializer=inputLayerInitializer,
                           bias_initializer=biasInitializer, kernel_regularizer='l1_l2')(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_], name="model1")
    model.summary()

    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])

    return model


if __name__ == '__main__':
    main()
