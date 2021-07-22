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
    y = [6.51778e-55,
         9.20148e-53,
         1.1754e-50,
         1.35858e-48,
         1.42087e-46,
         1.3446e-44,
         1.15134e-42,
         8.92042e-41,
         6.2537e-39,
         3.96697e-37,
         2.27694e-35,
         1.18254e-33,
         5.5571e-32,
         2.36294e-30,
         9.09133e-29,
         3.165e-27,
         9.96986e-26,
         2.84168e-24,
         7.3288e-23,
         1.71025e-21,
         3.61126e-20,
         6.89965e-19,
         1.1928e-17,
         1.86585e-16,
         2.64093e-15,
         3.38226e-14,
         3.91948e-13,
         4.1098e-12,
         3.89927e-11,
         3.34747e-10,
         2.60028e-09,
         1.82766e-08,
         1.16236e-07,
         6.6889e-07,
         3.4829e-06,
         1.64096e-05,
         6.99559e-05,
         0.00026985,
         0.000941867,
         0.0029746,
         0.00850037,
         0.0219795,
         0.0514242,
         0.108865,
         0.208536,
         0.361445,
         0.566858,
         0.80441,
         1.03288,
         1.20004,
         1.26157,
         1.20004,
         1.03288,
         0.80441,
         0.566858,
         0.361445,
         0.208536,
         0.108865,
         0.0514242,
         0.0219795,
         0.00850037,
         0.0029746,
         0.000941867,
         0.00026985,
         6.99559e-05,
         1.64096e-05,
         3.4829e-06,
         6.6889e-07,
         1.16236e-07,
         1.82766e-08,
         2.60028e-09,
         3.34747e-10,
         3.89927e-11,
         4.1098e-12,
         3.91948e-13,
         3.38226e-14,
         2.64093e-15,
         1.86585e-16,
         1.1928e-17,
         6.89965e-19,
         3.61126e-20,
         1.71025e-21,
         7.3288e-23,
         2.84168e-24,
         9.96986e-26,
         3.165e-27,
         9.09133e-29,
         2.36294e-30,
         5.5571e-32,
         1.18254e-33,
         2.27694e-35,
         3.96697e-37,
         6.2537e-39,
         8.92042e-41,
         1.15134e-42,
         1.3446e-44,
         1.42087e-46,
         1.35858e-48,
         1.1754e-50,
         9.20148e-53]
    x = np.linspace(-5, 5, 100)
    plt.plot(x, y)
    plt.show()
    int = sum(y) / 10;
    print(int)
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
