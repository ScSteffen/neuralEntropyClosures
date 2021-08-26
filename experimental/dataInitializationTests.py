### imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers

import src.utils as utils

# import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

from src.utils import finiteDiff, integrate, load_data, evaluateModel

plt.style.use("kitish")


# ------  Code starts here --------

def main():
    # --- Set Parameters ---
    batchSize = 64
    epochCount = 10
    filename1 = 'models/testSampling/best_model_linear.h5'
    filename2 = 'models/testSampling/best_model_tscheb.h5'

    filenameInit = 'models/testSampling/weight_initializer.h5'

    nwHeight = 4
    nwWidth = 4
    inputDim = 1
    nPts = 100

    # test Data
    [xTest, yTest] = createTrainingData(nPts * 2, -5, 5, mode="linear")
    xTest = xTest[1::2]
    yTest = yTest[1::2]

    ### linear data
    mc_best = tf.keras.callbacks.ModelCheckpoint(filename1, monitor='loss', mode='min',
                                                 save_best_only=True,
                                                 verbose=2)

    [xL, yL] = createTrainingData(nPts, -5, 5, mode="linear")  # samples data between -1 and 1
    [xT, yT] = [xL, yL]  # utils.shuffleTrainData(x, y)

    model1 = createModelRelu(nwWidth, nwHeight, inputDim)
    # model1.load_weights(filenameInit)

    history1 = model1.fit(x=xT, y=yT, validation_split=0.001,
                          epochs=epochCount,
                          batch_size=batchSize,
                          verbose=1, callbacks=mc_best)

    ypred = model1(xTest)

    # utils.plot1D(x, [y], ["y"], '../models/testPoly/prediction')

    utils.plot1D(xL, [yL, ypred], ["y", "ylinear"],
                 '../models/testSampling/prediction',
                 log=False)

    utils.plot1D(xL, [yL, yL], ["y", "Interpolation points"],
                 '../models/testSampling/dataptsLinear',
                 log=False, linetypes=['-', '*'])

    # print histories
    utils.plot1D(history1.epoch, [history1.history['loss']],
                 ["loss linear"],
                 '../models/testSampling/traininghistories',
                 log=False, linetypes=['-', '--'])
    return 0


def main2():
    # --- Set Parameters ---
    batchSize = 64
    epochCount = 200
    filename1 = 'models/testPoly/best_model_linear.h5'
    filename2 = 'models/testPoly/best_model_tscheb.h5'

    filenameInit = 'models/testPoly/weight_initializer.h5'

    nwHeight = 10
    nwWidth = 5
    inputDim = 1
    nPts = 1000

    # test Data
    [xTest, yTest] = createTrainingData(nPts * 2, -5, 5, mode="linear")
    xTest = xTest[1::2]
    yTest = yTest[1::2]

    ### linear data
    mc_best = tf.keras.callbacks.ModelCheckpoint(filename1, monitor='loss', mode='min',
                                                 save_best_only=True,
                                                 verbose=2)

    [xL, yL] = createTrainingData(nPts, -5, 5, mode="linear")  # samples data between -1 and 1
    [xT, yT] = [xL, yL]  # utils.shuffleTrainData(x, y)

    model1 = createModel(nwWidth, nwHeight, inputDim)
    model1.load_weights(filenameInit)

    history1 = model1.fit(x=xT, y=yT, validation_split=0.001,
                          epochs=epochCount,
                          batch_size=batchSize,
                          verbose=1, callbacks=mc_best)

    ypred = model1(xTest)

    ### tschebycheff data
    mc_best = tf.keras.callbacks.ModelCheckpoint(filename2, monitor='loss', mode='min',
                                                 save_best_only=True,
                                                 verbose=2)

    [xTscheb, yTscheb] = createTrainingData(nPts, -5, 5, mode="tscheb")  # samples data between -1 and 1

    [xT, yT] = [xTscheb, yTscheb]  # utils.shuffleTrainData(x, y)

    model2 = createModel(nwWidth, nwHeight, inputDim)
    model2.load_weights(filenameInit)

    history2 = model2.fit(x=xT, y=yT, validation_split=0.001,
                          epochs=epochCount,
                          batch_size=batchSize,
                          verbose=1, callbacks=mc_best)

    ypred2 = model2(xTest)
    # utils.plot1D(x, [y], ["y"], '../models/testPoly/prediction')
    utils.plot1D(xTest, [yTest, ypred, ypred2], ["y", "ylinear", "yTscheb"],
                 '../models/testPoly/prediction',
                 log=False)

    utils.plot1D(xTscheb, [yTscheb, yTscheb], ["y", "Interpolation points"],
                 '../models/testPoly/datapts',
                 log=False, linetypes=['-', '*'])

    utils.plot1D(xL, [yL, yL], ["y", "Interpolation points"],
                 '../models/testPoly/dataptsLinear',
                 log=False, linetypes=['-', '*'])

    # print histories
    utils.plot1D(history1.epoch, [history1.history['loss'], history2.history['loss']],
                 ["loss linear", "loss tschebycheff"],
                 '../models/testPoly/traininghistories',
                 log=False, linetypes=['-', '--'])
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


if __name__ == '__main__':
    main()
