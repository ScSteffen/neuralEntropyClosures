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

# plt.style.use("kitish")

from src.math import EntropyTools
import numpy as np
import scipy.stats


# ------  Code starts here --------

def main():
    # model = tf.keras.Sequential([
    #    tf.keras.layers.Dense(5, input_shape=(3,)),
    #    tf.keras.layers.Softmax()])
    # model.save('tmp/model')

    loaded_model = tf.keras.models.load_model('tmp/best_model')
    x = tf.random.uniform((10, 2))
    print(loaded_model.predict(x))
    print("end")
    # x = np.linspace(-10, 10, 100)
    # plt.plot(x, scipy.stats.norm(0, 10).pdf(x))
    # plt.show()

    """
    et2 = EntropyTools(polynomial_degree=4)
    alpha_1 = tf.constant([-1.42096, 1.40992, 0.1, 0.1, 2.0], shape=(1, 5), dtype=tf.float32)
    alpha_2 = tf.constant([-1.42096, 1.40992, 0.0], shape=(1, 3), dtype=tf.float32)
    # div = et2.KL_divergence(alpha_1, alpha_2)
    # print(div)

    et1 = EntropyTools(polynomial_degree=1)
    alpha = tf.constant([-1.42096, 1.40992], shape=(1, 2), dtype=tf.float32)
    pts = et2.quadPts

    f1 = et1.compute_kinetic_density(alpha)
    f2 = et2.compute_kinetic_density(alpha_1)
    plt.plot(pts[0, :], f1[0, :], '--')
    plt.plot(pts[0, :], f2[0, :], '-.')
    u1 = et1.compute_u(f1)
    u2 = et2.compute_u(f2)
    print("u1:")
    print(u1)
    print("u2:")
    print(u2)

    # plt.show()

    et3 = EntropyTools(polynomial_degree=3)
    alpha_full = np.asarray([-3.20452, -10.109, 1.60217, 14.1154])

    # alpha = tf.constant(alpha, shape=(1, 3), dtype=tf.float32)
    # alpha_full = et3.reconstruct_alpha(tf.constant(alpha))
    u = et3.reconstruct_u(tf.constant(alpha_full, shape=(1, 4), dtype=tf.float32))
    u_np = u.numpy()
    print(u_np)
    print(u_np / u_np[0, 0])
    """

    et4 = EntropyTools(polynomial_degree=1, spatial_dimension=2)
    alpha_generator = [[-0.20, 20]]
    alpha_full = et4.reconstruct_alpha(tf.constant(alpha_generator))
    opti4 = et4.reconstruct_u(alpha_full)
    alpha_init = [[0, 0, 0.0]]
    alpha3 = et4.minimize_entropy(u=opti4, start=tf.constant(alpha_init))
    f4 = et4.compute_kinetic_density(alpha3)
    pt4 = et4.quadPts

    et3 = EntropyTools(polynomial_degree=3)
    opti3 = tf.constant(opti4[:, :-1])
    alpha_init = [[0, 0, 0, 0]]
    alpha = et3.minimize_entropy(u=opti3, start=tf.constant(alpha_init))
    f3 = et3.compute_kinetic_density(alpha)
    pts3 = et3.quadPts

    et2 = EntropyTools(polynomial_degree=2)
    opti2 = tf.constant(opti3[:, :-1])
    alpha_init = [[0, 0, 0]]
    alpha = et2.minimize_entropy(u=opti2, start=tf.constant(alpha_init))
    f2 = et2.compute_kinetic_density(alpha)
    pts2 = et2.quadPts

    et1 = EntropyTools(polynomial_degree=1)
    opti1 = tf.constant(opti2[:, :-1])
    alpha_init = [[0, 0]]
    alpha = et1.minimize_entropy(u=opti1, start=tf.constant(alpha_init))
    f1 = et1.compute_kinetic_density(alpha)
    pts1 = et1.quadPts

    plt.plot(pt4[0, :], f4[0, :], '--')
    plt.plot(pts3[0, :], f3[0, :], '.')
    plt.plot(pts2[0, :], f2[0, :], '-.')
    plt.plot(pts1[0, :], f1[0, :], '-')
    plt.legend(['M4', 'M3', 'M2', 'M1'])
    plt.show()
    print("Here")
    """
    et2 = EntropyTools(polynomial_degree=2)
    alpha_full = np.asarray([1, 1, 1])
    # alpha = tf.constant(alpha, shape=(1, 2), dtype=tf.float32)
    # alpha_full = et2.reconstruct_alpha(tf.constant(alpha))
    u = et2.reconstruct_u(tf.constant(alpha_full, shape=(1, 3), dtype=tf.float32))
    u_np = u.numpy()
    print(u_np)
    print(u_np / u_np[0, 0])
    et1 = EntropyTools(polynomial_degree=1)
    alpha_full = np.asarray([1, 1])
    # alpha = tf.constant(alpha, shape=(1, 1), dtype=tf.float32)
    # alpha_full = et1.reconstruct_alpha(tf.constant(alpha))
    u = et1.reconstruct_u(tf.constant(alpha_full, shape=(1, 2), dtype=tf.float32))
    u_np = u.numpy()
    print(u_np)
    print(u_np / u_np[0, 0])
    """
    """
    ns = 1000
    x = np.linspace(-0.1, 0.1, ns)
    alpha = np.zeros((ns, 3))
    for i in range(len(x)):
        alpha[i] = np.asarray([4.0, 1.7, 2.3]) + np.asarray([0.1, 0.1, 0.1]) * x[i]
    alpha = tf.constant(alpha, dtype=tf.float32)
    alpha_full = et.reconstruct_alpha(tf.constant(alpha))
    u = et.reconstruct_u(alpha_full)
    u_np = u.numpy()
    fig, axis = plt.subplots(nrows=2, ncols=2)
    axis[0, 0].plot(x, u[:, 0])
    axis[0, 1].plot(x, u[:, 1])
    axis[1, 0].plot(x, u[:, 2])
    axis[1, 1].plot(x, u[:, 3])
    plt.show()
    print(u_np)
    """
    """
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
    """
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

        utils.plot_1d(np.asarray(xTList), [np.asarray(yTList), ypredArray, yDiff], ["y", "model", "difference"],

                      '../models/sandbox/prediction' + str(iter),
                      log=False)

        # sort points

        utils.plot_1d(xarr, [yarr], ["Interpolation points"],
                      '../models/sandbox/datapts' + str(iter),
                      log=False, linetypes=['*'])

        # print histories
        utils.plot_1d(history.epoch, [history.history['loss']],
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
