"""
Author: Steffen Schotthöfer
Date: 21.04.2020
Brief: Script to test the optimal convex data sampling alogrithm
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import tensorflow as tf

from src.networks.configmodel import init_neural_closure
from src import utils


def quad_func(x):
    return x * x


def quad_func_grad(x):
    return 2 * x


def exp_func(x):
    return np.exp(x)  # / np.exp(3)


def sample_data(min_x, max_x, tol, function, grad):
    """
    performs the sampling algorithm
    """

    s_x = [min_x, max_x]
    e_max = tol
    j = 0
    while e_max >= tol:

        e_max = 0
        i = 0

        [eL, tL] = get_errors(s_x, function, grad, 0.0001)
        plt.semilogy(tL, eL, '*')
        plt.savefig("figures/errors_" + str(j).zfill(3) + ".png")
        plt.clf()

        while i < len(s_x) - 1:

            # error, in case that the approximation undershoots
            t1 = 1 / (grad(s_x[i + 1]) - grad(s_x[i])) * (function(s_x[i]) - s_x[i] * grad(s_x[i]) - (
                    function(s_x[i + 1]) - s_x[i + 1] * grad(s_x[i + 1])))
            e1 = function(t1) - (function(s_x[i]) + (t1 - s_x[i]) * grad(s_x[i]))

            # error in case that the approximation overshoots
            grad_intermediate = (function(s_x[i + 1]) - function(s_x[i])) / (s_x[i + 1] - s_x[i])

            def F(x):
                return 0.5 * (grad_intermediate - grad(x)) * (grad_intermediate - grad(x))

            result = scipy.optimize.minimize_scalar(method="bounded", fun=F, bounds=[s_x[i], s_x[i + 1]])
            if not result.success:
                ValueError("Optimization unsuccessful!")
                print(grad_intermediate)
                print((s_x[i + 1] - s_x[i]) / 2)
                print("----")

            t2 = float(result.x)
            e2 = function(s_x[i]) + grad_intermediate * (t2 - s_x[i]) - function(t2)

            if e1 > e2:
                t = t1
                e = e1
                # print("undershoot worse")
            else:
                t = t2
                e = e2
                # print("overshoot worse")

            if e > e_max:
                e_max = e
            if e > tol:
                s_x.insert(i + 1, t)
                i = i + 1
            i = i + 1
        j = j + 1
    return s_x


def get_errors(s_x, function, grad, tol):
    eL = []
    tL = []
    for i in range(len(s_x) - 1):
        t1 = 1 / (grad(s_x[i + 1]) - grad(s_x[i])) * (function(s_x[i]) - s_x[i] * grad(s_x[i]) - (
                function(s_x[i + 1]) - s_x[i + 1] * grad(s_x[i + 1])))
        e1 = function(t1) - (function(s_x[i]) + (t1 - s_x[i]) * grad(s_x[i]))

        # error in case that the approximation overshoots
        tmp1 = function(s_x[i + 1])
        tmp2 = function(s_x[i])
        grad_intermediate = (tmp1 - tmp2) / (s_x[i + 1] - s_x[i])

        def F(x):
            return 0.5 * (grad_intermediate - grad(x)) * (grad_intermediate - grad(x))

        result = scipy.optimize.minimize_scalar(method="bounded", fun=F, bounds=[s_x[i], s_x[i + 1]])
        if not result.success:
            ValueError("Optimization unsuccessful!")
            print(grad_intermediate)
            print((s_x[i + 1] - s_x[i]) / 2)
            print("----")

        t2 = float(result.x)
        e2 = function(s_x[i]) + grad_intermediate * (t2 - s_x[i]) - function(t2)

        if e1 > e2:
            eL.append(e1)
            tL.append(t1)
        else:
            eL.append(e2)
            tL.append(t2)

    return [eL, tL]


def main():
    # x = sample_data(-10, 10, 0.0001, quad_func, quad_func_grad)
    tolerance = 0.1
    x = sample_data(0, 2, tolerance, exp_func, exp_func)

    y = []
    for i in range(len(x)):
        y.append(exp_func(x[i]))

    plt.plot(x, y, '*')
    plt.savefig("figures/sampling.png")
    plt.clf()

    # plot errors
    [e, tL] = get_errors(x, exp_func, exp_func, 0.001)
    plt.plot(tL, e, '*')
    plt.savefig("figures/errors.png")
    plt.clf()

    # show distance between sampling points
    d = []
    for i in range(len(x) - 1):
        d.append(abs(x[i + 1] - x[i]))

    plt.semilogy(x[1:], d, '*')
    plt.savefig("figures/distance.png")

    # Sample a comparative equidistant version:
    nS = len(x)
    x_uniform = np.linspace(0, 2, nS)
    y_uniform = []
    for i in range(len(x_uniform)):
        y_uniform.append(exp_func(x_uniform[i]))

    ### Turn everything in numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    x_uniform = np.asarray(x_uniform)
    y_uniform = np.asarray(y_uniform)

    '''
    ### Train two networks with same weight initialization. Once with tol sampling, once with uniform sampling.
    
    model_smart = initNeuralClosure(modelNumber=11, polyDegree=1, spatialDim=1, folderName="testFolder",
                                    optimizer='adam', width=10, depth=5, normalized=True)
    model_uniform = initNeuralClosure(modelNumber=11, polyDegree=1, spatialDim=1, folderName="testFolder",
                                      optimizer='adam', width=10, depth=5, normalized=True)

    mc_best_smart = tf.keras.callbacks.ModelCheckpoint('model_smart/best_model.h5', monitor='loss', mode='min',
                                                       save_best_only=True, verbose=0)
    csv_logger_smart = tf.keras.callbacks.CSVLogger('model_smart/history.csv')
    mc_best_uniform = tf.keras.callbacks.ModelCheckpoint('model_uniform/best_model.h5', monitor='loss', mode='min',
                                                         save_best_only=True, verbose=0)
    csv_logger_uniform = tf.keras.callbacks.CSVLogger('model_uniform/history.csv')

    # initialize both models with the same weight
    model_smart.model.load_weights('best_model.h5')
    model_uniform.model.load_weights('best_model.h5')
    # some params
    epochs = 100000
    batch = 16
    model_smart.model.fit(x=x, y=y, validation_split=0.0, epochs=epochs, batch_size=batch, verbose=2,
                          callbacks=[mc_best_smart, csv_logger_smart])
    model_uniform.model.fit(x=x_uniform, y=y_uniform, validation_split=0.0, epochs=epochs, batch_size=batch, verbose=2,
                            callbacks=[mc_best_uniform, csv_logger_uniform])

    ### Test the models.
    x_test = np.linspace(0, 2, 10000)
    y_test = exp_func(x_test)

    pred_smart = model_smart.model(x_test)
    pred_uniform = model_uniform.model(x_test)

    diff_y = pointwiseDiff(pred_smart[0], y_test)
    diff_dy = pointwiseDiff(pred_smart[1], y_test)
    utils.plot1D(x_test, [pred_smart[0], y_test], ['y', 'y_test'], 'y_over_x_smart', log=False,
                 folder_name="model_smart")
    utils.plot1D(x_test, [pred_smart[1], y_test], ['dy', 'dy_test'], 'dy_over_x_smart', log=False,
                 folder_name="model_smart")
    utils.plot1D(x_test, [diff_y, diff_dy], ['difference y', 'difference dy'], 'errors_smart', log=True,
                 folder_name="model_smart")

    diff_y = pointwiseDiff(pred_uniform[0], y_test)
    diff_dy = pointwiseDiff(pred_uniform[1], y_test)
    utils.plot1D(x_test, [pred_uniform[0], y_test], ['y', 'y_test'], 'y_over_x_uniform', log=False,
                 folder_name="model_uniform")
    utils.plot1D(x_test, [pred_uniform[1], y_test], ['dy', 'dy_test'], 'dy_over_x_uniform', log=False,
                 folder_name="model_uniform")
    utils.plot1D(x_test, [diff_y, diff_dy], ['difference y', 'difference dy'], 'errors_uniform', log=True,
                 folder_name="model_uniform")
   
    # Comparison Study for dense model
    model_smart_dense = initNeuralClosure(modelNumber=12, polyDegree=1, spatialDim=1, folderName="testFolder",
                                          optimizer='adam', width=10, depth=5, normalized=True)
    model_uniform_dense = initNeuralClosure(modelNumber=12, polyDegree=1, spatialDim=1, folderName="testFolder",
                                            optimizer='adam', width=10, depth=5, normalized=True)
    mc_best_smart_dense = tf.keras.callbacks.ModelCheckpoint('model_smart_dense/best_model.h5', monitor='loss',
                                                             mode='min',
                                                             save_best_only=True, verbose=0)
    csv_logger_smart_dense = tf.keras.callbacks.CSVLogger('model_smart_dense/history.csv')
    mc_best_uniform_dense = tf.keras.callbacks.ModelCheckpoint('model_uniform_dense/best_model.h5', monitor='loss',
                                                               mode='min',
                                                               save_best_only=True, verbose=0)
    csv_logger_uniform_dense = tf.keras.callbacks.CSVLogger('model_uniform_dense/history.csv')

    # initialize both models with the same weight
    model_smart_dense.model.load_weights('best_model_dense.h5')
    model_uniform_dense.model.load_weights('best_model_dense.h5')
    # some params
    epochs = 4000
    batch = 16
    model_smart_dense.model.fit(x=x, y=y, validation_split=0.0, epochs=epochs, batch_size=batch, verbose=1,
                                callbacks=[mc_best_smart_dense, csv_logger_smart_dense])
    model_uniform_dense.model.fit(x=x_uniform, y=y_uniform, validation_split=0.0, epochs=epochs, batch_size=batch,
                                  verbose=1,
                                  callbacks=[mc_best_uniform_dense, csv_logger_uniform_dense])

    ### Test the models.
    x_test = np.linspace(0, 2, 10000)
    y_test = exp_func(x_test)

    pred_smart = model_smart_dense.model(x_test)
    pred_uniform = model_uniform_dense.model(x_test)

    diff_y = pointwiseDiff(pred_smart[0], y_test)
    diff_dy = pointwiseDiff(pred_smart[1], y_test)
    utils.plot1D(x_test, [pred_smart[0], y_test], ['y', 'y_test'], 'y_over_x_smart_dense', log=False,
                 folder_name="model_smart_dense")
    utils.plot1D(x_test, [pred_smart[1], y_test], ['dy', 'dy_test'], 'dy_over_x_smart_dense', log=False,
                 folder_name="model_smart_dense")
    utils.plot1D(x_test, [diff_y, diff_dy], ['difference y', 'difference dy'], 'errors_smart_dense', log=True,
                 folder_name="model_smart_dense")

    diff_y = pointwiseDiff(pred_uniform[0], y_test)
    diff_dy = pointwiseDiff(pred_uniform[1], y_test)
    utils.plot1D(x_test, [pred_uniform[0], y_test], ['y', 'y_test'], 'y_over_x_uniform_dense', log=False,
                 folder_name="model_uniform_dense")
    utils.plot1D(x_test, [pred_uniform[1], y_test], ['dy', 'dy_test'], 'dy_over_x_uniform_dense', log=False,
                 folder_name="model_uniform_dense")
    utils.plot1D(x_test, [diff_y, diff_dy], ['difference y', 'difference dy'], 'errors_uniform_dense', log=True,
                 folder_name="model_uniform_dense")
    '''

    ### Compare all models
    model_smart = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1, folder_name="testFolder",
                                      optimizer='adam', nw_width=10, nw_depth=5, normalized=True)
    model_uniform = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1, folder_name="testFolder",
                                        optimizer='adam', nw_width=10, nw_depth=5, normalized=True)
    model_smart_dense = init_neural_closure(network_mk=12, poly_degree=1, spatial_dim=1, folder_name="testFolder",
                                            optimizer='adam', nw_width=10, nw_depth=5, normalized=True)
    model_uniform_dense = init_neural_closure(network_mk=12, poly_degree=1, spatial_dim=1, folder_name="testFolder",
                                              optimizer='adam', nw_width=10, nw_depth=5, normalized=True)

    model_smart.model.load_weights('model_smart_1e-1/best_model.h5')
    model_uniform.model.load_weights('model_uniform_1e-1/best_model.h5')
    model_smart_dense.model.load_weights('model_smart_dense_1e-1/best_model.h5')
    model_uniform_dense.model.load_weights('model_uniform_dense_1e-1/best_model.h5')

    ### Test the models.
    x_test = np.linspace(0, 2, 10000)
    y_test = exp_func(x_test)

    pred_smart = model_smart.model(x_test)
    pred_uniform = model_uniform.model(x_test)
    pred_smart_dense = model_smart_dense.model(x_test)
    pred_uniform_dense = model_uniform_dense.model(x_test)

    diff_y_smart = pointwiseDiff(pred_smart[0], y_test)
    diff_dy_smart = pointwiseDiff(pred_smart[1], y_test)
    diff_y_uniform = pointwiseDiff(pred_uniform[0], y_test)
    diff_dy_uniform = pointwiseDiff(pred_uniform[1], y_test)
    diff_y_smart_dense = pointwiseDiff(pred_smart_dense[0], y_test)
    diff_dy_smart_dense = pointwiseDiff(pred_smart_dense[1], y_test)
    diff_y_uniform_dense = pointwiseDiff(pred_uniform_dense[0], y_test)
    diff_dy_uniform_dense = pointwiseDiff(pred_uniform_dense[1], y_test)

    utils.plot_1d(x_test, [pred_smart[0], pred_uniform[0], pred_smart_dense[0], pred_uniform_dense[0], y_test],
                  ['y_smart', 'y_uniform', 'y_smart_dense', 'y_uniform_dense', 'y_test'], 'exp(x)_over_x', log=False,
                  folder_name="figures")
    utils.plot_1d(x_test, [pred_smart[1], pred_uniform[1], pred_smart_dense[1], pred_uniform_dense[1], y_test],
                  ['dy_smart', 'dy_uniform', 'dy_smart_dense', 'dy_uniform_dense', 'dy_test'], 'd_exp(x)_over_x',
                  log=False, folder_name="figures")
    utils.plot_1d(x_test, [diff_y_smart, diff_y_uniform, diff_y_smart_dense, diff_y_uniform_dense],
                  ['y_smart', 'y_uniform', 'y_smart_dense', 'y_uniform_dense'], 'errors_in_y', log=True,
                  folder_name="figures")
    utils.plot_1d(x_test, [diff_dy_smart, diff_dy_uniform, diff_dy_smart_dense, diff_dy_uniform_dense],
                  ['dy_smart', 'dy_uniform', 'dy_smart_dense', 'dy_uniform_dense'], 'errors_in_dy', log=True,
                  folder_name="figures")

    utils.plot_1d(x_test, [diff_y_smart, diff_y_smart_dense],
                  ['y_smart', 'y_smart_dense'], 'errors_in_y_smart', log=True,
                  folder_name="figures")
    utils.plot_1d(x_test, [diff_y_uniform, diff_y_uniform_dense],
                  ['y_uniform', 'y_uniform_dense'], 'errors_in_y_uniform', log=True,
                  folder_name="figures")
    utils.plot_1d(x_test, [diff_dy_smart, diff_dy_smart_dense],
                  ['dy_smart', 'dy_smart_dense'], 'errors_in_dy_smart', log=True,
                  folder_name="figures")
    utils.plot_1d(x_test, [diff_dy_uniform, diff_dy_uniform_dense],
                  ['dy_uniform', 'dy_uniform_dense'], 'errors_in_dy_uniform', log=True,
                  folder_name="figures")

    return 0


def pointwiseDiff(trueSamples, predSamples):
    """
    brief: computes the squared 2-norm for each sample point
    input: trueSamples, dim = (ns,N)
           predSamples, dim = (ns,N)
    returns: mse(trueSamples-predSamples) dim = (ns,)
    """
    err = []
    for i in range(trueSamples.shape[0]):
        err.append(np.abs(trueSamples[i] - predSamples[i]))
    loss_val = np.asarray(err)
    return loss_val


if __name__ == '__main__':
    main()
