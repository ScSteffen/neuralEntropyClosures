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
from src import math


def quad_func(x):
    return x * x


def quad_func_grad(x):
    return 2 * x


def exp_func(x):
    return np.exp(x)  # / np.exp(3)


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


def sample_data_entropy_M1(u, alpha, h, tol):
    """
    performs the smart sampling algorithm on the entropy closure problem
    """
    entropy_tools = math.EntropyTools(1)  # Tools for reconstruction

    e_max = tol
    j = 0
    while e_max >= tol:

        e_max = 0
        i = 0

        # [eL, tL] = get_errors(s_x, function, grad, 0.0001)
        # plt.semilogy(tL, eL, '*')
        # plt.savefig("figures/errors_" + str(j).zfill(3) + ".png")
        # plt.clf()

        while i < len(u) - 1:

            '''
            # error, in case that the approximation undershoots
            t1 = 1 / (grad(s_x[i + 1]) - grad(s_x[i])) * (function(s_x[i]) - s_x[i] * grad(s_x[i]) - (
                    function(s_x[i + 1]) - s_x[i + 1] * grad(s_x[i + 1])))
            e1 = function(t1) - (function(s_x[i]) + (t1 - s_x[i]) * grad(s_x[i]))
            '''

            # error in case that the approximation overshoots
            test = h[i + 1] - h[i]
            test2 = u[i + 1][1]
            test3 = u[i + 1][1] - u[i][1]
            alpha_o_1 = tf.reshape(((h[i + 1] - h[i]) / (u[i + 1][1] - u[i][1])), (1, 1))  # reshape
            alpha_o = entropy_tools.reconstruct_alpha(alpha_o_1)
            u_o = entropy_tools.reconstruct_u(alpha_o)
            h_o = entropy_tools.compute_h(u_o, alpha_o)
            e_o = h[i] + alpha_o_1 * (u_o[0, 1] - u[i][1]) - h_o

            # error in case that the approximation undershoots
            u_u = 1 / (alpha[i + 1][1] - alpha[i][1]) * (
                    h[i] - u[i][1] * alpha[i][1] - (h[i + 1] - u[i + 1][1] * alpha[i + 1][1]))
            # compute starting point
            alpha_1_start = tf.reshape((alpha[i + 1][1] + alpha[i][1]) / 2, shape=(1, 1))
            alpha_start = entropy_tools.reconstruct_alpha(alpha_1_start)
            u_0 = tf.constant([1.0], dtype=tf.float32)
            u_sol = tf.reshape(tf.concat([u_0, u_u], axis=0), shape=(1, 2))
            """
            alpha_t = entropy_tools.reconstruct_alpha(tf.constant([0.01], shape=(1, 1)))
            u_sol2 = entropy_tools.reconstruct_u(alpha_t)
            """
            alpha_u = entropy_tools.minimize_entropy(u_sol, alpha_start)
            # make realizable
            u_u = entropy_tools.reconstruct_u(alpha_u)
            h_u = entropy_tools.compute_h(u_u, alpha_u)
            e_u = h_u - (h[i] + (u_u[0, 1] - u[i][1]) * alpha[i][1])

            if e_u > e_o:
                u_new = tf.reshape(u_u, shape=(2,))
                alpha_new = tf.reshape(alpha_u, shape=(2,))
                h_new = tf.reshape(h_u, shape=(1,))
                e = e_u
                # print("undershoot worse")
            else:
                u_new = tf.reshape(u_o, shape=(2,))
                alpha_new = tf.reshape(alpha_o, shape=(2,))
                h_new = tf.reshape(h_o, shape=(1,))
                e = e_o
                # print("overshoot worse")

            if e > e_max:
                e_max = e
            if e > tol:
                u.insert(i + 1, u_new)
                alpha.insert(i + 1, alpha_new)
                h.insert(i + 1, h_new)
                i = i + 1
            i = i + 1
        j = j + 1

    # convert to tensorf with first dim = len(u)
    u_tensor = tf.convert_to_tensor(u)
    alpha_tensor = tf.convert_to_tensor(alpha)
    h_tensor = tf.convert_to_tensor(h)
    return [u_tensor, alpha_tensor, h_tensor]


def main():
    # x = sample_data(-10, 10, 0.0001, quad_func, quad_func_grad)

    ### Create alpha sampled values
    batchSize = 13
    N = 1
    entropy_tools = math.EntropyTools(N)

    alpha_1 = np.linspace(-50, 50, batchSize)
    alpha_1 = alpha_1.reshape((batchSize, N))
    alpha_1 = entropy_tools.convert_to_tensorf(alpha_1)
    alpha = entropy_tools.reconstruct_alpha(alpha_1)
    u = entropy_tools.reconstruct_u(alpha)
    h = entropy_tools.compute_h(u, alpha)

    # utils.plot1D([u[:, 1]], [alpha[:, 1], h], ['alpha', 'h'], 'sanity_check', log=False, folder_name="figures")

    ### Create u sampled values
    u_0 = np.ones((13, 1))
    u_1 = np.reshape(np.linspace(-0.98, 0.98, 13), (13, 1))
    u_ = np.concatenate((u_0, u_1), axis=1)
    u_in = tf.constant(u_, shape=(13, 2))
    alpha_ = []
    h_ = []
    u_ = []
    for i in range(13):
        alpha_curr = entropy_tools.minimize_entropy(tf.reshape(u_in[i], shape=(1, 2)),
                                                    tf.reshape(u_in[i], shape=(1, 2)))
        u_curr = entropy_tools.reconstruct_u(alpha_curr)
        h_curr = entropy_tools.compute_h(u_curr, alpha_curr)
        u_.append(u_curr[0])
        h_.append(h_curr[0])
        alpha_.append(alpha_curr[0])

    u_ = tf.stack(u_)
    h_ = tf.stack(h_)
    alpha_ = tf.stack(alpha_)
    # utils.plot1D([u_[:, 1]], [alpha_[:, 1], h_], ['alpha', 'h'], 'sanity_check2', log=False, folder_name="figures")

    ### Create smart sampled values

    tolerance = 0.1

    alpha_1 = entropy_tools.convert_to_tensorf(np.asarray([-50, 50]).reshape((2, N)))
    alpha_ini = entropy_tools.reconstruct_alpha(alpha_1)
    u_ini = entropy_tools.reconstruct_u(alpha_ini)
    h_ini = entropy_tools.compute_h(u_ini, alpha_ini)
    alpha_list = [alpha_ini[0, :], alpha_ini[1, :]]
    u_list = [u_ini[0, :], u_ini[1, :]]
    h_list = [h_ini[0, :], h_ini[1, :]]
    [u_train, alpha_train, h_train] = sample_data_entropy_M1(u_list, alpha_list, h_list, tolerance)

    utils.plot1D(xs=[u_train[:, 1]], ys=[alpha_train[:, 1], h_train], labels=['alpha', 'h'], linetypes=['+', '*'],
                 name='smart_sampled_entropy', log=False, folder_name="figures")

    utils.plot1D(xs=[u_train[:, 1], u[:, 1], u_[:, 1]], ys=[alpha_train[:, 1], alpha[:, 1], alpha_[:, 1]],
                 labels=['alpha_smart', 'alpha_alpha_sampled', 'alpha_u_sampled'],
                 linetypes=['*', '2', '3'],
                 name='alha_sampling_strategies', log=False, folder_name="figures")
    utils.plot1D(xs=[u_train[:, 1], u[:, 1], u_[:, 1]], ys=[h_train, h, h_],
                 labels=['h_smart', 'h_alpha_sampled', 'h_u_sampled'],
                 linetypes=['*', '2', '3'], name='h_sampling_strategies', log=False, folder_name="figures",
                 show_fig=False)

    ### Compare networks with different samplings

    model_smart = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1, folder_name="testFolder",
                                      nw_width=15, nw_depth=5, normalized=True, loss_combination=1)
    model_u = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1, folder_name="testFolder",
                                  nw_width=15, nw_depth=5, normalized=True, loss_combination=1)
    model_alpha = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1, folder_name="testFolder",
                                      nw_width=15, nw_depth=5, normalized=True, loss_combination=1)

    mc_best_smart = tf.keras.callbacks.ModelCheckpoint('model_smart/best_model.h5', monitor='loss', mode='min',
                                                       save_best_only=True, verbose=0)
    csv_logger_smart = tf.keras.callbacks.CSVLogger('model_smart/history.csv')
    mc_best_uniform = tf.keras.callbacks.ModelCheckpoint('model_u/best_model.h5', monitor='loss', mode='min',
                                                         save_best_only=True, verbose=0)
    csv_logger_uniform = tf.keras.callbacks.CSVLogger('model_u/history.csv')
    mc_best_alpha = tf.keras.callbacks.ModelCheckpoint('model_alpha/best_model.h5', monitor='loss', mode='min',
                                                       save_best_only=True, verbose=0)
    csv_logger_alpha = tf.keras.callbacks.CSVLogger('model_alpha/history.csv')

    # initialize both models with the same weight
    model_smart.model.load_weights('best_model.h5')
    model_u.model.load_weights('best_model.h5')
    model_alpha.model.load_weights('best_model.h5')

    # some params
    epochs = 200000
    batch = 16
    print("training starts")
    # model_smart.model.fit(x=u_train[:, 1], y=[h_train, alpha_train[:, 1]], validation_split=0.0, epochs=epochs,
    #                      batch_size=batch, verbose=0, callbacks=[mc_best_smart, csv_logger_smart])
    print("trained smart model")
    # model_u.model.fit(x=u[:, 1], y=[h, alpha[:, 1]], validation_split=0.0, epochs=epochs, batch_size=batch,
    #                  verbose=0, callbacks=[mc_best_uniform, csv_logger_uniform])
    print("trained u model")

    model_alpha.model.fit(x=u_[:, 1], y=[h_, alpha_[:, 1]], validation_split=0.0, epochs=epochs, batch_size=batch,
                          verbose=0, callbacks=[mc_best_alpha, csv_logger_alpha])
    print("trained alpha model")

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
