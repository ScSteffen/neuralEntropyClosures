"""
Script to call synthetic tests
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import csv
import src.utils
from src.utils import load_data
from optparse import OptionParser
from src.networks.configmodel import init_neural_closure

from src.math import EntropyTools
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def synth_tests2():
    print("---------- Start Synthetic test Suite ------------")
    print("Check combined regularization and nn approximation error")
    n_data = 200000
    imported_model = tf.keras.models.load_model("models/mk12_M3_2D_g2/best_model")
    [u, alpha, h] = load_data(filename="data/2D/Monomial_M3_2D_normal_gaussian.csv",
                              data_dim=10, selected_cols=[True, True, True])
    [u, alpha, h] = [u[:n_data, :], alpha[:n_data, :], h[:n_data, :]]  # mem overflow :(
    u_reduced = u[:, 1:]
    # evaluate network
    [h_pred, alpha_pred, u_pred] = imported_model(u_reduced)
    # compute regularized u
    et = EntropyTools(polynomial_degree=3, spatial_dimension=2, gamma=0.0)
    alpha_pred_full = et.reconstruct_alpha(alpha=tf.cast(alpha_pred, tf.float64))
    u_gamma_pred = et.reconstruct_u(alpha=alpha_pred_full)

    # compute errors
    m = tf.keras.metrics.MeanSquaredError()
    m.reset_state()
    m.update_state(h, h_pred)
    e_h = m.result().numpy()

    m.reset_state()
    m.update_state(alpha[:, 1:], alpha_pred)
    e_alpha = m.result().numpy()

    m.reset_state()
    m.update_state(u_reduced, u_pred)
    e_u = m.result().numpy()

    m.reset_state()
    m.update_state(u[:, 1:], u_gamma_pred[:, 1:])
    e_u_gamma = m.result().numpy()

    print("e_h:       " + str(e_h))
    print("e_alpha:   " +str(e_alpha))
    print("e_u:       " + str(e_u))
    print("e_u_gamma: " +str(e_u_gamma))

    return 0


def synth_tests():
    print("---------- Start Synthetic test Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-l", "--legacy", dest="legacy", default=1,
                      help="legacy mode for tf2.2 models", metavar="LEGACY")

    (options, args) = parser.parse_args()
    options.legacy = bool(int(options.legacy))

    # --- M1 1D synthetic tests  ----
    if options.legacy:
        # load network
        neural_closure = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1,
                                             folder_name="tmp",
                                             loss_combination=2,
                                             nw_width=10, nw_depth=7, normalized=True)
        neural_closure.create_model()
        ### Need to load this model as legacy code
        print("Load model in legacy mode. Model was created using tf 2.2.0")
        legacy_model = True
        imported = tf.keras.models.load_model("models/_simulation/mk11_M1_1D/best_model")
        neural_closure.model_legacy = imported
        test_model = neural_closure.model_legacy
    else:
        neural_closure = init_neural_closure(network_mk=15, poly_degree=1, spatial_dim=1,
                                             folder_name="_simulation/mk15_M1_1D",
                                             loss_combination=2, nw_width=30, nw_depth=2,
                                             normalized=True, input_decorrelation=True,
                                             scale_active=True)
        neural_closure.load_model()
        test_model = neural_closure.model

    # perform tests on normalized moments
    # load data
    [u_t, alpha_t, h_t] = src.utils.load_data(filename="data/test_data/Monomial_M1_1D_normal.csv", data_dim=2)
    u_tnsr = tf.constant(u_t[:, 1], shape=(u_t.shape[0], 1))

    if options.legacy:
        [h_pred, alpha_pred] = test_model(u_tnsr)
        alpha64 = tf.cast(alpha_pred, dtype=tf.float64, name=None)
        alpha_complete = neural_closure.model.reconstruct_alpha(alpha64)
        u_complete = neural_closure.model.reconstruct_u(alpha_complete)
        u_pred_np = u_complete.numpy()
        alpha_pred_np = alpha_complete.numpy()
        h_pred_np = h_pred.numpy()
    else:
        [alpha_pred, mono_loss, u_pred, h_pred] = neural_closure.model(u_tnsr)
        [u_rescaled, alpha_rescaled, h] = neural_closure.call_scaled_64(u_t)
        u_pred_np = u_rescaled.numpy()
        alpha_pred_np = alpha_rescaled.numpy()
        h_pred_np = h.numpy()

    # compute relative errors
    err_u = np.linalg.norm(u_t - u_pred_np, axis=1).reshape((u_t.shape[0], 1))
    rel_err_u = err_u / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))
    err_alpha = np.linalg.norm(alpha_t - alpha_pred_np, axis=1).reshape((u_t.shape[0], 1))
    rel_err_alpha = err_alpha / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))
    err_h = np.linalg.norm(h_t - h_pred_np, axis=1).reshape((u_t.shape[0], 1))
    rel_err_h = err_h / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))

    # print to file
    if options.legacy:
        filename = 'figures/synthetics/1D_M1_MK11_synthetic.csv'
    else:
        filename = 'figures/synthetics/1D_M1_MK15_synthetic.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["u_1", "err_u", "rel_err_u", "err_alpha", "rel_err_alpha", "err_h", "rel_err_h"])
        for i in range(u_t.shape[0]):
            writer.writerow(
                [u_t[i, 1], err_u[i, 0], rel_err_u[i, 0], err_alpha[i, 0], rel_err_alpha[i, 0], err_h[i, 0],
                 rel_err_h[i, 0]])

    # --- Synthetic test for  alpha vs u sampling ---

    [u_t, alpha_t, h_t] = src.utils.load_data(filename="data/test_data/Monomial_M1_1D_normal.csv", data_dim=2)
    u_tnsr = tf.constant(u_t[:, 1], shape=(u_t.shape[0], 1))

    # load network
    neural_closure = init_neural_closure(network_mk=11, poly_degree=1, spatial_dim=1,
                                         folder_name="tmp",
                                         loss_combination=2,
                                         nw_width=10, nw_depth=7, normalized=True)
    neural_closure.create_model()
    ### Need to load this model as legacy code
    print("Load model in legacy mode. Model was created using tf 2.2.0")
    imported = tf.keras.models.load_model("models/_simulation/mk11_M1_1D_normal/best_model")
    neural_closure.model_legacy = imported
    test_model_normal = neural_closure.model_legacy

    [h_pred, alpha_pred, u] = test_model_normal(u_tnsr)
    alpha64 = tf.cast(alpha_pred, dtype=tf.float64, name=None)
    alpha_complete = neural_closure.model.reconstruct_alpha(alpha64)
    u_complete = neural_closure.model.reconstruct_u(alpha_complete)
    u_pred_np_normal = u_complete.numpy()
    alpha_pred_np_normal = alpha_complete.numpy()
    h_pred_np_normal = h_pred.numpy()

    err_u = np.linalg.norm(u_t - u_pred_np_normal, axis=1).reshape((u_t.shape[0], 1))
    rel_err_u_normal = err_u / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))
    err_alpha = np.linalg.norm(alpha_t - alpha_pred_np_normal, axis=1).reshape((u_t.shape[0], 1))
    rel_err_alpha_normal = err_alpha / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))
    err_h = np.linalg.norm(h_t - h_pred_np_normal, axis=1).reshape((u_t.shape[0], 1))
    rel_err_h_normal = err_h / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))

    ### Need to load this model as legacy code
    imported = tf.keras.models.load_model("models/_simulation/mk11_M1_1D_alpha/best_model")
    neural_closure.model_legacy = imported
    test_model_alpha = neural_closure.model_legacy

    [h_pred, alpha_pred, u] = test_model_alpha(u_tnsr)
    alpha64 = tf.cast(alpha_pred, dtype=tf.float64, name=None)
    alpha_complete = neural_closure.model.reconstruct_alpha(alpha64)
    u_complete = neural_closure.model.reconstruct_u(alpha_complete)
    u_pred_np_alpha = u_complete.numpy()
    alpha_pred_np_alpha = alpha_complete.numpy()
    h_pred_np_alpha = h_pred.numpy()

    err_u = np.linalg.norm(u_t - u_pred_np_alpha, axis=1).reshape((u_t.shape[0], 1))
    rel_err_u_alpha = err_u / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))
    err_alpha = np.linalg.norm(alpha_t - alpha_pred_np_alpha, axis=1).reshape((u_t.shape[0], 1))
    rel_err_alpha_alpha = err_alpha / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))
    err_h = np.linalg.norm(h_t - h_pred_np_alpha, axis=1).reshape((u_t.shape[0], 1))
    rel_err_h_alpha = err_h / np.linalg.norm(u_t, axis=1).reshape((u_t.shape[0], 1))

    filename = 'figures/synthetics/1D_M1_normal_vs_alpha_synthetic.csv'
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["u_1", "rel_err_u_normal", "rel_err_alpha_normal", "rel_err_h_normal", "rel_err_u_alpha",
                         "rel_err_alpha_alpha", "rel_err_h_alpha"])
        for i in range(u_t.shape[0]):
            writer.writerow([u_t[i, 1], rel_err_u_normal[i, 0], rel_err_alpha_normal[i, 0], rel_err_h_normal[i, 0],
                             rel_err_u_alpha[i, 0], rel_err_alpha_alpha[i, 0], rel_err_h_alpha[i, 0]])
    return True


if __name__ == '__main__':
    synth_tests2()
    # synth_tests()
