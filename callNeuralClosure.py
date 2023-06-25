'''
This is the script that gets called from the C++ KiT-RT method MLOptimizer.cpp
It initializes and loads a neural Closure
The call method performs a prediction
Author: Steffen Schotthoefer
Version: 0.0
Date 29.10.2020
'''

import os
import statistics
import time
from optparse import OptionParser

import matplotlib.pyplot as plt
import numpy as np
# python modules
import tensorflow as tf

from src import utils
### imports ###
# internal modules
from src.networks.configmodel import init_neural_closure


### function definitions ###
def init_model(network_mk: int = 1, polynomial_degree: int = 0, spatial_dim: int = 3, folder_name: str = "testFolder",
               loss_combination: int = 0, width: int = 10, depth: int = 5, normalized: bool = False,
               input_decorrelation: bool = False, scale_active: bool = True, gamma_lvl: int = 0,
               basis: str = "monomial", rotated=False):
    '''
    modelNumber : Defines the used network model, i.e. MK1, MK2...
    maxDegree_N : Defines the maximal Degree of the moment basis, i.e. the "N" of "M_N"
    '''

    global neuralClosureModel
    neuralClosureModel = init_neural_closure(network_mk=network_mk, poly_degree=polynomial_degree,
                                             spatial_dim=spatial_dim, folder_name=folder_name,
                                             loss_combination=loss_combination, nw_depth=depth,
                                             nw_width=width, normalized=normalized,
                                             input_decorrelation=input_decorrelation, scale_active=scale_active,
                                             gamma_lvl=gamma_lvl, basis=basis, rotated=rotated)

    return 0


def main():
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-a", "--sampling", dest="sampling", default=0,
                      help="uses data sampled in alpha:\n 0: uniform in u\n 1: uniform in alpha\n 2: gaussian in alpha",
                      metavar="SAMPLING")
    parser.add_option("-b", "--batch", dest="batch", default=128,
                      help="batch size", metavar="BATCH")
    parser.add_option("-c", "--curriculum", dest="curriculum", default=1,
                      help="training curriculum", metavar="CURRICULUM")
    parser.add_option("-d", "--degree", dest="degree", default=0,
                      help="max degree of moment", metavar="DEGREE")
    parser.add_option("-e", "--epoch", dest="epoch", default=1000,
                      help="epoch count for neural network", metavar="EPOCH")
    parser.add_option("-f", "--folder", dest="folder", default="testFolder",
                      help="folder where the model is stored", metavar="FOLDER")
    parser.add_option("-g", "--scaledOutput", dest="scaledOutput", default="0",
                      help="train on scaled entropy values", metavar="SCALEDOUTPUT")
    parser.add_option("-i", "--decorrInput", dest="decorrInput", default="0",
                      help="train normalized and decorrelated input moments", metavar="SCALEDINPUT")
    parser.add_option("-l", "--loadModel", dest="loadmodel", default=1,
                      help="load model weights from file", metavar="LOADING")
    parser.add_option("-m", "--model", dest="model", default=11,
                      help="choice of network model", metavar="MODEL")
    parser.add_option("-n", "--normalized", dest="normalized", default=0,
                      help="train on normalized moments", metavar="NORMALIZED")
    parser.add_option("-o", "--objective", dest="objective", default=0,
                      help="choice of loss functions:\n 0=[h]\n 1 =[h,alpha]\n 2=[h,alpha,u]\n3=[rel_entropy_h]",
                      metavar="OBJECTIVE")
    parser.add_option("-p", "--processingmode", dest="processingmode", default=1,
                      help="gpu mode (1). cpu mode (0) ", metavar="PROCESSINGMODE")
    parser.add_option("-r", "--rotated", dest="rotated", default=0)
    parser.add_option("-s", "--spatialDimension", dest="spatialDimension", default=3,
                      help="spatial dimension of closure", metavar="SPATIALDIM")
    parser.add_option("-t", "--training", dest="training", default=1,
                      help="execution mode (0) training mode (1)  analysis mode (2) re-save mode (3)",
                      metavar="TRAINING")
    parser.add_option("-v", "--verbosity", dest="verbosity", default=1,
                      help="output verbosity keras (0 or 1)", metavar="VERBOSITY")
    parser.add_option("-w", "--networkwidth", dest="networkwidth", default=10,
                      help="width of each network layer", metavar="WIDTH")
    parser.add_option("-x", "--networkdepth", dest="networkdepth", default=5,
                      help="height of the network", metavar="HEIGHT")
    parser.add_option("-y", "--gammalevel", dest="gamma_level", default=0,
                      help="gamma for regularized entropy closure:\n 0= non regularized:\n 1 = 1e-1\n 2 = 1e-2\n 3 = "
                           "1e-3", metavar="GAMMA")
    parser.add_option("-z", "--basis", dest="basis", default="monomial",
                      help="moment basis", metavar="BASIS")
    parser.add_option("--max_alpha_norm", dest="max_alpha_norm", default=20,
                      help="max_alpha_norm", metavar="ALPHANORM")

    (options, args) = parser.parse_args()
    options.objective = int(options.objective)
    options.sampling = int(options.sampling)
    options.degree = int(options.degree)
    options.spatial_dimension = int(options.spatialDimension)
    options.model = int(options.model)
    options.epoch = int(options.epoch)
    options.curriculum = int(options.curriculum)
    options.batch = int(options.batch)
    options.verbosity = int(options.verbosity)
    options.scaledOutput = bool(int(options.scaledOutput))
    options.decorrInput = bool(int(options.decorrInput))
    options.loadmodel = int(options.loadmodel)
    options.training = int(options.training)
    options.processingmode = int(options.processingmode)
    options.normalized = bool(int(options.normalized))
    options.networkwidth = int(options.networkwidth)
    options.networkdepth = int(options.networkdepth)
    options.gamma_level = int(options.gamma_level)
    options.rotated = bool(int(options.rotated))
    options.max_alpha_norm = float(options.max_alpha_norm)
    # --- End Option Parsing ---

    # witch to CPU mode, if wished
    if options.processingmode == 0:
        # Set CPU as available physical device
        # Set CPU as available physical device
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if tf.test.gpu_device_name():
            print('GPU found. Using GPU')
            physical_devices = tf.config.list_physical_devices('GPU')
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        else:
            print("Disabled GPU. Using CPU")
    # Allow TF to use only part of GPU memory and grow this part at will
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # --- initialize model framework
    print("Initialize model")
    init_model(network_mk=options.model, polynomial_degree=options.degree, spatial_dim=options.spatial_dimension,
               folder_name=options.folder, normalized=options.normalized, loss_combination=options.objective,
               width=options.networkwidth, depth=options.networkdepth, input_decorrelation=options.decorrInput,
               scale_active=options.scaledOutput, gamma_lvl=options.gamma_level, basis=options.basis,
               rotated=options.rotated)

    # --- load model data before creating model (important for data scaling)
    if options.training == 1:
        # create training Data
        # Save options and runscript to file (only for training)
        utils.write_config_file(options, neuralClosureModel)
        neuralClosureModel.load_training_data(shuffle_mode=False, sampling=options.sampling,
                                              normalized_data=neuralClosureModel.normalized, train_mode=True,
                                              gamma_level=options.gamma_level, rotated=options.rotated,
                                              max_alpha_norm=options.max_alpha_norm)
    # create model after loading training data to get correct scaling in
    if options.loadmodel == 1 or options.training == 0 or options.training == 2 or options.training == 5:
        neuralClosureModel.load_model()  # also creates model
        # preprocess training data. Compute scalings
        neuralClosureModel.training_data_preprocessing(scaled_output=options.scaledOutput,
                                                       model_loaded=options.loadmodel)
    else:
        print("Start training with new weights")
        # preprocess training data. Compute scalings
        neuralClosureModel.training_data_preprocessing(scaled_output=options.scaledOutput,
                                                       model_loaded=options.loadmodel)
        neuralClosureModel.create_model()
    # neuralClosureModel.model.summary()

    if options.training == 1:

        # train model
        neuralClosureModel.config_start_training(val_split=0.1, epoch_count=options.epoch,
                                                 curriculum=options.curriculum,
                                                 batch_size=options.batch, verbosity=options.verbosity,
                                                 processing_mode=options.processingmode)

    elif options.training == 2:
        print("Analysis mode entered.")
        print("Evaluate Model on normalized data...")
        neuralClosureModel.load_training_data(shuffle_mode=False, sampling=options.sampling,
                                              normalized_data=neuralClosureModel.normalized, train_mode=True,
                                              gamma_level=options.gamma_level, rotated=options.rotated)
        [u, alpha, h] = neuralClosureModel.get_training_data()
        neuralClosureModel.evaluate_model_normalized(u, alpha, h)

        print("Evaluated Model")
        # neuralClosureModel.load_training_data(shuffle_mode=False, load_all=True, normalized_data=False,
        #                                      train_mode=False, gamma_level=options.gamma_level)
        # [u, alpha, h] = neuralClosureModel.get_training_data()
        # neuralClosureModel.evaluate_model(u, alpha, h)
    elif options.training == 3:
        print(
            "Re-Save mode entered.")  # if training was not finished, models are not safed to .pb. this can be done here
        neuralClosureModel.load_training_data(shuffle_mode=False,
                                              sampling=options.sampling,
                                              normalized_data=neuralClosureModel.normalized,
                                              train_mode=False, gamma_level=options.gamma_level)

        # normalize data (experimental)
        # neuralClosureModel.normalizeData()
        # train model

        neuralClosureModel.model(neuralClosureModel.training_data[0])
        # save model
        neuralClosureModel.save_model()

    elif options.training == 4:
        # timing measurement
        # startup
        u_in = tf.zeros([2, neuralClosureModel.input_dim], tf.float32)
        [u, alpha, h] = neuralClosureModel.model(u_in)

        u_in = tf.ones([1000000, neuralClosureModel.input_dim], tf.float32)

        # u_tf = tf.constant(u_in)
        totduration = 0
        durations = []
        for i in range(0, 100):
            print("Start computation")
            start = time.perf_counter()
            [u, alpha, h] = neuralClosureModel.model(u_in)
            end = time.perf_counter()
            totduration += end - start
            durations.append(end - start)
            print("Model executed. Elapsed time: " +
                  str(end - start) + " in iteration " + str(i) + ".")
        avg = totduration / 100
        print("Average duration: " + str(avg) + " seconds")
        stddev = statistics.stdev(durations)
        print("Standard deviation:" + str(stddev) + "")
    elif options.training == 5:  # print weights mode
        all_layers = neuralClosureModel.model.trainable_weights
        layer_list = []
        count = 0
        for layer in all_layers:
            t = layer
            # print(t)
            tn = t.numpy().flatten()
            layer_list.append(tn)
            print(layer.shape)
            print("max weight:  " + str(np.max(tn)) +
                  " min weight: " + str(np.min(tn)))
            # hist, bin_edges = np.histogram(tn, bins=10, density=True)
            plt.hist(tn, density=True)  # arguments are passed to np.histogram
            name = layer.name
            name = name.replace(':', '')
            name = name.replace('/', '_')
            plt.title("Histogram of weights in layer " + name)
            # Text(0.5, 1.0, "Histogram with 'auto' bins")
            plt.savefig(neuralClosureModel.folder_name + "/" + name + ".png")
            # plt.show()
            plt.clf()
            # if "nn_component" in name:
            #    tn_sm = tf.nn.relu(tn)
            #    print(max(tn_sm))
            #    print(min(tn_sm))
            #    plt.hist(tn_sm, density=True)
            #    name = name + "_relu"
            #    plt.title("Histogram of weights in layer " + name)
            #    plt.savefig(neuralClosureModel.folder_name + "/" + name + ".png")
            #    plt.clf()
            count += 1

        # print non trainable weights
        all_layers_nt = neuralClosureModel.model.non_trainable_weights
        layer_list = []
        count = 0
        for layer in all_layers_nt:
            t = layer
            # print(t)
            tn = t.numpy().flatten()
            layer_list.append(tn)
            print(layer.shape)
            print("max weight:  " + str(np.max(tn)) +
                  " min weight: " + str(np.min(tn)))
            # hist, bin_edges = np.histogram(tn, bins=10, density=True)
            plt.hist(tn, density=True)  # arguments are passed to np.histogram
            name = layer.name
            name = name.replace(':', '')
            name = name.replace('/', '_')
            plt.title("Histogram of weights in layer " + name)
            # Text(0.5, 1.0, "Histogram with 'auto' bins")
            plt.savefig(neuralClosureModel.folder_name + "/" + name + ".png")
            plt.clf()
            count += 1
        if options.decorrInput:
            print(all_layers_nt[0])
            print(all_layers_nt[1])
    else:
        # --- in execution mode,  call_network or call_network_batchwise get called from c++ directly ---
        print("pure execution mode")
    print("Neural Entropy Closure Suite finished successfully.")
    return 0


if __name__ == '__main__':
    main()
