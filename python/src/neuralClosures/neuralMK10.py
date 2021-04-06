'''
Derived network class "MK4" for the neural entropy closure.
It features the ICNN approach by Amos et al.
Author: Steffen Schotthöfer
Version: 0.0
Date 17.12.2020
'''
from .neuralBase import neuralBase
from .neuralBase import LossAndErrorPrintingCallback

import src.math as math

import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras import initializers


class neuralMK10(neuralBase):
    '''
    MK4 Model: Train u to h and alpha
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, polyDegree=0, spatialDim=0, folderName="testFolder", optimizer='adam', width=10, height=5):
        if (folderName == "testFolder"):
            tempString = "MK10_N" + str(polyDegree) + "_D" + str(spatialDim)
        else:
            tempString = folderName

        self.polyDegree = polyDegree
        self.spatialDim = spatialDim
        self.modelWidth = width
        self.modelHeight = height

        # --- Determine inputDim by MaxDegree ---
        if (spatialDim == 1):
            self.inputDim = polyDegree + 1
        elif (spatialDim == 3):
            if (self.polyDegree == 0):
                self.inputDim = 1
            elif (self.polyDegree == 1):
                self.inputDim = 4
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        elif (spatialDim == 2):
            if (self.polyDegree == 0):
                self.inputDim = 1
            elif (self.polyDegree == 1):
                self.inputDim = 3
            else:
                raise ValueError("Polynomial degeree higher than 1 not supported atm")
        else:
            raise ValueError("Saptial dimension other than 1,2 or 3 not supported atm")

        self.opt = optimizer
        self.filename = "models/" + tempString

        self.model = self.createModel()

    def createModel(self):

        # build model

        model = DerivativeNet(self.inputDim, self.modelWidth, self.modelHeight, name="ICNN_Derivative_Net")

        # implicitly build the model
        test_point = np.array([[0.5, 0.7], [-0.5, -0.7], [0.01, -0.01], [0.9, -0.9]], dtype=float)
        test_point = tf.constant(test_point, dtype=tf.float32)
        test_output = model.predict(test_point)

        model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])

        # model.summary()

        tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True)

        return model

    def trainModel(self, valSplit=0.1, epochCount=2, epochChunks=1, batchSize=500, verbosity=1, processingMode=0):
        '''
        Method to train network
        '''

        # Set full precision training for CPU training
        if processingMode == 0:
            tf.keras.backend.set_floatx('float32')

        # Create callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.filename + '/best_model.h5', monitor='loss', mode='min',
                                                     save_best_only=True,
                                                     verbose=verbosity)  # , save_weights_only = True, save_freq = 50, verbose=0)
        es = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', min_delta=0.0001, patience=10,
                                              verbose=1)
        # mc_checkpoint =  tf.keras.callbacks.ModelCheckpoint(filepath=self.filename + '/model_saved',
        #                                         save_weights_only=False,
        #                                         verbose=1)

        # Split Training epochs
        miniEpoch = int(epochCount / epochChunks)

        for i in range(0, epochChunks):
            #  perform a batch doublication every 1/10th of the epoch count
            print("Current Batch Size: " + str(batchSize))

            # assemble callbacks
            callbackList = []
            csv_logger = self.createCSVLoggerCallback()
            if verbosity == 1:
                callbackList = [mc_best, csv_logger]
            else:
                callbackList = [mc_best, LossAndErrorPrintingCallback(), csv_logger]

            # start Training
            self.history = self.model.fit(x=self.trainingData[0], y=[self.trainingData[2], self.trainingData[1]],
                                          validation_split=valSplit,
                                          epochs=miniEpoch,
                                          batch_size=batchSize,
                                          verbose=verbosity,
                                          callbacks=callbackList,
                                          )
            batchSize = 2 * batchSize

        self.concatHistoryFiles()

        return self.history

    def selectTrainingData(self):
        return [True, True, True]

    def trainingDataPostprocessing(self):
        return 0


class DerivativeNet(tf.keras.Model):

    def __init__(self, inputDim, modelWidth, modelDepth, **opts):
        super(DerivativeNet, self).__init__()

        # Specify integration weights and basis
        ### Compare u and reconstructed u
        [mu, quadWeights] = math.qGaussLegendre1D(100)  # Create quadrature
        mBasis = math.computeMonomialBasis1D(mu, 1)  # Create basis

        # Specify architecture and input shape
        self.mBasis = tf.constant(mBasis, dtype=float)
        self.quadWeights = tf.constant(quadWeights, dtype=float)

        self.inputDim = inputDim
        self.modelWidth = modelWidth
        self.modelDepth = modelDepth

        # Weight initializer for sofplus  after K Kumar
        input_stddev = np.sqrt((1 / inputDim) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        hidden_stddev = np.sqrt((1 / self.modelWidth) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))

        self.hiddenInitializer = initializers.RandomNormal(mean=0., stddev=hidden_stddev)
        self.inputLayerInitializer = initializers.RandomNormal(mean=0., stddev=input_stddev)

        # build the network
        self.input_layer = layers.Dense(self.inputDim, activation="softplus",
                                        kernel_initializer=self.inputLayerInitializer,
                                        use_bias=True,
                                        bias_initializer='zeros')
        self.ic_layers = list()

        for i in range(modelDepth):
            self.ic_layers.append(ICNNBlock(self.modelWidth, False))

        self.output_layer = ICNNBlock(self.modelWidth, True)

    def identity_func(self, tensor):
        return tensor

    def reconstructU(self, alpha, tol=1e-8):
        """
            imput: alpha, dims = (nS x N)
                   m    , dims = (N x nq)
                   w    , dims = nq
            returns u = <m*eta_*'(alpha*m)>, dim = (nS x N)
        """

        # Check the predicted alphas for +/- infinity or nan - raise error if found
        checked_alpha = tf.debugging.check_numerics(alpha, message='input tensor checking error', name='checked')

        # Clip the predicted alphas below the tf.exp overflow threshold
        clipped_alpha = tf.clip_by_value(checked_alpha, clip_value_min=-50, clip_value_max=50, name='checkedandclipped')

        # Calculate the closed density function at each point along velocity domain
        G_alpha = tf.math.exp(tf.tensordot(clipped_alpha[:, :], self.mBasis[:, :], axes=1))

        # Pointwise-multiply moment vector by closed density along velocity axis
        # m0G_alpha = tf.multiply(G_alpha, self.m0)
        # m1G_alpha = tf.multiply(G_alpha, self.m1)
        # m2G_alpha = tf.multiply(G_alpha, self.m2)

        # Compute integral by quadrature (dot-product with weights along velocity axis)
        # u0 = tf.tensordot(m0G_alpha, self.w, axes=1)
        # u1 = tf.tensordot(m1G_alpha, self.w, axes=1)
        # u2 = tf.tensordot(m2G_alpha, self.w, axes=1)

        # Stack-moments together
        # moment_pred = tf.stack([u0, u1, u2], axis=1)

        return alpha

    def reconstructFlux(self, alpha, tol=1e-8):
        return alpha

    def call(self, x, training=False):
        """
        Defines network function. Can be adapted to have different paths
        for training and non-training modes (not currently used).

        At each layer, applies, in order: (1) weights & biases, (2) batch normalization
        (current: commented out), then (3) activation.

        Inputs:
            (x,training = False,mask = False)
        Returns:
            returns [h(x),alpha(x),u(x)]
        """

        x = layers.Lambda(self.identity_func, name="input")(x)

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.input_layer(x)
            for ic_layer in self.ic_layers:
                y = ic_layer(y, x)
            h = self.output_layer(y, x)

        d_net = grad_tape.gradient(h, x)

        d_net = layers.Lambda(self.identity_func, name="d_net")(d_net)

        alpha = d_net

        # u = self.reconstructU(alpha)
        # flux = self.reconstructFlux(alpha)

        return [h, alpha]


class ICNNBlock(tf.keras.Model):
    def __init__(self, modelWidth, outputLayer=False):
        super(ICNNBlock, self).__init__(name='')

        self.outputLayer = outputLayer
        self.modelWidth = modelWidth
        # Weight initializer for sofplus  after K Kumar
        hidden_stddev = np.sqrt((1 / self.modelWidth) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        self.hiddenInitializer = initializers.RandomNormal(mean=0., stddev=hidden_stddev)

        # Create Layers
        self.Nonneg_layer = layers.Dense(self.modelWidth, kernel_constraint=NonNeg(), activation=None,
                                         kernel_initializer=self.hiddenInitializer,
                                         use_bias=True, bias_initializer='zeros')

        self.dense_layer = layers.Dense(self.modelWidth, activation=None,
                                        kernel_initializer=self.hiddenInitializer,
                                        use_bias=False)

        self.add_layer = layers.Add()
        self.bn_layer = layers.BatchNormalization()

    def call(self, layer_input, model_input, training=False):
        z_nonneg = self.Nonneg_layer(layer_input)
        x = self.dense_layer(model_input)
        intermediateSum = self.add_layer([x, z_nonneg])

        intermediateSum2 = self.bn_layer(intermediateSum, training=training)

        if self.outputLayer:
            out = intermediateSum2
        else:
            out = tf.keras.activations.softplus(intermediateSum2)

        return out
