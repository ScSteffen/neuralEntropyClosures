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

    def __init__(self, polyDegree=0, spatialDim=1, folderName="testFolder", lossCombi=0, width=10, depth=5,
                 normalized=False):
        if (folderName == "testFolder"):
            customFolderName = "MK1_N" + str(polyDegree) + "_D" + str(spatialDim)
        else:
            customFolderName = folderName

        super(neuralMK10, self).__init__(normalized, polyDegree, spatialDim, width, depth, lossCombi,
                                         customFolderName)

        self.model = self.createModel()

    def createModel(self):

        # build model

        model = customNet(self.inputDim, self.modelWidth, self.modelDepth, name="ICNN_Derivative_Net")

        # implicitly build the model
        batchSize = 3
        x_build = np.zeros((batchSize, self.inputDim), dtype=float)
        test_point = tf.constant(x_build, dtype=tf.float32)
        test_output = model.predict(test_point)

        # create the loss functions
        def mse_loss(h_true, h_pred):
            loss_val = tf.keras.losses.mean_squared_error(h_true, h_pred)
            return loss_val

        def alter_mse_loss(alpha_true, alpha_pred):
            loss_val = float(10) * tf.keras.losses.MeanSquaredError()(alpha_true, alpha_pred)
            # loss_val = 0
            return loss_val

        x_build = [[1], [2], [3]] * np.ones((batchSize, 1), dtype=float)
        test_point2 = tf.constant(x_build, dtype=tf.float32)

        x_build = np.zeros((batchSize, 1), dtype=float)
        test_point1 = tf.constant(x_build, dtype=tf.float32)

        print(alter_mse_loss(test_point, test_point2))
        print(alter_mse_loss(test_point2, test_point2))

        print(mse_loss(test_point, test_point2))
        print(mse_loss(test_point2, test_point2))

        model.compile(
            loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError()},
            loss_weights={'output_1': 1, 'output_2': 1},
            optimizer='adam',
            metrics=['mean_absolute_error'])

        model.summary()

        tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True,
                                  show_layer_names=True, rankdir='TB', expand_nested=True)
        return model

    def call_training(self, val_split=0.1, epoch_size=2, batch_size=128, verbosity_mode=1, callback_list=[]):
        '''
        Calls training depending on the MK model
        '''
        xData = self.trainingData[0]
        yData = [self.trainingData[2], self.trainingData[1]]
        self.model.fit(x=xData, y=yData,
                       validation_split=val_split, epochs=epoch_size,
                       batch_size=batch_size, verbose=verbosity_mode,
                       callbacks=callback_list, shuffle=True)
        return self.history

    def selectTrainingData(self):
        return [True, True, True]

    def trainingDataPostprocessing(self):
        return 0


class customNet(tf.keras.Model):

    def __init__(self, inputDim, modelWidth, modelDepth, **opts):
        # tf.keras.backend.set_floatx('float64')  # Full precision training
        super(customNet, self).__init__()

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
                                        bias_initializer='zeros',
                                        name='input_layer')
        self.ic_layers = list()

        for i in range(modelDepth):
            self.ic_layers.append(ICNNBlock(self.modelWidth, False, blocknumber=i))

        self.output_layer = ICNNBlock(1, True, blockName="icnn_output",
                                      blocknumber=0)  # outputsize 1, since h is scalar

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

        # x = layers.Lambda(self.identity_func, name="input")(x)

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.input_layer(x)
            for ic_layer in self.ic_layers:
                y = ic_layer(y, x)

            h = self.output_layer(y, x)

        d_net = grad_tape.gradient(h, x)

        alpha = layers.Lambda(self.identity_func, name="d_net")(d_net)

        # u = self.reconstructU(alpha)
        # flux = self.reconstructFlux(alpha)
        # net_out = tf.stack([h, alpha], axis=1)[:, :, 0]
        return [h, alpha]


class ICNNBlock(tf.keras.Model):
    def __init__(self, modelWidth, outputLayer=False, blockName="icnnBlock", blocknumber=0):
        super(ICNNBlock, self).__init__(name=blockName + str(blocknumber))

        self.outputLayer = outputLayer
        self.modelWidth = modelWidth
        # Weight initializer for sofplus  after K Kumar
        hidden_stddev = np.sqrt((1 / self.modelWidth) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        self.hiddenInitializer = initializers.RandomNormal(mean=0., stddev=hidden_stddev)

        # Create Layers
        self.Nonneg_layer = layers.Dense(self.modelWidth, kernel_constraint=NonNeg(), activation=None,
                                         kernel_initializer=self.hiddenInitializer,
                                         use_bias=True, bias_initializer='zeros',
                                         name="non_negative_layer_" + str(blocknumber))

        self.dense_layer = layers.Dense(self.modelWidth, activation=None,
                                        kernel_initializer=self.hiddenInitializer,
                                        use_bias=False,
                                        name="arb_negative_layer_" + str(blocknumber))

        self.add_layer = layers.Add(name="add_connection_" + str(blocknumber))
        self.bn_layer = layers.BatchNormalization(name="bn_" + str(blocknumber))

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
