'''
Derived network class "MK4" for the neural entropy closure.
It features the ICNN approach by Amos et al.
Author: Steffen Schotthöfer
Version: 0.0
Date 17.12.2020
'''
from .neuralBase import neuralBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import Tensor
from tensorflow.keras.constraints import NonNeg


class neuralMK7(neuralBase):
    '''
    MK4 Model: Train u to alpha
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, polyDegree=0, spatialDim=0, folderName="testFolder", optimizer='adam', width=10, height=5):
        if (folderName == "testFolder"):
            tempString = "MK7_N" + str(polyDegree) + "_D" + str(spatialDim)
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
        self.model = self.createModel()
        self.filename = "models/" + tempString

    def createModel(self):

        layerDim = self.modelWidth

        # Weight initializer
        initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
        initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

        def convexLayer(layerInput_z: Tensor, netInput_x: Tensor) -> Tensor:
            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = layers.Dense(layerDim, kernel_constraint=NonNeg(), activation=None,
                                               kernel_initializer=initializerNonNeg,
                                               use_bias=True,
                                               bias_initializer='zeros'
                                               # name='in_z_NN_Dense'
                                               )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = layers.Dense(layerDim, activation=None,
                                         kernel_initializer=initializer,
                                         use_bias=False
                                         # name='in_x_Dense'
                                         )(netInput_x)
            # Wz+Wx+b
            intermediateSum = layers.Add()([weightedSum_x, weightedNonNegSum_z])

            # activation
            out = tf.keras.activations.softplus(intermediateSum)
            # batch normalization
            # out = layers.BatchNormalization()(out)
            return out

        def convexLayerOutput(layerInput_z: Tensor, netInput_x: Tensor) -> Tensor:
            # Weighted sum of previous layers output plus bias
            weightedNonNegSum_z = layers.Dense(1, kernel_constraint=NonNeg(), activation=None,
                                               kernel_initializer=initializerNonNeg,
                                               use_bias=True,
                                               bias_initializer='zeros'
                                               # name='in_z_NN_Dense'
                                               )(layerInput_z)
            # Weighted sum of network input
            weightedSum_x = layers.Dense(1, activation=None,
                                         kernel_initializer=initializer,
                                         use_bias=False
                                         # name='in_x_Dense'
                                         )(netInput_x)
            # Wz+Wx+b
            intermediateSum = layers.Add()([weightedSum_x, weightedNonNegSum_z])

            # activation
            # out = tf.keras.activations.softplus(intermediateSum)
            # batch normalization
            # out = layers.BatchNormalization()(out)
            return intermediateSum

        # Number of basis functions used:
        input_ = keras.Input(shape=(self.inputDim,))

        ### Hidden layers ###
        # First Layer is a std dense layer
        hidden = layers.Dense(layerDim, activation="softplus",
                              kernel_initializer=initializer,
                              bias_initializer='zeros'
                              )(input_)
        # other layers are convexLayers
        for idx in range(0, self.modelHeight):
            hidden = convexLayer(hidden, input_)

        output_ = convexLayerOutput(hidden, input_)  # outputlayer

        # Create the model
        model = keras.Model(inputs=[input_], outputs=[output_], name="ICNN")
        # model.summary()

        # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
        model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])

        return model

    def selectTrainingData(self):
        return [True, False, True]

    def trainingDataPostprocessing(self):
        return 0


from tensorflow.keras import initializers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import BatchNormalization
import numpy as np


class Mk10Net(tf.keras.model):

    def __init__(self, N, nNode, nLayer, Q, **opts):

        super(Mk10Net, self).__init__()

        # Specify architecture and input shape

        self.N = N
        self.nNode = nNode
        self.nLayer = nLayer
        self.p = tf.constant(Q.p, dtype=float)
        self.w = tf.constant(Q.w, dtype=float)

        self.m0 = self.p[0, :]
        self.m1 = self.p[1, :]
        self.m2 = self.p[2, :]

        # Define variance for initialization

        # 1. This is a modified Kaiming inititalization with a first-order taylor expansion of the
        # softplus activation function (see S. Kumar "On Weight Initialization in
        # Deep Neural Networks").

        self.input_stddev = np.sqrt((1 / N) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))
        self.hidden_stddev = np.sqrt((1 / nNode) * (1 / ((1 / 2) ** 2)) * (1 / (1 + np.log(2) ** 2)))

        """

        #2. This is the He or Kaiming initialization 
        self.input_stddev = np.sqrt((2/N))
        self.hidden_stddev = np.sqrt((2/nNode))

        #Define the input layer, hidden layers, and output layer

        """

        # Standard variance for initialization: the "he" or "kaiming" init
        self.input_layer = Dense(nNode, use_bias=True, kernel_initializer= \
            initializers.RandomNormal(mean=0., stddev=self.input_stddev), \
                                 bias_initializer=initializers.Zeros())

        self.hidden_layers = dict()

        self.bn_layers = dict()

        self.bn_layers['bn_input'] = BatchNormalization(axis=0)

        for i in range(self.nLayer):
            self.hidden_layers['hidden_' + str(i)] = Dense(nNode, use_bias=True, kernel_initializer= \
                initializers.RandomNormal(mean=0., stddev=self.hidden_stddev), \
                                                           bias_initializer=initializers.Zeros(), )

            self.bn_layers['bn_' + str(i)] = BatchNormalization(axis=0)

        self.output_layer = Dense(1, use_bias=True, kernel_initializer= \
            initializers.RandomNormal(mean=0., stddev=self.hidden_stddev), \
                                  bias_initializer=initializers.Zeros(), name='function')

        self.activation = Activation('softplus')

        # batch normalization needs op conversion for function gradient to be accessible- not used here

    def identity_func(self, tensor):

        return tensor

    def alpha0surface(self, alpha_N):

        checked_alpha_N = tf.debugging.check_numerics(alpha_N, \
                                                      message='input tensor checking error', name='checked')
        clipped_alpha_N = tf.clip_by_value(checked_alpha_N, \
                                           clip_value_min=-50, clip_value_max=50, name='checkedandclipped')

        Ga1a2 = tf.math.exp(tf.tensordot(clipped_alpha_N, self.p[1:, :], axes=1))

        integral_Ga1a2 = tf.tensordot(Ga1a2, self.w, axes=1)

        alpha0_pred = - tf.math.log(integral_Ga1a2)

        return alpha0_pred

        # Define moment reconstruction function

    def moment_func(self, alpha, tol=1e-8):

        # Check the predicted alphas for +/- infinity or nan - raise error if found
        checked_alpha = tf.debugging.check_numerics(alpha, \
                                                    message='input tensor checking error', name='checked')

        # Clip the predicted alphas below the tf.exp overflow threshold
        clipped_alpha = tf.clip_by_value(checked_alpha, \
                                         clip_value_min=-50, clip_value_max=50, name='checkedandclipped')

        # Calculate the closed density function at each point along velocity domain
        G_alpha = tf.math.exp(tf.tensordot(clipped_alpha[:, :], self.p[:, :], axes=1))

        # Pointwise-multiply moment vector by closed denity along velocity axis
        m0G_alpha = tf.multiply(G_alpha, self.m0)
        m1G_alpha = tf.multiply(G_alpha, self.m1)
        m2G_alpha = tf.multiply(G_alpha, self.m2)

        # Compute integral by quadrature (dot-product with weights along velocity axis)
        u0 = tf.tensordot(m0G_alpha, self.w, axes=1)
        u1 = tf.tensordot(m1G_alpha, self.w, axes=1)
        u2 = tf.tensordot(m2G_alpha, self.w, axes=1)

        # Stack-moments together
        moment_pred = tf.stack([u0, u1, u2], axis=1)

        return moment_pred

    def call(self, x, training=False):
        """
        Defines network function. Can be adapted to have different paths
        for training and non-training modes (not currently used).

        At each layer, applies, in order: (1) weights & biases, (2) batch normalization
        (current: commented out), then (3) activation.

        Inputs:

            (x,training = False,mask = False)

        Returns:

            returns [h(x),alpha(x),u(x),hess(h)(x)]
        """

        x = Lambda(self.identity_func, name="input")(x)

        with tf.GradientTape(persistent=True) as hess_tape:
            hess_tape.watch(x)

            with tf.GradientTape() as grad_tape:
                grad_tape.watch(x)

                y = self.input_layer(x)

                y = self.activation(y)

                # y = self.bn_layers['bn_input'](y)

                for i in range(self.nLayer):
                    y = self.hidden_layers['hidden_' + str(i)](y)

                    y = self.activation(y)

                    # y = self.bn_layers['bn_'+str(i)](y)

                net = self.output_layer(y)

            d_net = grad_tape.gradient(net, x)

            d_net = Lambda(self.identity_func, name="d_net")(d_net)

        hess = hess_tape.batch_jacobian(d_net, x)

        dets = \
            tf.math.multiply(hess[:, 0, 0], hess[:, 1, 1]) - tf.math.multiply(hess[:, 1, 0], hess[:, 0, 1])

        hess_11 = hess[:, 0, 0]

        detpa = tf.stack([dets, hess_11], axis=1)

        alpha_N = d_net

        # Explicit quadrature equality for alpha_0; these are exact (up to quadrature) alpha_0 values for predicted alpha_N
        # alpha_0 = tf.expand_dims(self.alpha0surface(alpha_N),axis = 1)

        # Contraint equation for alpha_0
        alpha_0 = tf.constant(1, dtype=tf.float32) + net - tf.expand_dims(
            tf.reduce_sum(tf.multiply(x, d_net), axis=1), axis=1)

        alpha_out = tf.concat([alpha_0, alpha_N], axis=1)

        return [net, alpha_out, self.moment_func(alpha_out), detpa]
