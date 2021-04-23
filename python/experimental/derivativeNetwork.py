#### MK 4 Networks ####
'''
Exploration of convex Networks on a simple example
It includes the ICNN techniques (Amos et al)
Loss is |h-h_pred| + | alpha - alpha_pred |
'''
import src.math as math
import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg

from tensorflow.keras import initializers
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt

plt.style.use("kitish")


# ------  Code starts here --------

def main():
    # Training Parameters
    batchSize = 200
    epochCount = 200

    ### Dense Network
    filename = "legacyCode/models/ConvComparison_fcnn"

    model = create_modelMK4()
    model = trainModel(model, filename, batchSize, epochCount)
    # model.load_weights(filename + '/best_model.h5')
    # model = tf.keras.models.load_model(filename + '/model')

    ### Convex Network (nonnegative weights)
    model_nonneg = create_modelMK4_nonneg()
    filename = "legacyCode/models/ConvComparison_nonNeg"
    model_nonneg = trainModel(model_nonneg, filename, batchSize, epochCount)

    # model_nonneg.load_weights(filename + '/best_model.h5')
    # model_nonneg = tf.keras.models.load_model(filename + '/model')

    ### Convex Network ICNN architecture
    model_ICNN = create_modelMK4_ICNN()
    filename = "legacyCode/models/ConvComparison_ICNN"
    model_ICNN = trainModel(model_ICNN, filename, batchSize, epochCount)
    # model_nonneg.load_weights(filename + '/best_model.h5')
    # model_ICNN = tf.keras.models.load_model(filename + '/model')

    evaluateModel(model, model_nonneg, model_ICNN)

    printWeights(model)
    print("----")
    printWeights(model_nonneg)
    return 0


def printWeights(model):
    for layer in model.layers:
        weights = layer.get_weights()  # list of numpy arrays
        print(weights)
        # if weights:
        #    plt.plot(weights)

    # plt.ylabel('weight value')
    # plt.xlabel('weight index')
    # plt.show()

    return 0


def evaluateModel(model, model2, model3):
    x = np.arange(-12, 12, 0.001)

    y = createTrainingData(x)

    predictions = model.predict(x)
    predictions2 = model2.predict(x)
    predictions3 = model3.predict(x)

    plt.plot(x, y)
    plt.plot(x, predictions)
    plt.plot(x, predictions2)
    plt.plot(x, predictions3)

    plt.ylabel('function value')
    plt.xlabel('input value')
    # plt.ylim([30.9,31])
    plt.legend(['quadratic function', 'FCNN', 'naive convex', 'ICNN'])
    plt.show()

    return 0


def trainModel(model, filename, batchSize, epochCount):
    ### 0) Set variables #######################################################

    # Name of modelDirectory
    # filename = "models/Mk4_nnM_1"
    filenameAlpha = "trainingData_M1_alpha.csv"
    filenameU = "trainingData_M1_u.csv"

    ### 1)  Generate Training Data #############################################

    print("Create Training Data")
    # build training data!
    x = np.arange(-1.0, 1.0, 0.0001)
    y = createTrainingData(x)

    ### 2) Create Model ########################################################
    # print("Create Model")

    # Load weights
    # model.load_weights(filename + '/best_model.h5')

    ### 3) Setup Training and Train the model ##################################

    # Create Early Stopping callback
    es = EarlyStopping(monitor='loss', mode='min', min_delta=0.000000001, patience=500,
                       verbose=10)  # loss == custom_loss1dMBPrime by model definition
    mc_best = ModelCheckpoint(filename + '/best_model.h5', monitor='loss', mode='min', save_best_only=True)
    mc_500 = ModelCheckpoint(filename + '/model_quicksave.h5', monitor='loss', mode='min', save_best_only=False,
                             save_freq=500)

    # Train the model
    print("Train Model")
    history = model.fit(x, y, validation_split=0.01, epochs=epochCount, batch_size=batchSize, verbose=1,
                        callbacks=[es, mc_best, mc_500])  # batch size = 900000

    # View History
    # nnUtils.print_history(history.history)

    ### 4) Save trained model and history ########################################

    print("Save model and history")
    nnUtils.save_training(filename, model, history)
    print("Training successfully saved")

    # load history
    history1 = nnUtils.load_trainHistory(filename)
    # print history as a check
    # nnUtils.print_history(history1)

    print("Training Sequence successfully finished")
    return model


### Build the network:
def create_modelMK4():
    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05
    # Number of basis functions used:

    # Weight initializer
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
    #### input layer ####
    input_ = keras.Input(shape=(1,))
    # Hidden layers
    # hidden = layers.BatchNormalization()(input_)
    '''

    hidden = layers.Dense(3,kernel_constraint=NonNeg(), activation="relu")(input_)
    hidden = layers.Dense(3,kernel_constraint=NonNeg(), activation="relu")(hidden)
    hidden = layers.Dense(3, kernel_constraint=NonNeg(), activation="relu")(hidden)

    '''
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones')(input_)
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones'
                          )(hidden)
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones'
                          )(hidden)

    # Define the output
    output_ = layers.Dense(1,
                           kernel_initializer=initializer,
                           bias_initializer='ones'
                           )(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def create_modelMK4_nonneg():
    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05

    # Define LayerDimensions
    layerDim = 3

    # Weight initializer
    initializer = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)

    input_ = keras.Input(shape=(1,))

    # Hidden layers
    # hidden = layers.BatchNormalization()(input_)

    hidden = layers.Dense(layerDim, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(input_)
    hidden = layers.Dense(layerDim, kernel_constraint=NonNeg(), activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(hidden)
    hidden = layers.Dense(layerDim, kernel_constraint=NonNeg(), activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(hidden)

    # Define the ouput
    output_ = layers.Dense(1, kernel_constraint=NonNeg(),
                           kernel_initializer=initializer,
                           bias_initializer='zeros'
                           )(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def create_modelMK4_ICNN():
    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05

    # Define LayerDimensions
    # inputDim = 1
    layerDim = 3

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
        out = tf.keras.activations.softplus(intermediateSum)
        # batch normalization
        # out = layers.BatchNormalization()(out)
        return out

    # Number of basis functions used:

    input_ = keras.Input(shape=(1,))

    ### Hidden layers ###
    # First Layer is a std dense layer
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='zeros'
                          )(input_)
    # other layers are convexLayers
    hidden = convexLayer(hidden, input_)
    hidden = convexLayer(hidden, input_)
    output_ = convexLayerOutput(hidden, input_)  # outputlayer

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_])
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model


def createTrainingData(x):
    return x  # 0.5 * x * x


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

    def reconstructFlux(self, alpha, tol=1e-8):
        return 0

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

        x = Lambda(self.identity_func, name="input")(x)

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.input_layer(x)
            for ic_layer in self.ic_layers:
                y = ic_layer(y, x)
            h = self.output_layer(y, x)

        d_net = grad_tape.gradient(h, x)

        d_net = Lambda(self.identity_func, name="d_net")(d_net)

        alpha = d_net

        u = self.reconstructU(alpha)
        flux = self.reconstructFlux(alpha)

        return [h, alpha, u, flux]


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


class HaltWhen(tf.keras.callbacks.Callback):
    def __init__(self, quantity, tol):
        """
        Should be used in conjunction with
        the saving criterion for the model; otherwise
        training will stop without saving the model with quantity <= tol
        """
        super(HaltWhen, self).__init__()
        if type(quantity) == str:
            self.quantity = quantity
        else:
            raise TypeError('HaltWhen(quantity,tol); quantity must be a string for a monitored quantity')
        self.tol = tol

    def on_epoch_end(self, epoch, logs=None):
        if epoch > 1:
            if logs.get(self.quantity) < self.tol:
                print('\n\n', self.quantity, ' has reached', logs.get(self.quantity), ' < = ', self.tol,
                      '. End Training.')
                self.model.stop_training = True
        else:
            pass


if __name__ == '__main__':
    main()
