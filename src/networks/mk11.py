"""
Network class "MK11" for the neural entropy closure.
ICNN with sobolev wrapper.
Author: Steffen Schotthöfer
Version: 1.0
Date 09.04.2021
"""

import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras as keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg

from src.networks.basenetwork import BaseNetwork
from src.networks.customlayers import MeanShiftLayer, DecorrelationLayer
from src.networks.entropymodels import SobolevModel


class MK11Network(BaseNetwork):
    """
    MK11 Model: Multi purpose sobolev based convex model
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    """

    def __init__(
            self,
            normalized: bool,
            input_decorrelation: bool,
            polynomial_degree: int,
            spatial_dimension: int,
            width: int,
            depth: int,
            loss_combination: int,
            save_folder: str = "",
            scale_active: bool = True,
            gamma_lvl: int = 0,
            basis: str = "monomial",
            rotated=False,
    ):
        if save_folder == "":
            custom_folder_name = (
                    "MK11_N" + str(polynomial_degree) + "_D" + str(spatial_dimension)
            )
        else:
            custom_folder_name = save_folder
        super(MK11Network, self).__init__(
            normalized=normalized,
            polynomial_degree=polynomial_degree,
            spatial_dimension=spatial_dimension,
            width=width,
            depth=depth,
            loss_combination=loss_combination,
            save_folder=custom_folder_name,
            input_decorrelation=input_decorrelation,
            scale_active=scale_active,
            gamma_lvl=gamma_lvl,
            basis=basis,
            rotated=rotated,
        )

    def create_model(self) -> bool:
        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.5, maxval=0.5, seed=None
        )
        initializer_non_neg = tf.keras.initializers.RandomUniform(
            minval=0, maxval=0.1, seed=None
        )

        weight_regularizer = tf.keras.regularizers.L1L2(l2=5e-5, l1=5e-5)  # L1 + L2 penalties

        def convex_layer(
                layer_input_z: Tensor,
                nw_input_x: Tensor,
                layer_idx: int = 0,
                layer_dim: int = 10,
        ) -> Tensor:
            # Weighted sum of previous layers output plus bias
            weighted_non_neg_sum_z = layers.Dense(
                units=layer_dim,
                activation=None,
                kernel_constraint=NonNeg(),
                kernel_initializer=initializer_non_neg,
                kernel_regularizer=weight_regularizer,
                use_bias=True,
                bias_initializer=initializer,
                name="layer_" + str(layer_idx) + "nn_component",
            )(layer_input_z)
            # Weighted sum of network input
            weighted_sum_x = layers.Dense(
                units=layer_dim,
                activation=None,
                kernel_initializer=initializer,
                kernel_regularizer=weight_regularizer,
                use_bias=False,
                name="layer_" + str(layer_idx) + "dense_component",
            )(nw_input_x)
            # Wz+Wx+b + x
            intermediate_sum = layers.Add(name="add_component_" + str(layer_idx))(
                [weighted_sum_x, weighted_non_neg_sum_z]
            )

            # Batch normalization
            # intermediate_sum = layers.BatchNormalization()(intermediate_sum)

            # activation
            out = tf.keras.activations.elu(intermediate_sum)
            return out

        def convex_output_layer(
                layer_input_z: Tensor, net_input_x: Tensor, layer_idx: int = 0
        ) -> Tensor:
            weighted_nn_sum_z: Tensor = layers.Dense(
                1,
                activation=None,
                kernel_constraint=NonNeg(),
                kernel_initializer=initializer_non_neg,
                kernel_regularizer=weight_regularizer,
                use_bias=True,
                bias_initializer=initializer,
                name="layer_" + str(layer_idx) + "nn_component",
            )(layer_input_z)
            # Weighted sum of network input
            weighted_sum_x: Tensor = layers.Dense(
                1,
                activation=None,
                kernel_initializer=initializer,
                kernel_regularizer=weight_regularizer,
                use_bias=False,
                name="layer_" + str(layer_idx) + "dense_component",
            )(net_input_x)
            # Wz+Wx+b
            out: Tensor = layers.Add()([weighted_sum_x, weighted_nn_sum_z])

            if self.scale_active:  # if output is scaled, use relu.
                out = tf.keras.activations.relu(out)
            return out

            ### build the core network with icnn closure architecture ###

        input_ = keras.Input(shape=(self.input_dim,))

        if self.input_decorrelation:  # input data decorellation and shift
            hidden = MeanShiftLayer(input_dim=self.input_dim, mean_shift=self.mean_u, name="mean_shift")(input_)
            hidden = DecorrelationLayer(input_dim=self.input_dim, ev_cov_mat=self.cov_ev, name="decorrelation")(hidden)
            x = hidden
            # First Layer is a std dense layer
            hidden = layers.Dense(
                self.model_width,
                activation="elu",
                kernel_initializer=initializer,
                kernel_regularizer=weight_regularizer,
                use_bias=True,
                bias_initializer=initializer,
                bias_regularizer=None,
                name="layer_-1_input",
            )(hidden)
        else:
            x = input_
            # First Layer is a std dense layer
            hidden = layers.Dense(
                self.model_width,
                activation="elu",
                kernel_initializer=initializer,
                kernel_regularizer=weight_regularizer,
                use_bias=True,
                bias_initializer=initializer,
                bias_regularizer=None,
                name="layer_-1_input",
            )(x)
        # other layers are convexLayers
        for idx in range(0, self.model_depth):
            hidden = convex_layer(hidden, x, layer_idx=idx, layer_dim=self.model_width)
        pre_output = convex_output_layer(
            hidden, x, layer_idx=self.model_depth + 2
        )  # outputlayer

        # Create the core model
        core_model = keras.Model(
            inputs=[input_], outputs=[pre_output], name="Icnn_closure"
        )
        print("The core model overview")
        # core_model.summary()
        print("The sobolev wrapped model overview")

        # build sobolev wrapper
        model = SobolevModel(
            core_model,
            polynomial_degree=self.poly_degree,
            spatial_dimension=self.spatial_dim,
            reconstruct_u=bool(self.loss_weights[2]),
            scaler_max=self.scaler_max,
            scaler_min=self.scaler_min,
            scale_active=self.scale_active,
            gamma=self.regularization_gamma,
            name="sobolev_icnn_wrapper",
            basis=self.basis,
            rotated=self.rotated,
        )
        # build graph
        batch_size: int = 3  # dummy entry
        model.build(input_shape=(batch_size, self.input_dim))

        print(
            "Compile model with loss weights "
            + str(self.loss_weights[0])
            + "|"
            + str(self.loss_weights[1])
            + "|"
            + str(self.loss_weights[2])
        )
        model.compile(
            loss={
                "output_1": tf.keras.losses.MeanSquaredError(),
                "output_2": tf.keras.losses.MeanSquaredError(),
                "output_3": tf.keras.losses.MeanSquaredError(),
            },
            loss_weights={
                "output_1": self.loss_weights[0],
                "output_2": self.loss_weights[1],
                "output_3": self.loss_weights[2],
            },
            optimizer=self.optimizer,
            metrics=["mean_absolute_error", "mean_squared_error"],
        )

        self.model = model
        return True

    def call_training(
            self,
            val_split: float = 0.1,
            epoch_size: int = 2,
            batch_size: int = 128,
            verbosity_mode: int = 1,
            callback_list: list = [],
    ) -> list:
        """
        Calls training depending on the MK model
        """
        # u_in = self.training_data[0]
        # alpha_in = self.training_data[1]
        # [h, alpha, u_out] = self.model(alpha_in)
        ## alpha_complete_predicted = self.model.reconstruct_alpha(alpha_predicted)
        ## u_complete_reconstructed = self.model.reconstruct_u(alpha_complete_predicted)
        # u_in = self.training_data[0][:2]
        # alpha_in = self.training_data[1][:2]
        # h_in = self.training_data[2][:2]
        # [h, alpha, u] = self.model(alpha_in)

        x_data = self.training_data[0]
        y_data = [
            self.training_data[2],
            self.training_data[1],
            self.training_data[0],
        ]  # , self.trainingData[1]]
        self.history = self.model.fit(
            x=x_data,
            y=y_data,
            validation_split=val_split,
            epochs=epoch_size,
            batch_size=batch_size,
            verbose=verbosity_mode,
            callbacks=callback_list,
            shuffle=True,
        )
        return self.history

    def select_training_data(self):
        return [True, True, True]

    def call_network(self, u_complete):
        """
        brief: Only works for maxwell Boltzmann entropy so far.
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: u_complete, dims = (nS x N)
        returns: alpha_complete_predicted, dim = (nS x N)
                 u_complete_reconstructed, dim = (nS x N)
                 h_predicted, dim = (nS x 1)
        """
        u_reduced = u_complete[:, 1:]  # chop of u_0
        [h_predicted, alpha_predicted, u_0_predicted, tmp] = self.model(u_reduced)
        alpha_complete_predicted = self.model.reconstruct_alpha(alpha_predicted)
        u_complete_reconstructed = self.model.reconstruct_u(alpha_complete_predicted)

        return [u_complete_reconstructed, alpha_complete_predicted, h_predicted]

    def call_scaled(self, u_non_normal):
        """
        brief: Only works for maxwell Boltzmann entropy so far.
        Calls the network with non normalized moments. (first the moments get normalized, then the network gets called,
        then the upscaling mechanisms get called, then the original entropy gets computed.)
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: u_complete, dims = (nS x N)
        returns: [u,alpha,h], where
                 alpha_complete_predicted_scaled, dim = (nS x N)
                 u_complete_reconstructed_scaled, dim = (nS x N)
                 h_predicted_scaled, dim = (nS x 1)
        """
        u_non_normal = tf.constant(u_non_normal, dtype=tf.float32)
        u_downscaled = self.model.scale_u(
            u_non_normal, tf.math.reciprocal(u_non_normal[:, 0])
        )  # downscaling
        [
            u_complete_reconstructed,
            alpha_complete_predicted,
            h_predicted,
        ] = self.call_network(u_downscaled)
        u_rescaled = self.model.scale_u(
            u_complete_reconstructed, u_non_normal[:, 0]
        )  # upscaling
        alpha_rescaled = self.model.scale_alpha(
            alpha_complete_predicted, u_non_normal[:, 0]
        )  # upscaling
        h_rescaled = self.model.compute_h(u_rescaled, alpha_rescaled)

        return [u_rescaled, alpha_rescaled, h_rescaled]

    def call_scaled_64(self, u_non_normal, legacy_mode=False):
        """
        brief: Only works for maxwell Boltzmann entropy so far.
        Calls the network with non normalized moments. (first the moments get normalized, then the network gets called,
        then the upscaling mechanisms get called, then the original entropy gets computed.)
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: u_complete, dims = (nS x N)
        returns: [u,alpha,h], where
                 alpha_complete_predicted_scaled, dim = (nS x N)
                 u_complete_reconstructed_scaled, dim = (nS x N)
                 h_predicted_scaled, dim = (nS x 1)
        """
        u_non_normal = tf.constant(u_non_normal, dtype=tf.float32)
        u_downscaled = self.model.scale_u(
            u_non_normal, tf.math.reciprocal(u_non_normal[:, 0])
        )  # downscaling
        #
        #
        #
        u_reduced = u_downscaled[:, 1:]  # chop of u_0
        u_0 = tf.cast(u_non_normal[:, 0], dtype=tf.float64, name=None)
        if legacy_mode:
            if self.poly_degree > 1:
                [h_predicted, alpha_predicted, u_predicted] = self.model_legacy(
                    u_reduced
                )
            else:
                [h_predicted, alpha_predicted] = self.model_legacy(u_reduced)
        else:
            [h_predicted, alpha_predicted, u_predicted] = self.model(u_reduced)

        ### cast to fp64 ###
        alpha64 = tf.cast(alpha_predicted, dtype=tf.float64, name=None)
        alpha_complete = self.model.reconstruct_alpha(alpha64)
        u_complete = self.model.reconstruct_u(alpha_complete)
        u_rescaled = self.model.scale_u(u_complete, u_0)  # upscaling
        alpha_rescaled = self.model.scale_alpha(alpha_complete, u_0)
        h = self.model.compute_h(u_rescaled, alpha_rescaled)
        # u_non_2 = self.model.reconstruct_u(alpha_rescaled)

        return [u_rescaled, alpha_rescaled, h]
