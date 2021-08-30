"""
brief: Custom Keras Layers for own networks
author: Steffen Schotthöfer
date: 30.08.2021
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import nn_ops


class MeanShiftLayer(layers.Layer):
    """
    Layer that substracts the a given value (dataset mean) from the input
    """

    def __init__(self, input_dim=32, mean_shift=np.zeros(32, ), **kwargs):
        super(MeanShiftLayer, self).__init__()

        mu_init = tf.zeros_initializer()
        self.mu = tf.Variable(initial_value=mu_init(shape=(input_dim,)), trainable=False)
        self.mu.assign(mean_shift)

    def call(self, inputs):
        return inputs - self.mu


class PositiveWeightLayer(layers.Dense):
    """
    Layer that obtains a positive weight matrix by pointwise applying softmax to it.
    The idea is that this also stabilizes training (compared to a projection)
    Layer structure: y = softmax(w)*x
    """

    def __init__(self, units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, **kwargs):
        super(PositiveWeightLayer, self).__init__(
            units=units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint, **kwargs)

    def call(self, inputs):
        # perform a softmax activation on only the weights
        kernel_sm = tf.nn.relu(self.kernel)
        # proceed dense computation as usual (copy pasted from tensorflow codebase)
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = math_ops.cast(inputs, dtype=self._compute_dtype_object)

        rank = inputs.shape.rank
        if rank == 2 or rank is None:
            # We use embedding_lookup_sparse as a more efficient matmul operation for
            # large sparse input tensors. The op will result in a sparse gradient, as
            # opposed to sparse_ops.sparse_tensor_dense_matmul which results in dense
            # gradients. This can lead to sigfinicant speedups, see b/171762937.
            if isinstance(inputs, sparse_tensor.SparseTensor):
                # We need to fill empty rows, as the op assumes at least one id per row.
                inputs, _ = sparse_ops.sparse_fill_empty_rows(inputs, 0)
                # We need to do some munging of our input to use the embedding lookup as
                # a matrix multiply. We split our input matrix into separate ids and
                # weights tensors. The values of the ids tensor should be the column
                # indices of our input matrix and the values of the weights tensor
                # can continue to the actual matrix weights.
                # The column arrangement of ids and weights
                # will be summed over and does not matter. See the documentation for
                # sparse_ops.sparse_tensor_dense_matmul a more detailed explanation
                # of the inputs to both ops.
                ids = sparse_tensor.SparseTensor(
                    indices=inputs.indices,
                    values=inputs.indices[:, 1],
                    dense_shape=inputs.dense_shape)
                weights = inputs
                outputs = embedding_ops.embedding_lookup_sparse_v2(
                    kernel_sm, ids, weights, combiner='sum')
            else:
                outputs = gen_math_ops.MatMul(a=inputs, b=kernel_sm)
        # Broadcast kernel to inputs.
        else:
            outputs = standard_ops.tensordot(inputs, kernel_sm, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.shape.as_list()
                output_shape = shape[:-1] + [kernel_sm.shape[-1]]
                outputs.set_shape(output_shape)

        if self.use_bias:
            outputs = nn_ops.bias_add(outputs, self.bias)

        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    """
            def shifted_convex_layer(layer_input_z: Tensor, nw_input_x: Tensor, layer_idx: int = 0,
                                     layer_dim: int = 10, input_shape: int = 10) -> Tensor:
                initializer = tf.keras.initializers.LecunNormal()
                shift_tensor = tf.ones(shape=(layer_dim, input_shape), dtype=tf.dtypes.float32, name=None)
    
                # Weighted sum of previous layers output plus bias
                weighted_non_neg_sum_z = layers.Dense(layer_dim,  # kernel_constraint=ClipByValueConstraint(-1.0),
                                                      activation=None, kernel_initializer=initializer,
                                                      kernel_regularizer=l2_regularizer_nn, use_bias=True,
                                                      bias_initializer='zeros',
                                                      name='non_neg_component_' + str(layer_idx))(layer_input_z)
                # Weighted sum of network input
                weighted_sum_x = layers.Dense(layer_dim, activation=None, kernel_initializer=initializer,
                                              kernel_regularizer=l1l2_regularizer, use_bias=False,
                                              name='dense_component_' + str(layer_idx))(nw_input_x)
                # Wz+Wx+b
                intermediate_sum = layers.Add(name='add_component_' + str(layer_idx))(
                    [weighted_sum_x, weighted_non_neg_sum_z])
                # input shift: ones*z == [sum(z),...]
                shift = tf.math.reduce_sum(layer_input_z, axis=1, keepdims=False,
                                           name='shift_component_' + str(layer_idx))
                # shift_bc = tf.broadcast_to(shift, shape=(None, layer_dim), name="broadcaster")
    
                # shift: Tensor = layers.Dot(axes=1, name='shift_component_' + str(layer_idx))(
                #   [layer_input_z, shift_tensor])
                # shift = tf.reshape(shift, shape=(shift.shape[1], layer_dim))
                # add to current layer: # Wz+Wx+b + 1z
                intermediate_sum = layers.Add(name='add_shift_' + str(layer_idx))(
                    [intermediate_sum, shift])
                # activation
                out = tf.keras.activations.softplus(intermediate_sum)
                # out = tf.keras.activations.selu(intermediate_sum)
                # batch normalization
                # out = layers.BatchNormalization(name='bn_' + str(layerIdx))(out)
                return out
                """
