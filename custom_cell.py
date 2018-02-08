#!/usr/lib/env python
#coding=utf-8
"""
  description:
    实现了RNN 自定义的 Cell 结构，GRU
"""
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell

class CustomCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, num_weights):
        self._num_units = num_units
        self._num_weights = num_weights

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                ru = rnn_cell._linear([inputs, state],
                                        2 * self._num_units, True, 1.0)
                ru = tf.nn.sigmoid(ru)
                r, u = tf.split(1, 2, ru)
            with tf.variable_scope("Candidate"):
                lambdas = rnn_cell._linear([inputs, state], self._num_weights, True)
                lambdas = tf.split(1, self._num_weights, tf.nn.softmax(lambdas))

                Ws = tf.get_variable("Ws",
                        shape = [self._num_weights, inputs.get_shape()[1], self._num_units])
                Ws = [tf.squeeze(i) for i in tf.split(0, self._num_weights, Ws)]

                candidate_inputs = []

                for idx, W in enumerate(Ws):
                    candidate_inputs.append(tf.matmul(inputs, W) * lambdas[idx])

                Wx = tf.add_n(candidate_inputs)

                c = tf.nn.tanh(Wx + rnn_cell._linear([r * state],
                                            self._num_units, True, scope="second"))
            new_h = u * state + (1 - u) * c
        return new_h, new_h
