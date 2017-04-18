"""
Example implementation of a vanilla LSTM network 
based on https://www.tensorflow.org/tutorials/recurrent.
"""

import inspect
import numpy as np
import tensorflow as tf

from data_processing import input_data


TRAIN = "train"
VALID = "validate"
TEST = "test"

flags = tf.flags
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class InputData:
    """The input data."""

    class __init__(self, config, data_type, name=None):
        self.batch_size = config.batch_size
        self.num_steps = config.num_steps
        self.type = data_type
        self.raw_data, self.input_data, self.targets = input_data(
            batch_size, num_steps, name=name)
        self.epoch_size = ((len(self.raw_data) // batch_size) - 1) // num_steps


class LSTMModel:
    """Basic LSTM model."""

    def __init__(self, training, config, input_):
        self._input = input_
        batch_size = input_.batch_size
        num_steps = input_.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size
        
        def lstm_cell():
            # With the latest TensorFlow source code (as of Apr 17, 2017),
            # the BasicLSTMCell will need a reuse parameter which is not 
            # defined in TensorFlow 1.0. To maintain backwards compatibility,
            # we add an argument check here:
            if "reuse" in inspect.getargspec(
                    tf.contrib.rnn.BasicLSTMCell.__init__).args:
                return contrib.rnn.BasicLSTMCell(
                    size, forget_bias=1, state_is_tuple=True,
                    reuse=tf.get_variable_scope().reuse)
            else:
                return contrib.rnn.BasicLSTMCell(
                    size, forget_bias=1, state_is_tuple=True)
        
        attn_cell = lstm_cell
        # Apply dropout
        if training and config.keep_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.keep_prob)
        # Add multiple layers
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)
        
        self._initial_state = cell.zero_state(batch_size, data_type())
        
        # Get word vectors
        with tf.device("/cpi:0"):
            embedding = tf.get_variable(
                "embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, input_.input_data)
        
        # Apply dropout
        if training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
        
        # Unroll LSTM
        inputs = tf.unstack(inputs, num=num_steps, axis=1)
        outputs, state = tf.contrib.rnn.static_rnn(
            cell, inputs, initial_state=self._initial_state)
        
        output = tf.reshape(tf.concat(values=outputs, axis=1), [-1, size])
        softmax_w = tf.get_variable(
            "softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.contrib.seq2seq.sequence_loss(
            logits=[logits], targets=[tf.reshape(input_.targets, [-1])],
            weights=[tf.ones([batch_size * num_steps], dtype=data_type())],
            average_across_batch=False)
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        if not training:
            return
        else:
            self._learning_rate = tf.Variable(0.0, trainable=False)
            trainable_vars = tf.trainable_variables()
            gradients, _ = tf.clip_by_global_norm(
                tf.gradients(cost, trainable_vars), config.max_grad_norm)
            optimizer = tf.train.GradientDescentOptiizer(self._learning_rate)
            self._train_op = optimizer.apply_gradients(
                zip(gradients, trainable_vars),
                global_step=tf.contrib.framework.get_or_create_global_step())
            self._new_learning_rate = tf.placeholder(
                tf.float32, shape=[], name="new_learning_rate")
            self._learning_rate_update = tf.assign(
                self._learning_rate, self._new_learning_rate)

    def assign_learning_rate(self, session, learning_rate_value):
        session.run(self._learning_rate_update, 
                    feed_dict={self._new_learning_rate: learning_rate_value})

    @property
    def input(self):
        return self._input

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def train_op(self):
        return self._train_op

