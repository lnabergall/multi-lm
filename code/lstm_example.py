"""
Example implementation of a vanilla LSTM network 
based on https://www.tensorflow.org/tutorials/recurrent.
"""

import time
import inspect

import numpy as np
import tensorflow as tf

import data_processing as dp


TRAIN = "train"
VALID = "validate"
TEST = "test"

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("model", "small", 
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory.")
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
        self.raw_data, self.input_data, self.targets = dp.input_data(
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

      
class Config:
    
      def __init__(self, config_type):
          if config_type == "small":
              init_scale = 0.1
              learning_rate = 1.0
              max_grad_norm = 5
              num_layers = 2
              num_steps = 20
              hidden_size = 200
              max_epoch = 4
              max_max_epoch = 13
              keep_prob = 1.0
              learning_rate_decay = 0.5
              batch_size = 20
              vocab_size = 10000
          elif config_type == "medium":
              init_scale = 0.05
              learning_rate = 1.0
              max_grad_norm = 5
              num_layers = 2
              num_steps = 35
              hidden_size = 650
              max_epoch = 6
              max_max_epoch = 39
              keep_prob = 0.5
              learning_rate_decay = 0.8
              batch_size = 20
              vocab_size = 10000
          elif config_type == "large":
              init_scale = 0.04
              learning_rate = 1.0
              max_grad_norm = 10
              num_layers = 2
              num_steps = 35
              hidden_size = 1500
              max_epoch = 14
              max_max_epoch = 44
              keep_prob = 0.35
              learning_rate_decay = 1 / 1.15
              batch_size = 20
              vocab_size = 10000
          elif config_type == "test":
              init_scale = 0.1
              learning_rate = 1.0
              max_grad_norm = 1
              num_layers = 1
              num_steps = 2
              hidden_size = 2
              max_epoch = 1
              max_max_epoch = 1
              keep_prob = 1.0
              learning_rate_decay = 0.5
              batch_size = 20
              vocab_size = 10000
          else:
              raise ValueError("Invalid config type: '%s'", config_type)
              

def run_epoch(session, model, eval_op=None, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
    costs = 0.0
    iterations = 0
    state = session.run(model.initial_state)
    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }
    if eval_op is not None:
        fetches["eval_op"] = eval_op
    
    for step in range(model.input.epoch_size):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        
        values = session.run(fetches, feed_dict)
        cost = values["cost"]
        state = values["final_state"]
        
        cost += cost
        iterations += model.input.num_steps
        
        if verbose and step % (model.input.epoch_size // 10) == 10:
            print("perplexity %.3f; speed %.3f; wps %.0f" % 
                  (step * 1.0 / model.input.epoch_size, np.exp(costs / iterations),
                   iterations * model.input.batch_size / (time.time() - start_time)))
            
    return np.exp(costs / iterations)


def get_config():
    return Config(FLAGS.model)


def main():
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path.")

    raw_data = dp.raw_data(FLAGS.data_path)
    train_data, validate_data, test_data, _ = raw_data

    config = get_config()
    eval_config = get_config()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(-config.init_scale, 
                                                    config.init_scale)

        with tf.name_scope("Train"):
            train_input = InputData(
                config=config, data=train_data, name="TrainInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model = LSTMModel(
                    training=True, config=config, input_=train_input)
            tf.summary.scalar("Training Loss", model.cost)
            tf.summary.scalar("Learning Rate", model.learning_rate)

        with tf.name_scope("Validate"):
            validate_input = InputData(
                config=config, data=validate_data, name="ValidateInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model_valid = LSTMModel(
                    training=False, config=config, input_=validate_input)
            tf.summary.scalar("Validation Loss", model_valid.cost)

        with tf.name_scope("Test"):
            test_input = InputData(
                config=eval_config, data=test_data, name="TestInput")
            with tf.variable_scope("Model", reuse=None, initializer=initializer):
                model_test = LSTMModel(
                    training=False, config=eval_config, input_=test_input)

        supervisor = tf.train.Supervisor(logdir=FLAGS.save_path)
        with supervisor.managed_session() as session:
            for i in range(config.max_max_epoch):
                learning_rate_decay = (
                    config.learning_rate_decay ** max(i + 1 - config.max_epoch, 0.0))
                model.assign_learning_rate(
                    session, config.learning_rate * learning_rate_decay)

                print("Epoch: %d, Learning rate: %.3f" % (
                    i+1, session.run(model.learning_rate)))
                train_perplexity = run_epoch(session, model, 
                    eval_op=model.train_op, verbose=True)
                print("Epoch %d, Train Perplexity: %.3f" % (i+1, train_perplexity))
                validate_perplexity = run_epoch(session, model_valid)
                print("Epoch: %d, Validate Perplexity: %.3f" % (i+1, validate_perplexity))

            test_perplexity = run_epoch(session, model_test)
            print("Test Perplexity: %.3f" % test_perplexity)

            if FLAGS.save_path:
                print("Saving model to %s." % FLAGS.save_path)
                supervisor.saver.save(session, FLAGS.save_path, 
                                      global_step=supervisor.global_step)


if __name__ == '__main__':
    tf.app.run()

