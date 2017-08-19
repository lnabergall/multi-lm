"""
Classes and functions for multi-domain language modeling 
using Tensor2Tensor.

Four primary models: LSTM, LSTM + Mixture-of-Experts, Attention Network,
                     Attention Network + Mixture-of-Experts

TODO: Add character convolution neural network input layer (modality?)
      Implement LSTM + MoE by adding MoE + residual connections to BasicLSTMCell
      May need to implement importance sampling (or some other scheme...)
"""
import tensorflow as tf
from tensor2tensor.models import lstm, attention_lm, attention_lm_moe
from tensor2tensor.layers import common_layers, common_hparams
from tensor2tensor.utils import registry, t2t_model


@registry.register_model
class LSTMLm(t2t_model.T2TModel):
    """Basic LSTM recurrent neural network."""

    def model_fn_body(self, features):
        train = self._hparams.mode == tf.contrib.learn.ModeKeys.TRAIN
        with tf.variable_scope("lstm_lm"):
            # Flatten inputs.
            inputs = common_layers.flatten4d3d(features["inputs"])
            outputs, _ = lstm.lstm(tf.reverse(inputs, axis=[1]), 
                                   self._hparams, train, "lstm")
            return tf.expand_dims(outputs, axis=2)


@registry.register_hparams
def lstm_literature_base():
    """Set of base hyperparameters for LSTM from Jozefowicz et al."""
    hparams = common_hparams.basic_params1()
    hparams.clip_grad_norm = 1.0
    hparams.label_smoothing = 0.0
    hparams.batch_size = 2048
    hparams.optimizer = "Adagrad"
    hparams.learning_rate = 0.2
    return hparams


@registry.register_hparams
def lstm_literature_small():
    """Set of hyperparameters for small LSTM from Jozefowicz et al."""
    hparams = lstm_literature_base()
    hparams.hidden_size = 512
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    return hparams


@registry.register_hparams
def lstm_literature_large():
    """Set of hyperparameters for largest LSTM from Jozefowicz et al."""
    hparams = lstm_literature_base()
    hparams.hidden_size = 8192
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.25
    return hparams


@registry.register_hparams
def lstm_base():
    """Set of hyperparameters for our LSTM."""
    hparams = common_hparams.basic_params1()
    hparams.batch_size = 2048
    hparams.label_smoothing = 0.0
    hparams.shared_embedding_and_softmax_weights = int(True)
    return hparams


@registry.register_hparams
def lstm_small():
    """Set of hyperparameters for our LSTM."""
    hparams = lstm_base()
    hparams.hidden_size = 256
    hparams.num_hidden_layers = 1
    return hparams


@registry.register_hparams
def lstm_medium():
    """Set of hyperparameters for our LSTM."""
    hparams = lstm_base()
    hparams.hidden_size = 1024
    params.num_hidden_layers = 2
    return hparams


@registry.register_hparams
def lstm_large():
    """Set of hyperparameters for our LSTM."""
    hparams = lstm_base()
    hparams.hidden_size = 8192
    params.num_hidden_layers = 4
    return hparams


@registry.register_hparams
def attention_lm_tiny():
    """Set of hyperparameters for the attention network."""
    hparams = attention_lm.attention_lm_base()
    hparams.batch_size = 2048
    hparams.num_hidden_layers = 1
    hparams.hidden_size = 256
    hparams.filter_size = 1024
    hparams.layer_prepostprocess_dropout = 0.5
    # hparams.shared_embedding_and_softmax_weights = int(True)
    return hparams