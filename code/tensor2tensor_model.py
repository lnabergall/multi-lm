"""
Classes and functions for multi-domain language modeling 
using Tensor2Tensor.

Four primary models: LSTM, LSTM + Mixture-of-Experts, Attention Network,
                     Attention Network + Mixture-of-Experts

TODO: Add character convolution neural network input layer (modality?)
      Implement LSTM + MoE by adding MoE + residual connections to BasicLSTMCell
"""

from tensor2tensor.models import (lstm, attention_lm, attention_lm_moe, 
                                  common_layers)
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



