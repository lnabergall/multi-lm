"""
Classes and functions for multi-domain language modeling 
using Tensor2Tensor.

Four primary models: LSTM, LSTM + Mixture-of-Experts, Attention Network,
                     Attention Network + Mixture-of-Experts
"""

from tensor2tensor.models import lstm, attention_lm, attention_lm_moe
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
            return tf.expand_dims(decoder_outputs, axis=2)


@registry.register_model
class LSTMLmMoe(t2t_model.T2TModel):
    """LSTM recurrent neural network with MoE layers."""

    def model_fn_body_sharded(self, sharded_features):
        pass