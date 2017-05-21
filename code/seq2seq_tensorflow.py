"""Sequence-to-sequence model implemented using tf.contrib.seq2seq"""

from math import sqrt

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import (LSTMCell, MultiRNNCell, LSTMStateTuple, 
                                    DropoutWrapper, ResidualWrapper)
from tensorflow.contrib.seq2seq import (BasicDecoder, dynamic_decode, 
                                        TrainingHelper, GreedyEmbeddingHelper,
                                        sequence_loss)

import data_processing as dp
from data_processing import (PAD_VALUE, MAX_DESCRIPTION_LENGTH, 
                             MAX_SCRIPT_LENGTH)


HIDDEN_DIM = 128
LEARNING_RATE = 0.005
BACKPROP_TIMESTEPS = 50
BATCH_SIZE = 128
SUPER_BATCHES = 50


class Seq2SeqModel():

    PAD = 0
    EOS = 1

    def __init__(self, encoder_cell, decoder_cell, encoder_vocab_size, 
                 decoder_vocab_size, encoder_embedding_size, decoder_embedding_size, 
                 bidirectional=False, attention=False):
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size

        self.bidirectional = bidirectional
        self.attention = attention

        self._make_graph()

    def _make_graph(self):
        self._init_placeholders()
        self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()
        self._init_optimizer()


    def _init_placeholders(self):
        """Everything is time-major."""
        self.encoder_inputs = tf.placeholder(
            shape=(None, None), dtype=tf.int32, name="encoder_inputs")
        self.encoder_inputs_length = tf.placeholder(
            shape=(None,), dtype=tf.int32, name="encoder_inputs_length")

        # required for training, not for testing
        self.decoder_targets = tf.placeholder(
            shape=(None, None), dtype=tf.int32, name="decoder_targets")
        self.decoder_targets_length = tf.placeholder(
            shape=(None,), dtype=tf.int32, name="decoder_targets_length")

    def _init_decoder_train_connectors(self):
        with tf.name_scope("DecoderTrainFeeds"):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat(
                [EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat(
                [self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(
                tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(
                self.decoder_train_length-1, decoder_train_targets_seq_len,
                on_value=self.EOS, off_value=self.PAD, dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(
                decoder_train_targets_eos_mask, [1, 0])
            # Put EOS symbol at the end of target sequence
            decoder_train_targets = tf.add(decoder_train_targets,
                                           decoder_train_targets_eos_mask)
            self.decoder_train_targets = decoder_train_targets

            self.loss_weights = tf.ones(
                [batch_size, tf.reduce_max(self.decoder_train_length)], 
                dtype=tf.float32, name="loss_weights")

    def _init_embeddings(self):
        with tf.variable_scope("embedding") as scope:
            sqrt3 = sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.encoder_embedding_matrix = tf.get_variable(
                name="encoder_embedding_matrix", 
                shape=[self.encoder_vocab_size, self.encoder_embedding_size],
                initializer=initializer,
                dtype=tf.float32)
            self.decoder_embedding_matrix = tf.get_variable(
                name="decoder_embedding_matrix", 
                shape=[self.decoder_vocab_size, self.decoder_embedding_size],
                initializer=initializer,
                dtype=tf.float32)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(
                self.encoder_embedding_matrix, self.encoder_inputs)
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(
                self.decoder_embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):
        with tf.variable_scope("Encoder") as scope:
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                cell=self.encoder_cell, inputs=self.encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length, time_major=True,
                dtype=tf.float32)

    def _init_bidirectional_encoder(self):
        with tf.variable_scope("BidirectionalEncoder") as scope:
            ((encoder_fw_outputs, encoder_bw_outputs), 
             (encoder_fw_state, encoder_bw_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=self.encoder_cell, cell_bw=self.encoder_cell,
                inputs=self.encoder_inputs_embedded, 
                sequence_length=self.encoder_inputs_length, 
                time_major=True, dtype=tf.float32)
             self.encoder_outputs = tf.concat(
                (encoder_fw_outputs, encoder_bw_outputs), 2)

             if isinstance(encoder_fw_state, LSTMStateTuple):
                encoder_state_c = tf.concat(
                    (encoder_fw_state.c, encoder_bw_state.c), 
                    1, name="bidirectional_concat_c")
                encoder_state_h = tf.concat(
                    (encoder_fw_state.h, encoder_bw_state.h), 
                    1, name="bidirectional_concat_h")
                self.encoder_state = LSTMStateTuple(
                    c=encoder_state_c, h=encoder_state_h)
            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state),
                                               1, name="bidirectional_concat")

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(
                    outputs, self.decoder_vocab_size, scope=scope)

            train_helper = TrainingHelper(self.decoder_train_inputs_embedded, 
                                          self.decoder_train_length, time_major=True)

            if self.attention:
                pass    # To implement...

            train_decoder = BasicDecoder(cell=self.decoder_cell, 
                                         helper=train_helper, 
                                         initial_state=self.encoder_state)
            self.decoder_outputs_train, self.decoder_state_train = dynamic_decode(
                train_decoder, output_time_major=True, scope=scope)

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(
                self.decoder_logits_train, axis=-1, name="decoder_prediction_train")

            scope.reuse_variables()

            _, batch_size = tf.unstack(tf.shape(self.decoder_targets))
            start_tokens = tf.ones(batch_size, dtype=int32) * self.EOS
            inference_helper = GreedyEmbeddingHelper(self.decoder_embedding_matrix, 
                                                     start_tokens=start_tokens, 
                                                     end_token=self.EOS)
            inference_decoder = BasicDecoder(cell=self.decoder_cell, 
                                             helper=inference_helper,
                                             initial_state=self.encoder_state)
            self.decoder_logits_inference, self.decoder_state_inference = dynamic_decode(
                inference_decoder, output_time_major=True, scope=scope)
            self.decoder_prediction_inference = tf.argmax(
                self.decoder_logits_inference, axis=-1, 
                name="decoder_prediction_inference")

        def _init_optimizer(self):
            logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
            targets = tf.transpose(self.decoder_train_targets, [1, 0])
            self.loss = sequence_loss(logits=logits, targets=targets,
                                      weights=self.loss_weights)
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        def make_train_inputs(self, input_seq, target_seq):
            inputs_, inputs_length_ = dp.batch(input_seq)
            targets_, targets_length_ = dp.batch(target_seq)
            return {
                self.encoder_inputs: inputs_,
                self.encoder_inputs_length: inputs_length_,
                self.decoder_targets: targets_,
                self.decoder_targets_length: targets_length_,
            }

        def make_inference_inputs(self, input_seq):
            inputs_, inputs_length_ = dp.batch(input_seq)
            return {
                self.encoder_inputs: inputs_,
                self.encoder_inputs_length: inputs_length_,
            }


def main():
    print("\nFetching data...")
    data = dp.get_data()
    print("\nProcessing data...")
    processed_data = dp.process_data(data)
    description_chars = processed_data["description_chars"]
    script_chars = processed_data["python_chars"]
    training_data_dict = OrderedDict(processed_data["training_data"])
    validation_data_dict = processed_data["validation_data"]
    test_data_dict = processed_data["test_data"]

    tf.reset_default_graph()
    tf.set_random_seed()
    with tf.Session() as session:
        model = Seq2SeqModel(encoder_cell=LSTMCell(HIDDEN_DIM),
                             decoder_cell=LSTMCell(HIDDEN_DIM),
                             encoder_vocab_size=len(description_chars),
                             decoder_vocab_size=len(script_chars),
                             encoder_embedding_size=len(description_chars),
                             decoder_embedding_size=len(script_chars),
                             attention=False,
                             bidirectional=False)
        session.run(tf.global_variables_initializer())
        