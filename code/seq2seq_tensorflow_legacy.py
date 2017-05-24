"""
Sequence-to-sequence model implemented 
using tf.contrib.seq2seq from tensorflow 1.0.
"""

from math import sqrt

import numpy as np
import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow.contrib.rnn import (LSTMCell, MultiRNNCell, LSTMStateTuple, 
                                    DropoutWrapper)
from tensorflow.contrib.seq2seq import (dynamic_rnn_decoder, simple_decoder_fn_train, 
                                        simple_decoder_fn_inference, sequence_loss)

import data_processing as dp
from data_processing import (PAD_VALUE, MAX_DESCRIPTION_LENGTH, 
                             MAX_SCRIPT_LENGTH)


HIDDEN_DIM = 64
LEARNING_RATE = 0.005
BACKPROP_TIMESTEPS = 50
BATCH_SIZE = 128
EPOCHS = 50


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

    @property
    def decoder_hidden_units(self):
        return self.decoder_cell.output_size

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
                [EOS_SLICE, self.decoder_targets], 0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat(
                [self.decoder_targets, PAD_SLICE], 0)
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

            decoder_fn_train = simple_decoder_fn_train(encoder_state=self.encoder_state)

            if self.attention:
                pass    # To implement...

            (self.decoder_outputs_train, self.decoder_state_train, 
             self.decoder_context_state_train) = dynamic_rnn_decoder(
                cell=self.decoder_cell, decoder_fn=decoder_fn_train,
                inputs=self.decoder_train_inputs_embedded, 
                sequence_length=self.decoder_train_length,
                time_major=True, scope=scope)

            self.decoder_logits_train = output_fn(self.decoder_outputs_train)
            self.decoder_prediction_train = tf.argmax(
                self.decoder_logits_train, axis=-1, name="decoder_prediction_train")

            scope.reuse_variables()

            decoder_fn_inference = simple_decoder_fn_inference(
                output_fn=output_fn, encoder_state=self.encoder_state,
                embeddings=self.decoder_embedding_matrix, 
                start_of_sequence_id=self.EOS, end_of_sequence_id=self.EOS,
                maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                num_decoder_symbols=self.decoder_vocab_size)
            (self.decoder_logits_inference, self.decoder_state_inference, 
             self.decoder_context_state_inference) = dynamic_rnn_decoder(
                cell=self.decoder_cell, decoder_fn=decoder_fn_inference,
                time_major=True, scope=scope)
            self.decoder_prediction_inference = tf.argmax(
                self.decoder_logits_inference, axis=-1, 
                name="decoder_prediction_inference")

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = sequence_loss(logits=logits, targets=targets,
                                  weights=self.loss_weights)
        self.train_op = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(self.loss)

    def make_train_inputs(self, inputs_, inputs_length_, 
                          targets_, targets_length_):
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_,
        }

    def make_inference_inputs(self, inputs_, inputs_length_):
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
        }

    def save(self):
        pass


def randomize_insync(*args):
    state = np.random.get_state()
    for collection in args:
        np.random.shuffle(collection)
        np.random.set_state(state)


def main():
    print("\nFetching data...")
    data = dp.get_data()
    print("\nProcessing data...")
    processed_data = dp.process_data(data)
    description_chars = processed_data["description_chars"]
    script_chars = processed_data["python_chars"]
    training_data_dict = processed_data["training_data"]
    validation_data_dict = processed_data["validation_data"]
    test_data_dict = processed_data["test_data"]
    print("\nVectorizing data...")
    train_inputs, train_targets, train_input_lengths, train_target_lengths \
        = dp.vectorize_data(training_data_dict, description_chars, 
                            script_chars, dense=True, 
                            backprop_timesteps=BACKPROP_TIMESTEPS)
    description_values_count = len(description_chars) + 2
    script_values_count = len(script_chars) + 2

    tf.reset_default_graph()
    tf.set_random_seed(1)
    with tf.Session() as session:
        print("\nCreating model...")
        model = Seq2SeqModel(encoder_cell=LSTMCell(HIDDEN_DIM),
                             decoder_cell=LSTMCell(HIDDEN_DIM),
                             encoder_vocab_size=description_values_count,
                             decoder_vocab_size=script_values_count,
                             encoder_embedding_size=description_values_count,
                             decoder_embedding_size=script_values_count,
                             attention=False,
                             bidirectional=False)
        session.run(tf.global_variables_initializer())
        loss_track = []
        try:
            for epoch in range(EPOCHS):
                randomize_insync(train_inputs, train_targets, 
                                 train_input_lengths, train_target_lengths)
                print("\nBatching data...")
                input_batches = []
                target_batches = []
                input_batch_lengths = []
                target_batch_lengths = []
                batch_count = (train_inputs.shape[1]+BATCH_SIZE-1) // BATCH_SIZE
                for i in range(batch_count):
                    input_batches.append(train_inputs[
                        :,i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                    target_batches.append(train_targets[
                        :,i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                    input_batch_lengths.append(train_input_lengths[
                        i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                    target_batch_lengths.append(train_target_lengths[
                        i*BATCH_SIZE:(i+1)*BATCH_SIZE])
                for i in range(batch_count):
                    feed_dict = model.make_train_inputs(
                        input_batches[i], input_batch_lengths[i], 
                        target_batches[i], target_batch_lengths[i])
                    print("Epoch", epoch, ": training on batch", i)
                    _, loss = session.run([model.train_op, model.loss], feed_dict)
                    print("Loss:", loss, "\n")
                    loss_track.append(loss)
        except KeyboardInterrupt:
            print("Training interrupted!")
        else:
            print("Training completed!")

        plotter.plot(loss_track)
        plotter.show()
        print("Final loss:", loss_track[-1])


if __name__ == '__main__':
    main()