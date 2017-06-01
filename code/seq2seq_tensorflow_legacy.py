"""
Sequence-to-sequence model implemented 
using tf.contrib.seq2seq from tensorflow 1.0.
"""

import os
from math import sqrt, isnan
from time import time

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
import data_processing_copy_task as dp_copy


COPY_TASK = False

# Hyperparameters
ENCODER_VOCAB_SIZE = 0
DECODER_VOCAB_SIZE = 0
HIDDEN_DIM = 64
LEARNING_RATE = 0.005
BACKPROP_TIMESTEPS = 50
BATCH_SIZE = 128
EPOCHS = 1

MODEL_DIR = os.path.join(os.pardir, "models")


class Seq2SeqModel():

    PAD = 0
    EOS = 1

    def __init__(self, encoder_cell, decoder_cell, encoder_vocab_size, 
                 decoder_vocab_size, encoder_embedding_size, decoder_embedding_size,
                 encoder_inputs_shape, encoder_inputs_length_shape,
                 decoder_targets_shape, decoder_targets_length_shape, 
                 bidirectional=False, attention=False):
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell

        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size

        self.encoder_inputs_shape = encoder_inputs_shape
        self.encoder_inputs_length_shape = encoder_inputs_length_shape
        self.decoder_targets_shape = decoder_targets_shape
        self.decoder_targets_length_shape = decoder_targets_length_shape

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
        # Inputs are batch-major
        self.inputs_initializer = tf.placeholder(
            shape=self.encoder_inputs_shape, 
            dtype=tf.int32, name="inputs_initializer")
        self.inputs_length_initializer = tf.placeholder(
            shape=self.encoder_inputs_length_shape, 
            dtype=tf.int32, name="inputs_length_initializer")

        # required for training, not for testing
        self.targets_initializer = tf.placeholder(
            shape=self.decoder_targets_shape, 
            dtype=tf.int32, name="targets_initializer")
        self.targets_length_initializer = tf.placeholder(
            shape=self.decoder_targets_length_shape, 
            dtype=tf.int32, name="targets_length_initializer")

        self.inputs = tf.Variable(
            self.inputs_initializer, trainable=False, collections=[])
        self.inputs_length = tf.Variable(
            self.inputs_length_initializer, trainable=False, collections=[])

        self.targets = tf.Variable(
            self.targets_initializer, trainable=False, collections=[])
        self.targets_length = tf.Variable(
            self.targets_length_initializer, trainable=False, collections=[])

        # input_, input_length, target, target_length = tf.train.slice_input_producer(
        #     [self.inputs, self.inputs_length, self.targets, self.targets_length], 
        #     num_epochs=EPOCHS, capacity=5000)
        (self.encoder_inputs, self.encoder_inputs_length, 
         self.decoder_targets, self.decoder_targets_length) = tf.train.batch(
           [self.inputs, self.inputs_length, self.targets, self.targets_length], 
            batch_size=BATCH_SIZE, num_threads=8, 
            capacity=self.encoder_inputs_shape[0], enqueue_many=True)

        # Switch to time-major
        self.encoder_inputs = tf.transpose(self.encoder_inputs, [1, 0])
        self.decoder_targets = tf.transpose(self.decoder_targets, [1, 0])

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

            decoder_fn_train = simple_decoder_fn_train(
                encoder_state=self.encoder_state)

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

    def initialize_train_inputs(self, session, inputs, inputs_length, 
                                targets, targets_length):
        session.run(self.inputs.initializer, 
                    feed_dict={self.inputs_initializer: inputs})
        session.run(self.inputs_length.initializer, 
                    feed_dict={self.inputs_length_initializer: inputs_length})
        session.run(self.targets.initializer, 
                    feed_dict={self.targets_initializer: targets})
        session.run(self.targets_length.initializer, 
                    feed_dict={self.targets_length_initializer: targets_length})

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


def plot_loss(loss_track):
    plotter.plot(loss_track)
    average_loss_track = [sum(loss_track[3*(i+1)//4 : i+1])/((i+4)//4)
                          for i in range(len(loss_track))]
    plotter.plot(average_loss_track)
    plotter.show()


def train_on_copy_task(session, length_from=3, length_to=8,
                       vocab_lower=2, vocab_upper=10,
                       batch_size=128, batches=2000,
                       verbose=True):
    print("\nFetching data...")
    sequences = dp_copy.random_sequences(length_from, length_to, vocab_lower, 
                                         vocab_upper, batch_size, batches)
    print("\nVectorizing data...")
    train_inputs, train_input_lengths = dp_copy.vectorize(sequences)
    print("\nCreating model...")
    model = Seq2SeqModel(encoder_cell=LSTMCell(HIDDEN_DIM),
                         decoder_cell=LSTMCell(HIDDEN_DIM),
                         encoder_vocab_size=vocab_upper-vocab_lower+2,
                         decoder_vocab_size=vocab_upper-vocab_lower+2,
                         encoder_embedding_size=vocab_upper-vocab_lower+2,
                         decoder_embedding_size=vocab_upper-vocab_lower+2,
                         encoder_inputs_shape=train_inputs.shape,
                         encoder_inputs_length_shape=(len(train_input_lengths),),
                         decoder_targets_shape=train_inputs.shape,
                         decoder_targets_length_shape=(len(train_input_lengths),),
                         attention=False,
                         bidirectional=False)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    model.initialize_train_inputs(session, train_inputs, train_input_lengths, 
                                  train_inputs, train_input_lengths)
    summary_writer = tf.summary.FileWriter(MODEL_DIR, session.graph)
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coordinator)
    loss_track = []
    try:
        batch = 0
        while not coordinator.should_stop():
            start_time = time()
            if batch % 500 != 0:
                _, loss = session.run([model.train_op, model.loss])
            else:
                _, loss, inputs, prediction = session.run([
                    model.train_op, model.loss, 
                    model.encoder_inputs, model.decoder_prediction_inference])
            duration = time() - start_time
            if verbose:
                if batch % 100 == 0:
                    print("batch:", batch, "-- loss:", loss, "-- duration:", duration)
                    summary_string = session.run(summary_op)
                    summary_writer.add_summary(summary_string, batch)
                    if batch % 500 == 0:
                        for i in range(2):
                            if batch != batches:
                                print("\nInput: ", inputs[:,i])
                                print("Output:", prediction[:,i])
            loss_track.append(loss)
            batch += 1
    except KeyboardInterrupt:
        print("Training interrupted!")
    except tf.errors.OutOfRangeError:
        print("Training completed!")
    finally:
        saver.save(session, os.path.join(MODEL_DIR, "copy_task"), 
                   global_step=batch)
        coordinator.request_stop()

    coordinator.join(threads)

    return loss_track


def main():
    if not COPY_TASK:
        print("\nFetching data...")
        data = dp.get_data()
        print("\nProcessing data...")
        processed_data = dp.process_data(data)
        description_chars = processed_data["description_char_counts"]
        script_chars = processed_data["python_char_counts"]
        training_data_dict = processed_data["training_data"]
        validation_data_dict = processed_data["validation_data"]
        test_data_dict = processed_data["test_data"]
        print("\nVectorizing data...")
        train_inputs, train_targets, train_input_lengths, train_target_lengths \
            = dp.vectorize_data(training_data_dict, description_chars, 
                                script_chars, backprop_timesteps=BACKPROP_TIMESTEPS,
                                description_vocab_size=ENCODER_VOCAB_SIZE, 
                                python_vocab_size=DECODER_VOCAB_SIZE)

        if ENCODER_VOCAB_SIZE:
            description_values_count = ENCODER_VOCAB_SIZE + 4
        else:
            description_values_count = len(description_chars) + 3
        if DECODER_VOCAB_SIZE:
            script_values_count = DECODER_VOCAB_SIZE + 4
        else:
            script_values_count = len(script_chars) + 3

    tf.reset_default_graph()
    tf.set_random_seed(1)
    with tf.Session() as session:
        if COPY_TASK:
            plot_loss(train_on_copy_task(session, length_from=3, length_to=8,
                                         vocab_lower=2, vocab_upper=10,
                                         batch_size=BATCH_SIZE))
            return
        else:
            print("\nCreating model...")
            model = Seq2SeqModel(encoder_cell=LSTMCell(HIDDEN_DIM),
                                 decoder_cell=LSTMCell(HIDDEN_DIM),
                                 encoder_vocab_size=description_values_count,
                                 decoder_vocab_size=script_values_count,
                                 encoder_embedding_size=description_values_count,
                                 decoder_embedding_size=script_values_count,
                                 encoder_inputs_shape=train_inputs.shape,
                                 encoder_inputs_length_shape=(len(train_input_lengths),),
                                 decoder_targets_shape=train_targets.shape,
                                 decoder_targets_length_shape=(len(train_target_lengths),),
                                 attention=False,
                                 bidirectional=False)
        summary_op = tf.summary.merge_all()
        saver = tf.train.Saver()
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())
        model.initialize_train_inputs(session, train_inputs, train_input_lengths, 
                                      train_targets, train_target_lengths)
        summary_writer = tf.summary.FileWriter(MODEL_DIR, session.graph)
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coordinator)
        loss_track = []
        try:
            batch = 0
            while not coordinator.should_stop():
                start_time = time()
                if batch % 500 != 0:
                    _, loss = session.run([model.train_op, model.loss])
                else:
                    _, loss, prediction = session.run([
                        model.train_op, model.loss, model.decoder_prediction_inference])
                duration = time() - start_time
                if batch % 100 == 0:
                    print("batch:", batch, "-- loss:", loss, "-- duration:", duration)
                    summary_string = session.run(summary_op)
                    summary_writer.add_summary(summary_string, batch)
                    if batch % 500 == 0:
                        print("\nGenerated script:")
                        generated_script = dp.devectorize(
                            prediction[:,0], "script", description_chars, script_chars)
                        print(generated_script)
                        print("Length:", len(generated_script))
                        print("Vector:", prediction[:,0])
                loss_track.append(loss)
                batch += 1
        except KeyboardInterrupt:
            print("Training interrupted!")
        except tf.errors.OutOfRangeError:
            print("Training completed!")
        finally:
            saver.save(session, os.path.join(MODEL_DIR, "desc2code_task"), 
                       global_step=batch)
            coordinator.request_stop()

        coordinator.join(threads)

        plot_loss(loss_track)
        print("Final loss:", loss_track[-1])


if __name__ == '__main__':
    main()