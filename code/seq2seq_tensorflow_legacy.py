"""
Sequence-to-sequence model implemented 
using tf.contrib.seq2seq from tensorflow 1.0.
"""

import os
import sys
import re
from math import sqrt, isnan
from time import time
from datetime import datetime
from random import shuffle

import numpy as np
import matplotlib.pyplot as plotter
import tensorflow as tf
from tensorflow.contrib.rnn import (LSTMCell, MultiRNNCell, LSTMStateTuple, 
                                    DropoutWrapper)
from tensorflow.contrib.seq2seq import (dynamic_rnn_decoder, simple_decoder_fn_train, 
                                        simple_decoder_fn_inference, prepare_attention,
                                        attention_decoder_fn_train, sequence_loss,
                                        attention_decoder_fn_inference)

import data_processing as dp
from data_processing import (PAD_VALUE, MAX_DESCRIPTION_LENGTH, 
                             MAX_SCRIPT_LENGTH)
import data_processing_copy_task as dp_copy
import training_storage as storage
import helper_functions as helper


COPY_TASK = False

# Hyperparameters
ENCODER_VOCAB_SIZE = 0
DECODER_VOCAB_SIZE = 0
HIDDEN_DIM = 200
LEARNING_RATE = 0.005
BACKPROP_TIMESTEPS = 50
BATCH_SIZE = 16
EPOCHS = 50
OPTIMIZER = tf.train.RMSPropOptimizer(LEARNING_RATE)

MODEL_DIR = os.path.join(os.pardir, "models")


class Seq2SeqModel():

    PAD = 0
    EOS = 1

    def __init__(self, encoder_cell=None, decoder_cell=None, encoder_vocab_size=None, 
                 decoder_vocab_size=None, encoder_embedding_size=None, 
                 decoder_embedding_size=None, hidden_dim=HIDDEN_DIM, 
                 batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, 
                 optimizer=OPTIMIZER, bidirectional=False, attention=False, 
                 layers=1, model_dir=MODEL_DIR, graph=None, model=None):
        self.graph = graph
        if self.graph:
            self.model_id = model.model_id
            self.model_graph_file = model.model_graph_file

            self.description = model.description
            self.input_description = model.input_type
            self.output_description = model.output_type

            self.encoder_cell = model.encoder_cell
            self.decoder_cell = model.decoder_cell
            self.layers = model.layers
            self.bidirectional = model.bidirectional_encoder
            self.attention = model.attention

            self.hidden_dim = model.hidden_dimension
            self.encoder_vocab_size = model.encoder_vocab_size
            self.decoder_vocab_size = model.decoder_vocab_size
            self.encoder_embedding_size = model.encoder_embedding_size
            self.decoder_embedding_size = model.decoder_embedding_size

            self.batch_size = model.batch_size
            self.truncated_backprop = model.truncated_backprop
            self.learning_rate = model.learning_rate
            self.optimizer = model.optimization_algorithm
            if "RMSProp" in self.optimizer:
                self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

            self._init_placeholders()
        else:
            if layers == 1:
                self.encoder_cell = encoder_cell
                self.decoder_cell = decoder_cell
            else:
                self.encoder_cell = MultiRNNCell(
                    [encoder_cell for i in range(layers)])
                self.decoder_cell = MultiRNNCell(
                    [decoder_cell for i in range(layers)])

            self.description = ("Sequence-to-sequence recurrent neural network model" 
                                " that generates Python scripts from programming" 
                                " challenge descriptions")
            self.input_description = "sequence of characters"
            self.output_description = "sequence of characters"

            self.hidden_dim = hidden_dim
            self.encoder_vocab_size = encoder_vocab_size
            self.decoder_vocab_size = decoder_vocab_size
            self.encoder_embedding_size = encoder_embedding_size
            self.decoder_embedding_size = decoder_embedding_size

            self.layers = layers
            self.bidirectional = bidirectional
            self.attention = attention

            self.batch_size = batch_size
            self.truncated_backprop = False
            self.learning_rate = learning_rate
            self.optimizer = optimizer

            self._make_graph()
            self._store_model_architecture(model_dir)

        self.saver = tf.train.Saver()

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
        if self.graph:
            self.encoder_inputs = self.graph.get_tensor_by_name(
                "encoder_inputs:0")
            self.encoder_inputs_length = self.graph.get_tensor_by_name(
                "encoder_inputs_length:0")

            # Required for training, not for testing
            self.decoder_targets = self.graph.get_tensor_by_name(
                "decoder_targets:0")
            self.decoder_targets_length = self.graph.get_tensor_by_name(
                "decoder_targets_length:0")
        else:
            # Everything is time-major
            self.encoder_inputs = tf.placeholder(
                shape=(None, None), dtype=tf.int32, name="encoder_inputs")
            self.encoder_inputs_length = tf.placeholder(
                shape=(None,), dtype=tf.int32, name="encoder_inputs_length")

            # Required for training, not for testing
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
            self.decoder_train_targets = tf.identity(
                decoder_train_targets, name="decoder_train_targets")

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
                self.encoder_state = tf.concat(
                    (encoder_fw_state, encoder_bw_state),
                    1, name="bidirectional_concat")
            elif isinstance(encoder_fw_state, tuple):
                self.encoder_state = [None, None, None]
                for i in range(len(encoder_fw_state)):
                    encoder_fw_state_part = encoder_fw_state[i]
                    encoder_bw_state_part = encoder_bw_state[i]
                    if isinstance(encoder_fw_state_part, LSTMStateTuple):
                        encoder_state_c = tf.concat(
                            (encoder_fw_state_part.c, encoder_bw_state_part.c), 
                            1, name="bidirectional_concat_c")
                        encoder_state_h = tf.concat(
                            (encoder_fw_state_part.h, encoder_bw_state_part.h), 
                            1, name="bidirectional_concat_h")
                        encoder_state_part = LSTMStateTuple(
                            c=encoder_state_c, h=encoder_state_h)
                    elif isinstance(encoder_fw_state_part, tf.Tensor):
                        encoder_state_part = tf.concat(
                            (encoder_fw_state_part, encoder_bw_state_part),
                            1, name="bidirectional_concat")
                    self.encoder_state[i] = encoder_state_part
                self.encoder_state = tuple(self.encoder_state)

    def _init_decoder(self):
        with tf.variable_scope("Decoder") as scope:
            def output_fn(outputs):
                return tf.contrib.layers.linear(
                    outputs, self.decoder_vocab_size, scope=scope)

            if not self.attention:
                decoder_fn_train = simple_decoder_fn_train(
                    encoder_state=self.encoder_state)
                decoder_fn_inference = simple_decoder_fn_inference(
                    output_fn=output_fn, encoder_state=self.encoder_state,
                    embeddings=self.decoder_embedding_matrix, 
                    start_of_sequence_id=self.EOS, end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.decoder_vocab_size)
            else:
                # attention_states: size [batch_size, max_time, num_units]
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])
                (attention_keys, attention_values, attention_score_fn, 
                 attention_construct_fn) = prepare_attention(
                    attention_states=attention_states,
                    attention_option="bahdanau",
                    num_units=self.decoder_hidden_units)
                decoder_fn_train = attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name="attention_decoder")
                decoder_fn_inference = attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.decoder_embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.decoder_vocab_size)

            (self.decoder_outputs_train, self.decoder_state_train, 
             self.decoder_context_state_train) = dynamic_rnn_decoder(
                cell=self.decoder_cell, decoder_fn=decoder_fn_train,
                inputs=self.decoder_train_inputs_embedded, 
                sequence_length=self.decoder_train_length,
                time_major=True, scope=scope)

            self.decoder_logits_train = tf.identity(
                output_fn(self.decoder_outputs_train), name="decoder_logits_train")
            self.decoder_prediction_train = tf.argmax(
                self.decoder_logits_train, axis=-1, name="decoder_prediction_train")

            scope.reuse_variables()

            (self.decoder_logits_inference, self.decoder_state_inference, 
             self.decoder_context_state_inference) = dynamic_rnn_decoder(
                cell=self.decoder_cell, decoder_fn=decoder_fn_inference,
                time_major=True, scope=scope)
            self.decoder_logits_inference = tf.identity(
                self.decoder_logits_inference, name="decoder_logits_inference")
            self.decoder_prediction_inference = tf.argmax(
                self.decoder_logits_inference, axis=-1, 
                name="decoder_prediction_inference")

    def _init_optimizer(self):
        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = sequence_loss(logits=logits, targets=targets,
                                  weights=self.loss_weights, name="sequence_loss")
        self.train_op = self.optimizer.minimize(self.loss, name="train_op")

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

    def _store_model_architecture(self, base_directory):
        # Store in database
        self.model_id = storage.store_model_info(
            self.description, self.input_description, self.output_description,
            str(type(self.encoder_cell)), str(type(self.decoder_cell)), 
            self.layers, self.bidirectional, self.attention, self.hidden_dim, 
            self.encoder_vocab_size, self.decoder_vocab_size, 
            self.encoder_embedding_size, self.decoder_embedding_size,
            self.batch_size, self.truncated_backprop, self.learning_rate, 
            str(type(self.optimizer)))

        # then store in text file
        utcnow_string = str(datetime.utcnow().replace(microsecond=0))
        utcnow_string = utcnow_string.replace(" ", "_").replace(":", "-")
        self.model_dir = os.path.join(base_directory, 
                                      "model_trained_on_" + utcnow_string)
        print("\nModel info/data directory:", self.model_dir)   # For logging purposes
        os.makedirs(self.model_dir)
        with open(os.path.join(self.model_dir, "model_definition.txt"), "w") as file:
            print(self.description, file=file)
            print("\nInput:", self.input_description, file=file)
            print("Output:", self.output_description, file=file)
            print("\nEncoder cell:", type(self.encoder_cell), file=file)
            print("Decoder cell:", type(self.decoder_cell), file=file)
            print("Hidden dimension:", self.hidden_dim, file=file)
            print("\nEncoder vocabulary size:", self.encoder_vocab_size, file=file)
            print("Decoder vocabulary size:", self.decoder_vocab_size, file=file)
            print("\nEncoder embedding size:", 
                  self.encoder_embedding_size, file=file)
            print("Decoder embedding size:", 
                  self.decoder_embedding_size, file=file)
            print("\nLayers:", self.layers, file=file)
            print("Bidirectional encoder:", self.bidirectional, file=file)
            print("Attention:", self.attention, file=file)
            print("\nBatch size:", self.batch_size, file=file)
            print("Truncated backprop:", self.truncated_backprop, file=file)
            print("Learning rate:", self.learning_rate, file=file)
            print("\nOptimization algorithm:", type(self.optimizer), file=file)

    def save(self, session, step):
        self.model_save_path = (
            os.path.join(self.model_dir, "desc2code_model") 
            + "-" + str(step) + ".meta")
        storage.store_model_graph_file(self.model_id, self.model_save_path)
        checkpoint_path = self.saver.save(
            session, os.path.join(self.model_dir, "desc2code_model"), 
            global_step=step)
        return checkpoint_path

    def load(self):
        self.decoder_train_targets = self.graph.get_tensor_by_name(
            "DecoderTrainFeeds/decoder_train_targets:0")
        self.decoder_logits_train = self.graph.get_tensor_by_name(
            "Decoder/decoder_logits_train:0")
        self.decoder_prediction_train = self.graph.get_tensor_by_name(
            "Decoder/decoder_prediction_train:0")
        self.decoder_logits_inference = self.graph.get_tensor_by_name(
            "Decoder/decoder_logits_inference:0")
        self.decoder_prediction_inference = self.graph.get_tensor_by_name(
            "Decoder/decoder_prediction_inference:0")
        self.loss_weights = self.graph.get_tensor_by_name(
            "DecoderTrainFeeds/loss_weights:0")
        # self._init_optimizer()

    def infer(self, session, input_array, target_array):
        start_time = time()
        if input_array.ndim == 1:
            # Assumes row vector input
            input_array = np.reshape(input_array, (-1, 1))
        elif input_array.shape[0] == 1:
            input_array = np.transpose(input_array)
        if target_array.ndim == 1:
            # Assumes row vector target
            target_array = np.reshape(target_array, (-1, 1))
        elif target_array.shape[0] == 1:
            target_array = np.transpose(target_array)
        feed_dict = self.make_train_inputs(input_array, [input_array.shape[0]], 
                                           target_array, [target_array.shape[0]])
        prediction_inference, prediction_train = session.run(
            [self.decoder_prediction_inference, self.decoder_prediction_train], 
            feed_dict)
        duration = time() - start_time
        return prediction_inference, prediction_train, duration


def train_on_copy_task(session, length_from=3, length_to=8, vocab_lower=2, 
                       vocab_upper=10, batch_size=128, batches=2000,
                       hidden_dim=HIDDEN_DIM, verbose=True):
    print("\nFetching data...")
    sequences = dp_copy.random_sequences(length_from, length_to, vocab_lower, 
                                         vocab_upper, batch_size, batches)
    print("\nVectorizing data...")
    train_inputs, train_input_lengths = dp_copy.vectorize(sequences)
    print("\nCreating model...")
    model = Seq2SeqModel(encoder_cell=LSTMCell(hidden_dim),
                         decoder_cell=LSTMCell(hidden_dim),
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
    summary_writer = tf.summary.FileWriter(
        os.path.join(MODEL_DIR, "copy_task"), session.graph)
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
        print("\nTraining interrupted!")
    except tf.errors.OutOfRangeError:
        print("\nTraining completed!")
    finally:
        saver.save(session, os.path.join(MODEL_DIR, "copy_task"), 
                   global_step=batch)
        coordinator.request_stop()

    coordinator.join(threads)

    return loss_track


def train_on_desc2code_task(session, training_data, validation_data, 
                            description_chars, script_chars, layers=1, 
                            hidden_dim=HIDDEN_DIM, attention=False, 
                            bidirectional=False, batch_size=BATCH_SIZE, 
                            epochs=EPOCHS, learning_rate=LEARNING_RATE, 
                            optimizer=OPTIMIZER):
    model_dir = os.path.join(MODEL_DIR, "desc2code_task")

    print("\nVectorizing data...")
    description_arrays, script_arrays = dp.vectorize_examples(
        training_data, description_chars, script_chars,
        description_vocab_size=ENCODER_VOCAB_SIZE, 
        python_vocab_size=DECODER_VOCAB_SIZE)
    description_values_count, script_values_count = helper.get_vocab_sizes(
        description_chars, script_chars, ENCODER_VOCAB_SIZE, 
        ENCODER_VOCAB_SIZE)

    if bidirectional:
        encoder_cell = LSTMCell(hidden_dim//2)
    else:
        encoder_cell = LSTMCell(hidden_dim)

    print("\nCreating model...")
    model = Seq2SeqModel(encoder_cell=encoder_cell,
                         decoder_cell=LSTMCell(hidden_dim),
                         encoder_vocab_size=description_values_count,
                         decoder_vocab_size=script_values_count,
                         encoder_embedding_size=description_values_count,
                         decoder_embedding_size=script_values_count,
                         layers=layers,
                         hidden_dim=hidden_dim, 
                         batch_size=batch_size,
                         learning_rate=learning_rate,
                         optimizer=optimizer,
                         attention=attention, 
                         bidirectional=bidirectional,
                         model_dir=model_dir)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    start_training_timestamp = datetime.utcnow()
    save_path = None
    train_loss_track = []
    validation_loss_track = []
    try:
        step = 0
        for epoch in range(epochs):
            print("\nBatching data...")
            shuffle(description_arrays)
            shuffle(script_arrays)
            train_inputs, train_targets, train_input_lengths, train_target_lengths \
                = dp.make_batches(description_arrays, script_arrays, batch_size)
            for batch in range((len(script_arrays) - 1)//batch_size + 1):
                start_time = time()
                feed_dict = model.make_train_inputs(
                    train_inputs[batch], train_input_lengths[batch], 
                    train_targets[batch], train_target_lengths[batch])
                if step % 10 == 0:
                    _, loss, prediction_train, prediction_inference = session.run(
                        [model.train_op, model.loss, 
                         model.decoder_prediction_train,
                         model.decoder_prediction_inference], feed_dict)
                else:
                    _, loss = session.run(
                        [model.train_op, model.loss], feed_dict)
                duration  = time() - start_time
                print("\nepoch:", epoch, "-- batch:", batch, "-- loss:", loss, 
                      "-- duration:", duration)
                if step % 10 == 0:
                    print("\nGenerated Script Train:")
                    generated_script_train = dp.devectorize(
                        prediction_train[:,0], "script", 
                        description_chars, script_chars)
                    print(generated_script_train.rstrip())
                    print("Trailing whitespace:", len(generated_script_train) 
                          - len(generated_script_train.rstrip()))
                    print("\nGenerated Script Inference:")
                    generated_script_inference = dp.devectorize(
                        prediction_inference[:,0], "script", 
                        description_chars, script_chars)
                    print(generated_script_inference.rstrip())
                    print("Trailing whitespace:", len(generated_script_inference) 
                          - len(generated_script_inference.rstrip()))
                train_loss_track.append((loss, datetime.utcnow(), epoch, batch))
                step += 1
            print("\nEvaluating on validation dataset...")
            (validation_loss, validate_targets, 
             validate_logits, validate_prediction) = (
                validate_on_desc2code_task(session, model, validation_data, 
                                           description_chars, script_chars))
            generated_script_validation = dp.devectorize(
                    validate_prediction[:,1], "script", 
                    description_chars, script_chars)
            print("\nValidation Greedy Prediction:\n", 
                  generated_script_validation.rstrip())
            target_script = dp.devectorize(
                    validate_targets[:,1], "script", 
                    description_chars, script_chars)
            print("\nTarget Script:\n", target_script.rstrip())
            generated_script_validation = dp.devectorize(
                    validate_prediction[:,-2], "script", 
                    description_chars, script_chars)
            print("\nValidation Greedy Prediction:\n", 
                  generated_script_validation.rstrip())
            target_script = dp.devectorize(
                    validate_targets[:,-2], "script", 
                    description_chars, script_chars)
            print("\nTarget Script:\n", target_script.rstrip())
            print("\nValidation loss:", validation_loss)
            validation_loss_track.append(
                (validation_loss, datetime.utcnow(), epoch, batch))
            if (min(eval_step[0] for eval_step in validation_loss_track) 
                    == validation_loss):
                print("New best validation loss.")
                save_path = model.save(session, step)
                print("Saved model at", save_path)
            if helper.early_stop(validation_loss_track):
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\nTraining interrupted!")
    else:
        print("\nTraining completed!")


    return (model, train_loss_track, validation_loss_track, 
            save_path, start_training_timestamp)


def validate_on_desc2code_task(session, model, validation_data, 
                               description_chars, script_chars):
    print("\nVectorizing data...")
    description_arrays, script_arrays = dp.vectorize_examples(
        validation_data, description_chars, script_chars,
        description_vocab_size=ENCODER_VOCAB_SIZE, 
        python_vocab_size=DECODER_VOCAB_SIZE)
    validation_input_lengths, validation_target_lengths = [], []
    for desc_array, script_array in zip(description_arrays, script_arrays):
        validation_input_lengths.append(desc_array.shape[1])
        validation_target_lengths.append(script_array.shape[1])

    description_values_count, script_values_count = helper.get_vocab_sizes(
        description_chars, script_chars, ENCODER_VOCAB_SIZE, 
        ENCODER_VOCAB_SIZE)

    validation_inputs, validation_targets = dp.merge_arrays(
        description_arrays, script_arrays)

    start_time = time()
    feed_dict = model.make_train_inputs(
        validation_inputs, validation_input_lengths, 
        validation_targets, validation_target_lengths)
    loss, logits, prediction = session.run(
        [model.loss, model.decoder_logits_inference, 
         model.decoder_prediction_inference], feed_dict)
    duration = time() - start_time

    return loss, validation_targets, logits, prediction


def load_model(session, stored_model, stored_run):
    saver = tf.train.import_meta_graph(stored_model.model_graph_file)
    if not stored_run.model_parameters_file:
        model_parameters_file = input("Model parameters file path: ").strip()
        saver.restore(session, model_parameters_file)
    else:
        saver.restore(session, stored_run.model_parameters_file)
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    graph = tf.get_default_graph()
    model = Seq2SeqModel(graph=graph, model=stored_model)
    model.load()

    return model


def perform_inference(session, model, input_strings, target_strings, 
                      description_chars, script_chars):
    vectorized_inputs = []
    vectorized_targets = []
    for input_string, target_string in zip(input_strings, target_strings):
        input_array, target_array = dp.vectorize_example(
            input_string, target_string, description_chars, script_chars)
        vectorized_inputs.append(input_array)
        vectorized_targets.append(target_array)

    predictions = []
    total_duration = 0
    for i, (input_array, target_array) in enumerate(zip(
            vectorized_inputs, vectorized_targets)):
        prediction_inference, prediction_train, duration = model.infer(
            session, input_array, target_array)
        prediction_inference_text = dp.devectorize(
            prediction_inference, "script", description_chars, script_chars)
        prediction_train_text = dp.devectorize(
            prediction_train, "script", description_chars, script_chars)
        with open("temp_inference_file.txt", "a") as infer_file:
            print("Prediction Inference:", file=infer_file)
            print(prediction_inference_text, file=infer_file)
            print("\n", file=infer_file)
            print("Prediction Train:", file=infer_file)
            print(prediction_train_text, file=infer_file)
            print("\n", file=infer_file)
            print("Target:", file=infer_file)
            print(target_strings[i], file=infer_file)
            print("\n\n\n", file=infer_file)
        predictions.append(prediction_inference_text)
        total_duration += duration

    return predictions, total_duration


def infer_using_model(stored_model=None, stored_run=None, model=None, session=None):
    print("\nFetching data...")
    data = dp.get_data()
    print("\nProcessing data...")
    processed_data = dp.process_data(data)
    description_chars = processed_data["description_char_counts"]
    script_chars = processed_data["python_char_counts"]
    input_strings = []
    target_strings = []
    for i, description in enumerate(processed_data["validation_data"]):
        if i < 5:
            script = processed_data["validation_data"][description]["python"][0]
            input_strings.append(description)
            target_strings.append(script)

    if session is None or model is None:
        tf.reset_default_graph()
        tf.set_random_seed(1)
        with tf.Session() as session:
            model = load_model(session, stored_model, stored_run)
            predictions, duration = perform_inference(
                session, model, input_strings, target_strings,
                description_chars, script_chars)
    else:
        predictions, duration = perform_inference(
            session, model, input_strings, target_strings,
            description_chars, script_chars)

    for input_string, target_string, prediction in zip(
            input_strings, target_strings, predictions):
        storage.store_model_run(prediction, input_text=input_string, 
                                target_text=target_string, model=model)


def train_model(plot_losses=True, training_description_count=0, layers=1,
                hidden_dim=HIDDEN_DIM, attention=False, bidirectional=False, 
                batch_size=BATCH_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE, 
                optimizer=OPTIMIZER):
    if not COPY_TASK:
        print("\nFetching data...")
        data = dp.get_data()
        print("\nProcessing data...")
        processed_data = dp.process_data(data)
        description_chars = processed_data["description_char_counts"]
        script_chars = processed_data["python_char_counts"]
        training_data_dict = processed_data["training_data"]
        training_data_dict = helper.truncate_data(
            training_data_dict, training_description_count)
        validation_data_dict = processed_data["validation_data"]
        test_data_dict = processed_data["test_data"]
        (input_sequences, average_input_length, smallest_input_length, 
         largest_input_length, output_sequences, average_output_length,
         smallest_output_length, largest_output_length) = dp.get_dataset_statistics(
            training_data_dict, validation_data_dict, test_data_dict)

    tf.reset_default_graph()
    tf.set_random_seed(1)
    with tf.Session() as session:
        if COPY_TASK:
            loss_track = train_on_copy_task(session, length_from=3, length_to=8,
                                            vocab_lower=2, vocab_upper=10,
                                            batch_size=batch_size)
        else:
            (model, train_loss_track, validation_loss_track, 
             parameters_file_path, start_training_timestamp) = train_on_desc2code_task(
                session, training_data_dict, validation_data_dict, 
                description_chars, script_chars, layers=layers, hidden_dim=hidden_dim, 
                attention=attention, bidirectional=bidirectional,
                batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, 
                optimizer=optimizer)
            run_id = storage.store_training_run(
                model.model_id, parameters_file_path, start_training_timestamp,
                input_sequences, average_input_length, smallest_input_length,
                largest_input_length, output_sequences, average_output_length,
                smallest_output_length, largest_output_length)
            validation_loss_track_expanded = []
            if validation_loss_track:
                expansion_factor = len(train_loss_track) // len(validation_loss_track)
                for eval_step in validation_loss_track:
                    validation_loss_track_expanded.extend([eval_step]*expansion_factor)
            storage.store_evaluation_track(run_id, "training", train_loss_track)
            storage.store_evaluation_track(run_id, "validation", validation_loss_track)
        if plot_losses:
            helper.plot_loss(train_loss_track, validation_loss_track_expanded,
                             labels=["Training loss", "Validation loss"])
            plotter.show()
        print("\nFinal training loss:", train_loss_track[-1][0])
        if validation_loss_track:
            print("Best validation loss:", 
                  min(eval_step[0] for eval_step in validation_loss_track))

        # Test inference
        infer_using_model(model=model, session=session)


if __name__ == '__main__':
    action = "train"
    if action == "train":
        log_file_name = "train_run_log_default2"
        with open(log_file_name + ".txt", "w") as log_file:
            sys.stdout = log_file
            train_model(plot_losses=False)
        # configs = [(3, True, False), (3, True, True)]
        # for i, (layers, bidirectional, attention) in enumerate(configs):
        #     log_file_name = "train_run_log"
        #     if bidirectional:
        #         log_file_name += "_bidirectional_encoder"
        #     if attention:
        #         log_file_name += "_attention"
        #     log_file_name += "_layers-" + str(layers)
        #     with open(log_file_name + ".txt", "w") as log_file:
        #         sys.stdout = log_file
        #         train_model(plot_losses=False, bidirectional=bidirectional, 
        #                     attention=attention, layers=layers)
    elif action == "infer":
        timestamp = datetime.utcnow()
        timestamp = timestamp.replace(day=timestamp.day-2)
        stored_models = storage.get_model_info(
            timestamp=timestamp, batch_size=16, 
            attention=False, bidirectional_encoder=False)
        stored_run = storage.get_latest_training_run(model=stored_models[-1])
        infer_using_model(stored_models[-1], stored_run)