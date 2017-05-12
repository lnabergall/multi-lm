"""
Example implementation of a vanilla LSTM network in Keras.
"""

from random import shuffle
from collections import OrderedDict

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, RepeatVector
from keras.optimizers import (RMSprop, SGD, Adagrad, 
                              Adadelta, Adam, Adamax, Nadam)

import data_processing as dp


HIDDEN_DIM = 128
LEARNING_RATE = 0.01
BACKPROP_TIMESTEPS = 50
BATCH_SIZE = 128
SUPER_BATCHES = 50


def shuffle_dict(dictionary):
    keys = list(dictionary.keys())
    shuffle(keys)
    shuffled_dictionary = {}
    for key in keys:
        shuffled_dictionary[key] = dictionary[key]
    return shuffled_dictionary


def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, input_shape=(
        BACKPROP_TIMESTEPS, input_dim), return_sequences=False))
    model.add(RepeatVector(BACKPROP_TIMESTEPS))
    model.add(LSTM(HIDDEN_DIM, input_shape=(
        BACKPROP_TIMESTEPS, HIDDEN_DIM), return_sequences=True))
    model.add(Dense(output_dim))
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model


def sample(probability_array, temperature=1.0):
    predications = np.asarray(probability_array).astype("float64")
    predications = np.log(predications) / temperature
    exp_predictions = np.exp(predications)
    predications = exp_predictions / np.sum(exp_predictions)
    return np.random.multinomial(1, predications, 1)


def main():
    print("\nFetching data...")
    data = dp.get_data()
    print("\nProcessing data...")
    processed_data = dp.process_data(data, BACKPROP_TIMESTEPS)
    description_chars = processed_data["description_chars"]
    script_chars = processed_data["python_chars"]
    training_data_dict = OrderedDict(processed_data["training_data"])
    validation_data_dict = processed_data["validation_data"]
    test_data_dict = processed_data["test_data"]
    # training_data, training_data_labels = dp.vectorize_data(
    #     training_data_dict, description_chars, script_chars)
    print("\nBuilding model...")
    model = build_model(len(description_chars), len(script_chars))
    for iteration in range(1, 60):
        training_data_dict = shuffle_dict(training_data_dict)
        for index in range(SUPER_BATCHES):
            print("\nVectorizing data...")
            description_count = len(training_data_dict.keys())
            super_batch_dict = {desc: training_data_dict[desc] 
                for i, desc in enumerate(training_data_dict) 
                if i in range((index*description_count)//SUPER_BATCHES, 
                              ((index+1)*description_count)//SUPER_BATCHES + 1)}
            super_batch, super_batch_labels = dp.vectorize_data(
                super_batch_dict, description_chars, script_chars, BACKPROP_TIMESTEPS)
            print("Training model...")
            model.fit(super_batch, super_batch_labels, batch_size=BATCH_SIZE, epochs=1)
            # training_loss = model.evaluate(
            #     training_data, training_data_labels, batch_size=1)
            # print("Training loss:", training_loss)
    print("\nTraining stopped.")
    # print("\nVectorizing data...")
    # validation_data, validation_data_labels = dp.vectorize_data(
    #     validation_data_dict, description_chars, script_chars)
    # validation_loss = model.evaluate(
    #     validation_data, validation_data_labels, batch_size=1)
    # print("Validation loss:", validation_loss)


if __name__ == '__main__':
    main()

