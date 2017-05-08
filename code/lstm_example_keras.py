"""
Example implementation of a vanilla LSTM network in Keras.
"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import (RMSprop, SGD, Adagrad, 
                              Adadelta, Adam, Adamax, Nadam)

import data_processing as dp


HIDDEN_DIM = 128
LEARNING_RATE = 0.01


def build_model(training_data, training_data_labels):
    print("Building model...")
    model = Sequential()
    model.add(LSTM(HIDDEN_DIM, input_shape=(training_data.shape[-1],)))
    model.add(Dense(training_data_labels.shape[-1]))
    model.add(Activation("softmax"))
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)


def sample(probability_array, temperature=1.0):
    predications = np.asarray(probability_array).astype("float64")
    predications = np.log(predications) / temperature
    exp_predictions = np.exp(predications)
    predications = exp_predictions / np.sum(exp_predictions)
    return np.random.multinomial(1, predications, 1)


def train(model, training_data, training_data_labels, 
          validation_data, validation_data_labels):
    print("Training model...")
    for iteration in range(1, 60):
        print("Iteration", iteration, "...")
        model.fit(training_data, training_data_labels, batch_size=128, epochs=1)
        training_loss = model.evaluate(
            training_data, training_data_labels, batch_size=1)
        print("Training loss:", training_loss)
    print("Training stopped.\n")
    validation_loss = model.evaluate(
        validation_data, validation_data_labels, batch_size=1)
    print("Validation loss:", validation_loss)


def main():
    data = dp.get_data()
    processed_data = dp.process_data(data)
    vectorized_data = dp.vectorize_data(processed_data)
    model = build_model(vectorized_data[0], vectorized_data[1])
    train(model, vectorized_data[0], vectorized_data[1], 
          vectorized_data[2], vectorized_data[3])


if __name__ == '__main__':
    main()

