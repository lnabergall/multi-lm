"""
Example implementation of a vanilla LSTM network in Keras.
"""

from statistics import mean
from random import shuffle
from collections import OrderedDict
from datetime import datetime

import numpy as np
from keras.models import Sequential
from keras.layers import (Dense, Activation, LSTM, Masking,
                          RepeatVector, TimeDistributed)
from keras.optimizers import (RMSprop, SGD, Adagrad, 
                              Adadelta, Adam, Adamax, Nadam)
from keras.callbacks import TensorBoard

import data_processing as dp
from data_processing import (PAD_VALUE, MAX_DESCRIPTION_LENGTH, 
                             MAX_SCRIPT_LENGTH)


HIDDEN_DIM = 128
LEARNING_RATE = 0.005
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


def get_output_masks(data_labels_array):
    output_masks = np.zeros((data_labels_array.shape[0], 
                             data_labels_array.shape[1]), dtype=np.float32)
    for i in range(data_labels_array.shape[0]):
        for j in range(data_labels_array.shape[1]):
            if np.any(data_labels_array[i, j, :]):
                output_masks[i, j] = 1
    return output_masks


def build_model(input_dim, output_dim):
    model = Sequential()
    model.add(Masking(mask_value=dp.PAD_VALUE, 
                      input_shape=(BACKPROP_TIMESTEPS, input_dim)))
    model.add(LSTM(HIDDEN_DIM, input_dim=input_dim, return_sequences=False))
    model.add(RepeatVector(dp.MAX_SCRIPT_LENGTH))
    model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(output_dim)))
    model.add(TimeDistributed(Activation("softmax")))
    optimizer = RMSprop(lr=LEARNING_RATE, clipnorm=5)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, 
                  sample_weight_mode="temporal")
    model.summary()
    return model


def save_model(model):
    model_json = model.to_json()
    datetime_string = str(datetime.utcnow()).replace(":", "-")
    datetime_string = datetime_string.replace(".", "-").replace(" ", "_")
    with open("../models/model_" + datetime_string + ".json", "w") as model_file:
        print(model_json, file=model_file)
    model.save_weights("../models/model_weights_" + datetime_string + ".h5")


def sample(probability_array, temperature=1.0):
    predictions = np.asarray(probability_array).astype("float64")
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    return np.argmax(np.random.multinomial(1, predictions, 1))


def evaluate_model(model, data_dict, description_chars, script_chars):
    data, data_labels = [], []
    for description, solution_dict in data_dict.items():
        script_arrays = []
        for i, script in enumerate(solution_dict["python"]):
            description_array, script_array = dp.vectorize_example(
                description, script, description_chars, script_chars, 
                as_collection=True)
            script_arrays.append(script_array)
            if i == 0:
                data.append(description_array)
        data_labels.append(script_arrays)

    loss, count = 0, 0
    for desc_array, script_arrays in zip(data, data_labels):
        for script_array in script_arrays:
            loss += model.evaluate(desc_array, script_array)
            count += 1

    return loss / count


def generate_output(model, description_array, diversity, script_chars):
    script_indices_char = {i: char for i, char in enumerate(script_chars)}
    predictions = model.predict(description_array)
    output = ""
    for i in range(predictions.shape[1]):
        next_index = sample(predictions[0,i,:], diversity)
        next_char = script_indices_char[next_index]
        output += next_char
    return output


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

    print("\nBuilding model...")
    model = build_model(len(description_chars), len(script_chars))
    tensorboard_callback = TensorBoard(
        log_dir='../models/tensorboard', histogram_freq=1, 
        write_graph=True, write_images=True)
    try:
        for iteration in range(1, 60):
            training_data_dict = shuffle_dict(training_data_dict)
            for index in range(SUPER_BATCHES):
                print("\n\n-------- Iteration " + str(iteration) 
                      + ", Super Batch " + str(index) + "...")
                print("Vectorizing data...")
                description_count = len(training_data_dict.keys())
                super_batch_dict = {desc: training_data_dict[desc] 
                    for i, desc in enumerate(training_data_dict) 
                    if i in range((index*description_count)//SUPER_BATCHES, 
                                  ((index+1)*description_count)//SUPER_BATCHES + 1)}
                super_batch, super_batch_labels = dp.vectorize_data(
                    super_batch_dict, description_chars, 
                    script_chars, BACKPROP_TIMESTEPS)
                output_masks = get_output_masks(super_batch_labels)
                print("Training model...")
                model.fit(super_batch, super_batch_labels, batch_size=BATCH_SIZE, 
                          epochs=1, sample_weight=output_masks, 
                          callbacks=[tensorboard_callback])
                print("\n---- Sample Output:")
                description = list(validation_data_dict.keys())[-1]
                print("-- Description:")
                print(description)
                for diversity in [0.2, 0.4, 0.6, 1.0]:
                    print("\n-- diversity:", diversity)
                    script = validation_data_dict[description]["python"][0]
                    description_array, _ = dp.vectorize_example(
                        description, script, description_chars, 
                        script_chars, as_collection=True)
                    generated_script = generate_output(
                        model, description_array, diversity, script_chars)
                    print("-- Generated Script:")
                    print(generated_script)
                print("\nCalculating training loss...")
                training_loss = evaluate_model(
                    model, training_data_dict, description_chars, script_chars)
                print("Training loss:", training_loss)
    except KeyboardInterrupt:
        save_model(model)
        print("Model saved!")
        raise
    print("\nTraining stopped.")

    validation_loss = evaluate_model(
        model, validation_data_dict, description_chars, script_chars)
    print("Validation loss:", validation_loss)


if __name__ == '__main__':
    main()

