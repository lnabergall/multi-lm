"""Some functions for processing log files."""

import re
import math

import numpy as np

from seq2seq_tensorflow_legacy import plot_loss


def sigmoid(number):
    return 1 / (1 + math.exp(-number))


def get_numbers(string, integers=False):
    if integers:
        numbers = re.findall(r"\d+", string)
    else:
        numbers = re.findall(r"-?\d+\.\d+(?:e[+-]\d\d)?", string)
    if not numbers:
        raise NotImplementedError
    else:
        return [float(number) for number in numbers]


def get_number(string):
    return get_numbers(string)[0]


def compactify_log(log_file_name):
    include = True
    with open(log_file_name[:-4] + "_compact.txt", "w") as compact_log_file:
        with open(log_file_name, "r") as log_file:
            for line in log_file:
                if "Validation Sample Logits" in line:
                    include = False
                if "Validation loss" in line:
                    include = True
                if include:
                    print(line, file=compact_log_file)


def get_loss_statistics(log_file_name):
    train_loss_track = []
    validation_loss_track = []
    with open(log_file_name, "r") as log_file:
        for line in log_file:
            if "-- loss:" in line:
                for i in range(3):
                    train_loss_track.append(get_number(line))
            if "Validation loss" in line:
                while len(validation_loss_track) < len(train_loss_track):
                    validation_loss_track.append(get_number(line))

    plot_loss(train_loss_track, validation_loss_track)
    return train_loss_track, validation_loss_track


def extract_validation_samples(log_file_name):
    """Extracts all validation sample outputs and associated targets."""
    logits_sequences = []
    target_sequences = []
    working_on = None
    with open(log_file_name, "r") as log_file:
        for line in log_file:
            if "Validation Sample Logits" in line:
                logits = []
                working_on = "logits"
            elif "Target Values" in line:
                targets = []
                working_on = "targets"
            if working_on == "logits":
                if "[" in line:
                    logits.append(get_numbers(line))
                else:
                    logits[-1].extend(get_numbers(line))
                if "]]" in line:
                    logits_sequences.append(logits)
                    working_on = None
            elif working_on == "targets":
                targets.extend(get_numbers(line, integers=True))
                if "]" in line:
                    target_sequences.append(targets)
                    working_on = None

    return logits_sequences, target_sequences


def get_prediction_info(logits, targets):
    """
    Returns a list containing entries of the form 
        ((true_value, prob), (top_value1, prob1), (top_value2, prob2)).
    """
    prediction_info = []
    for logit_array, target in zip(logits, targets):
        probability_array = [
            (i, sigmoid(logit)) for i, logit in enumerate(logit_array)]
        probabilities_sorted = sorted(probability_array, key=lambda pair: pair[1])
        for pair in probability_array:
            if pair[0] == target:
                break
        prediction_info.append(
            (pair, probabilities_sorted[-1], probabilities_sorted[-2]))

    matches = 0
    for triple in prediction_info:
        if triple[0][0] == triple[1][0] or triple[0][0] == triple[2][0]:
            matches += 1

    return prediction_info, matches/len(prediction_info)


def output_predict_tuple(logits_sequences, target_sequences, index, file):
    prediction_info, prediction_percentage = get_prediction_info(
        logits_sequences[index], target_sequences[index])
    print("Prediction percentage:", str(prediction_percentage) + "%", file=file)
    print("[", file=file)
    for predict_tuple in prediction_info:
        print(predict_tuple, file=file)
    print("]", file=file)


if __name__ == '__main__':
    # compactify_log("train_run_log2.txt")
    # get_loss_statistics("train_run_log2_compact.txt")
    logits_sequences, target_sequences = extract_validation_samples(
        "train_run_log2.txt")
    with open("train_run_log2_prediction_info.txt", "w") as info_file:
        print("\n", file=info_file)
        output_predict_tuple(logits_sequences, target_sequences, 0, info_file)
        print("\n\n", file=info_file)
        output_predict_tuple(logits_sequences, target_sequences, 1, info_file)
        print("\n\n", file=info_file)
        output_predict_tuple(logits_sequences, target_sequences, -2, info_file)
        print("\n\n", file=info_file)
        output_predict_tuple(logits_sequences, target_sequences, -1, info_file)