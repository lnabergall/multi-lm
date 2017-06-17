"""Helper functions for modifying, analyzing, and visualizing data."""

import numpy as np
import matplotlib.pyplot as plotter

import training_storage as storage


def get_vocab_sizes(description_chars, script_chars, encoder_vocab_size, 
                    decoder_vocab_size):
    if encoder_vocab_size:
        description_values_count = encoder_vocab_size + 4
    else:
        description_values_count = len(description_chars) + 2
    if decoder_vocab_size:
        script_values_count = decoder_vocab_size + 4
    else:
        script_values_count = len(script_chars) + 2

    return description_values_count, script_values_count


def truncate_data(training_data_dict, description_count=0):
    if not description_count:
        return training_data_dict
    else:
        truncated_training_data = {}
        for i, desc in enumerate(training_data_dict):
            if i < description_count:
                truncated_training_data[desc] = training_data_dict[desc]

    return truncated_training_data


def randomize_insync(*args):
    state = np.random.get_state()
    for collection in args:
        np.random.shuffle(collection)
        np.random.set_state(state)


def early_stop(validation_loss_track):
    if len(validation_loss_track) < 6:
        return False
    else:
        return (min(validation_loss_track[-i][0] for i in range(1, 7)) 
                == validation_loss_track[-6][0])


def moving_average(sequence, window_size):
    sum_ = 0
    sequence_average = [0 for x in sequence]
    for i in range(len(sequence)):
        if i < window_size:
            sum_ = sum_ + sequence[i]
            sequence_average[i] = sum_ / (i+1)
        else:
            sum_ = sum_ - sequence[i-window_size] + sequence[i]
            sequence_average[i] = sum_ / window_size

    return sequence_average


def plot_loss(*loss_tracks, labels=None, plot_first_average=True):
    if type(loss_tracks[0][0]) != float:
        loss_only_tracks = []
        for loss_track in loss_tracks:
            loss_only_track = [eval_step[0] for eval_step in loss_track]
            loss_only_tracks.append(loss_only_track)
    else:
        loss_only_tracks = loss_tracks

    plotter.figure()
    for i, loss_track in enumerate(loss_only_tracks):
        try:
            if labels:
                plotter.plot(loss_track, label=labels[i])
            else:
                plotter.plot(loss_track)
        except ValueError:
            print("Problem with displaying a loss track!")

    if plot_first_average:
        average_loss_track = moving_average(loss_only_tracks[0], 100)
        if labels:
            plotter.plot(average_loss_track, label=labels[0] + " average")
        else:
            plotter.plot(average_loss_track)
    if labels:
        plotter.legend(loc="best")


def get_model_file_paths(**kwargs):
    models = storage.get_model_info(**kwargs)
    return [(model.model_id, model.model_graph_file, run.model_parameters_file) 
            for model in models for run in model.training_runs]


def compare_loss_tracks(dataset_type=None, **kwargs):
    # Get loss tracks
    models = storage.get_model_info(**kwargs)
    loss_tracks = []
    for model in models:
        training_run = get_training_run_info(model=model)
        if dataset_type is None:
            training_evaluation_track = get_evaluation_track(
                "training", training_run=training_run)
            validation_evaluation_track = get_evaluation_track(
                "validation", training_run=training_run)
            test_evaluation_track = get_evaluation_track(
                "test", training_run=training_run)
            training_loss_track = [
                model_eval.loss for model_eval in training_evaluation_track]
            validation_loss_track = [
                model_eval.loss for model_eval in validation_evaluation_track]
            test_loss_track = [
                model_eval.loss for model_eval in test_evaluation_track]
            loss_tracks.append(
                (training_loss_track, validation_loss_track, test_loss_track))
        else:
            evaluation_track = get_evaluation_track(
                dataset_type, training_run=training_run)
            loss_track = [model_eval.loss for model_eval in evaluation_track]
            loss_tracks.append((loss_track, [], []))

    # Plot loss tracks in separate figures
    all_loss_tracks = []
    all_labels = []
    for i, loss_track_tuple in enumerate(loss_tracks):
        labels = ("Model " + str(model.id) + " training loss", 
                  "Model " + str(model.id) + " validation loss", 
                  "Model " + str(model.id) + " test loss")
        loss_tracks = [loss_track for loss_track in loss_track_tuple if loss_track]
        labels = [label for i, label in enumerate(labels) if loss_track_tuple[i]]
        plot_loss(*loss_tracks, labels=labels, plot_first_average=False)
        all_loss_tracks.extend(loss_tracks)
        all_labels.extend(all_labels)

    # If single dataset type, include summary plot
    if dataset_type:
        plot_loss(*all_loss_tracks, labels=all_labels, plot_first_average=False)
