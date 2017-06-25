"""Helper functions for modifying, analyzing, and visualizing data."""

import re
from datetime import datetime

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
    if len(validation_loss_track) < 5:
        return False
    else:
        return (min(validation_loss_track[-i][0] for i in range(1, 6)) + .004 
                >= validation_loss_track[-5][0])


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
    if not loss_tracks:
        return
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


def filtered_by_trained(models):
    trained_models = []
    for model in models:
        latest_training_run = storage.get_latest_training_run(model=model)
        if latest_training_run:
            loss_track = storage.get_evaluation_track(
                "training", training_run=latest_training_run)
            if len(loss_track) >= 100:
                trained_models.append(model)

    return trained_models


def get_optimizer_label(optimizer_string):
    possible_labels = re.findall(r"[a-zA-Z]+", optimizer_string) 
    for label in possible_labels:
        if "Optimizer" in label:
            optimizer_label = label
            break
    return optimizer_label


def get_unique_hyperparameters(models):
    hyperparameters = {}
    hyperparameter_sets = {}
    for model in models:
        hyperparameters[model.model_id] = model.as_dict()
        del hyperparameters[model.model_id]["model_id"]
        del hyperparameters[model.model_id]["timestamp"]
        del hyperparameters[model.model_id]["model_graph_file"]
        for hyperparameter in hyperparameters[model.model_id]:
            values = hyperparameter_sets.get(hyperparameter, set())
            values.add(hyperparameters[model.model_id][hyperparameter])
            hyperparameter_sets[hyperparameter] = values

    for hyperparameter in hyperparameter_sets:
        if len(hyperparameter_sets[hyperparameter]) == 1:
            for model in models:
                del hyperparameters[model.model_id][hyperparameter]

    return hyperparameters


def make_labels(models):
    unique_hyperparameters = get_unique_hyperparameters(models)
    labels = []
    for model in models:
        model_id = model.model_id
        label = "Model " + str(model_id)
        for hyperparameter, value in unique_hyperparameters[model_id].items():
            if type(value) == str and "optimizer" in value.lower():
                value = get_optimizer_label(value)
            label += " - " + hyperparameter + " " + str(value)
            label = label.replace("_", " ")
        labels.append(label)

    return labels


def compare_loss_tracks(dataset_type=None, only_summary=False, 
                        models=None, **kwargs):
    # Get loss tracks
    if not models:
        models = storage.get_model_info(**kwargs)
    models = filtered_by_trained(models)
    loss_tracks = []
    for model in models:
        training_run = storage.get_latest_training_run(model=model)
        if dataset_type is None:
            training_evaluation_track = storage.get_evaluation_track(
                "training", training_run=training_run)
            validation_evaluation_track = storage.get_evaluation_track(
                "validation", training_run=training_run)
            test_evaluation_track = storage.get_evaluation_track(
                "test", training_run=training_run)
            training_loss_track = [
                model_eval.loss for model_eval in training_evaluation_track]
            validation_loss_track = [
                model_eval.loss for model_eval in validation_evaluation_track]
            test_loss_track = [
                model_eval.loss for model_eval in test_evaluation_track]
            loss_tracks.append((model.model_id, 
                (training_loss_track, validation_loss_track, test_loss_track)))
        else:
            evaluation_track = storage.get_evaluation_track(
                dataset_type, training_run=training_run)
            loss_track = [model_eval.loss for model_eval in evaluation_track]
            if dataset_type == "training":
                loss_tracks.append((model.model_id, (loss_track, [], [])))
            elif dataset_type == "validation":
                loss_tracks.append((model.model_id, ([], loss_track, [])))
            elif dataset_type == "test":
                loss_tracks.append((model.model_id, ([], [], loss_track)))

    labels = make_labels(models)

    # Plot loss tracks in separate figures
    all_loss_tracks = []
    all_labels = []
    for loss_track_tuple, model_label in zip(loss_tracks, labels):
        track_labels = [model_label + ": training loss", 
                        model_label + ": validation loss", 
                        model_label + ": test loss"]
        loss_tracks = [loss_track for loss_track in loss_track_tuple[1] if loss_track]
        track_labels = [label for i, label in enumerate(track_labels) 
                        if loss_track_tuple[1][i]]
        if not only_summary:
            plot_loss(*loss_tracks, labels=track_labels, plot_first_average=False)
        all_loss_tracks.extend(loss_tracks)
        all_labels.extend(track_labels)

    # If single dataset type, include summary plot
    if dataset_type:
        plot_loss(*all_loss_tracks, labels=all_labels, plot_first_average=False)

    return models


def get_model_clusters(*hyperparameters):
    """
    Get sets of models with all hyperparameters equivalent 
    except for those in **kwargs.
    """ 
    all_models = storage.get_model_info()
    model_clusters = []
    for model in all_models:
        model_dict = model.as_dict()
        match_found = False
        for i, model_cluster in enumerate(model_clusters):
            match = True
            representative = model_cluster[0].as_dict()
            for hyperparameter in representative:
                if (hyperparameter in hyperparameters 
                        or hyperparameter == "model_id"
                        or hyperparameter == "timestamp"
                        or hyperparameter == "model_graph_file"):
                    continue
                if representative[hyperparameter] != model_dict[hyperparameter]:
                    match = False
                    break
            if match:
                model_clusters[i].append(model)
                match_found = True
        if not match_found:
            model_clusters.append([model])

    return model_clusters


def output_training_times(models):
    labels = make_labels(models)
    for model, label in zip(models, labels):
        training_run = storage.get_latest_training_run(model=model)
        duration = abs(training_run.begin_timestamp.timestamp() 
                       - training_run.end_timestamp.timestamp())
        print(label + ": duration", duration, "seconds")


def analyze_training_run_clusters(dataset_type, *hyperparameters,
                                  models=None, 
                                  minimal_cluster_size=None,
                                  analyze="loss_tracks"):
    if models is not None:
        model_clusters = [models]
    else:
        model_clusters = get_model_clusters(*hyperparameters)
    if minimal_cluster_size:
        model_clusters = [model_cluster for model_cluster in model_clusters 
                          if len(model_cluster) >= minimal_cluster_size]
    for model_cluster in model_clusters:
        if analyze == "loss_tracks":
            compare_loss_tracks(dataset_type=dataset_type, 
                                only_summary=True, models=model_cluster)
        elif analyze == "training_time":
            output_training_times(model_cluster)
    plotter.show()


def compare_recent_training_runs(dataset_type):
    timestamp = datetime.utcnow()
    timestamp = timestamp.replace(day=timestamp.day-2)
    models = compare_loss_tracks(dataset_type=dataset_type, only_summary=True, 
                                 timestamp=timestamp)
    plotter.show()
    return models


if __name__ == '__main__':
    models = compare_recent_training_runs("training")
    analyze_training_run_clusters(
        "training", models=models, analyze="training_time")
    compare_recent_training_runs("validation")
    # compare_loss_tracks(dataset_type="training", only_summary=True, 
    #                     learning_rate=0.001)
    # compare_loss_tracks(dataset_type="validation", only_summary=True, 
    #                     learning_rate=0.001)
    # plotter.show()
    # analyze_training_run_clusters(
    #     "training", "bidirectional", "attention",
    #     minimal_cluster_size=2, analyze="training_time")
    # analyze_training_run_clusters(
    #     "training", "bidirectional", "attention",
    #     minimal_cluster_size=2)
    # analyze_training_run_clusters(
    #     "validation", "bidirectional", "attention",
    #     minimal_cluster_size=2)