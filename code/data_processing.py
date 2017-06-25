
import os
from time import sleep
from random import shuffle
from statistics import mean, median
from itertools import product

import numpy as np
import textacy


BASE_DIR = os.path.join(os.pardir, "data")
if not os.path.exists(BASE_DIR):
    raise NotImplementedError("Can't work from current directory!")


PAD_CHARACTER = "\n"
PAD_VALUE = 0
MAX_DESCRIPTION_LENGTH = 4244
MAX_SCRIPT_LENGTH = 1500


def detect_language(code_text):
    # Note: could be Java too apparently, but focuses on  
    # correctly classifying Python code for now.
    if ("#include" in code_text or "int main" in code_text 
            or "void main" in code_text or "for(" in code_text 
            or "for (" in code_text or "public class" in code_text
            or "public static" in code_text or "java." in code_text):
        language = "c++"
    else:
        language = "python"

    return language


def get_data():
    """
    Fetch all data, producing a dictionary of natural language descriptions
    as keys and corresponding code solutions as values.
    """
    data_path = os.path.join(BASE_DIR, "description2code_current", 
                             "description2code_current", "codechef", "easy")
    data_by_problem = {}
    for path, dir_names, file_names in os.walk(data_path):
        path_parts = os.path.split(path)
        if path_parts[1] in ["description", "solutions_c++", "solutions_python"]:
            with open(os.path.join(path_parts[0], "part_of.txt"), "r") as dataset_file:
                dataset_assignment = dataset_file.read().strip()
            problem_id = os.path.split(path_parts[0])[1]
            data_by_problem.setdefault(
                problem_id, {"dataset_assignment": dataset_assignment, 
                             "description": "", "c++": [], "python": []})
        if path_parts[1] == "description":
            for file_name in file_names:
                with open(os.path.join(path, file_name), "r", encoding="utf8") as desc_file:
                    description = desc_file.read()
                    data_by_problem[problem_id]["description"] = description
        elif path_parts[1] == "solutions_c++":
            problem_id = os.path.split(path_parts[0])[1]
            for file_name in file_names:
                with open(os.path.join(path, file_name), "r", 
                          encoding="latin-1") as code_file:
                    code = code_file.read()
                    language = detect_language(code)
                    if code and language == "c++":
                        data_by_problem[problem_id]["c++"].append(code)
                    elif code and language == "python":
                        data_by_problem[problem_id]["python"].append(code)
        elif path_parts[1] == "solutions_python":
            problem_id = os.path.split(path_parts[0])[1]
            for file_name in file_names:
                with open(os.path.join(path, file_name), "r", 
                          encoding="latin-1") as code_file:
                    code = code_file.read()
                    language = detect_language(code)
                    if code and language == "python":
                        data_by_problem[problem_id]["python"].append(code)
                    elif code and language == "c++":
                        data_by_problem[problem_id]["c++"].append(code)
    data = {}
    for problem_id in data_by_problem:
        description = data_by_problem[problem_id]["description"]
        code_dict = {
            "dataset_assignment": data_by_problem[problem_id]["dataset_assignment"],
            "c++": data_by_problem[problem_id]["c++"], 
            "python": data_by_problem[problem_id]["python"],
        }
        data[description] = code_dict

    return data


def get_spell_corrector():
    spell_correct_dict = {}
    with open(os.path.join(BASE_DIR, 
            "wikipedia_common_misspellings.txt"), "r") as spell_file:
        for line in spell_file:
            misspelled_word = line.split("->")[0]
            correct_spelling = line.split("->")[1]
            spell_correct_dict[misspelled_word] = correct_spelling
    return spell_correct_dict


def correct_spelling(spell_correct_dict, textacy_doc):
    text = textacy_doc.text
    for word in textacy_doc:
        word = word.text
        if word in spell_correct_dict:
            corrected_word = spell_correct_dict[word]
        else:
            corrected_word = word
        text.replace(word, corrected_word)
    return text


def split_processed_data(processed_data_dict):
    training_size = ((len(processed_data_dict["data"]) * 8) // 10) + 1
    validation_size = len(processed_data_dict["data"]) // 10
    test_size = len(processed_data_dict["data"]) // 10
    training_data = {}
    validation_data = {}
    test_data = {}
    if "dataset_assignment" in processed_data_dict["data"][
            list(processed_data_dict["data"].keys())[0]]:
        for i, description in enumerate(processed_data_dict["data"]):
            assignment = processed_data_dict["data"][description]["dataset_assignment"]
            if assignment == "training":
                training_data[description] = processed_data_dict["data"][description]
            elif assignment == "validation":
                validation_data[description] = processed_data_dict["data"][description]
            elif assignment == "test":
                test_data[description] = processed_data_dict["data"][description]
    else:
        random_descs = list(processed_data_dict["data"].keys())
        shuffle(random_descs)
        for i, description in enumerate(random_descs):
            if i < training_size:
                training_data[description] = processed_data_dict["data"][description]
            elif training_size <= i < training_size + validation_size:
                validation_data[description] = processed_data_dict["data"][description]
            else:
                test_data[description] = processed_data_dict["data"][description]

    return training_data, validation_data, test_data


def process_data(data_dict):
    """
    Collect character information and add special beginning 
    and end symbols.
    """
    desc_characters = set()
    desc_char_counts = {}
    python_characters = set()
    python_char_counts = {}
    cplusplus_characters = set()
    cplusplus_char_counts = {}
    for description in data_dict:
        cplusplus_solutions = data_dict[description]["c++"]
        python_solutions = data_dict[description]["python"]
        description_orig = description
        description = description.strip()
        desc_characters |= set(description)
        for char in description:
            if char not in desc_char_counts:
                desc_char_counts[char] = 0
            desc_char_counts[char] += 1
        cplusplus_solutions_new = []
        for script in cplusplus_solutions:
            script = script.strip()
            if len(script) > MAX_SCRIPT_LENGTH:
                continue
            cplusplus_characters |= set(script)
            for char in script:
                if char not in cplusplus_char_counts:
                    cplusplus_char_counts[char] = 0
                cplusplus_char_counts[char] += script.count(char)
            cplusplus_solutions_new.append(script)
        data_dict[description_orig]["c++"] = cplusplus_solutions_new
        python_solutions_new = []
        for script in python_solutions:
            script = script.strip()
            if len(script) > MAX_SCRIPT_LENGTH:
                continue
            python_characters |= set(script)
            for char in script:
                if char not in python_char_counts:
                    python_char_counts[char] = 0
                python_char_counts[char] += script.count(char)
            python_solutions_new.append(script)
        data_dict[description_orig]["python"] = python_solutions_new
    desc_characters.discard("")
    python_characters.discard("")
    cplusplus_characters.discard("")
    desc_characters = sorted(list(desc_characters))
    python_characters = sorted(list(python_characters))
    cplusplus_characters = sorted(list(cplusplus_characters))
    processed_data_dict = {
        "description_chars": desc_characters,
        "description_char_counts": desc_char_counts,
        "python_chars": python_characters,
        "python_char_counts": python_char_counts,
        "c++_chars": cplusplus_characters,
        "c++_char_counts": cplusplus_char_counts,
        "data": data_dict,
    }

    # Calculate some dataset statistics
    description_lengths = [len(description) 
                           for description in processed_data_dict["data"]]
    min_desc_length = min(description_lengths)
    max_desc_length = max(description_lengths)
    avg_desc_length = mean(description_lengths)
    median_desc_length = median(description_lengths)
    python_lengths = [[len(script) for script 
                       in processed_data_dict["data"][desc]["python"]]
                      for desc in processed_data_dict["data"]]
    min_python_length = min(
        min(lengths, default=1000) for lengths in python_lengths)
    max_python_length = max(
        max(lengths, default=0) for lengths in python_lengths)
    avg_python_length = mean(
        mean(lengths) for lengths in python_lengths if lengths)
    median_python_length = median(
        length for lengths in python_lengths for length in lengths)
    print(min_desc_length, max_desc_length, 
          avg_desc_length, median_desc_length)
    print(min_python_length, max_python_length, 
          avg_python_length, median_python_length)

    # Split into training, validation, and test sets
    training_data, validation_data, test_data = split_processed_data(
        processed_data_dict)
    processed_data_dict["training_data"] = training_data
    processed_data_dict["validation_data"] = validation_data
    processed_data_dict["test_data"] = test_data

    return processed_data_dict


def get_dataset_statistics(training_data_dict, validation_data_dict, 
                           test_data_dict, language="python"):
    inputs = (list(training_data_dict.keys()) 
        + list(validation_data_dict.keys()) + list(test_data_dict.keys()))
    outputs = []
    for description in inputs:
        if description in training_data_dict:
            outputs.append(training_data_dict[description][language])
        elif description in validation_data_dict:
            outputs.append(validation_data_dict[description][language])
        elif description in test_data_dict:
            outputs.append(test_data_dict[description][language])

    input_lengths = [len(description) for description in inputs]
    output_lengths = [len(script) for script in outputs]

    inputs = sum(1 for input_length in input_lengths)
    average_input_length = mean(input_lengths)
    smallest_input_length = min(input_lengths)
    largest_input_length = max(input_lengths)

    outputs = sum(1 for output_length in output_lengths)
    average_output_length = mean(output_lengths)
    smallest_output_length = min(output_lengths)
    largest_output_length = max(output_lengths)

    return (inputs, average_input_length, smallest_input_length, 
            largest_input_length, outputs, average_output_length, 
            smallest_output_length, largest_output_length)


def generate_char_labels(desc_char_counts, python_char_counts, dense=True,
                         description_vocab_size=0, python_vocab_size=0):
    desc_chars, python_chars = list(desc_char_counts), list(python_char_counts)
    desc_chars.sort(key=lambda char: desc_char_counts[char], reverse=True)
    python_chars.sort(key=lambda char: python_char_counts[char], reverse=True)
    if dense:
        desc_char_values = {char: i+2 for i, char in enumerate(desc_chars)}
        python_char_values = {char: i+2 for i, char in enumerate(python_chars)}
        if description_vocab_size:
            for i, char in enumerate(desc_chars):
                if i >= description_vocab_size:
                    desc_char_values[char] = description_vocab_size + 2
        if python_vocab_size:
            for i, char in enumerate(python_chars):
                if i >= python_vocab_size:
                    python_char_values[char] = python_vocab_size + 2
    else:
        desc_char_values = {char: i for i, char in enumerate(desc_chars)}
        python_char_values = {char: i for i, char in enumerate(python_chars)}

    return desc_char_values, python_char_values


def devectorize(vector, data_type, desc_char_counts, python_char_counts):
    desc_char_values, python_char_values = generate_char_labels(
        desc_char_counts, python_char_counts)
    desc_value_chars = {i: char for char, i in desc_char_values.items()}
    python_value_chars = {i: char for char, i in python_char_values.items()}

    if len(vector.shape) >= 2:
        vector = vector.flatten()
    if type(vector) == np.ndarray:
        length = vector.size
    elif type(vector) == list:
        length = len(vector)
    else:
        raise NotImplementedError("Unexpected type: " + type(vector))
    text = ""
    if data_type == "description":
        for i in range(length):
            if vector[i] == 0:
                text += "\n"
            elif vector[i] == 1:
                text += "<eos>"
            else:
                text += desc_value_chars[vector[i]]
    elif data_type == "script":
        for i in range(length):
            if vector[i] == 0:
                text += "\n"
            elif vector[i] == 1:
                text += "<eos>"
            else:
                text += python_value_chars[vector[i]]
    else:
        raise ValueError("Cannot interpret this data type: " + data_type)

    return text


def vectorize_example(description, python_solution, desc_char_counts, 
                      python_char_counts, dense=True, description_vocab_size=0,
                      python_vocab_size=0):
    desc_char_values, python_char_values = generate_char_labels(
        desc_char_counts, python_char_counts, dense, 
        description_vocab_size, python_vocab_size)
    if dense:
        description_array = np.zeros((1, len(description)), dtype=np.int8)
        python_solution_array = np.zeros((1, len(python_solution)), dtype=np.int8)
    else:
        description_array = np.zeros(
            (len(description), len(desc_chars)), dtype=np.bool)
        python_solution_array = np.zeros(
            (MAX_SCRIPT_LENGTH, len(python_chars)), dtype=np.bool)

    for i, char in enumerate(description):
        if dense:
            description_array[0, i] = desc_char_values[char]
        else:
            description_array[i, desc_char_values[char]] = 1
    for i, char in enumerate(python_solution):
        if dense:
            python_solution_array[0, i] = python_char_values[char]
        else:
            python_solution_array[i, python_char_values[char]] = 1

    return description_array, python_solution_array


def vectorize_examples(processed_data_dict, desc_char_counts, 
                       python_char_counts, dense=True, 
                       description_vocab_size=0, python_vocab_size=0):
    description_arrays = []
    script_arrays = []
    for description in processed_data_dict:
        for script in processed_data_dict[description]["python"]:
            description_array, script_array = vectorize_example(
                description, script, desc_char_counts, python_char_counts,
                description_vocab_size=description_vocab_size, 
                python_vocab_size=python_vocab_size)
            description_arrays.append(description_array)
            script_arrays.append(script_array)

    return description_arrays, script_arrays


def merge_arrays(description_arrays, script_arrays):
    if len(description_arrays) != len(script_arrays):
        raise ValueError("Unexpected sequence lengths!")
    max_desc_length = max(array.shape[1] for array in description_arrays)
    max_script_length = max(array.shape[1] for array in script_arrays)
    merged_description_array = np.zeros(
        (max_desc_length, len(description_arrays)), dtype=np.int8)
    merged_script_array = np.zeros(
        (max_script_length, len(script_arrays)), dtype=np.int8)
    for i, (desc_array, script_array) in enumerate(
            zip(description_arrays, script_arrays)):
        for j in range(desc_array.shape[1]):
            merged_description_array[j, i] = desc_array[0, j]
        for j in range(script_array.shape[1]):
            merged_script_array[j, i] = script_array[0, j]

    return merged_description_array, merged_script_array


def make_batches(description_arrays, script_arrays, batch_size):
    description_batches = []
    script_batches = []
    description_batches_lengths = []
    script_batches_lengths = []
    for i in range((len(script_arrays) - 1)//batch_size + 1):
        description_batches_lengths.append([])
        script_batches_lengths.append([])
        descriptions = description_arrays[i*batch_size:(i+1)*batch_size]
        scripts = script_arrays[i*batch_size:(i+1)*batch_size]
        max_desc_length = max(desc.shape[1] for desc in descriptions)
        max_script_length = max(script.shape[1] for script in scripts)
        if i != len(script_arrays)//batch_size:
            # Time-major arrays
            description_batch = np.zeros(
                (max_desc_length, batch_size), dtype=np.int8)
            script_batch = np.zeros(
                (max_script_length, batch_size), dtype=np.int8)
        else:
            description_batch = np.zeros(
                (max_desc_length, len(script_arrays) % batch_size), dtype=np.int8)
            script_batch = np.zeros(
                (max_script_length, len(script_arrays) % batch_size), dtype=np.int8)
        for j, (desc, script) in enumerate(zip(descriptions, scripts)):
            description_batches_lengths[-1].append(desc.shape[1])
            script_batches_lengths[-1].append(script.shape[1])
            for k in range(desc.shape[1]):
                description_batch[k, j] = desc[0, k]
            for k in range(script.shape[1]):
                script_batch[k, j] = script[0, k]
        description_batches.append(description_batch)
        script_batches.append(script_batch)

    return (description_batches, script_batches, 
            description_batches_lengths, script_batches_lengths)


def vectorize_data(processed_data_dict, desc_char_counts, python_char_counts, 
                   backprop_timesteps=50, dense=True, description_vocab_size=0,
                   python_vocab_size=0):
    desc_char_values, python_char_values = generate_char_labels(
        desc_char_counts, python_char_counts, dense, 
        description_vocab_size, python_vocab_size)
    desc_chars = list(desc_char_values)
    python_chars = list(python_char_values)
    print(desc_char_values)
    print(python_char_values)

    # Cut the data into sequences of backprop_timesteps characters
    desc_sequences = []
    python_sequences = []
    for description in processed_data_dict:
        if backprop_timesteps:
            if len(description) < backprop_timesteps:
                if dense:
                    desc_sequences.append([description])
                else:
                    padding = PAD_CHARACTER * (backprop_timesteps - len(description))
                    desc_sequences.append([padding + description])
            else:
                sequences = [description[i: i+backprop_timesteps] for i in 
                    range(0, len(description)-backprop_timesteps, backprop_timesteps//2)]
                desc_sequences.append(sequences)
                leftover_index = ((len(description)-backprop_timesteps) 
                                  % (backprop_timesteps//2))
                last_desc_sequence = description[len(description)-1-leftover_index:]
                if dense:
                    desc_sequences[-1].append(last_desc_sequence)
                else:
                    padding = PAD_CHARACTER * (
                        backprop_timesteps-len(description)+leftover_index)
                    desc_sequences[-1].append(padding + last_desc_sequence)
        else:
            desc_sequences.append([description])
        python_sequences.append([])
        python_solutions = processed_data_dict[description]["python"]
        for script in python_solutions:
            if not backprop_timesteps or len(script) < backprop_timesteps:
                python_sequences[-1].append([script])
            else:
                script_sequences = [script[i: i+backprop_timesteps] for i in 
                    range(0, len(script)-backprop_timesteps, backprop_timesteps//2)]
                python_sequences[-1].append(script_sequences)
                leftover_index = ((len(script)-backprop_timesteps) 
                                  % (backprop_timesteps//2))
                last_script_sequence = script[len(script)-1-leftover_index:]
                if dense:
                    python_sequences[-1][-1].append(last_script_sequence)
                else:
                    padding = PAD_CHARACTER * (
                        backprop_timesteps-len(script)+leftover_index)
                    python_sequences[-1][-1].append(last_script_sequence + padding)

    # Vectorize
    training_example_count = 0
    for desc_list, desc_script_list in zip(desc_sequences, python_sequences):
        for sequences in desc_script_list:
            training_example_count += len(desc_list) * len(sequences)
    print("Training examples:", training_example_count)
    if backprop_timesteps:
        if dense:
            description_array = np.zeros((training_example_count, 
                backprop_timesteps), dtype=np.int8)
            python_solution_array = np.zeros((training_example_count, 
                backprop_timesteps), dtype=np.int8)
        else:
            description_array = np.zeros((training_example_count, 
                backprop_timesteps, len(desc_chars)), dtype=np.bool)
            python_solution_array = np.zeros((training_example_count, 
                MAX_SCRIPT_LENGTH, len(python_chars)), dtype=np.bool)
    else:
        if dense:
            description_array = np.zeros((training_example_count, 
                MAX_DESCRIPTION_LENGTH), dtype=np.int8)
            python_solution_array = np.zeros((training_example_count, 
                MAX_SCRIPT_LENGTH), dtype=np.int8)
        else:
            description_array = np.zeros((training_example_count, 
                MAX_DESCRIPTION_LENGTH, len(desc_chars)), dtype=np.bool)
            python_solution_array = np.zeros((training_example_count, 
                MAX_SCRIPT_LENGTH, len(python_chars)), dtype=np.bool)
    if not dense:
        solution_output_masks = np.zeros(
            (training_example_count, MAX_SCRIPT_LENGTH), dtype=np.float32)
    print("Description array size:", description_array.size*8)
    print("Python script array size:", python_solution_array.size*8)
    desc_sequence_lengths = []
    python_sequence_lengths = []
    outer_index = 0
    for i, (desc_seqs, desc_script_list) in enumerate(
            zip(desc_sequences, python_sequences)):
        for j, script_seqs in enumerate(desc_script_list):
            for k, (desc_seq, script_seq) in enumerate(product(desc_seqs, script_seqs)):
                if dense:
                    desc_sequence_lengths.append(len(desc_seq))
                    python_sequence_lengths.append(len(script_seq))
                for m, char in enumerate(desc_seq):
                    if dense:
                        description_array[outer_index, m] = desc_char_values[char]
                    else:
                        description_array[outer_index, m, desc_char_values[char]] = 1
                for m, char in enumerate(script_seq):
                    if dense:
                        python_solution_array[outer_index, m] = python_char_values[char]
                    else:
                        python_solution_array[
                            outer_index, m, python_char_values[char]] = 1
                    if not dense:
                        solution_output_masks[outer_index, m] = 1
                outer_index += 1

    if dense:
        # Change from batch-major to time-major
        # description_array = description_array.swapaxes(0, 1)
        # python_solution_array = python_solution_array.swapaxes(0, 1)
        return (description_array, python_solution_array, 
                desc_sequence_lengths, python_sequence_lengths)
    else:
        return description_array, python_solution_array, solution_output_masks


if __name__ == '__main__':
    data = get_data()
    print(len(data))
    cplusplus_count = 0
    python_count = 0
    for description in data:
        cplusplus_count += len(data[description]["c++"])
        python_count += len(data[description]["python"])
    for i, desc in enumerate(data):
        if i == 1000:
            print("Description:")
            print(desc, "\n")
            print("C++ solutions:")
            for solution in data[desc]["c++"]:
                print(solution, "\n")
            print("Python solutions:")
            for solution in data[desc]["python"]:
                print(solution, "\n")
            print("\n")
    print(cplusplus_count)
    print(python_count)

