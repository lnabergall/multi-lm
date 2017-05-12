
import os
from random import shuffle
from statistics import mean, median
from itertools import product

import numpy as np
import textacy


BASE_DIR = os.path.join(os.pardir, "data")
if not os.path.exists(BASE_DIR):
    raise NotImplementedError("Can't work from current directory!")


PAD_CHARACTER = "§"


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
            problem_id = os.path.split(path_parts[0])[1]
            data_by_problem.setdefault(
                problem_id, {"description": "", "c++": [], "python": []})
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
                    data_by_problem[problem_id]["c++"].append(code)
        elif path_parts[1] == "solutions_python":
            problem_id = os.path.split(path_parts[0])[1]
            for file_name in file_names:
                with open(os.path.join(path, file_name), "r", 
                          encoding="latin-1") as code_file:
                    code = code_file.read()
                    data_by_problem[problem_id]["python"].append(code)
    data = {}
    for problem_id in data_by_problem:
        description = data_by_problem[problem_id]["description"]
        code_dict = {
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


def process_data(data_dict, backprop_timesteps):
    """
    Collect character information and add special beginning 
    and end symbols.
    """
    desc_characters = set()
    desc_char_counts = {}
    python_characters = set("<boc><eoc>§")
    python_char_counts = {}
    cplusplus_characters = set("<boc><eoc>§")
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
            if len(script) < backprop_timesteps-len("<boc><eoc>"):
                continue    # Remove very short scripts
            cplusplus_characters |= set(script)
            for char in script:
                if char not in cplusplus_char_counts:
                    cplusplus_char_counts[char] = 0
                cplusplus_char_counts[char] += script.count(char)
            script = "<boc>" + script + "<eoc>"
            cplusplus_solutions_new.append(script)
        data_dict[description_orig]["c++"] = cplusplus_solutions_new
        python_solutions_new = []
        for script in python_solutions:
            script = script.strip()
            if len(script) < backprop_timesteps-len("<boc><eoc>"):
                continue    # Remove very short scripts
            python_characters |= set(script)
            for char in script:
                if char not in python_char_counts:
                    python_char_counts[char] = 0
                python_char_counts[char] += script.count(char)
            script = "<boc>" + script + "<eoc>"
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
    return processed_data_dict


def vectorize_data(processed_data_dict, backprop_timesteps):
    # Collect metadata
    desc_chars = processed_data_dict["description_chars"]
    desc_char_indices = {char: i for i, char in enumerate(desc_chars)}
    python_chars = processed_data_dict["python_chars"]
    python_char_indices = {char: i for i, char in enumerate(python_chars)}
    description_lengths = [len(description) 
                           for description in processed_data_dict["data"]]
    min_desc_length = min(description_lengths)
    python_lengths = [[len(script) for script 
                       in processed_data_dict["data"][desc]["python"]]
                      for desc in processed_data_dict["data"]]
    min_python_length = min(min(lengths, default=1000) for lengths in python_lengths)
    if (min_desc_length < backprop_timesteps 
            or min_python_length < backprop_timesteps):
        raise ValueError("Need to backprop for fewer timesteps! "
                         "Or change implementation...")

    # Cut the data into sequences of backprop_timesteps characters
    desc_sequences = []
    python_sequences = []
    for description in processed_data_dict["data"]:
        desc_sequences.append([description[i: i+backprop_timesteps] 
            for i in range(0, len(description)-backprop_timesteps, backprop_timesteps//2)])
        leftover_index = (len(description)-backprop_timesteps) % (backprop_timesteps//2)
        last_desc_sequence = description[len(description)-1-leftover_index:]
        for i in range(backprop_timesteps - len(description) + leftover_index):
            last_desc_sequence += "§"
        desc_sequences[-1].append(last_desc_sequence)
        python_sequences.append([])
        python_solutions = processed_data_dict["data"][description]["python"]
        for script in python_solutions:
            python_sequences[-1].append([script[i: i+backprop_timesteps] 
                for i in range(0, len(script)-backprop_timesteps, backprop_timesteps//2)])
            leftover_index = (len(script)-backprop_timesteps) % (backprop_timesteps//2)
            last_script_sequence = script[len(script)-1-leftover_index:]
            for i in range(backprop_timesteps - len(script) + leftover_index):
                last_script_sequence += "§"
            if len(last_script_sequence) > 50:
                print(last_script_sequence)
                print(len(last_script_sequence))
                raise NotImplementedError
            python_sequences[-1][-1].append(last_script_sequence)

    # Vectorize
    training_example_count = 0
    for desc_list, desc_script_list in zip(desc_sequences, python_sequences):
        for sequences in desc_script_list:
            training_example_count += len(desc_list) * len(sequences)
    print(training_example_count)
    description_array = np.zeros(
        (training_example_count, backprop_timesteps, len(desc_chars)), dtype=np.bool)
    python_solution_array = np.zeros(
        (training_example_count, backprop_timesteps, len(python_chars)), dtype=np.bool)
    print(description_array.size)
    print(python_solution_array.size)
    outer_index = 0
    for i, (desc_seqs, desc_script_list) in enumerate(
            zip(desc_sequences, python_sequences)):
        for j, script_seqs in enumerate(desc_script_list):
            for k, (desc_seq, script_seq) in enumerate(product(desc_seqs, script_seqs)):
                for m, char in enumerate(desc_seq):
                    description_array[outer_index, m, desc_char_indices[char]] = 1
                for m, char in enumerate(script_seq):
                    python_solution_array[outer_index, m, python_char_indices[char]] = 1
                outer_index += 1

    print("Splitting into training, validation, and test sets...")
    # Split into training, validation, and test sets
    training_size = ((training_example_count * 8) // 10) + 1
    validation_size = training_example_count // 10
    test_size = training_example_count // 10
    description_array_train = np.zeros(
        (training_size, max_desc_length, len(desc_chars)), dtype=np.bool)
    solution_array_train = np.zeros(
        (training_size, max_python_length, len(python_chars)), dtype=np.bool)
    description_array_valid = np.zeros(
        (validation_size, max_desc_length, len(desc_chars)), dtype=np.bool)
    solution_array_valid = np.zeros(
        (validation_size, max_python_length, len(python_chars)), dtype=np.bool)
    description_array_test = np.zeros(
        (test_size, max_desc_length, len(desc_chars)), dtype=np.bool)
    solution_array_test = np.zeros(
        (test_size, max_python_length, len(python_chars)), dtype=np.bool)
    random_range = list(range(training_example_count))
    shuffle(random_range)
    for i in range(description_array.shape[0]):
        print(i, "out of", description_array.shape[0])
        for j in range(description_array.shape[1]):
            for k in range(description_array.shape[2]):
                if i <= training_size - 1:
                    index = i
                    i = random_range[i]
                    description_array_train[index, j, k] = description_array[i, j, k]
                elif training_size <= i <= training_size + validation_size - 1:
                    index = i - training_size
                    i = random_range[i]
                    description_array_valid[index, j, k] = description_array[i, j, k]
                else:
                    index = i - training_size - validation_size
                    i = random_range[i]
                    description_array_test[index, j, k] = description_array[i, j, k]
        for j in range(python_solution_array.shape[1]):
            for k in range(python_solution_array.shape[2]):
                if i <= training_size - 1:
                    index = i
                    i = random_range[i]
                    solution_array_train[index, j, k] = python_solution_array[i, j, k]
                elif training_size <= i <= training_size + validation_size - 1:
                    index = i - training_size
                    i = random_range[i]
                    solution_array_valid[index, j, k] = python_solution_array[i, j, k]
                else:
                    index = i - training_size - validation_size
                    i = random_range[i]
                    solution_array_test[index, j, k] = python_solution_array[i, j, k]

    return (description_array_train, solution_array_train, description_array_valid,
            solution_array_valid, description_array_test, solution_array_test)


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

