
import os

import textacy


BASE_DIR = os.path.join(os.pardir, "data")
if not os.path.exists(BASE_DIR):
    raise NotImplementedError("Can't work from current directory!")


def get_data():
    """
    Fetch all data, producing a dictionary of natural language descriptions
    as keys and corresponding code solutions as values.
    """
    data_path = os.path.join(BASE_DIR, "description2code_current", 
                             "description2code_current", "codechef")
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


def process_data(data_dict):
    desc_characters = set()
    python_characters = set()
    cplusplus_characters = set()
    for description in data_dict:
        cplusplus_solutions = data_dict[description]["c++"]
        python_solutions = data_dict[description]["python"]
        for char in description:
            desc_characters.add(char)
        for script in cplusplus_solutions:
            for char in script:
                cplusplus_characters.add(char)
        for script in python_solutions:
            for char in script:
                python_characters.add(char)
    return {
        "descriptions_chars": desc_characters,
        "python_characters": python_characters,
        "c++_characters": cplusplus_characters,
        "data": data_dict,
    }


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

