
import os


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
                with open(os.path.join(path, file_name), "r") as desc_file:
                    description = desc_file.read()
                    data_by_problem[problem_id]["description"] = description 
        elif path_parts[1] == "solutions_c++":
            problem_id = os.path.split(path_parts[0])[1]
            for file_name in file_names:
                with open(os.path.join(path, file_name), "r") as code_file:
                    code = code_file.read()
                    data_by_problem[problem_id]["c++"].append(code)
        elif path_parts[1] == "solutions_python":
            problem_id = os.path.split(path_parts[0])[1]
            for file_name in file_names:
                with open(os.path.join(path, file_name), "r") as code_file:
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

if __name__ == '__main__':
    data = get_data()
    for i, desc in enumerate(data):
        if i <= 1:
            print("Description:")
            print(desc, "\n")
            print("C++ solutions:")
            for solution in data[desc]["c++"]:
                print(solution, "\n")
            print("Python solutions:")
            for solution in data[desc]["python"]:
                print(solution, "\n")
            print("\n")

