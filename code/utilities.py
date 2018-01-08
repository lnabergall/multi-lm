"""
Utility functions for handling files, directories, and text, 
as well as performing basic object manipulations.
"""

import os
import tarfile
import io
from zipfile import ZipFile
from zipfile2 import LeanZipFile

import numpy as np
import textacy
import cchardet


MEMORY_LIMIT = 3000000000


def split(sequence, partition_size):
    k, m = divmod(len(sequence), partition_size)
    return (sequence[i*k + min(i, m) : (i+1)*k + min(i+1, m)] 
            for i in range(partition_size))


def invert_mapping(mapping):
    inverse_mapping = {}
    for key, values in mapping.items():
        for value in values:
            inverse_mapping.setdefault(value, []).append(key)

    return inverse_mapping


def get_statistics(sequence):
    """
    Args:
        sequence: List or Tuple of floats or integers.
    Returns:
        A dictionary with statistic names as keys and statistic values
        as values. 
    """
    return {
        "mean": np.mean(sequence), 
        "standard_deviation": np.std(sequence), 
        "median": np.percentile(sequence, 50), 
        "lower_quartile": np.percentile(sequence, 25), 
        "upper_quartile": np.percentile(sequence, 75), 
        "minimum": np.percentile(sequence, 0), 
        "maximum": np.percentile(sequence, 100)
    }


class CustomTarFile(tarfile.TarFile):
    """
    Higher-level interface for manipulating tar files; built upon TarFile.
    """
    def read_file(self, file_name):
        text_file = self.extractfile(file_name)
        return text_file.read().decode()

    def read_files(self):
        texts = []
        file_names = []
        for file_name in self.getnames():
            text_file = self.extractfile(file_name)
            file_names.append(file_name)
            texts.append(text_file.read().decode())

        return texts, file_names

    def read_lines(self, encoding=None, file_name=None):
        file_names = [file_name] if file_name else self.get_names()
        for file_name in file_names:
            text_file = self.extractfile(file_name)
            for line in text_file.read().decode().splitlines():
                yield line

    def add_text_file(self, string, file_name):
        string_file = io.BytesIO(string.encode("utf-8"))
        tarinfo = tarfile.TarInfo(file_name)
        tarinfo.size = len(string)
        self.addfile(tarinfo, string_file)

    def partial_copy(self, copy_name, file_names, encoding="utf-8"):
        texts, text_names = self.read_files()
        with CustomTarFile.open(copy_name, "w", encoding=encoding) as tar:
            for text, name in zip(texts, text_names):
                if name in file_names:
                    tar.add_text_file(text, name)


def open_tarfile(file_path, encoding="utf-8"):
    with CustomTarFile(file_path, "r", encoding=encoding) as tar_file:
        return tar_file.read_files()


def open_text_from_tarfile(file_path, file_name, encoding="utf-8"):
    with CustomTarFile(file_path, "r", encoding=encoding) as tar_file:
        return tar_file.read_file(file_name)


def store_tarfile_data(data_dictionary, file_path, overwrite=False):
    if not os.path.exists(file_path) or overwrite:
        mode = "w"
    else:
        mode = "a"
    with CustomTarFile(file_path, mode, encoding="utf-8") as tar_file:
        for key, value in data_dictionary.items():
            if not isinstance(value, str):
                raise ValueError("Expected a string!")
            else:
                tar_file.add_text_file(value, key)


def open_zipfile(file_path):
    with ZipFile(file_path, "r") as zip_file:
        return [zip_file.read(file_name).decode() 
                for file_name in zip_file.namelist()]


def get_filenames(zip_file_path):
    with ZipFile(zip_file_path, "r") as zip_file:
        return zip_file.namelist()


def open_text_from_zipfile(file_path, file_name):
    with LeanZipFile(file_path) as zip_file:
        return zip_file.read(file_name).decode()


def store_zipfile_data(data_dictionary, file_path, overwrite=False):
    if not os.path.exists(file_path) or overwrite:
        mode = "w"
    else:
        mode = "a"
    with ZipFile(file_path, mode) as zip_file:
        for key, value in data_dictionary.items():
            if not isinstance(value, str):
                raise ValueError("Expected a string!")
            else:
                zip_file.writestr(key, value)


def is_excessively_large(file_path):
    file_size = os.stat(file_path).st_size
    return file_size > MEMORY_LIMIT


def get_text_chunks(text_file):
    while True:
        text_chunk = text_file.read(MEMORY_LIMIT)
        if not text_chunk:
            break
        yield text_chunk


def open_file(file_path=None, file_object=None, encoding=None, large=None):
    if file_path is None and file_object is None:
        raise ValueError("Need a file path or file object to open!")
    if file_object is not None:
        return get_text_chunks(file_object)
    file_too_large = is_excessively_large(file_path)
    if not encoding:
        encoding = get_encoding(file_path)
    try:
        with open(file_path, "r", encoding=encoding) as text_file:
            if large or (file_too_large and large is not False):
                return get_text_chunks(text_file)
            else:
                text = text_file.read()
    except (UnicodeDecodeError, LookupError):
        try:
            with open(file_path, "r") as text_file:
                if large or (file_too_large and large is not False):
                    return get_text_chunks(text_file)
                else:
                    text = text_file.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as text_file:
                if large or (file_too_large and large is not False):
                    return get_text_chunks(text_file)
                else:
                    text = text_file.read()

    return text


def store_text(text, file_path, encoding="utf-8", append=False):
    if append:
        mode = "a"
    else:
        mode = "w"
    with open(file_path, mode, encoding=encoding) as output_file:
        output_file.write(text)


def get_folder_index(path, folder_name):
    return path.split("\\").index(folder_name)


def get_files_from_directory(root_path, extensions=None):
    files = [file for file in os.listdir(root_path) 
             if os.path.isfile(os.path.join(root_path, file))]
    if extensions is not None:
        files = [file for file in files 
                 if any(file.lower().endswith(extension.lower()) 
                        for extension in extensions)]
    return files


def remove_extension(file_name):
    return file_name[:file_name.rfind(".")]


def add_filename_suffix(file_path, suffix):
    return (file_path[:file_path.rfind(".")] + suffix 
            + file_path[file_path.rfind("."):])


def get_file_path(root_path, identifiers):
    file_name = [file_name for file_name in os.listdir(root_path)
                 if all(identifier in file_name for identifier in identifiers)][0]
    return os.path.join(root_path, file_name)


def get_encoding(file_path=None, file_object=None):
    if not file_object:
        with open(file_path, "rb") as file:
            text = file.read()
    else:
        text = file_object.read()
    return cchardet.detect(text)["encoding"]


def detect_language(file_path, text=None, source_code=False):
    from .data_preparation import PYTHON, C, FORTRAN, LISP  # For circular import
    language = None
    file_path_lower = file_path.lower()
    if file_path_lower.endswith(".py") or file_path_lower.endswith(".pyw"):
        language = PYTHON
    elif file_path_lower.endswith(".c") or file_path_lower.endswith(".h"):
        language = C
    elif (file_path_lower.endswith(".f") or file_path_lower.endswith(".for") 
            or file_path_lower.endswith(".f90") or file_path_lower.endswith(".f95")
            or file_path_lower.endswith(".f03") or file_path_lower.endswith(".f08")
            or file_path_lower.endswith(".f15")):
        language = FORTRAN
    elif (file_path_lower.endswith(".lisp") or file_path_lower.endswith(".lsp")
            or file_path_lower.endswith(".l") or file_path_lower.endswith(".cl")
            or file_path_lower.endswith(".fasl")):
        language = LISP
    else:
        if not source_code:
            if text is None:
                text = open_file(file_path)
            if textacy.text_utils.detect_language(text).lower() == "en":
                language = ENGLISH
            elif textacy.text_utils.detect_language(text).lower() == "fr":
                language = FRENCH
            elif textacy.text_utils.detect_language(text).lower() == "de":
                language = GERMAN
            elif textacy.text_utils.detect_language(text).lower() == "zh":
                language = CHINESE

    return language


def get_spell_corrector():
    from .data_preparation import BASE_DIR  # for circular import
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
        text = text.replace(word, corrected_word)

    return text