"""
Classes and functions for processing and generating data.

TODO: Implement data oversampling and/or undersampling
"""

import os
import re
import shutil
import tarfile
import tokenize as py_token
from collections import Counter, namedtuple
from random import shuffle

import textacy
import tensorflow as tf
from pygments import lex
from pygments.lexers.python import PythonLexer
from pygments.lexers.c_cpp import CLexer
from pygments.lexers.fortran import FortranFixedLexer, FortranLexer
from pygments.lexers.lisp import CommonLispLexer
from bs4 import UnicodeDammit
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators.text_encoder import (
    TokenTextEncoder, EOS, RESERVED_TOKENS)
from tensor2tensor.data_generators.problem import Problem
from tensor2tensor.utils import registry

from data_preparation import (get_files_from_directory, open_file,
                              get_encoding, detect_language, BASE_DIR, 
                              CustomTarFile, open_tarfile)


BASE_DIR = ("C:\\Users\\lnabe\\Dropbox\\Artificial Intelligence and Robotics\\"
            "learning-language\\data\\language_modeling")

NATURAL_LANGUAGE = "natural language"
PROGRAMMING_LANGUAGE = "programming language"
MARKUP_LANGUAGE = "markup language"

C = "c"
PYTHON = "python"
FORTRAN = "fortran"
LISP = "lisp"

ENGLISH = "english"
FRENCH = "french"
GERMAN = "german"
CHINESE = "chinese"

HTML = "html"
LATEX = "latex"
YAML = "yaml"
MARKDOWN = "markdown"

# Classification type tuples: ("folder-name", "corpus-name", 
#                              "category1", "category2", ...)

# English datasets
AMAZON_REVIEWS = ("amazon_reviews", "amazon review corpus", "amazon review")
BLOG_CORPUS = ("blog_authorship_corpus", "blog authorship corpus", "blog")
BROWN_CORPUS = ("brown_corpus", "brown corpus", None)
ENGLISH_WIKI = ("enwiki-20140707-corpus.xml", 
                "english wikipedia corpus", "wikipedia")
GUTENBERG = ("gutenberg", "gutenberg dataset", "book")

# French datasets
CHAMBERS_ROSTAND_CORPUS = ("chambers_rostand_journalistic_corpus", 
                           "chambers-rostand journalistic corpus", "news")
FRENCH_WIKI = ("frwiki-20140804-corpus.xml", 
               "french wikipedia corpus", "wikipedia")
ORAL_NARRATIVE_CORPUS = ("oral_narrative_corpus", "french oral narrative corpus", 
                         "speech", "narrative")
ABU_CORPUS = ("abu_corpus", "abu corpus", None)

# German datasets
GERMAN_BIBLE = ("german_bible", "german bible", "bible")
GERMAN_WIKI = ("dewiki-20140725-corpus.xml", 
               "german wikipedia corpus", "wikipedia")
GERMANC = ("GerManC", "germanc corpus", None)
PAROLE_CORPUS = ("parole_corpus", "german parole corpus", None)

# Chinese datasets
LANCASTER_CORPUS = ("lancaster_mandarin_corpus", 
                    "lancaster mandarin corpus", None)
CHINESE_WIKI = ("zhwiki-20140804-corpus.xml", 
                "chinese wikipedia corpus", "wikipedia")
LEIDEN_WEIBO_CORPUS = ("leiden_weibo_corpus-messages", 
                       "leiden weibo corpus", "microblog")


UNKNOWN_TOKEN = "<UNK>"


def get_all_corpora_info():
    corpora_info = []
    for name, value in globals().items():
        if (re.fullmatch(r"[A-Z_]+", name) 
                and type(value) == tuple and len(value) >= 3):
            corpora_info.append(value)

    return corpora_info


def get_corpora(language_types=None, languages=None, corpora=None):
    corpora_info = get_all_corpora_info()
    corpora = []
    for dir_path, dir_names, file_names in os.walk(directory_path):
        if "processed" in dir_names:
            classification = get_classification(dir_path)
            corpus_directory = dir_path.split("\\")[-1]
            name = [dataset_info[1] for dataset_info in datasets 
                    if dataset_info[0] == corpus_directory][0]
            if ((language_types is not None 
                    and classification[0] in language_types)
                    or (languages is not None 
                        and classification[1] in languages)
                    or (corpora is not None and name in corpora)):
                corpora.append(corpus_directory, name, classification)

    return corpora


def get_classification(directory_path):
    dataset_directories = [dataset_info[0] for dataset_info 
                           in get_all_corpora_info()]
    directories = directory_path.split("\\")
    base_index = [i for i, directory in enumerate(directories) 
                  if directory == "language_modeling"][0]
    return [directory for i, directory in enumerate(directories) 
            if i > base_index and directory != "processed" 
            and directory not in dataset_directories]


class MultiLmProblem(Problem):

    def generate_data(self, data_dir, tmp_dir):
        data_paths = [os.path.join(BASE_DIR, "t2t_data")]
        generator_utils.generate_dataset_and_shuffle(
            self.dataset_collection.training_generator(), data_paths, 
            self.dataset_collection.validation_generator(), data_paths)

    def dataset_filename(self):
        return BASE_DIR

    def hparams(self, defaults, model_hparams):
        defaults.input_modality = {"inputs": (registry.Modalities.SYMBOL,
                                              self.vocab_size)}
        defaults.target_modality = (registry.Modalities.SYMBOL, self.vocab_size)

    def feature_encoders(self, data_dir):
        return {"inputs": self.text_encoder, "targets": self.text_encoder}


def get_large_dataset():
    """All corpora and a vocabulary size of 2000000."""
    vocab_size = 2000000
    directory_path = BASE_DIR
    corpora_info = get_corpora(
        language_types=[NATURAL_LANGUAGE, PROGRAMMING_LANGUAGE])

    return directory_path, corpora_info, vocab_size


def get_small_dataset():
    """
    All programming language corpora plus a corpora 
    from each natural language; a vocabulary size of 1500000.
    """
    vocab_size = 1500000
    directory_path = BASE_DIR
    corpora_info = get_corpora(
        language_types=[PROGRAMMING_LANGUAGE,],
        corpora=[ENGLISH_WIKI[1], CHAMBERS_ROSTAND_CORPUS[1], 
                 GERMAN_BIBLE[1], LEIDEN_WEIBO_CORPUS[1]])

    return directory_path, corpora_info, vocab_size


def get_natural_language_dataset():
    """All natural language corpora and a vocabulary size of 1000000."""
    pavocab_size = 1000000
    directory_path = BASE_DIR
    corpora_info = get_corpora(language_types=[NATURAL_LANGUAGE,])

    return directory_path, corpora_info, vocab_size


@registry.register_problem("multi_lm_natural_lang")
class MultiLmNaturalLang(MultiLmProblem):

    def __init__(self, *args, *kwargs):
        super(MultiLmProblem, self).__init__(*args, **kwargs)
        self.dataset_collection = DatasetCollection(
            get_natural_language_dataset()*)
        self.text_encoder = self.dataset_collection.text_encoder
        self.vocab_size = self.text_encoder.vocab_size


def get_programming_language_dataset():
    """All programming language corpora and a vocabulary size of 1000000."""
    pavocab_size = 1000000
    directory_path = BASE_DIR
    corpora_info = get_corpora(language_types=[PROGRAMMING_LANGUAGE,])

    return directory_path, corpora_info, vocab_size


@registry.register_problem("multi_lm_programming_lang")
class MultiLmProgrammingLang(MultiLmProblem):

    def __init__(self, *args, *kwargs):
        super(MultiLmProblem, self).__init__(*args, **kwargs)
        self.dataset_collection = DatasetCollection(
            get_programming_language_dataset()*)
        self.text_encoder = self.dataset_collection.text_encoder
        self.vocab_size = self.text_encoder.vocab_size


def get_test_natural_language_dataset():
    """A corpus for each natural language and a vocabulary size of 1000000."""
    vocab_size = 1000000
    directory_path = BASE_DIR
    corpora_info = get_corpora(
        corpora=[ENGLISH_WIKI[1], CHAMBERS_ROSTAND_CORPUS[1], 
                 GERMAN_BIBLE[1], LEIDEN_WEIBO_CORPUS[1]])

    return directory_path, corpora_info, vocab_size


@registry.register_problem("multi_lm_natural_lang_test")
class MultiLmNaturalLangTest(MultiLmProblem):

    def __init__(self, *args, *kwargs):
        super(MultiLmProblem, self).__init__(*args, **kwargs)
        self.dataset_collection = DatasetCollection(
            get_test_natural_language_dataset()*)
        self.text_encoder = self.dataset_collection.text_encoder
        self.vocab_size = self.text_encoder.vocab_size


def get_gutenberg_dataset():
    """The Gutenberg corpus and a vocabulary size of 250000"""
    vocab_size = 250000
    directory_path = BASE_DIR
    corpora_info = get_corpora(corpora=[GUTENBERG[1],])

    return directory_path, corpora_info, vocab_size


@registry.register_problem("multi_lm_gutenberg")
class MultiLmGutenberg(MultiLmProblem):

    def __init__(self, *args, *kwargs):
        super(MultiLmProblem, self).__init__(*args, **kwargs)
        self.dataset_collection = DatasetCollection(get_gutenberg_dataset()*)
        self.text_encoder = self.dataset_collection.text_encoder
        self.vocab_size = self.text_encoder.vocab_size


def get_english_wiki_dataset():
    """The English Wikipedia corpus and a vocabulary size of 500000"""
    vocab_size = 500000
    directory_path = BASE_DIR
    corpora_info = get_corpora(corpora=[ENGLISH_WIKI[1],])

    return directory_path, corpora_info, vocab_size


@registry.register_problem("multi_lm_enwiki")
class MultiLmEnWiki(MultiLmProblem):

    def __init__(self, *args, *kwargs):
        super(MultiLmProblem, self).__init__(*args, **kwargs)
        self.dataset_collection = DatasetCollection(get_english_wiki_dataset()*)
        self.text_encoder = self.dataset_collection.text_encoder
        self.vocab_size = self.text_encoder.vocab_size


class CustomTokenTextEncoder(TokenTextEncoder):

    def __init__(self, *args, extra_tokens=[], **kwargs):
        super(CustomTokenTextEncoder, self).__init__(*args, **kwargs)
        self._extra_tokens = extra_tokens

    def _load_vocab_from_file(self, file_path):
        self._token_to_id = {}
        self._id_to_token = {}

        for idx, tok in enumerate(RESERVED_TOKENS):
            self._token_to_id[tok] = idx
            self._id_to_token[idx] = tok

        token_start_idx = self._num_reserved_ids
        with tf.gfile.Open(filename) as file:
            for i, line in enumerate(file):
                idx = token_start_idx + i
                if line.count(", ") != 1:
                    raise ValueError("Unexpected vocabulary file formatting!")
                tok, _ = line.rstrip().split(", ")
                self._token_to_id[tok] = idx
                self._id_to_token[idx] = tok
            for tok in self._extra_tokens:
                idx += 1
                self._token_to_id[tok] = idx
                self._id_to_token[idx] = tok

class Document:

    def __init__(self, file_name, file_path, classification, 
                 text=None, tokenize=True):
        self.text = text
        self.classification = classification
        self.language = self.classification[1]
        self.file_name = file_name
        self.file_path = file_path
        if text and tokenize:
            self._tokenize()

    def _tokenize(self):
        self.tokens = tokenize(self.text, self.language, self.file_name)

    def _detokenize(self):
        raise NotImplementedError

    def _encode(self, encoder):
        self.ids = encoder.encode(self.tokens)

    def _decode(self, encoder):
        self.tokens = encoder.decode(self.ids)

    def tokenize(self, string):
        return tokenize(string, self.language, self.file_name)

    def encode(self, tokens, encoder):
        return encoder.encoder(tokens)


class Corpus:

    def __init__(self, name, directory_path, classification, load_texts=False):
        self.name = name
        self.directory_path = directory_path
        self.classification = classification

        if load_texts:
            self._load_documents()
            self._build_vocabulary()
            self._save_vocabulary()
        else:
            self._load_document_metadata()

    def _add_to_partition(self, directory_path, document):
        if directory_path.endswith("training"):
            self.training_documents.append(document)
        elif directory_path.endswith("validation"):
            self.validation_documents.append(document)
        elif directory_path.endswith("test"):
            self.test_documents.append(document)

    def _load_documents(self):
        self.documents = []
        self.training_documents = []
        self.validation_documents = []
        self.test_documents = []
        for dir_path, dir_names, file_names in os.walk(self.directory_path):
            for file_name in file_names:
                if file_name.split(".")[-2].endswith("_processed"):
                    file_path = os.path.join(dir_path, file_name)
                    classification = get_classification(dir_path)
                    if file_name.endswith(".tar"):
                        texts, file_names = open_tarfile(file_path, encoding="utf-8")
                        for text, file_name in zip(texts, file_names):
                            self.documents.append(
                                Document(file_name, file_path, 
                                         classification, text=text))
                            self._add_to_partition(dir_path, self.documents[-1])
                    else:
                        text = open_file(file_path, encoding="utf-8")
                        self.documents.append(
                            Document(file_name, file_path, 
                                     classification, text=text))
                        self._add_to_partition(dir_path, self.documents[-1])
        self.token_count = sum(len(document.tokens) for document in self.documents)

    def _load_document_metadata(self):
        self.documents = []
        self.training_documents = []
        self.validation_documents = []
        self.test_documents = []
        for dir_path, dir_names, file_names in os.walk(self.directory_path):
            for file_name in file_names:
                if file_name.split(".")[-2].endswith("_processed"):
                    file_path = os.path.join(dir_path, file_name)
                    classification = get_classification(dir_path)
                    if file_name.endswith(".tar"):
                        with CustomTarFile(os.path.join(
                                dir_path, file_name), "r") as tar:
                            file_names = tar.getnames()
                        for file_name in file_names:
                            self.documents.append(
                                Document(file_name, file_path, classification))
                            self._add_to_partition(dir_path, self.documents[-1])
                    else:
                        self.documents.append(
                            Document(file_name, file_path, classification))
                        self._add_to_partition(dir_path, self.documents[-1])

    def _build_vocabulary(self):
        self.vocabulary = Counter()
        for document in self.documents:
            self.vocabulary.update(Counter(document.tokens))

    def _save_vocabulary(self):
        vocab_file_path = os.path.join(self.directory_path, "vocabulary.txt")
        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for token, count in self.vocabulary.items():
                print(token + ", " + count, file=vocab_file)

    def partition(self, training=0.8, validation=0.1):
        shuffle(self.documents)
        training_count = int(training*len(self.documents))
        validation_count = int(validation*len(self.documents))
        self.training_documents = [
            document for i, document in enumerate(self.documents) 
            if i in range(training_count)]
        self.validation_documents = [
            document for i, document in enumerate(self.documents) 
            if i in range(training_count, training_count + validation_count)]
        self.test_documents = [
            document for i, document in enumerate(self.documents) 
            if i in range(training_count + validation_count, len(self.documents))]

        tar_file_indicators = [".tar" in document.file_path 
                               for document in self.documents]
        if all(tar_file_indicators):
            if len(set(document.file_path for document in self.documents)) != 1:
                raise NotImplementedError("Can't partition multiple tar files!")
            with CustomTarFile(self.documents[0].file_path, "r") as tar:
                for document_set, folder_name in [
                        (self.training_documents, "training"), 
                        (self.validation_documents, "validation"),
                        (self.test_documents, "test")]:
                    new_file_path = new_file_path = (
                        "\\".join(document.file_path.split("\\")[:-1]) + "\\"
                        + folder_name + "\\" + document.file_path.split("\\")[-1])
                    file_names = [document.file_name for document in document_set]
                    tar.partial_copy(new_file_path, file_names)
        elif not any(tar_file_indicators):
            for document in self.documents:
                if document in self.training_documents:
                    folder_name = "training"
                elif document in self.validation_documents:
                    folder_name = "validation"
                elif document in self.test_documents:
                    folder_name = "test"
                else:
                    raise RuntimeError("Unexpected exception!")
                new_file_path = (
                    "\\".join(document.file_path.split("\\")[:-1]) + "\\"
                    + folder_name + "\\" + document.file_path.split("\\")[-1])
                shutil.move(document.file_path, new_file_path)
                document.file_path = new_file_path
        else:
            raise RuntimeError("Unexpected file types!")

    def _detokenize(self):
        raise NotImplementedError

    def _encode(self, encoder):
        for document in self.documents:
            document.encode(encoder)

    def _decode(self, encoder):
        for document in self.documents:
            document.decode(encoder)


class DatasetCollection:
    
    def __init__(directory_path, corpora_info, vocab_size=None, load_texts=False):
        self.directory_path = directory_path
        self.corpora = []
        for corpus_directory, name, classification in corpora_info:
            corpus_directory_path = os.path.join(
                self.directory_path, corpus_directory)
            self.corpora.append(
                Corpus(name, corpus_directory_path, 
                       classification, load_texts=load_texts))

        self.unknown_token = UNKNOWN_TOKEN
        if load_texts:
            self.token_count = sum(corpus.token_count for corpus in self.corpora) 
            self._build_vocabulary(vocab_size)
            self._save_vocabulary()
        else:
            self._load_vocabulary()
            self.text_encoder = CustomTokenTextEncoder(
                self.vocab_file_path, 
                extra_tokens=[self.tag_token, self.unknown_token])

    def _build_vocabulary(self, vocab_size=None):
        self.vocabulary = Counter()
        for corpus in self.corpora:
            self.vocabulary.update(corpus.vocabulary)
        if vocab_size:
            vocab_size -= 2  # accounts for tag + unknown tokens
            self.truncated_vocabulary = {
                token: count for token, count 
                in self.vocabulary.most_common(vocab_size)}
        self._assign_tag_token()

    def _save_vocabulary(self):
        vocab_size = len(self.truncated_vocabulary)
        self.vocab_file_path = os.path.join(
            self.directory_path, "vocabulary_" + str(vocab_size) + ".txt")
        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for token, count in self.truncated_vocabulary.items():
                print(token + ", " + count, file=vocab_file)
        self.text_encoder = CustomTokenTextEncoder(self.vocab_file_path)

    def _load_vocabulary(self):
        # Assumes the path of the vocabulary file
        self.vocab_file_path = os.path.join(
            self.directory_path, "vocabulary_" + str(vocab_size) + ".txt")
        self.truncated_vocabulary = Counter()
        with open(self.vocab_file_path, "r", encoding="utf-8") as vocab_file:
            for line in vocab_file:
                if line.count(", ") != 1:
                    raise ValueError("Unexpected vocabulary file formatting!")
                token, count = line.rstrip().split(", ")
                self.truncated_vocabulary[token] = int(count)
        self._assign_tag_token()

    def _assign_tag_token(self):
        self.tag_token = None
        token_id = 0
        while self.tag_token is None:
            if chr(token_id) not in self.vocabulary:
                self.tag_token = chr(token_id)

    def partition_datasets(self):
        for corpus in self.corpora:
            corpus.partition()

    @property
    def unknown_token_count(self):
        return sum(count for token, count in self.vocabulary.items() 
                   if token not in self.truncated_vocabulary)

    def train_generator(self):
        return self._generator("training")

    def valid_generator(self):
        return self._generator("validation")

    def test_generator(self):
        raise NotImplementedError

    def encode_line(self, line, document, first_line=False):
        tag_tokens = []
        if first_line:
            for tag in tags:
                tag_tokens.extend([self.tag_token, tag, self.tag_token])
        tokens = document.tokenize(line)
        # Replace OOV tokens
        tokens = [token if token in self.truncated_vocabulary 
                  else self.unknown_token for token in tokens]  
        return document.encode(tag_tokens + tokens, self.encoder)

    def _generator(self, dataset_type):
        for corpus in self.corpora:
            if dataset_type == "training":
                documents = corpus.training_documents
            elif dataset_type == "validation":
                documents = corpus.validation_documents
            for document in documents:
                if document.file_path.endswith(".tar"):
                    with CustomTarFile(file_path, "r") as tar:
                        for i, line in enumerate(tar.read_lines(encoding="utf-8")):
                            first_line = True if i == 0 else False
                            encoded_tokens = self.encode_line(
                                line, document, first_line=first_line)
                            yield {"inputs": encoded_tokens[:-1], 
                                   "targets": encoded_tokens[1:]}
                else:
                    with open(file_path, "r", encoding="utf-8") as file:
                        for i, line in enumerate(file):
                            first_line = True if i == 0 else False
                            encoded_tokens = self.encode_line(
                                line, document, first_line=first_line)
                            yield {"inputs": encoded_tokens[:-1], 
                                   "targets": encoded_tokens[1:]}


def tokenize(text, language, file_name):
    if language == PYTHON[0]:
        tokens = [token_pair[1] for token_pair 
                  in lex(text, PythonLexer(encoding="utf-8"))]
    elif language == C[0]:
        tokens = [token_pair[1] for token_pair 
                  in lex(text, CLexer(encoding="utf-8"))]
    elif language == FORTRAN[0]:
        if file_name.lower().endswith(".f95") or file_name.lower().endswith(".f03"):
            tokens = [token_pair[1] for token_pair 
                      in lex(text, FortranLexer(encoding="utf-8"))]
        else:
            tokens = [token_pair[1] for token_pair 
                      in lex(text, FortranFixedLexer(encoding="utf-8"))]
    elif language == LISP[0]:
        tokens = [token_pair[1] for token_pair 
                  in lex(text, CommonLispLexer(encoding="utf-8"))]
    elif language in [ENGLISH[0], FRENCH[0], GERMAN[0], CHINESE[0]]:
        document = textacy.doc.Doc(text)
        tokens = [str(token) for token in document]

    # Break apart whitespace tokens and remove spaces
    cleaned_tokens = []
    for token in tokens:
        if token == " ":
            continue
        if len(token) != 1 and token.strip() == "":
            whitespace_tokens = token
            cleaned_tokens.extend(
                [token for token in whitespace_tokens if token != " "])
        else:
            cleaned_tokens.append(token)

    return tokens


def replace_rare_tokens(text_list_tokens, threshold=3):
    """
    Args:
        text_list_tokens: List, expects a list of lists of tokens.
    """
    # Collect token counts
    token_counts = {}
    for text in text_list_tokens:
        for token in text:
            prev_count = token_counts.get(token, 0)
            token_counts[token] = prev_count + 1

    # Replace rare tokens
    for i, text in enumerate(text_list_tokens):
        for j, token in enumerate(text):
            if token_counts[token] <= threshold:
                text[j] = UNKNOWN_TOKEN
        text_list_tokens[i] = text

    return text_list_tokens


