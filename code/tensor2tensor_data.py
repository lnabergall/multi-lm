"""Classes and functions for processing and generating data."""

import os
import shutil
import tarfile
import tokenize as py_token
from collections import Counter
from random import shuffle

import textacy
import tensorflow as tf
from pygments import lex
from pygments.lexers.python import PythonLexer
from pygments.lexers.c_cpp import CLexer
from pygments.lexers.fortran import FortranFixedLexer, FortranLexer
from pygments.lexers.lisp import CommonLispLexer
from bs4 import UnicodeDammit
from tensor2tensor.data_generators.text_encoder import (
    TokenTextEncoder, EOS, RESERVED_TOKENS)

from data_preparation import (get_files_from_directory, open_file,
                              get_encoding, detect_language, BASE_DIR, 
                              CustomTarFile, open_tarfile)


# Classification type tuples: ("folder-name", "category1", "category2", ...)

NATURAL_LANGUAGE = ("natural_language", "natural language")
PROGRAMMING_LANGUAGE = ("programming_language", "programming language")
MARKUP_LANGUAGE = ("markup_language", "markup language")

C = ("c", "c")
PYTHON = ("python", "python")
FORTRAN = ("fortran", "fortran")
LISP = ("lisp", "lisp")

ENGLISH = ("english", "english")
FRENCH = ("french", "french")
GERMAN = ("german", "german")
CHINESE = ("chinese", "chinese")

HTML = ("html", "html")
LATEX = ("latex", "latex")
YAML = ("yaml", "yaml")
MARKDOWN = ("markdown", "markdown")

# English datasets
AMAZON_REVIEWS = ("amazon_reviews", "amazon review")
BLOG_CORPUS = ("blog_authorship_corpus", "blog")
BROWN_CORPUS = ("brown_corpus", None)
ENGLISH_WIKI = ("enwiki-20140707-corpus.xml", "wikipedia")
GUTENBERG = ("gutenberg", "book")

# French datasets
CHAMBERS_ROSTAND_CORPUS = ("chambers_rostand_journalistic_corpus", "news")
FRENCH_WIKI = ("frwiki-20140804-corpus.xml", "wikipedia")
ORAL_NARRATIVE_CORPUS = ("oral_narrative_corpus", "speech", "narrative")
ABU_CORPUS = ("abu_corpus", None)

# German datasets
GERMAN_BIBLE = ("german_bible", "bible")
GERMAN_WIKI = ("dewiki-20140725-corpus.xml", "wikipedia")
GERMANC = ("GerManC", None)
PAROLE_CORPUS = ("parole_corpus", None)

# Chinese datasets
LANCASTER_CORPUS = ("lancaster_mandarin_corpus", None)
CHINESE_WIKI = ("zhwiki-20140804-corpus.xml", "wikipedia")
LEIDEN_WEIBO_CORPUS = ("leiden_weibo_corpus-messages", "microblog")


class CustomTokenTextEncoder(TokenTextEncoder):

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


class Document:

    def __init__(self, text, classification, file_name, file_path):
        self.text = text
        self.classification = classification
        self.language = self.classification[1]
        self.file_name = file_name
        self.file_path = file_path
        self._tokenize()

    def _tokenize(self):
        self.tokens = tokenize(text, self.language, self.file_path)

    def _detokenize(self):
        raise NotImplementedError

    def encode(self, encoder):
        self.ids = encoder.encode(self.tokens)
        return self.ids

    def decode(self, encoder):
        self.tokens = decoder.decode(self.ids)
        return encoder.decode(self.ids)


class Corpus:

    def __init__(self, name, directory_path, classification):
        self.name = name
        self.directory_path = directory_path
        self.classification = classification

        self._load_processed_text()
        self._build_vocabulary()
        self._save_vocabulary()

    def _add_to_partition(self, directory_path, document):
        if directory_path.endswith("training"):
            self.training_documents.append(document)
        elif directory_path.endswith("validation"):
            self.validation_documents.append(document)
        elif directory_path.endswith("test"):
            self.test_documents.append(document)

    def _load_processed_text(self):
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
                        texts, file_names = open_tarfile(
                            os.path.join(dir_path, file_name),
                            encoding="utf-8")
                        for text, file_name in zip(texts, file_names):
                            self.documents.append(
                                Document(text, classification, file_name, file_path))
                            self._add_to_partition(dir_path, self.documents[-1])
                    else:
                        text = open_file(os.path.join(dir_path, file_name), 
                                         encoding="utf-8")
                        self.documents.append(
                            Document(text, classification, file_name, file_path))
                        self._add_to_partition(dir_path, self.documents[-1])
        self.token_count = sum(len(document.tokens) for document in self.documents)

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

    def encode(self, encoder):
        for document in self.documents:
            document.encode(encoder)

    def decode(self, encoder):
        for document in self.documents:
            document.decode(encoder)


class DatasetCollection:
    
    def __init__(directory_path, corpora_info, vocab_size=None):
        self.directory_path = directory_path
        self.corpora = []
        for corpus_directory, name, classification in corpora_info:
            corpus_directory_path = os.path.join(
                self.directory_path, corpus_directory)
            self.corpora.append(
                Corpus(name, corpus_directory_path, classification))
        self.token_count = sum(corpus.token_count for corpus in self.corpora) 

        self._build_vocabulary(vocab_size)
        self._save_vocabulary()

    def _build_vocabulary(self, vocab_size=None):
        self.vocabulary = Counter()
        for corpus in self.corpora:
            self.vocabulary.update(corpus.vocabulary)
        self.truncated_vocabulary = {token: count for token, count 
                                     in self.vocabulary.most_common(vocab_size)}

    def _save_vocabulary(self):
        vocab_size = len(self.truncated_vocabulary)
        self.vocab_file_path = os.path.join(
            self.directory_path, "vocabulary_" + str(vocab_size) + ".txt")
        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for token, count in self.truncated_vocabulary.items():
                print(token + ", " + count, file=vocab_file)
        self.text_encoder = CustomTokenTextEncoder(self.vocab_file_path)

    def partition_datasets(self):
        for corpus in self.corpora:
            corpus.partition()

    @property
    def unknown_token_count(self):
        return sum(count for token, count in self.vocabulary.items() 
                   if token not in self.truncated_vocabulary)

    def train_generator(self):
        pass

    def valid_generator(self):
        pass

    def test_generator(self):
        raise NotImplementedError

    def _generator(self):
        pass


def get_classification(directory_path):
    directories = directory_path.split("\\")
    base_index = [i for i, directory in enumerate(directories) 
                  if directory == "language_modeling"][0]
    return [directory for i, directory in enumerate(directories) 
            if i > base_index and directory != "processed"]


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
            if token_counts[token] <= 3:
                text[j] = "<UNK>"
        text_list_tokens[i] = text

    return text_list_tokens


def store_tokens(tokens, file_path):
    extension = file_path.split(".")[-1]
    new_file_path = ".".join(file_path.split(".")[:-1]) + "_processed" + extension
    if detect_language(file_path) in [ENGLISH, FRENCH, GERMAN, CHINESE]:
        processed_text = " ".join(tokens)
    else:
        processed_text = "".join(tokens)
    with open(new_file_path, "w") as processed_text_file:
        processed_text_file.write(processed_text)


