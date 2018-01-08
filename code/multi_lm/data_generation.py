""""""

import os
import random
import codecs
from time import time
from zipfile import ZipFile
from itertools import chain, groupby
from collections import Counter
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
from spacy.lang.en import English
from spacy.lang.fr import French
from spacy.lang.de import German
from spacy.lang.zh import Chinese
from pygments import lex
from pygments.lexers.python import PythonLexer
from pygments.lexers.c_cpp import CLexer
from pygments.lexers.fortran import FortranFixedLexer, FortranLexer
from pygments.lexers.lisp import CommonLispLexer
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators.text_encoder import (
    EOS_ID, RESERVED_TOKENS, TextEncoder, 
    TokenTextEncoder, SubwordTextEncoder)
from cached_property import cached_property
from bs4 import UnicodeDammit

from . import utilities as utils
from . import data_preparation as data


NATURAL_LANGUAGE_PROCESSORS = {
    data.ENGLISH: English(),
    data.FRENCH: French(),
    data.GERMAN: German(),
    data.CHINESE: Chinese(),
}


ROOT_CLASS = "language"
UNKNOWN_TOKEN = "<UNK>"
TOKEN_FILENAME_SUFFIX = "_tokens"
TOKEN_SEPARATOR = " "
ENCODING = "utf-8"
DOCUMENT_LIMIT = 500000
DOCUMENT_SEPARATOR = "doc121sep191ara121tor"  # To guarantee uniqueness


#### SUPPORT ONE BILLION WORD DATASET (AND, GENERALLY, "ARBITRARY" DATASETS)!!!


def clean_tokens(tokens):
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

    return cleaned_tokens


def tokenize(text, language, file_name):
    if language == data.PYTHON:
        tokens = [token_pair[1] for token_pair 
                  in lex(text, PythonLexer(encoding="utf-8"))]
    elif language == data.C:
        tokens = [token_pair[1] for token_pair 
                  in lex(text, CLexer(encoding="utf-8"))]
    elif language == data.FORTRAN:
        if file_name.lower().endswith(".f95") or file_name.lower().endswith(".f03"):
            tokens = [token_pair[1] for token_pair 
                      in lex(text, FortranLexer(encoding="utf-8"))]
        else:
            tokens = [token_pair[1] for token_pair 
                      in lex(text, FortranFixedLexer(encoding="utf-8"))]
    elif language == data.LISP:
        tokens = [token_pair[1] for token_pair 
                  in lex(text, CommonLispLexer(encoding="utf-8"))]
    elif language in [data.ENGLISH, data.FRENCH, data.GERMAN, data.CHINESE]:
        document = NATURAL_LANGUAGE_PROCESSORS[language].tokenizer(text)
        tokens = [token.text for token in document]

    return clean_tokens(tokens)


def tokenize_document(document):
    document.tokenize()


class CustomTokenTextEncoder(TokenTextEncoder):

    def encode(self, tokens):
        """Converts a list of tokens to a list of token ids."""
        if self._replace_oov is not None:
            tokens = [token if token in self._token_to_id 
                      else self._replace_oov for token in tokens]
        ret = [self._token_to_id[tok] for tok in tokens]
        return ret[::-1] if self._reverse else ret


class CustomSubwordTextEncoder(SubwordTextEncoder):

    def encode(self, tokens):
        """Converts a list of tokens to a list of subtoken ids."""
        return self._tokens_to_subtoken_ids(tokens)


class CustomCharacterTextEncoder(TextEncoder):

    def encode(self, characters):
        return [int(char) + self._num_reserved_ids for char in characters]


class DatasetConfiguration:

    def __init__(self, name, category_config, sampling_method, 
                 vocab_type, vocab_size=None, vocab_min_count=None, 
                 vocab_file_path=None, root_directory=None, 
                 total_token_count=None, min_doc_length=None, 
                 max_doc_length=None, use_categories=True, use_lines=False):
        """
        Args:
            name: String, name of this dataset configuration.
            category_config: List of 2-tuples of the form 
                (domain, subcategory_count, partition_type, doc_languages), 
                where domain is a string, subcategory_count is an integer 
                representing the number of subcategories to sample, 
                partition_type is either 'training', 'validation', 'test', 
                'split', 'split_as_one', 'top_domain', 'bottom_domain', 
                or 'document', and doc_languages is either None or 
                a tuple of languages.
            sampling_method: String, either 'truncate' or 'equalize'.
            vocab_type: String, either 'token', 'subtoken', or 'character'.
            vocab_size: Integer, defaults to None.
            vocab_min_count: Integer, minimum count required for a token 
                or subtoken to be included in the vocabulary. 
                Defaults to None.
            vocab_file_path: String, defaults to None.
            root_directory: String, defaults to None.
            total_token_count: Integer, defaults to None.
            min_doc_length: Integer, defaults to None.
            max_doc_length: Integer, defaults to None.
            use_categories: Boolean, defaults to True. Determines whether 
                category tags are included.
            use_lines: Boolean, defaults to False. Determines whether 
                documents are split into lines.
        """
        self.name = name
        self.category_config = category_config
        self.sampling_method = sampling_method
        self.vocab_type = vocab_type
        self.vocab_size = vocab_size
        self.vocab_min_count = vocab_min_count
        self.vocab_file_path = vocab_file_path
        self.root_directory = root_directory
        self.total_token_count = total_token_count
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.use_categories = use_categories
        self.use_lines = use_lines

        self.vocabulary = None
        self.truncated_vocabulary = None

    def __repr__(self):
        return "<DatasetConfig(name={}, categories={})>".format(
            self.name, self.category_config)

    @classmethod
    def from_file(cls, file_path):
        root_directory = os.path.split(file_path)[0]
        dataset_description = utils.open_file(file_path)
        (name, category_config, sampling_method, vocab_type, vocab_size, 
         vocab_min_count, vocab_file_path, total_token_count, use_lines) = tuple(
            dataset_description.splitlines())
        vocab_size = int(vocab_size)
        category_config = eval(category_config)
        use_lines = eval(use_lines)
        vocab_file_path = (file_path.replace("config", "vocab") 
            if vocab_file_path == "None" else vocab_file_path)
        parameters = [vocab_size, vocab_min_count, total_token_count]
        for i, parameter in enumerate(parameters):
            if parameter == "None":
                parameters[i] = None
            else:
                parameters[i] = int(parameter)
        return cls(name, category_config, sampling_method, vocab_type, 
                   parameters[0], parameters[1], vocab_file_path, 
                   total_token_count=parameters[2], root_directory=root_directory,
                   use_lines=use_lines)

    def sample(self, root_category, sample_largest=True):
        categories = root_category.all_categories
        determined_category_config = []
        for (domain, subcategory_count, 
                partition_type, doc_languages) in self.category_config:
            if subcategory_count < 1:
                determined_category_config.append(
                    (domain, subcategory_count, 
                     partition_type, doc_languages))
                continue
            try:
                category = [category for category in categories 
                            if category.category == domain][0]
            except IndexError:
                raise ValueError("Domain not found in category tree!")
            else:
                if sample_largest:
                    # sample subcategories with most documents
                    sampled_categories = sorted(
                        category.subcategories, 
                        key=lambda subcategory: len(subcategory.all_documents),
                        reverse=True)[:subcategory_count]
                    sampled_categories = [cat.category for cat in sampled_categories]
                else:
                    # sample randomly
                    sampled_categories = random.sample(
                        [subcategory.category for subcategory 
                         in category.subcategories], subcategory_count)
                determined_category_config.extend(
                    [(category, 0, partition_type, doc_languages) 
                     for category in sampled_categories])

        self.category_config = determined_category_config

    @property
    def file_path(self):
        file_name = self.name.replace(" ", "_") + "_" + self.vocab_type
        file_name += (str(self.vocab_size) if self.vocab_size 
                      else str(self.vocab_min_count))
        file_name += "_size" + str(self.total_token_count) + ".txt"
        file_path = os.path.join(self.root_directory or data.BASE_DIR, file_name)

        return file_path

    def store(self):
        file_path = utils.add_filename_suffix(self.file_path, "_config")
        dataset_description = "\n".join([
            self.name, str(self.category_config), self.sampling_method, 
            str(self.vocab_type), str(self.vocab_size), 
            str(self.vocab_min_count), str(self.vocab_file_path),
            str(self.total_token_count), str(self.use_lines)])
        utils.store_text(dataset_description, file_path)

    @property
    def unknown_token_count(self):
        return sum(count for token, count in self.vocabulary.items() 
                   if token not in self.truncated_vocabulary)

    def _assign_tag_token(self):
        self.tag_token = None
        token_id = 32
        while self.tag_token is None:
            if chr(token_id) not in self.vocabulary:
                self.tag_token = chr(token_id)
            else:
                token_id += 1
        self.vocabulary[self.tag_token] = 100000
        if self.truncated_vocabulary is not None:
            self.truncated_vocabulary[self.tag_token] = 100000

    def generate_truncated_vocab(self):
        if self.vocab_size is not None:
            self.truncated_vocabulary = {
                token: count for token, count 
                in self.vocabulary.most_common(self.vocab_size)}
        else:
            self.truncated_vocabulary = {
                token: count for token, count
                in self.vocabulary if count >= self.vocab_min_count}
        self.truncated_vocabulary.setdefault(
            UNKNOWN_TOKEN, self.unknown_token_count)

    def _store_vocabulary(self, truncated=False):
        if truncated:
            vocabulary = self.truncated_vocabulary
        else:
            vocabulary = self.vocabulary
        if self.vocab_file_path is None:
            self.vocab_file_path = utils.add_filename_suffix(
                self.file_path, "_vocab")
            if not truncated:
                self.vocab_file_path = utils.add_filename_suffix(
                    self.vocab_file_path, "_complete")
        vocab_description = "\n".join(token + ", " + str(count) for token, count 
                                      in vocabulary.items())
        utils.store_text(vocab_description, self.vocab_file_path)

    def _load_vocabulary(self):
        if self.vocab_file_path is None:
            self.vocab_file_path = utils.add_filename_suffix(
                self.file_path, "_vocab")
            if not os.path.exists(self.vocab_file_path):
                try:
                    self.vocab_file_path = utils.get_file_path(
                        self.root_directory, 
                        [self.name, str(self.total_token_count), "vocab_complete"])
                except IndexError:
                    raise RuntimeError("Vocabulary file not found!")
        if self.vocab_type in ["token", "subtoken"]:
            vocab_description = utils.open_file(self.vocab_file_path)
            self.vocabulary = Counter()
            for line in vocab_description.splitlines():
                values = line.split(", ")
                if len(values) > 2:
                    raise RuntimeError
                elif len(values) == 2:
                    token, count = values
                else:
                    continue
                token = "\n" if not token else token
                self.vocabulary[token] = int(count) if count.strip() else None
        if self.vocab_type == "subtoken" and len(self.vocabulary) > 2000000:
            self.vocabulary = Counter(dict(self.vocabulary.most_common(2000000)))

    def create_text_encoder(self):
        load_vocab = self.vocabulary is None and self.truncated_vocabulary is None
        if load_vocab:
            self._load_vocabulary()
        else:
            self._store_vocabulary()
            if self.vocab_type == "token":
                self._store_vocabulary(truncated=True)
        self._assign_tag_token()
        if self.vocab_type == "token":
            self.generate_truncated_vocab()
            encoder = CustomTokenTextEncoder(
                None, vocab_list=list(self.truncated_vocabulary.keys()), 
                replace_oov=UNKNOWN_TOKEN)
        elif self.vocab_type == "subtoken":
            if (load_vocab and self.vocab_type in self.vocab_file_path 
                    and "complete" not in self.vocab_file_path):
                encoder = CustomSubwordTextEncoder(self.vocab_file_path)
            else:
                if self.vocab_size is not None:
                    encoder = CustomSubwordTextEncoder.build_to_target_size(
                        self.vocab_size, self.vocabulary, 3, 200000)
                else:
                    encoder = CustomSubwordTextEncoder()
                    encoder.build_from_token_counts(
                        self.vocabulary, self.vocab_min_count)
                self.vocab_file_path = utils.add_filename_suffix(
                    self.file_path, "_vocab")
                encoder.store_to_file(self.vocab_file_path)
        elif self.vocab_type == "character":
            encoder = CustomCharacterTextEncoder()

        self.encoder = encoder

    def encode_classification(self, classification):
        classification_tokens = list(chain.from_iterable(
            [domain, self.tag_token] 
            for domain in classification))
        # extra token to signal start of document
        classification_tokens += [self.tag_token]
        return self.encoder.encode(classification_tokens)


def get_directory_dataset(directory_path):
    directory = os.path.split(directory_path)[1].lower()
    if "train" in directory:
        dataset_name = "training"
    elif "dev" in directory or "valid" in directory:
        dataset_name = "validation"
    elif "test" or "holdout" in directory:
        dataset_name = "test"
    else:
        raise ValueError("Directory path doesn't match any known dataset name!")

    return dataset_name


class Category:

    def __init__(self, category, category_type=None, directory_paths=[], 
                 dataset_config=None, documents=[], subcategories=[]):
        self.category = category
        self.category_type = category_type
        self.directory_paths = list(directory_paths)  # To document-containing folders
        self.dataset_config = dataset_config
        self.documents = list(documents)
        self.subcategories = list(subcategories)

    def __repr__(self):
        return "<Category({})>".format(self.category)

    @classmethod                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    def from_classifications(cls, classifications):
        root_category = cls(ROOT_CLASS, directory_paths=[data.BASE_DIR])
        for classification in classifications:
            categories = root_category.all_categories
            top_category = root_category
            for i, domain in enumerate(classification):
                if i == len(classification)-1:
                    directory_path = classification.directory_path
                else:
                    directory_path = None
                try:
                    top_category = [category for category in categories 
                                    if category.category == domain][0]
                    if (directory_path is not None 
                            and directory_path not in top_category.directory_paths):
                        top_category.directory_paths.append(directory_path)
                except IndexError:
                    if domain == classification.language_type:
                        category_type = "language_type"
                    elif domain == classification.language:
                        category_type = "language"
                    else:
                        category_type = "domain"
                    directory_paths = (
                        [directory_path] if directory_path is not None else [])
                    category = cls(domain, category_type=category_type, 
                                   directory_paths=directory_paths)
                    top_category.add_subcategory(category)
                    top_category = category

        return root_category

    @property
    def all_categories(self):
        return [self] + list(chain.from_iterable(
            [category.all_categories for category in self.subcategories]))

    @cached_property
    def all_documents(self):
        return self.documents + list(chain.from_iterable(
            [category.all_documents for category in self.subcategories]))

    def dataset_categories(self, include_doc_language=False):
        all_categories = self.all_categories
        dataset_categories = []
        for domain, _, _, doc_languages in self.dataset_config.category_config:
            try:
                category = [category for category in all_categories 
                            if category.category == domain][0]
            except IndexError:
                continue
            else:
                if include_doc_language:
                    dataset_categories.append((category, doc_languages))
                else:
                    dataset_categories.append(category)

        return dataset_categories

    @cached_property
    def dataset_documents(self):
        categories = self.dataset_categories(include_doc_language=True)
        documents = list(chain.from_iterable(
            [[document for document in category.all_documents 
              if document.classification.language in languages] 
              for category, languages in categories]))
        min_length = self.dataset_config.min_doc_length 
        max_length = self.dataset_config.max_doc_length
        return Document.filter(documents, min_length, max_length)

    @cached_property
    def training_documents(self):
        return [document for document in self.all_documents 
                if document.dataset_assigned == "training"
                and document.copies > 0]

    @cached_property
    def validation_documents(self):
        return [document for document in self.all_documents 
                if document.dataset_assigned == "validation"
                and document.copies > 0]

    @property
    def tokens(self):
        tokens = 0
        for document in self.all_documents:
            tokens += document.token_count
        return tokens

    @property
    def training_tokens(self):
        tokens = 0
        for document in self.training_documents:
            tokens += document.copies*document.token_count
        return tokens

    @property
    def validation_tokens(self):
        tokens = 0
        for document in self.validation_documents:
            tokens += document.copies*document.token_count
        return tokens

    def get_subcategories(self, criterion=None, depth=None):
        subcategories = []
        if depth is not None:
            if depth == 0:
                subcategories = [self]
            else:
                subcategories = list(chain.from_iterable(
                    [category.get_subcategories(depth=depth-1) 
                     for category in self.subcategories]))
        elif criterion == "split":
            subcategories = self.subcategories
        elif criterion == "top_domain":
            if self.category_type == "language":
                subcategories = self.subcategories
            else:
                for category in self.subcategories:
                    subcategories.extend(
                        category.get_subcategories(criterion=criterion))
        elif criterion == "bottom_domain" or criterion == "document":
            if not self.subcategories:
                subcategories = [self]
            else:
                for category in self.subcategories:
                    subcategories.extend(
                        category.get_subcategories(criterion=criterion))
        else:
            raise ValueError("Unexpected subcategory filter criterion!")

        return subcategories

    def add_subcategory(self, subcategory):
        self.subcategories.append(subcategory)

    def add_document(self, document):
        for i, domain in enumerate(document.classification):
            if self.category == domain:
                try:
                    next_domain = document.classification[i+1]
                except IndexError:
                    self.documents.append(document)
                else:
                    matching_subcategories = [
                        subcategory for subcategory in self.subcategories 
                        if subcategory.category == next_domain]
                    if matching_subcategories:
                        if len(matching_subcategories) > 1:
                            raise NotImplementedError(
                                "Multiple matching subcategories, can't disambiguate!")
                        else:
                            matching_subcategories[0].add_document(document)
                    else:
                        # Make new category to accommodate this document
                        category = Category(next_domain)
                        self.add_subcategory(category)
                        category.add_document(document)
        else:
            # Assumes currently in the (placeholder) 'root' category
            category = Category(document.classification[0])
            self.add_subcategory(category)
            category.add_document(document)

    def load_documents(self, all_documents=False, already_partitioned=False):
        if all_documents:
            categories = self.all_categories 
        else:
            categories = chain.from_iterable(
                [category.all_categories for category in self.dataset_categories()])
        for category in categories:
            for directory_path in category.directory_paths:
                if directory_path is None or directory_path == data.BASE_DIR:
                    continue
                if already_partitioned:
                    # assumes directories for training, validation, and test
                    directory_paths = [os.path.join(directory_path, directory) 
                        for directory in os.listdir(directory_path) 
                        if os.path.isdir(os.path.join(directory_path, directory))]
                else:
                    directory_paths = [directory_path]
                for dir_path in directory_paths:
                    for file_name in os.listdir(dir_path):
                        if utils.remove_extension(
                                file_name).endswith(TOKEN_FILENAME_SUFFIX):
                            continue
                        file_path = os.path.join(dir_path, file_name)
                        if already_partitioned:
                            classification = BILLION_BENCHMARK  # assumed
                        else:
                            classification = data.TextClassification.from_path(dir_path)
                        if file_name.endswith(".zip"):
                            try:
                                file_names = utils.get_filenames(file_path)
                            except Exception as e:
                                print("Problem loading:", file_name)
                                continue
                            for file_name in file_names:
                                if file_name.endswith(TOKEN_FILENAME_SUFFIX):
                                    continue
                                document = Document(file_name, classification, file_path)
                                category.documents.append(document)
                        else:
                            document = Document(file_name, classification, file_path)
                            if already_partitioned:
                                dataset_assigned = get_directory_dataset(dir_path)
                                document.dataset_assigned = dataset_assigned
                                document.separate_token_file = False
                            category.documents.append(document)

    def get_document_statistics(self, max_length=None, min_length=None):
        documents = self.all_documents
        zipfile_documents, other_documents = group_by_zipfile(documents)
        token_counts = []
        for file_path, document_list in zipfile_documents.items():
            Document.bulk_load("token", document_list)
            token_counts.extend([document.token_count for document in document_list])
        token_counts.extend([document.token_count for document in other_documents])
        statistics = utils.get_statistics(token_counts)
        if max_length:
            very_large_count = sum(1 for token_count in token_counts 
                                   if token_count > max_length)
            statistics[">" + str(max_length)] = very_large_count
        if min_length:
            very_small_count = sum(1 for token_count in token_counts 
                                   if token_count < min_length)
            statistics["<" + str(min_length)] = very_small_count

        return statistics

    def set_dataset(self, dataset_config):
        for category in self.all_categories:
            category.dataset_config = dataset_config

    def tokenize(self, all_documents=False, retokenize=False):
        if all_documents or self.dataset_config is None:
            documents = self.all_documents
        else:
            documents = self.dataset_documents
        Document.bulk_tokenize(documents, retokenize=retokenize)
        zipfile_documents, other_documents = group_by_zipfile(documents)
        for file_path, document_list in zipfile_documents.items():
            if os.path.exists(document_list[0].get_token_file_info()[1]):
                continue  # Skip if token zip file found
            document_list = [document for document in document_list 
                             if document.tokens is None]
            if not document_list:
                continue 
            for i in range(30):
                document_sublist = document_list[i*len(document_list)//50
                                                 :(i+1)*len(document_list)//50]
                token_sequences = Document.bulk_load("token", document_sublist)
                for document, tokens in zip(document_sublist, token_sequences):
                    document.tokens = tokens
                    document.generate_vocabulary()
                    document.unload_tokens()
        for document in other_documents:
            document.load_tokens()
            document.unload_tokens()

    def _build_vocabulary(self):
        documents = self.dataset_documents
        vocabulary = Counter()
        for i, document in enumerate(documents):
            vocabulary.update(document.vocabulary)
            vocabulary.update(chain.from_iterable(document.classification))
        self.dataset_config.vocabulary = vocabulary
        if self.dataset_config.vocab_type == "subtoken" and len(vocabulary) > 3000000:
            self.dataset_config.vocabulary = Counter(dict(
                self.dataset_config.vocabulary.most_common(3000000)))

    def generate_vocabulary(self):
        self._build_vocabulary()
        self.dataset_config.create_text_encoder()

    def store_partition(self):
        # Partition dictionary
        partition = {"training": [], "validation": [], "test": []}
        for i, document in enumerate(self.dataset_documents):
            if document.dataset_assigned is not None:
                partition[document.dataset_assigned].append(
                    (document.file_path, document.file_name, document.copies))

        # Partition description
        partition_description = self.dataset_config.name + "\n"
        for part, assigned_documents in partition.items():
            partition_description += "\n\n" + part + "\n\n"
            partition_description += "".join(
                doc_path + " | " + doc_file_name + " | " + str(copies) + "\n" for 
                (doc_path, doc_file_name, copies) in assigned_documents)

        # Store
        file_path = utils.add_filename_suffix(
            self.dataset_config.file_path, "_partition")
        utils.store_text(partition_description, file_path)

    def load_partition(self):
        file_path = utils.add_filename_suffix(
            self.dataset_config.file_path, "_partition")
        if not os.path.exists(file_path):
            try:
                file_path = utils.get_file_path(
                    self.dataset_config.root_directory, 
                    [self.dataset_config.name, 
                     str(self.dataset_config.total_token_count), 
                     "partition"])
            except IndexError:
                raise RuntimeError("Partition file not found!")
        partition_description = utils.open_file(file_path)
        documents = self.dataset_documents
        documents_by_file = {(document.file_path, document.file_name) : document
                             for document in documents}
        for line in partition_description.splitlines():
            if not line or line == self.dataset_config.name:
                continue
            elif line in ["training", "validation", "test"]:
                dataset_part = line
            else:
                file_path, file_name, copies = line.split(" | ")
                document = documents_by_file[(file_path, file_name)]
                document.dataset_assigned = dataset_part
                document.copies = int(copies)

    def partition(self, training=0.8, validation=0.1, test=0.1):
        dataset_documents = self.dataset_documents
        all_categories = self.all_categories
        category_config = self.dataset_config.category_config
        for domain, _, partition_type, doc_languages in category_config:
            category = [category for category in all_categories 
                        if category.category == domain][0]
            if partition_type in ["training", "validation", "test"]:
                documents = list(set(dataset_documents) 
                                 & set(category.all_documents))
                for document in documents:
                    document.dataset_assigned = partition_type
            elif partition_type == "split_as_one":
                documents = list(set(dataset_documents) 
                                 & set(category.all_documents))
                Document.partition(documents, training, validation, test)
            else:
                split_subcategories = category.get_subcategories(partition_type)
                if partition_type == "document":
                    for category in split_subcategories:
                        documents = list(set(dataset_documents) 
                                         & set(category.documents))
                        Document.partition(documents)
                else:
                    Document.partition(
                        [list(set(dataset_documents) & set(category.documents)) 
                         for category in split_subcategories])

    def sample(self, sampling_method=None, 
               required_tokens=None, force_equalize=None):
        print("Sampling:", self.category)
        if required_tokens is None:
            required_tokens = self.dataset_config.total_token_count
        if sampling_method is None:
            sampling_method = self.dataset_config.sampling_method
        if sampling_method == "equalize":
            if self.subcategories:
                dataset_categories = set(self.dataset_categories())
                dataset_parent_categories = {
                    category for category in self.all_categories
                    if dataset_categories & set(category.all_categories)}
                if force_equalize:
                    sampled_categories = self.subcategories
                else:
                    sampled_categories = [category for category in self.subcategories
                                          if category in dataset_parent_categories]
                tokens_per = required_tokens // len(sampled_categories)
                for category in sampled_categories:
                    force_equalize = category in dataset_categories
                    category.sample(sampling_method, tokens_per, force_equalize)
            else:
                random.shuffle(self.training_documents)
                if self.training_tokens > required_tokens:
                    while True:
                        tokens = self.training_tokens
                        for document in self.training_documents:
                            if tokens > required_tokens:
                                document.copies -= 1
                                tokens -= document.token_count
                            else:
                                break
                        if self.training_tokens <= required_tokens:
                            break
                    remaining_documents = [
                        document for document in self.training_documents
                        if document.copies]
                    if not remaining_documents:
                        raise ValueError("Documents are too large " 
                                         "for the required token count!")
                elif self.training_tokens < required_tokens:
                    while True:
                        tokens = self.training_tokens
                        for document in self.training_documents:
                            if tokens < required_tokens:
                                document.copies += 1
                                tokens -= document.token_count
                            else:
                                break
                        if self.training_tokens >= required_tokens:
                            break
        else:
            raise NotImplementedError("Sampling method not supported or recognized!")
    
    def training_generator(self):
        print("Generating training data...")
        return self._generator("training")

    def validation_generator(self):
        print("Generating validation data...")
        return self._generator("validation")

    def _generator(self, dataset_type):
        # Get documents
        if dataset_type == "training":
            documents = self.training_documents
        elif dataset_type == "validation":
            documents = self.validation_documents
        # Load tokens
        zipfile_documents, other_documents = group_by_zipfile(documents)
        for file_path, document_list in zipfile_documents.items():
            token_sequences = Document.bulk_load("token", document_list)
            for document, tokens in zip(document_list, token_sequences):
                document.tokens = tokens
        for document in other_documents:
            document.load_tokens()
        # Generate examples
        for document in documents:
            if document.tokens is None:
                continue
            for i in range(document.copies):
                document_ids_list = document.get_ids(
                    self.dataset_config.encoder, 
                    split_lines=self.dataset_config.use_lines)
                for document_ids in document_ids_list:
                    encoded_document = document_ids + [EOS_ID]
                    if self.dataset_config.use_categories:
                        classifications_ids = self.dataset_config.encode_classification(
                            document.classification)
                        encoded_document = classifications_ids + encoded_document
                    yield {"inputs": [0], "targets": encoded_document}


def group_by_zipfile(documents):
    zipfile_documents, other_documents = {}, []
    for document in documents:
        if document._in_zipfile:
            zipfile_documents.setdefault(document.file_path, []).append(document)
        else:
            other_documents.append(document)

    return zipfile_documents, other_documents


class Document:

    def __init__(self, file_name, classification, file_path=None, 
                 text=None, tokens=None, dataset_assigned=None, copies=1, 
                 separate_token_file=True):
        self.file_name = file_name
        self.classification = classification
        self.file_path = file_path
        self.separate_token_file = separate_token_file

        self.text = text
        self.tokens = tokens

        self.dataset_assigned = dataset_assigned
        self.copies = copies

        self._in_zipfile = self.file_name != os.path.split(self.file_path)[1]
        self.vocabulary = None

    def __repr__(self):
        return "<Document(file_name={}, classification={})>".format(
            self.file_name, self.classification)

    @classmethod
    def filter(cls, documents, min_length=None, max_length=None):
        if min_length is None and max_length is None:
            return documents
        if min_length is None:
            min_length = 0
        if max_length is None:
            max_length = 10000000
        filtered_documents = [document for document in documents 
                              if document.tokens is not None and 
                              min_length <= document.token_count <= max_length]
        documents = [document for document in documents if document.tokens is None]
        zipfile_documents, other_documents = group_by_zipfile(documents)
        for _, document_list in zipfile_documents.items():
            count = len(document_list)
            parts = 30 if count >= 100000 else 1
            for i in range(parts):
                print(str(i) + "/" + str(parts) + "...")
                document_slice = document_list[i*count//parts:(i+1)*count//parts]
                token_list = Document.bulk_load("token", document_slice)
                for document, tokens in zip(document_slice, token_list):
                    if (tokens is not None 
                            and min_length <= len(tokens) <= max_length):
                        document.tokens = tokens
                        document.generate_vocabulary()
                        filtered_documents.append(document)
        filtered_documents.extend([
            document for document in other_documents 
            if min_length <= document.token_count <= max_length])

        return filtered_documents

    @classmethod
    def bulk_filter_notokens(cls, documents, file_path=None):
        if file_path is None:
            _, file_path = documents[0].get_token_file_info()
        token_file_names = set(utils.get_filenames(file_path))
        return [document for document in documents 
                if document.get_token_file_info()[0] not in token_file_names]

    @classmethod
    def bulk_load(cls, data_type, documents, file_path=None, 
                  page_count=None, page=None):
        if file_path is None:
            # Assumes all documents in same zip file
            if data_type == "text":
                file_path = documents[0].file_path
            elif data_type == "token":
                _, file_path = documents[0].get_token_file_info()
        with ZipFile(file_path, "r") as zip_file:
            if data_type == "text":
                file_names = [document.file_name for document in documents]
                if page_count:
                    file_names = file_names[(page-1)*len(file_names)//page_count
                                            :page*len(file_names)//page_count]
                document_data = [zip_file.read(file_name).decode() 
                                 for file_name in file_names]
            elif data_type == "token":
                file_names = [document.get_token_file_info()[0] 
                              for document in documents]
                if page_count:
                    file_names = file_names[(page-1)*len(file_names)//page_count
                                            :page*len(file_names)//page_count]
                document_data = []
                for file_name in file_names:
                    try:
                        tokens = zip_file.read(file_name).decode().split(
                            TOKEN_SEPARATOR)
                    except:
                        document_data.append(None)
                    else:
                        document_data.append(tokens)
            else:
                raise ValueError("Invalid input argument:", data_type)

        return document_data

    @classmethod
    def bulk_tokenize(cls, documents, retokenize=False):
        text_cleaner = data.TextCleaner()
        zipfile_documents, other_documents = group_by_zipfile(documents)
        for file_path, document_list in zipfile_documents.items():
            if (os.path.exists(document_list[0].get_token_file_info()[1]) 
                    and not retokenize):
                continue  # Skip if token zip file found
            random.shuffle(document_list)
            print("\nTokenizing:", file_path)
            for i in range(50):
                print(str(i+1) + "/50...")
                print(time())
                document_texts = cls.bulk_load(
                    "text", document_list, page_count=50, page=i+1)
                document_tokens = []
                for j, text in enumerate(document_texts):
                    # log document name to catch error-producing document
                    document = document_list[(i*len(document_list)//50)+j]
                    utils.store_text(document.file_name + "\n", 
                                     data.BASE_DIR + "\\tokenize_log.txt",
                                     append=True)
                    if j % 500 == 0:
                        print("Tokenizing document", 
                              str(j) + "/" + str(len(document_texts)))
                    if text is None:
                        document_tokens.append(["None"])
                    else:
                        document_tokens.append(tokenize(
                            text, document_list[0].classification.language, 
                            file_path))
                if not document_tokens:
                    continue
                if ["None"] in document_tokens:
                    print("At least one document couldn't be processed!")
                token_data_dictionary = {}
                for j, tokens in enumerate(document_tokens):
                    if document_texts[j] is None:
                        continue
                    token_text = TOKEN_SEPARATOR.join(tokens)
                    document = document_list[(i*len(document_list)//50)+j]
                    token_file_name, token_file_path = document.get_token_file_info()
                    token_data_dictionary[token_file_name] = token_text
                utils.store_zipfile_data(token_data_dictionary, token_file_path)
        with Pool(8) as pool:
            pool.map(tokenize_document, other_documents)

    @classmethod
    def partition(cls, documents, training=0.8, validation=0.1, test=0.1):
        if isinstance(documents[0], cls) and len(documents) < 30:
            for document in documents:
                document.dataset_assigned =  "training"
        else:
            partition_indices = list(range(len(documents)))
            random.shuffle(partition_indices)
            validation_size = min((len(documents) * int(validation * 10)) // 10, 10000)
            test_size = (len(documents) * int(validation * 10)) // 10
            for i in partition_indices:
                document_list = documents[i]
                if isinstance(document_list, cls):
                    document_list = [document_list]
                if i < validation_size:
                    for document in document_list:
                        document.dataset_assigned = "validation"
                elif validation_size <= i < validation_size + test_size:
                    for document in document_list:
                        document.dataset_assigned = "test"
                else:
                    for document in document_list:
                        document.dataset_assigned = "training"

    @cached_property
    def token_count(self):
        if self.tokens is None and self.vocabulary is None:
            raise RuntimeError("Tokens must be generated/loaded first!")
        elif self.tokens is None:
            token_count = sum(self.vocabulary.values())
        else:
            token_count = len(self.tokens)
        return token_count

    def generate_vocabulary(self):
        if self.tokens is None:
            raise RuntimeError("Require tokens to generate the vocabulary!")
        else:
            self.vocabulary = Counter(self.tokens)

    def load_text(self):
        if self._in_zipfile:
            self.text = utils.open_text_from_zipfile(
                self.file_path, self.file_name)
        else:
            self.text = utils.open_file(self.file_path, encoding=ENCODING)

    def unload_text(self):
        self.text = None

    def get_token_file_info(self):
        if self.separate_token_file:
            token_file_name = utils.add_filename_suffix(
                self.file_name, TOKEN_FILENAME_SUFFIX)
            token_file_path = utils.add_filename_suffix(
                self.file_path, TOKEN_FILENAME_SUFFIX)
        else:
            token_file_name, token_file_path = (self.file_name, self.file_path)
        return token_file_name, token_file_path

    def load_tokens(self):
        if self.tokens is None:
            token_file_name, token_file_path = self.get_token_file_info()
            if self._in_zipfile:
                token_text = utils.open_text_from_zipfile(
                    token_file_path, token_file_name)
            else:
                token_text = utils.open_file(token_file_path, encoding=ENCODING)

            self.tokens = token_text.replace("\n", " \n ").split(TOKEN_SEPARATOR)
            self.generate_vocabulary()

    def unload_tokens(self):
        self.tokens = None

    def _store_tokens(self):
        token_file_name, token_file_path = self.get_token_file_info()
        token_text = TOKEN_SEPARATOR.join(self.tokens)
        if self._in_zipfile:
            utils.store_zipfile_data({token_file_name: token_text}, token_file_path)
        else:
            utils.store_text(token_text, token_file_path)

    def tokenize(self, store=True):
        if os.path.exists(self.get_token_file_info()[1]):
            return
        if self.text is None and self.tokens is None:
            self.load_text()
        if self.tokens is None:
            self.tokens = tokenize(
                self.text, self.classification.language, self.file_name)
        self.generate_vocabulary()
        if store:
            self._store_tokens()
            self.unload_tokens()
        self.unload_text()

    def get_ids(self, encoder, split_lines=False):
        try:
            if self.tokens is None:
                self.load_tokens()
        except FileNotFoundError:
            raise RuntimeError("Must generate and store tokens" 
                               " before encoding them!")
        else:
            if split_lines:
                token_sequences = [
                       list(token_sequence) for seperator, token_sequence 
                       in groupby(self.tokens, lambda element: element == "\n")
                       if not seperator]
                token_sequences = [
                    tokens for tokens in token_sequences if 
                    tokens and any([token.strip() for token in tokens])]
                ids = [encoder.encode(tokens) for tokens in token_sequences]
            else:
                ids = [encoder.encode(self.tokens)]
            self.unload_tokens()

        return ids


def load_category_graph(root_path=data.BASE_DIR, classification=None):
    if classification is None:
        data_crawler = data.DataCrawler(root_path)
        text_classifications = data_crawler.crawl(
            process=False, crawl_processed=True)
    else:
        text_classifications = [classification]
    root_category = Category.from_classifications(text_classifications)
    return root_category


def prepare_data(category, generate_vocabulary=True, load_partition=True, 
                 already_partitioned=False):
    if generate_vocabulary:
        if already_partitioned:
            print("\nLoading tokens...")
            for document in category.all_documents:
                document.load_tokens()
                document.unload_tokens()
        else:
            print("\nTokenizing...")
            category.tokenize(all_documents=False)
        print("\nGenerating vocabulary...")
        category.generate_vocabulary()
    else:
        print("Creating text encoder...")
        category.dataset_config.create_text_encoder()
    if load_partition:
        print("Loading partition...")
        category.load_partition()
    else:
        if not already_partitioned:
            print("\nPartitioning...")
            category.partition()
        print("\nSampling...")
        category.sample()
        print("\nStoring partition...")
        category.store_partition()
    print("Data prepared.\n")


def setup_dataset(name, category_config, vocab_size=None, vocab_min_count=None, 
                  vocab_type="token", sampling_method="equalize", 
                  total_token_count=None, min_doc_length=None, 
                  max_doc_length=None, use_lines=False, classification=None, 
                  root_path=data.BASE_DIR, already_partitioned=False):
    category_graph = load_category_graph(root_path, classification)
    dataset_config = DatasetConfiguration(name, category_config, sampling_method, 
                                          vocab_type, vocab_size, vocab_min_count,
                                          root_directory=root_path, 
                                          total_token_count=total_token_count,
                                          min_doc_length=min_doc_length,
                                          max_doc_length=max_doc_length,
                                          use_lines=use_lines)
    category_graph.set_dataset(dataset_config)
    print("\nLoading documents...")
    category_graph.load_documents(already_partitioned=already_partitioned)
    dataset_config.sample(category_graph)
    dataset_config.store()
    prepare_data(category_graph, generate_vocabulary=True, load_partition=False, 
                 already_partitioned=already_partitioned)


def load_dataset(dataset_config_filename, root_path=data.BASE_DIR,
                 classification=None, already_partitioned=False, 
                 use_categories=True, training=False, 
                 vocab_file_path=None):
    category_graph = load_category_graph(root_path, classification)
    dataset_config = DatasetConfiguration.from_file(
        os.path.join(root_path, dataset_config_filename))
    dataset_config.use_categories = use_categories
    if vocab_file_path is not None:
        dataset_config.vocab_file_path = vocab_file_path
    category_graph.set_dataset(dataset_config)
    if not training:
        print("\nLoading documents...")
        category_graph.load_documents(already_partitioned=already_partitioned)
        prepare_data(category_graph, generate_vocabulary=False, load_partition=True, 
                     already_partitioned=already_partitioned)

    return category_graph


