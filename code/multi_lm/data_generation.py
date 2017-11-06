""""""

import os
import random
import numpy as np
from zipfile import ZipFile
from itertools import chain, groupby
from collections import Counter
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

import spacy
from pygments import lex
from pygments.lexers.python import PythonLexer
from pygments.lexers.c_cpp import CLexer
from pygments.lexers.fortran import FortranFixedLexer, FortranLexer
from pygments.lexers.lisp import CommonLispLexer
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators.text_encoder import (
    EOS, RESERVED_TOKENS, TextEncoder, 
    TokenTextEncoder, SubwordTextEncoder)
from tensor2tensor.data_generators import problem
from tensor2tensor.utils import registry

import utilities as utils
import data_preparation as data


with ThreadPool(4) as pool:
    ENGLISH_PROCESSOR, FRENCH_PROCESSOR, GERMAN_PROCESSOR, CHINESE_PROCESSOR = (
        tuple(pool.map(spacy.load, ["en", "fr", "de", "zh"])))

NATURAL_LANGUAGE_PROCESSORS = {
    data.ENGLISH: ENGLISH_PROCESSOR,
    data.FRENCH: FRENCH_PROCESSOR,
    data.GERMAN: GERMAN_PROCESSOR,
    data.CHINESE: CHINESE_PROCESSOR,
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
                 vocab_file_name=None, root_directory=None, 
                 total_token_count=None, use_categories=True, use_lines=False):
        """
        Args:
            name: String, name of this dataset configuration.
            category_config: List of 2-tuples of the form 
                (domain, subcategory_count, partition_type), where 
                domain is a string, subcategory_count is an integer 
                representing the number of subcategories to sample, and
                partition_type is either 'training', 'validation', 'test', 
                'split', 'split_as_one', 'top_domain', 'bottom_domain', 
                or 'document'.
            sampling_method: String, either 'truncate' or 'equalize'.
            vocab_type: String, either 'token', 'subtoken', or 'character'.
            vocab_size: Integer, defaults to None.
            vocab_min_count: Integer, minimum count required for a token 
                or subtoken to be included in the vocabulary. 
                Defaults to None.
            vocab_file_name: String, defaults to None.
            root_directory: String, defaults to None.
            total_token_count: Integer, defaults to None.
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
        self.vocab_file_name = vocab_file_name
        self.root_directory = root_directory
        self.total_token_count = total_token_count
        self.use_categories = use_categories

        self.vocabulary = None
        self.truncated_vocabulary = None

    def __repr__(self):
        return "<DatasetConfig(name={}, categories={})>".format(
            self.name, self.category_config)

    @classmethod
    def from_file(cls, file_path):
        dataset_description = utils.open_file(file_path)
        (name, category_config, sampling_method, vocab_type, vocab_size, 
         vocab_min_count, vocab_file_name, total_token_count) = tuple(
            dataset_description.splitlines())
        vocab_size = int(vocab_size)
        category_config = eval(category_config)
        vocab_file_name = None if vocab_file_name == "None" else vocab_file_name
        parameters = [vocab_size, vocab_min_count, total_token_count]
        for i, parameter in enumerate(parameters):
            if parameter == "None":
                parameters[i] = None
            else:
                parameters[i] = int(parameter)
        return cls(name, category_config, sampling_method, vocab_type, 
                   parameters[0], parameters[1], vocab_file_name, 
                   total_token_count=parameters[2])

    def sample(self, root_category, sample_largest=True):
        categories = root_category.all_categories
        determined_category_config = []
        for domain, subcategory_count, partition_type in self.category_config:
            if subcategory_count < 1:
                determined_category_config.append(
                    (domain, subcategory_count, partition_type))
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
                    [(category, 0, partition_type) 
                     for category in sampled_categories])

        self.category_config = determined_category_config

    @property
    def file_path(self):
        file_name = self.name.replace(" ", "_") + "_" + self.vocab_type
        if self.vocab_type == "token":
            file_name += (str(self.vocab_size) if self.vocab_size 
                          else str(self.vocab_min_count))
        elif self.vocab_type == "subtoken":
            file_name += str(self.vocab_min_count)
        file_name += "_size" + str(self.total_token_count) + ".txt"
        file_path = os.path.join(self.root_directory or data.BASE_DIR, file_name)

        return file_path

    def store(self):
        file_path = utils.add_filename_suffix(self.file_path, "_config")
        dataset_description = "\n".join([
            self.name, str(self.category_config), self.sampling_method, 
            str(self.vocab_type), str(self.vocab_size), 
            str(self.vocab_min_count), str(self.vocab_file_name),
            str(self.total_token_count)])
        utils.store_text(dataset_description, file_path)

    @property
    def unknown_token_count(self):
        return sum(count for token, count in self.vocabulary.items() 
                   if token not in self.truncated_vocabulary)

    def _assign_tag_token(self):
        self.tag_token = None
        token_id = 32
        while self.tag_token is None:
            if chr(token_id) not in self.truncated_vocabulary:
                self.tag_token = chr(token_id)
            else:
                token_id += 1
        if self.vocab_min_count is not None:
            self.vocabulary.update([self.tag_token]*self.vocab_min_count)
            self.truncated_vocabulary.update([self.tag_token]*self.vocab_min_count)
        else:
            self.vocabulary[self.tag_token] = 200
            self.truncated_vocabulary[self.tag_token] = 200

    def generate_truncated_vocab(self):
        if self.vocab_size is not None:
            self.truncated_vocabulary = {
                token: count for token, count 
                in self.vocabulary.most_common(self.vocab_size)}
        else:
            self.truncated_vocabulary = {
                token: count for token, count
                in self.vocabulary if count >= self.vocab_min_count}
        self._assign_tag_token()

    def _store_vocabulary(self):
        if self.vocab_file_name is None:
            self.vocab_file_name = utils.add_filename_suffix(
                self.file_path, "_vocab")
        vocab_file_path = os.path.join(self.root_directory or data.BASE_DIR, 
                                       self.vocab_file_name)
        vocab_description = "\n".join(token + ", " + str(count) for token, count 
                                      in self.vocabulary.items())
        utils.store_text(vocab_description, vocab_file_path)

    def _load_vocabulary(self):
        if self.vocab_file_name is None:
            self.vocab_file_name = utils.add_filename_suffix(
                self.file_path, "_vocab")
        vocab_file_path = os.path.join(self.root_directory or data.BASE_DIR, 
                                       self.vocab_file_name)
        if self.vocab_type in ["token", "subtoken"]:
            vocab_description = utils.open_file(vocab_file_path)
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

    def create_text_encoder(self):
        load_vocab = self.vocabulary is None and self.truncated_vocabulary is None
        if load_vocab:
            self._load_vocabulary()
        else:
            self._store_vocabulary()
        if self.vocab_type == "token":
            self.generate_truncated_vocab()
            self.truncated_vocabulary.setdefault(UNKNOWN_TOKEN, 1)
            encoder = CustomTokenTextEncoder(
                None, vocab_list=list(self.truncated_vocabulary.keys()), 
                replace_oov=UNKNOWN_TOKEN)
        elif self.vocab_type == "subtoken":
            if load_vocab:
                vocab_file_path = os.path.join(
                    self.root_directory or data.BASE_DIR, 
                    self.vocab_file_name)
                encoder = CustomSubwordTextEncoder(vocab_file_path)
            else:
                encoder = CustomSubwordTextEncoder()
                encoder.build_from_token_counts(
                    self.vocabulary, self.vocab_min_count)
                vocab_file_name = utils.add_filename_suffix(
                    self.file_path, "_vocab_subtoken." + str(encoder.vocab_size)) 
                encoder.store_to_file(vocab_file_name)
        elif self.vocab_type == "character":
            encoder = CustomCharacterTextEncoder()

        self.encoder = encoder

    def encode_classification(self, classification):
        classification_tokens = chain.from_iterable(
            [self.tag_token, domain, self.tag_token] 
            for domain in classification)
        return self.encoder.encode(classification_tokens)


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
            for domain in classification:
                if (os.path.split(classification.directory_path)[0].endswith(domain)
                        or classification.directory_path.endswith(domain)):
                    directory_path = classification.directory_path
                else:
                    directory_path = None
                try:
                    top_category = [category for category in categories 
                                    if category.category == domain][0]
                    if directory_path not in top_category.directory_paths:
                        top_category.directory_paths.append(directory_path)
                except IndexError:
                    if domain == classification.language_type:
                        category_type = "language_type"
                    elif domain == classification.language:
                        category_type = "language"
                    else:
                        category_type = "domain"
                    category = cls(domain, category_type=category_type, 
                                   directory_paths=[directory_path])
                    top_category.add_subcategory(category)
                    top_category = category

        return root_category

    @property
    def all_categories(self):
        return [self] + list(chain.from_iterable(
            [category.all_categories for category in self.subcategories]))

    @property
    def all_documents(self):
        return self.documents + list(chain.from_iterable(
            [category.all_documents for category in self.subcategories]))

    @property
    def dataset_categories(self):
        all_categories = self.all_categories
        dataset_categories = []
        for domain, _, _ in self.dataset_config.category_config:
            try:
                category = [category for category in all_categories 
                            if category.category == domain][0]
            except IndexError:
                continue
            else:
                dataset_categories.append(category)

        return dataset_categories

    @property
    def dataset_documents(self):
        categories = self.dataset_categories
        return list(chain.from_iterable(
            [category.all_documents for category in categories]))

    @property
    def training_documents(self):
        return [document for document in self.all_documents 
                if document.dataset_assigned == "training"]

    @property
    def validation_documents(self):
        return [document for document in self.all_documents 
                if document.dataset_assigned == "validation"]

    @property
    def tokens(self):
        tokens = 0
        for document in self.all_documents:
            document.load_tokens()
            tokens += len(document.tokens)
            document.unload_tokens()
        return tokens

    @property
    def training_tokens(self):
        tokens = 0
        for document in self.training_documents:
            document.load_tokens()
            tokens += document.copies*len(document.tokens)
            document.unload_tokens()
        return tokens

    @property
    def validation_tokens(self):
        tokens = 0
        for document in self.validation_documents:
            document.load_tokens()
            tokens += document.copies*len(document.tokens)
            document.unload_tokens()
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
                        # Make new category to accomodate this document
                        category = Category(next_domain)
                        self.add_subcategory(category)
                        category.add_document(document)
        else:
            # Assumes currently in the (placeholder) 'root' category
            category = Category(document.classification[0])
            self.add_subcategory(category)
            category.add_document(document)

    def load_documents(self, all_documents=False):
        if all_documents:
            categories = self.all_categories 
        else:
            categories = chain.from_iterable(
                [category.all_categories for category in self.dataset_categories])
        for category in categories:
            for directory_path in category.directory_paths:
                if directory_path is None or directory_path == data.BASE_DIR:
                    continue
                for file_name in os.listdir(directory_path):
                    if utils.remove_extension(file_name).endswith(TOKEN_FILENAME_SUFFIX):
                        continue
                    file_path = os.path.join(directory_path, file_name)
                    classification = data.TextClassification.from_path(directory_path)
                    if file_name.endswith(".zip"):
                        try:
                            file_names = utils.get_filenames(file_path)
                        except Exception as e:
                            print("Problem loading:", file_name)
                            print(str(e))
                            continue
                        for file_name in file_names:
                            if file_name.endswith(TOKEN_FILENAME_SUFFIX):
                                continue
                            document = Document(file_name, classification, file_path)
                            category.documents.append(document)
                    else:
                        document = Document(file_name, classification, file_path)
                        category.documents.append(document)

    def get_document_statistics(self, max_length=None, min_length=None):
        documents = self.all_documents
        zipfile_documents, other_documents = group_by_zipfile(documents)
        token_counts = []
        for file_path, document_list in zipfile_documents.items():
            Document.bulk_load(document_list, "token")
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

    def tokenize(self, all_documents=False):
        if all_documents or self.dataset_config is None:
            documents = self.all_documents
        else:
            documents = self.dataset_documents
        Document.bulk_tokenize(documents)

    def _build_vocabulary(self):
        documents = self.dataset_documents
        vocabulary = Counter()
        for document in documents:
            vocabulary.update(document.vocabulary)
            vocabulary.update(chain.from_iterable(document.classification))
        self.dataset_config.vocabulary = vocabulary
        self.dataset_config.generate_truncated_vocab()

    def generate_vocabulary(self):
        self._build_vocabulary()
        self.dataset_config.create_text_encoder()

    def store_partition(self):
        # Partition dictionary
        partition = {"training": [], "validation": [], "test": []}
        for document in self.dataset_documents:
            partition[document.dataset_assigned].append(
                (document.file_path, document.file_name, document.copies))

        # Partition description
        partition_description = self.dataset_config.name + "\n"
        for part in partition:
            partition_description += "\n\n" + part + "\n\n"
            for doc_path, doc_file_name, copies in partition[part]:
                partition_description += (doc_path + " | " + doc_file_name 
                                          + " | " + str(copies) + "\n")

        # Store
        file_path = utils.add_filename_suffix(
            self.dataset_config.file_path, "_partition")
        utils.store_text(partition_description, file_path)

    def load_partition(self):
        file_path = utils.add_filename_suffix(
            self.dataset_config.file_path, "_partition")
        partition_description = utils.open_file(file_path)
        documents = self.dataset_documents
        for line in partition_description.splitlines():
            if not line:
                continue
            elif line in [self.dataset_config.name, "training", "validation", "test"]:
                dataset_part = line
            else:
                file_path, file_name, copies = line.split(" | ")
                for document in documents:
                    if (document.file_path == file_path 
                            and document.file_name == file_name):
                        document.dataset_assigned = dataset_part
                        document.copies = copies

    def partition(self, training=0.8, validation=0.1, test=0.1):
        all_categories = self.all_categories
        for domain, _, partition_type in self.dataset_config.category_config:
            category = [category for category in all_categories 
                        if category.category == domain][0]
            if partition_type in ["training", "validation", "test"]:
                for document in category.all_documents:
                    document.dataset_assigned = partition_type
            elif partition_type == "split_as_one":
                Document.partition(category.all_documents, 
                                   training, validation, test)
            else:
                split_subcategories = category.get_subcategories(partition_type)
                if partition_type == "document":
                    for category in split_subcategories:
                        Document.partition(category.documents)
                else:
                    Document.partition(
                        [category.documents for category in split_subcategories])

    def sample(self, sampling_method=None, 
               required_tokens=None, force_equalize=None):
        print("Sampling:", self.category)
        if required_tokens is None:
            required_tokens = self.dataset_config.total_token_count
        if sampling_method is None:
            sampling_method = self.dataset_config.sampling_method
        if sampling_method == "equalize":
            if self.subcategories:
                dataset_categories = set(self.dataset_categories)
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
                random.shuffle(self.documents)
                if self.training_tokens > required_tokens:
                    while self.training_tokens > required_tokens:
                        for document in self.training_documents:
                            if self.training_tokens > required_tokens:
                                document.copies -= 1
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
                        for document in self.training_documents:
                            if self.training_tokens < required_tokens:
                                document.copies += 1
                            else:
                                break
                        if self.training_tokens >= required_tokens:
                            break
        else:
            raise NotImplementedError("Sampling method not supported or recognized!")
    
    def training_generator(self):
        return self._generator("training")

    def validation_generator(self):
        return self._generator("validation")

    def _generator(self, dataset_type):
        if dataset_type == "training":
            documents = self.training_documents
        elif dataset_type == "validation":
            documents = self.validation_documents
        for document in documents:
            document_ids_list = document.get_ids(
                self.dataset_config.encoder, 
                split_lines=self.dataset_config.use_lines)
            for document_ids in document_ids_list:
                encoded_document = document_ids + self.dataset_config.encoder.EOS_ID
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
                 text=None, tokens=None, dataset_assigned=None, copies=1):
        self.file_name = file_name
        self.classification = classification
        self.file_path = file_path

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
    def bulk_load(cls, data_type, documents, file_path=None):
        if file_path is None:
            # Assumes all documents in same zip file
            if data_type == "text":
                file_path = documents[0].file_path
            elif data_type == "token":
                _, file_path = documents[0].get_token_file_info()
        with ZipFile(file_path, "r") as zip_file:
            if data_type == "text":
                file_names = [document.file_name for document in documents]
                document_data = [str(zip_file.read(file_name)) 
                                 for file_name in file_names]
            elif data_type == "token":
                file_names = [document.get_token_file_info()[0] 
                              for document in documents]
                document_data = [str(zip_file.read(file_name)).split(TOKEN_SEPARATOR)
                                 for file_name in file_names]
            else:
                raise ValueError("Invalid input argument:", data_type)

        return document_data

    @classmethod
    def bulk_tokenize(cls, documents):
        zipfile_documents, other_documents = group_by_zipfile(documents)
        for file_path, document_list in zipfile_documents.items():
            if os.path.exists(document_list[0].get_token_file_info()[1]):
                continue  # Skip if token zip file found
            print("\nTokenizing:", file_path)
            document_texts = cls.bulk_load("text", document_list, file_path)
            # To deal with spacy string size limit (2**30)
            text_lists = utils.split(document_texts, 80)
            first_document_index = 0
            for i, texts in enumerate(text_lists):
                print(str(i) + "/" + str(80) + "...")
                concatenated_text = (" " + DOCUMENT_SEPARATOR + " ").join(texts)
                # Assumes all documents in zip file are same language
                tokens = tokenize(concatenated_text, 
                                  document_list[0].classification.language, file_path)
                del concatenated_text
                document_tokens = [
                    list(token_sequence) for seperator, token_sequence 
                    in groupby(tokens, lambda element: element == DOCUMENT_SEPARATOR)
                    if not seperator]
                token_data_dictionary = {}
                for j, tokens in enumerate(document_tokens):
                    token_text = TOKEN_SEPARATOR.join(tokens)
                    document = document_list[first_document_index+j]
                    token_file_name, token_file_path = document.get_token_file_info()
                    token_data_dictionary[token_file_name] = token_text
                utils.store_zipfile_data(token_data_dictionary, token_file_path)
                first_document_index += len(document_tokens)
        with Pool(8) as pool:
            pool.map(tokenize_document, other_documents)

    @classmethod
    def partition(cls, documents, training=0.8, validation=0.1, test=0.1):
        if isinstance(documents[0], cls) and len(documents) < 30:
            for document in documents:
                document.dataset_assigned =  "training"
        else:
            partition_indices = random.shuffle(list(range(len(documents))))
            training_size = ((len(documents) * int(training * 10)) // 10) + 1
            validation_size = (len(documents) * int(validation * 10)) // 10
            for i, document_list in enumerate(documents):
                if isinstance(document_list, cls):
                    document_list = [document_list]
                if i < training_size:
                    for document in document_list:
                        document.dataset_assigned = "training"
                elif training_size <= i < training_size + validation_size:
                    for document in document_list:
                        document.dataset_assigned = "validation"
                else:
                    for document in document_list:
                        document.dataset_assigned = "test"

    @property
    def token_count(self):
        if self.tokens is None:
            self.load_tokens()
        token_count = len(self.tokens)
        self._unload_tokens()
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
        token_file_name = utils.add_filename_suffix(
            self.file_name, TOKEN_FILENAME_SUFFIX)
        token_file_path = utils.add_filename_suffix(
            self.file_path, TOKEN_FILENAME_SUFFIX)
        return token_file_name, token_file_path

    def load_tokens(self):
        token_file_name, token_file_path = self.get_token_file_info()
        if self._in_zipfile:
            token_text = utils.open_text_from_zipfile(
                token_file_path, token_file_name)
        else:
            token_text = utils.open_file(token_file_path, encoding=ENCODING)

        self.tokens = token_text.split(TOKEN_SEPARATOR)

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

    def get_ids(encoder, split_lines=False):
        try:
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
            self._unload_tokens()

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


def prepare_data(category, generate_vocabulary=True, load_partition=True):
    if generate_vocabulary:
        print("\nTokenizing and generating vocabulary...")
        category.tokenize(all_documents=False)
        category.generate_vocabulary()
    else:
        category.dataset_config.create_text_encoder()
    if load_partition:
        category.load_partition()
    else:
        print("\nPartitioning...")
        category.partition()
        print("\nSampling...")
        category.sample()
        print("\nStoring partition...")
        category.store_partition()
    print("\nData prepared\n.")


def setup_dataset(name, category_config, vocab_size=None, vocab_min_count=None, 
                  vocab_type="token", sampling_method="equalize", 
                  total_token_count=None, root_path=data.BASE_DIR):
    category_graph = load_category_graph(root_path)
    dataset_config = DatasetConfiguration(name, category_config, sampling_method, 
                                          vocab_type, vocab_size, vocab_min_count,
                                          root_directory=root_path, 
                                          total_token_count=total_token_count)
    category_graph.set_dataset(dataset_config)
    print("\nLoading documents...")
    category_graph.load_documents()
    dataset_config.sample(category_graph)
    dataset_config.store(root_path)
    prepare_data(category_graph, generate_vocabulary=False, load_partition=False)


def load_dataset(dataset_config_filename, root_path=data.BASE_DIR,
                 classification=None):
    category_graph = load_category_graph(root_path, classification)
    dataset_config = DatasetConfiguration.from_file(
        os.path.join(root_path, dataset_config_filename))
    category_graph.dataset_config = dataset_config
    prepare_data(category_graph, generate_vocabulary=False, load_partition=True)
    return category_graph


class MultiLmProblem(problem.Text2TextProblem):
    """Base class for multi-domain language modeling problems."""

    @property
    def is_character_level(self):
        return False

    @property
    def targeted_vocab_size(self):
        return self.dataset_config.vocab_size

    @property
    def input_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def target_space_id(self):
        return self.input_space_id

    @property
    def num_shards(self):
        return 1

    @property
    def vocab_file(self):
        return self.dataset_config.vocab_file_name

    @property 
    def use_subword_tokenizer(self):
        return True

    @property
    def has_inputs(self):
        return False

    def load_dataset(self, dataset_config_filename, root_path=data.BASE_DIR):
        self.document_graph = load_dataset(dataset_config_filename, root_path)
        self.dataset_config = self.document_graph.dataset_config

    def generator(self, data_dir, temp_dir, is_training):
        if is_training:
            return self.document_graph.training_generator()
        else:
            return self.document_graph.validation_generator()


@registry.register_problem("multi_lm_gutenberg")
class MultiLmGutenberg(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_dataset("gutenberg_token100000_size50000000_config.txt")

    @property 
    def use_subword_tokenizer(self):
        return False


BILLION_BENCHMARK = data.TextClassification(
    language_type=data.NATURAL_LANGUAGE, language=data.ENGLISH, domain=["news"], 
    folder_name="1-billion-word-language-modeling-benchmark",
    corpus_name="1 billion word benchmark dataset",
    directory_path=os.path.join(data.BASE_DIR, data.NATURAL_LANGUAGE, 
        data.ENGLISH, "1-billion-word-language-modeling-benchmark",
        "training-monolingual.tokenized.shuffled"))


@registry.register_problem("multi_lm_1billion")
class MultiLm1Billion(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_dataset("1billion_token100000_size50000000_config.txt",
                          classification=BILLION_BENCHMARK)

    @property 
    def use_subword_tokenizer(self):
        return False