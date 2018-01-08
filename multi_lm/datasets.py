""""""

import os

from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators.lm1b import LanguagemodelLm1b32k

from . import data_preparation as data
from .data_generation import load_dataset, setup_dataset


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
        return self.dataset_config.vocab_file_path

    @property 
    def use_subword_tokenizer(self):
        return True

    @property
    def has_inputs(self):
        return False

    def load_dataset(self, dataset_config_filename, root_path=data.BASE_DIR, 
                     classification=None, already_partitioned=False, 
                     use_categories=True, training=False, vocab_file_path=None):
        self.document_graph = load_dataset(
            dataset_config_filename, root_path=root_path, 
            classification=classification, already_partitioned=already_partitioned, 
            use_categories=use_categories, training=training, 
            vocab_file_path=vocab_file_path)
        self.dataset_config = self.document_graph.dataset_config

    def generator(self, data_dir, temp_dir, is_training):
        if is_training:
            return self.document_graph.training_generator()
        else:
            return self.document_graph.validation_generator()


BILLION_BENCHMARK = data.TextClassification(
    language_type=data.NATURAL_LANGUAGE, language=data.ENGLISH, domains=["news"], 
    folder_name="1-billion-word-language-modeling-benchmark",
    corpus_name="1 billion word benchmark dataset",
    directory_path=os.path.join(data.BASE_DIR, data.NATURAL_LANGUAGE, 
        data.ENGLISH, "1-billion-word-language-modeling-benchmark"))


@registry.register_problem("languagemodel_lm1b8k")
class LanguagemodelLm1b8k(LanguagemodelLm1b32k):

    @property
    def targeted_vocab_size(self):
        return 2**13  # 8192


@registry.register_problem("multi_lm_1billion_subword_small")
class MultiLm1BillionSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        try:
            self.load_dataset("1billion_subtoken8192_size30000000_config.txt",
                              classification=BILLION_BENCHMARK, 
                              already_partitioned=True, training=True)
        except:
            setup_dataset("1billion", 
                          [("news", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**13, vocab_type="subtoken", 
                          total_token_count=30000000, use_lines=True,
                          classification=BILLION_BENCHMARK, 
                          already_partitioned=True)
            self.load_dataset("1billion_subtoken8192_size30000000_config.txt",
                              classification=BILLION_BENCHMARK, 
                              already_partitioned=True)


@registry.register_problem("multi_lm_1billion_nocats_subword_small")
class MultiLm1BillionSmallNoCats(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        try:
            self.load_dataset("1billion_nocats_subtoken8192_size30000000_config.txt",
                              classification=BILLION_BENCHMARK, 
                              already_partitioned=True, use_categories=False,
                              training=False)
        except:
            setup_dataset("1billion_nocats", 
                          [("news", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**13, vocab_type="subtoken", 
                          total_token_count=30000000, use_lines=True,
                          classification=BILLION_BENCHMARK, 
                          already_partitioned=True)
            self.load_dataset("1billion_nocats_subtoken8192_size30000000_config.txt",
                              classification=BILLION_BENCHMARK, 
                              use_categories=False, already_partitioned=True)


@registry.register_problem("multi_lm_1billion_subword_medium")
class MultiLm1BillionMedium(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        try:
            self.load_dataset("1billion_subtoken16384_size100000000_config.txt",
                              classification=BILLION_BENCHMARK, 
                              already_partitioned=True, training=True)
        except:
            setup_dataset("1billion", 
                          [("news", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**14, vocab_type="subtoken", 
                          total_token_count=100000000, use_lines=True,
                          classification=BILLION_BENCHMARK, 
                          already_partitioned=True)
            self.load_dataset("1billion_subtoken16384_size100000000_config.txt",
                              classification=BILLION_BENCHMARK,
                              already_partitioned=True)


@registry.register_problem("multi_lm_1billion_subword_large")
class MultiLm1BillionLarge(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 
        try:
            self.load_dataset("1billion_subtoken16384_size250000000_config.txt",
                              classification=BILLION_BENCHMARK,
                              already_partitioned=True, training=False)
        except:
            setup_dataset("1billion", 
                          [("news", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**14, vocab_type="subtoken", 
                          total_token_count=250000000, use_lines=True,
                          classification=BILLION_BENCHMARK,
                          already_partitioned=True)
            self.load_dataset("1billion_subtoken16384_size250000000_config.txt",
                              classification=BILLION_BENCHMARK,
                              already_partitioned=True)


@registry.register_problem("multi_lm_enwiki_token_small")
class MultiLmEnWikiTokenSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enwiki_token100000_size30000000_config.txt")
        except FileNotFoundError:
            setup_dataset("enwiki", 
                          [("wikipedia", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=100000, vocab_type="token", 
                          total_token_count=30000000, min_doc_length=20, 
                          max_doc_length=200)
            self.load_dataset("enwiki_token100000_size30000000_config.txt")

    @property 
    def use_subword_tokenizer(self):
        return False


@registry.register_problem("multi_lm_enwiki_subword_small")
class MultiLmEnWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enwiki_subtoken8192_size30000000_config.txt", 
                              training=True)
        except FileNotFoundError:
            setup_dataset("enwiki", 
                          [("wikipedia", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**13, vocab_type="subtoken", 
                          total_token_count=30000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enwiki_subtoken8192_size30000000_config.txt")


@registry.register_problem("multi_lm_enwiki_subword_small_enfrwikivocab")
class MultiLmEnWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            vocab_file_path = os.path.join(data.BASE_DIR,  
                "enfrwiki_subtoken10000_size50000000_vocab.txt")
            self.load_dataset("enwiki_subtoken8192_size30000000_config.txt", 
                              training=True, vocab_file_path=vocab_file_path)
        except FileNotFoundError:
            setup_dataset("enwiki", 
                          [("wikipedia", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**13, vocab_type="subtoken", 
                          total_token_count=30000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enwiki_subtoken8192_size30000000_config.txt")


@registry.register_problem("multi_lm_enwiki_subword_small_wikivocab")
class MultiLmEnWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            vocab_file_path = os.path.join(data.BASE_DIR,  
                "wiki_subtoken16384_size50000000_vocab.txt")
            self.load_dataset("enwiki_subtoken8192_size30000000_config.txt", 
                              training=True, vocab_file_path=vocab_file_path)
        except FileNotFoundError:
            setup_dataset("enwiki", 
                          [("wikipedia", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**13, vocab_type="subtoken", 
                          total_token_count=30000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enwiki_subtoken8192_size30000000_config.txt")


@registry.register_problem("multi_lm_enwiki_nocats_subword_small")
class MultiLmEnWikiSubwordSmallNoCats(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enwiki_nocats_subtoken8192_size30000000_config.txt", 
                              training=False, use_categories=False)
        except FileNotFoundError:
            setup_dataset("enwiki_nocats", 
                          [("wikipedia", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**13, vocab_type="subtoken", 
                          total_token_count=30000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enwiki_nocats_subtoken8192_size30000000_config.txt", 
                              use_categories=False)


@registry.register_problem("multi_lm_enwiki_subword_medium")
class MultiLmEnWikiSubwordMedium(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enwiki_subtoken16384_size100000000_config.txt",
                              training=True)
        except FileNotFoundError:
            setup_dataset("enwiki", 
                          [("wikipedia", 0, "split_as_one", data.ENGLISH)],
                          vocab_size=2**14, vocab_type="subtoken", 
                          total_token_count=100000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enwiki_subtoken16384_size100000000_config.txt")


@registry.register_problem("multi_lm_enfrwiki_subword_small")
class MultiLmEnFrWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enfrwiki_subtoken10000_size50000000_config.txt", 
                              training=True)
        except FileNotFoundError:
            setup_dataset("enfrwiki", 
                          [("wikipedia", 0, "split_as_one", (data.ENGLISH, data.FRENCH))],
                          vocab_size=10000, vocab_type="subtoken", 
                          total_token_count=50000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enfrwiki_subtoken10000_size50000000_config.txt")


@registry.register_problem("multi_lm_enfrwiki_subword_small_wikivocab")
class MultiLmEnFrWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            vocab_file_path = os.path.join(data.BASE_DIR,  
                "wiki_subtoken16384_size50000000_vocab.txt")
            self.load_dataset("enfrwiki_subtoken10000_size50000000_config.txt", 
                              training=True, vocab_file_path=vocab_file_path)
        except FileNotFoundError:
            setup_dataset("enfrwiki", 
                          [("wikipedia", 0, "split_as_one", (data.ENGLISH, data.FRENCH))],
                          vocab_size=10000, vocab_type="subtoken", 
                          total_token_count=50000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enfrwiki_subtoken10000_size50000000_config.txt")


@registry.register_problem("multi_lm_enfrwiki_nocats_subword_small")
class MultiLmEnFrWikiSubwordNoCats(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enfrwiki_nocats_subtoken10000_size50000000_config.txt", 
                              training=True, use_categories=False)
        except FileNotFoundError:
            setup_dataset("enfrwiki_nocats", 
                          [("wikipedia", 0, "split_as_one", (data.ENGLISH, data.FRENCH))],
                          vocab_size=10000, vocab_type="subtoken", 
                          total_token_count=50000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enfrwiki_nocats_subtoken10000_size50000000_config.txt",
                              use_categories=False)


@registry.register_problem("multi_lm_enfrwiki_subword_medium")
class MultiLmEnFrWikiSubwordMedium(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enfrwiki_subtoken16384_size100000000_config.txt", 
                              training=True)
        except FileNotFoundError:
            setup_dataset("enfrwiki", 
                          [("wikipedia", 0, "split_as_one", (data.ENGLISH, data.FRENCH))],
                          vocab_size=2**14, vocab_type="subtoken", 
                          total_token_count=100000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enfrwiki_subtoken16384_size100000000_config.txt")


@registry.register_problem("multi_lm_enfrwiki_nocats_subword_medium")
class MultiLmEnFrWikiSubwordMediumNoCats(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("enfrwiki_nocats_subtoken16384_size100000000_config.txt", 
                              training=False, use_categories=False)
        except FileNotFoundError:
            setup_dataset("enfrwiki_nocats", 
                          [("wikipedia", 0, "split_as_one", (data.ENGLISH, data.FRENCH))],
                          vocab_size=2**14, vocab_type="subtoken", 
                          total_token_count=100000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("enfrwiki_nocats_subtoken16384_size100000000_config.txt", 
                              use_categories=False)


@registry.register_problem("multi_lm_wiki_subword_small")
class MultiLmWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("wiki_subtoken16384_size50000000_config.txt", 
                              training=True)
        except FileNotFoundError:
            setup_dataset("wiki", 
                          [("wikipedia", 0, "split_as_one", 
                            (data.ENGLISH, data.FRENCH, data.GERMAN, data.CHINESE))],
                          vocab_size=2**14, vocab_type="subtoken", 
                          total_token_count=50000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("wiki_subtoken16384_size50000000_config.txt")


@registry.register_problem("multi_lm_wiki_nocats_subword_small")
class MultiLmWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("wiki_subtoken16384_size50000000_config.txt", 
                              training=False, use_categories=False)
        except FileNotFoundError:
            setup_dataset("wiki", 
                          [("wikipedia", 0, "split_as_one", 
                            (data.ENGLISH, data.FRENCH, data.GERMAN, data.CHINESE))],
                          vocab_size=2**14, vocab_type="subtoken", 
                          total_token_count=50000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("wiki_subtoken16384_size50000000_config.txt",
                              use_categories=False)


@registry.register_problem("multi_lm_wiki_subword_medium")
class MultiLmEnFrWikiSubwordSmall(MultiLmProblem):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.load_dataset("wiki_subtoken24576_size200000000_config.txt", 
                              training=False)
        except FileNotFoundError:
            setup_dataset("wiki", 
                          [("wikipedia", 0, "split_as_one", 
                            (data.ENGLISH, data.FRENCH, data.GERMAN, data.CHINESE))],
                          vocab_size=24576, vocab_type="subtoken", 
                          total_token_count=200000000, min_doc_length=20,
                          max_doc_length=150)
            self.load_dataset("wiki_subtoken24576_size200000000_config.txt")