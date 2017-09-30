"""
Tools for preparaing natural language datasets---currently 
assumes a fixed set of datasets (see code).

Classes:

    TextClassification, TextCleaner, TextProcessor, 
    DataHandler, DataCrawler

Usage:

    $ python data_preparation.py

                or

    >>> data_crawler = DataCrawler(root_path)
    >>> data_crawler.crawl(process=True)
"""

import os
import re
import tarfile
import io
from time import time

from bs4 import UnicodeDammit

import utilities as utils


BASE_DIR = "\\\\?\\" + os.path.join(os.path.join(
    os.path.abspath(os.pardir), "data"), "language_modeling")
if not os.path.exists(BASE_DIR):
    raise NotImplementedError("Can't work from current directory!")


class TextClassification:

    def __init__(self, language_type, language, domains=[], 
                 folder_name=None, directory_path=None, corpus_name=None):
        self.language_type = language_type
        self.language = language
        self.domains = []
        self.classification = [self.language_type, self.language] + self.domains
        self.folder_name = folder_name
        self.directory_path = directory_path
        self.corpus_name = corpus_name

    @classmethod
    def from_path(cls, directory_path):
        if NATURAL_LANGUAGE in directory_path:
            classifications = get_defined_classifications(NATURAL_LANGUAGE)
            classifications = [classification for classification in classification 
                if "\\" + classification.folder_name + "\\" in directory_path]
            if not classifications:
                classification = None
            else:
                classification = classifications[0] 
        elif PROGRAMMING_LANGUAGE in directory_path:
            language_type_index = utils.get_folder_index(
                directory_path, PROGRAMMING_LANGUAGE)
            language_type = directory_path[language_type_index]
            try:
                language = directory_path[language_type_index+1]
                domain1 = directory_path[language_type_index+2]  # software category
                domain2 = directory_path[language_type_index+3]  # library
            except IndexError:
                classification = None
            else:
                classification = cls(language_type, language, [domain1, domain2],
                                     folder_name=domain2, corpus_name=domain2)
        else:
            raise NotImplementedError("Unsupported language type (or a typo)!")

        classification.directory_path = directory_path

        return classification


def get_defined_classifications(language_type=None):
    return [value for value in globals().values() 
            if isinstance(value, TextClassification) 
            and (language_type is None or value.language_type == language_type)]


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


WIKIPEDIA = "wikipedia"


# English datasets
AMAZON_REVIEWS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=ENGLISH, domains=["amazon review"],
    folder_name="amazon_reviews", corpus_name="amazon review corpus")
BLOG_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=ENGLISH, domains=["blog"],
    folder_name="blog_authorship_corpus", corpus_name="blog authorship corpus")
BROWN_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=ENGLISH,
    folder_name="brown_corpus", corpus_name="brown corpus")
ENGLISH_WIKI = TextClassification(
    language_type=NATURAL_LANGUAGE, language=ENGLISH, domains=[WIKIPEDIA],
    folder_name="enwiki-20140707-corpus.xml", corpus_name="english wikipedia corpus")
GUTENBERG = TextClassification(
    language_type=NATURAL_LANGUAGE, language=ENGLISH, domains=["book"],
    folder_name="gutenberg", corpus_name="gutenberg dataset")
ENGLISH_DATASETS = [AMAZON_REVIEWS, BLOG_CORPUS, BROWN_CORPUS, 
                    ENGLISH_WIKI, GUTENBERG]

# French datasets
CHAMBERS_ROSTAND_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=FRENCH, domains=["news"],
    folder_name="chambers_rostand_journalistic_corpus", 
    corpus_name="chambers-rostand journalistic corpus")
FRENCH_WIKI = TextClassification(
    language_type=NATURAL_LANGUAGE, language=FRENCH, domains=[WIKIPEDIA],
    folder_name="frwiki-20140804-corpus.xml", corpus_name="french wikipedia corpus")
ORAL_NARRATIVE_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=FRENCH, domains=["speech", "narrative"],
    folder_name="oral_narrative_corpus", corpus_name="french oral narrative corpus")
ABU_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=FRENCH,
    folder_name="abu_corpus", corpus_name="abu corpus")

# German datasets
GERMAN_BIBLE = TextClassification(
    language_type=NATURAL_LANGUAGE, language=GERMAN, domains=["bible"],
    folder_name="german_bible", corpus_name="german bible")
GERMAN_WIKI = TextClassification(
    language_type=NATURAL_LANGUAGE, language=GERMAN, domains=[WIKIPEDIA],
    folder_name="dewiki-20140725-corpus.xml", corpus_name="german wikipedia corpus")
GERMANC = TextClassification(
    language_type=NATURAL_LANGUAGE, language=GERMAN,
    folder_name="GerManC", corpus_name="germanc corpus")
PAROLE_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=GERMAN,
    folder_name="parole_corpus", corpus_name="german parole corpus")

# Chinese datasets
LANCASTER_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=CHINESE,
    folder_name="lancaster_mandarin_corpus", corpus_name="lancaster mandarin corpus")
CHINESE_WIKI = TextClassification(
    language_type=NATURAL_LANGUAGE, language=CHINESE, domains=[WIKIPEDIA],
    folder_name="zhwiki-20140804-corpus.xml", corpus_name="chinese wikipedia corpus")
LEIDEN_WEIBO_CORPUS = TextClassification(
    language_type=NATURAL_LANGUAGE, language=CHINESE, domains=["microblog"],
    folder_name="leiden_weibo_corpus-messages", corpus_name="leiden weibo corpus")


DOCUMENT_SEPARATOR = "\n\n<document_separator>\n\n"


class TextCleaner:

    whitespace_regexes = [
        r"\n?([ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*)+", 
        r"\t+", r"\r+", r"\f+", r"\v+", r"[ ]{2,}", r"(?<=\n)[ ]+"
    ]
    whitespace_regexes = [re.compile(regex) for regex in whitespace_regexes]
    extra_whitespace_regex = re.compile(r"(?<!\n)[ \t\r\f\v]*\n[ \t\r\f\v]*(?!\n)")
    spaces_to_tabs_regex = re.compile(r"[ ]{4}")
    image_tag_regex = re.compile(r"{.*?:.*?}")
    markup_tags_regex = re.compile(r"<.*?>")
    sound_regex = re.compile(r"\[.*?\]")
    docstring_regexes = {
        PYTHON: (re.compile(r"^\s*?\"\"\".*?\"\"\"\s*?\n?"), 
                   re.compile(r"^\s*?\'\'\'.*?\'\'\'\s*?\n?")),
        C: (re.compile(r"^\s*?/\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*?\n?"),),
        FORTRAN: (),
        LISP: (),
    }
    comment_regexes = {
        PYTHON: (re.compile(r"^\s*?#.*?\n"),),
        C: (re.compile(r"^\s*?//.*?\n"),),
        FORTRAN: (re.compile(r"^[\*cCdD]\s.*?\n"), re.compile(r"^\s*?!.*?\n"),),
        LISP: (re.compile(r"^\s*?;.*?\n"), 
                 re.compile(r"^\s*?#\|(?:(?!#\|).)*?\|#\s*?\n?")),
    }
    spell_correct_map = utils.get_spell_corrector()

    def __init__(self, classification):
        self.classification = classification

    def clean(self, document, correct_spelling=False):
        document = document.strip()
        if self.classification in [BROWN_CORPUS]:
            document = self.remove_pos_tags(document)
        if self.classification in [GUTENBERG]:
            document = image_tag_regex.sub("", document)
        if self.classification in [ORAL_NARRATIVE_CORPUS, PAROLE_CORPUS]:
            document = self.remove_markup(document, all_tags=True)
        if self.classification == ORAL_NARRATIVE_CORPUS:
            document = sound_regex.sub("", document)  # Remove sound annotation
        if self.classification.language == CHINESE:
            document = self.convert_chinese_punctuation(document)

        if self.classification.language_type == NATURAL_LANGUAGE:
            document = self.remove_markup(document)
        if self.classification.language_type == PROGRAMMING_LANGUAGE:
            document = self.convert_spaces_to_tabs(
                self.remove_multiline_comments(document))

        return self.remove_excess_whitespace(document)

    def remove_excess_whitespace(self, text):
        text = text.strip()
        for i, regex in enumerate(whitespace_regexes):
            if i == 0:
                text = regex.sub("\n\n", text)
            elif i == 5:
                text = regex.sub("  ", text)
            else:
                text = regex.sub("", text)
        if self.classification == ORAL_NARRATIVE_CORPUS:
            text = extra_whitespace_regex.sub(" ", text_cleaned)

        return text

    def remove_markup(self, text, all_tags=False):
        text = text.replace("&nbsp;", "").replace("urlLink", "")
        text = text.replace("<h>", "").replace("</h>", "")
        text = text.replace("[Illustration]", "")
        if all_tags:
            text = markup_tags_regex.sub("", text)
        return text

    def remove_pos_tags(self, text):
        return re.sub(r"/[^ \t\n\r\f\v]+", "", text)

    def convert_punctuation(self, text):
        conversion_map = {
            "。": ". ",
            "。\n": ".\n",
            "，": ", ",
            "，\n": ",\n", 
            "、": ", ",
            "、\n": ",\n",
            "：": ": ", 
            "；": "; ", 
            "？": "? ",
            "！": "! ",
            "。</": ".</", 
            "？</": "?</",
            "：</": ":</",
            "＂": "\"",
            "　": " ",
        }
        for punc_char, converted_punc_char in conversion_map.items():
            text = text.replace(punc_char, converted_punc_char)

        return text

    def correct_spelling(self, natural_text):
        return utils.correct_spelling(spell_correct_map, natural_text)

    def remove_multiline_comments(self, code_text):
        text = code_text
        if self.language == PYTHON:
            for regex in docstring_regexes[PYTHON]:
                text = regex.sub("", text, flags=re.MULTILINE|re.DOTALL)
            text = comment_regexes[PYTHON][0].sub("", text, flags=re.MULTILINE)
        elif self.language == C:
            text = docstring_regexes[C][0].sub("", text, flags=re.MULTILINE|re.DOTALL)
            text = comment_regexes[C][0].sub("", text, flags=re.MULTILINE)
        elif self.language == FORTRAN:
            for regex in comment_regexes[FORTRAN]:
                text = regex.sub("", text, flags=re.MULTILINE)
        elif self.language == LISP:
            # Note: doesn't remove 'docstrings'
            text = comment_regexes[LISP][0].sub("", text, flags=re.MULTILINE)
            text_prev = ""
            while text != text_prev:
                text_prev = text
                text = comment_regexes[LISP][1].sub(
                    "", text, flags=re.MULTILINE|re.DOTALL)
        else:
            raise NotImplementedError("Can't remove comments for "
                                      "source code in that language!")

        return text

    def convert_spaces_to_tabs(self, code_text):
        return re.sub(spaces_to_tabs_regex, "\t", code_text)


class TextProcessor:

    def __init__(self, classification):
        self.classification = classification
        self.cleaner = TextCleaner(classification)

    def process(self, text, clean=True):
        documents = []
        if self.classification == AMAZON_REVIEWS:
            documents = self.extract_amazon_reviews(text)
        elif self.classification == BLOG_CORPUS:
            documents = self.extract_blogs(text)
        elif WIKIPEDIA in self.classification.domains:
            documents = self.extract_wiki_articles(text)
        elif self.classification == ORAL_NARRATIVE_CORPUS:
            text = self.extract_oral_narrative(text)
        elif self.classification == ABU_CORPUS:
            text = self.extract_abu_corpus_text(text)
        elif self.classification == PAROLE_CORPUS:
            documents = self.extract_parole_corpus_documents(text)
        elif self.classification == LANCASTER_CORPUS:
            documents = self.extract_lancaster_documents(text)
        elif self.classification == LEIDEN_WEIBO_CORPUS:
            documents = self.extract_leiden_weibo_messages(text)

        if clean:
            if not documents:
                text = self.cleaner.clean(document)
            if isinstance(documents, list):
                for i, document in enumerate(documents):
                    documents[i] = self.cleaner.clean(document)
            elif isinstance(documents, dict):
                for key in documents:
                    for i, document in enumerate(documents[key]):
                        documents[key][i] = self.cleaner.clean(document)
            else:
                raise NotImplementedError(
                    "Implementation error: unexpected type!")

        if documents:
            return documents
        else:
            return text

    def extract_amazon_reviews(self, text, by_user=True):
        if by_user:
            reviews = {}
        else:
            reviews = []
        for i, line in enumerate(text.split("\n")):
            if i % 200000 == 0:
                print("Completed processing", i, "reviews...")
            if not line.strip():
                continue
            review_dict = eval(line)
            if by_user:
                prev_reviews = reviews.get(review_dict["reviewerID"], [])
                reviews[review_dict["reviewerID"]] = (
                    prev_reviews + [review_dict["reviewText"]])
            else:
                reviews.append(review_dict["reviewText"])

        return reviews

    def extract_blogs(self, text):
        blogs = []
        for fragment in text.split("<post>")[1:]:
            if not fragment.strip():
                continue
            blog = fragment.split("</post>")[0]
            if "<date>" in blog or "</date>" in blog:
                raise ValueError("Not implemented correctly!")
            else:
                blogs.append(blog)

        return blogs

    def extract_wiki_articles(self, text):
        articles = text.split("</article>")
        article_by_name = {}
        for i, article in enumerate(articles):
            if i == 0 or i == len(articles) - 1:
                continue  # First and last elements may not be articles
            if i % 100000 == 0:
                print("Completed processing", i, "articles...")
            # Get name
            names = re.findall(r"<article name=\"(.+?)\">", article)
            if len(names) != 1:
                print(names)
                raise ValueError("No unique article tag!")
            name = names[0]

            # Remove article tag
            article_cleaned = re.sub(r"<article name=\"(.+?)\">", "", article)

            # Remove 'footnote' material: notes, references, bibliography, etc.
            regex = (r"<h>\s*(see also|voir aussi|siehe auch|\\u53c2\\u89c1|note"
                     + r"|reference|bibliograph|literatur|einzelnachweise"
                     + r"|\\u53c2\\u8003\\u8457\\u4f5c|\\u76f8\\u95dc)\s*</h>")
            article_cleaned = re.split(regex, article_cleaned, flags=re.IGNORECASE)[0]

            article_by_name[name] = (name + "\n\n" + article_cleaned)

        return article_by_name

    def extract_oral_narrative(self, html_text):
        return UnicodeDammit(html_text).unicode_markup.split("<body>")[1]

    def extract_abu_corpus_text(self, text):
        return re.split(r"-{25} DEBUT DU FICHIER .*? -{32}", text)[1]

    def extract_parole_corpus_documents(self, text):
        text = UnicodeDammit(text).unicode_markup
        if "<div1" in text:
            documents = re.split(r"<div1 type=.*?>", text, flags=re.IGNORECASE)[1:]
        else:
            documents = re.split(r"<body>", text, flags=re.IGNORECASE)[1:]

        return documents

    def extract_lancaster_documents(self, text):
        documents = re.split(r"<file ID=.*?>", text)[1:]
        cleaned_documents = []
        for document in documents:
            if "<p>" in document:
                tokens = []
                paragraphs = document.split("<p>")
                for paragraph in paragraphs:
                    if "<s n=" in paragraph:
                        sentences = re.split(r"<s n=.*?>", paragraph)
                        for j, sentence in enumerate(sentences):
                            if "<w POS=" in sentence:
                                words = re.split(r"<[w,c] POS=.*?>", sentence)
                                for word in words:
                                    tokens.append(re.sub(r"<.*?>", "", word).strip())
                                if j == len(sentences) - 1:
                                    tokens.append("\n")
                document = "".join(token for token in tokens 
                                   if token.strip() or token == "\n")
                cleaned_documents.append(document)

        return cleaned_documents

    def extract_leiden_weibo_messages(self, csv_text):
        messages_with_metadata = (line for line in csv_text.split("\n") 
                                  if line.strip())
        messages = []
        for i, message in enumerate(messages_with_metadata):
            if i % 100000 == 0:
                print("Completed processing", i, "messages...")
            try:
                message = message.split("\",\"")[5].replace("\"", "")
            except IndexError:
                continue  # Likely indicates message with no words, so skipped
            # Skip very short messages or messages with no words
            if len(message) >= 15 and message.count("\\N") < 1:
                messages.append(message)

        return messages

    def get_brown_corpus_categories(category_text):
        ids_by_category = {}
        for line in category_text.split("\n"):
            if line.strip():
                doc_id, category = line.strip().split(" ")
                prev_ids = ids_by_category.get(category, [])
                ids_by_category[category] = prev_ids + [doc_id]

        return ids_by_category

    def get_author_genre_abu_text(self, text):
        author_names = re.findall(r"<IDENT_AUTEURS (.*?)>", text)
        genres = re.findall(r"<GENRE (.*?)>", text)
        if len(author_names) != 1:
            raise ValueError("No unique author name found!")
        elif len(genres) != 1:
            raise ValueError("No unique genre found!")
        else:
            return author_names[0], genres[0]

    def get_lancaster_domain(self, text):
        domains = re.findall(r"<text ID=.*? TYPE=\"(.*?)\">", text)
        if len(domains) != 1:
            raise ValueError("No unique domain found!")
        return domains[0]

    def get_document_identifier(self, file_name):
        if self.classification == BLOG_CORPUS:
            return file_name.split(".")[0]  # string
        elif self.classification == GUTENBERG:
            author = file_name.split("___")[0]
            title = utils.remove_extension(file_name.split("___")[1])   
            return author, title   # pair of strings
        elif self.classification == CHAMBERS_ROSTAND_CORPUS:
            categories = {
                "c": "cultural", 
                "e": "editorial",
                "f": "finance",
                "i": "international news",
                "n": "national news",
                "s": "sports",
            }
            return categories[file_name.split("_")[2].lower()]  # string
        elif self.classification == ORAL_NARRATIVE_CORPUS:
            return file_name.split("_")[0]  # string
        elif self.classification == GERMANC:
            genres = {
                "DRAM": "drama", 
                "HUMA": "humanities",
                "LETT": "letter",
                "LEGA": "legal",
                "NARR": "narrative",
                "NEWS": "newspaper",
                "SCIE": "science",
                "SERM": "sermon",
            }
            return genres[file_name.split("_")[0]]  # string


class DataHandler:

    def __init__(self, classification, root_path):
        self.classification = classification
        self.text_processor = TextProcessor(classification)
        self.root_path = root_path

    def process(self):
        self.get_file_names()
        for file_name in self.file_names:
            texts = self.retrieve_texts(file_name)
            for text in texts:
                output_file_path = self.make_output_file_path(file_name, text)
                documents = self.text_processor.process(text, clean=True)
                self.store_documents(documents, output_file_path)

    def get_file_names(self):
        extensions = [".txt", ".json", ".xml", "-stripped.html", "sgm"]
        self.file_names = utils.get_files_from_directory(
            self.root_path, extensions=extensions)
        self.file_names = [file_name for file_name in file_name 
            if not utils.remove_extension(file_name).endswith("_processed")]
        if WIKIPEDIA in self.classification.domains:
            self.file_names = [file_name for file_name in self.file_names 
                               if "corpus-processed.txt" in file_name]
        if self.classification == BROWN_CORPUS:
            self.file_names = [file_name for file_name in self.file_names
                               if re.fullmatch(r"[a-zA-Z]{2}[0-9]{2}", file_name)]
        if self.classification.language_type == PROGRAMMING_LANGUAGE:
            self.file_names = [file_name for file_name in file_name 
                if (utils.detect_language(file_name, source_code=True) 
                    == self.classification.language)]

    def retrieve_texts(file_name):
        for text in utils.open_file(os.path.join(self.root_path, file_name)):
            yield UnicodeDammit(text).unicode_markup()

    def make_output_file_path(self, file_name, text):
        # Make folder path
        directories = self.root_path.split("\\")
        try:
            base_index = [i for i, directory in enumerate(directories) 
                          if directory == "language_modeling"][0]
        except IndexError:
            raise RuntimeError("Unexpected file path names!")

        file_path = "\\".join(
            directory for i, directory in enumerate(directories) 
            if i >= base_index and directory in self.classification.classification)
        file_path = os.path.join(file_path, "processed")
        for domain in self.classification.domains:
            if "\\" + domain + "\\" not in file_path:
                file_path = os.path.join(file_path, domain)

        # Make file name
        document_identifier = self.text_processor.get_document_identifier(file_name)
        extension_index = file_name.rfind(".")
        if self.classification == BLOG_CORPUS:
            file_name_root = document_identifier
        elif self.classification == GUTENBERG:
            file_name_root = document_identifier[1]
            document_identifier = document_identifier[0]
        else:
            file_name_root = file_name[:extension_index]
        output_file_name = file_name_root + "_processed" + file_name[extension_index:]

        # Add additional domain info
        if document_identifier and self.classification != BLOG_CORPUS:
            file_path = os.path.join(file_path, document_identifier.lower())
        if self.classification in [BROWN_CORPUS, ABU_CORPUS, LANCASTER_CORPUS]:
            if self.classification == BROWN_CORPUS:
                categories_text = utils.open_file(
                    os.path.join(self.root_path, "cats.txt"))
                categories = invert_mapping(
                    self.text_processor.get_brown_corpus_categories(categories_text))
                file_path_suffix = categories[file_name][0]
            elif self.classification == ABU_CORPUS:
                author, genre = self.text_processor.get_author_genre_abu_text(text)
                file_path_suffix = os.path.join(genre, author)
            elif self.classification == LANCASTER_CORPUS:
                file_path_suffix = self.text_processor.get_lancaster_domain(text)
            file_path = os.path.join(file_path, file_path_suffix)

        try:
            os.makedirs(file_path)
        except FileExistsError:
            pass

        return os.path.join(file_path, output_file_name)

    def store_documents(self, documents, output_file_path):
        if isinstance(documents, list) and len(documents) == 1:
            documents = documents[0]
        if isinstance(documents, str):
            utils.store_text(documents, output_file_path)
        elif isinstance(documents, list):
            utils.store_text(DOCUMENT_SEPARATOR.join(documents), output_file_path)
        elif isinstance(documents, dict):
            # Store values in a tar file by key
            output_data = {}
            for key, value in documents.items():
                file_name = key + ".txt"
                if isinstance(value, list):
                    text = DOCUMENT_SEPARATOR.join(value)
                output_data[file_name] = text
            utils.store_tarfile(output_data, output_file_path)


class DataCrawler:

    def __init__(self, root_path):
        self.root_path = root_path

    def crawl(self, process=False, crawl_processed=False):
        text_classifications = []
        for directory_path, directory_names, file_names in os.walk(self.root_path):
            if ((crawl_processed and "\\processed\\" not in directory_path) 
                    or (not crawl_processed and "\\processed\\" in directory_path)):
                continue
            try:
                classification = TextClassification.from_path(directory_path)
            except NotImplementedError as e:
                print(str(e))
            if classification is not None:
                if ((classification.language_type == NATURAL_LANGUAGE 
                        and not directory_names 
                        and (classification != CHAMBERS_ROSTAND_CORPUS 
                             or "Plain text" not in directory_path)
                        and (classification != ORAL_NARRATIVE_CORPUS 
                             or "HTML" not in directory_path)
                        and (classification != GERMANC 
                             or "RAW" not in directory_path)
                        and (classification != LANCASTER_CORPUS 
                             or "character" not in directory_path)) 
                        or classification.language_type != NATURAL_LANGUAGE
                        ):
                    text_classifications.append(classification)
                    if process:
                        data_handler = DataHandler(classification, directory_path)
                        data_handler.process()

        return text_classifications


if __name__ == '__main__':
    data_crawler = DataCrawler(BASE_DIR)
    data_crawler.crawl(process=True)