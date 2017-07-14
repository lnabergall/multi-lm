"""Functions for preparing text data of various formats for processing."""

import os
import re
import tokenize as py_token
from xml.etree import ElementTree

import textacy
from pygments import lex
from pygments.lexer.python import PythonLexer
from pygments.lexer.c_cpp import CLexer
from pygments.lexer.fortran import FortranFixedLexer, FortranLexer
from pygments.lexer.lisp import CommonLispLexer
from bs4 import BeautifulSoup


BASE_DIR = os.path.join(os.path.join(os.pardir, "data"), "language_modeling")
if not os.path.exists(BASE_DIR):
    raise NotImplementedError("Can't work from current directory!")


C = "c"
PYTHON = "python"
FORTRAN = "fortran"
LISP = "common lisp"

ENGLISH = "english"
FRENCH = "french"
GERMAN = "german"
CHINESE = "chinese"

HTML = "html"
LATEX = "latex"
YAML = "yaml"
MARKDOWN = "markdown"


def detect_language(file_path):
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
        with open(file_path, "r") as text_file:
            text = text_file.read()
        if textacy.text_utils.detect_language(text).lower() == "en":
            language = ENGLISH
        elif textacy.text_utils.detect_language(text).lower() == "fr":
            language = FRENCH
        elif textacy.text_utils.detect_language(text).lower() == "de":
            language = GERMAN
        elif textacy.text_utils.detect_language(text).lower() == "zh":
            language = CHINESE
        else:
            language = None

    return language


def tokenize(file_path, use_native_python=False):
    with open(file_path) as text_file:
        text = text_file.read()
    if detect_language(file_path) == PYTHON:
        if use_native_python:
            with open(file_path, "rb") as source_file:
                file_reader = source_file.readline
                try:
                    tokens = py_token.tokenize(file_reader)
                except py_token.TokenError:
                    tokens = None
                else:
                    tokens = list(token.string for token in tokens)
        else:
            tokens = [token_pair[1] for token_pair in lex(text, PythonLexer())]
    elif detect_language(file_path) == C:
        tokens = [token_pair[1] for token_pair in lex(text, CLexer())]
    elif detect_language(file_path) == FORTRAN:
        if file_path.lower().endswith(".f95") or file_path.lower().endswith(".f03"):
            tokens = [token_pair[1] for token_pair in lex(text, FortranLexer())]
        else:
            tokens = [token_pair[1] for token_pair in lex(text, FortranFixedLexer())]
    elif detect_language(file_path) == LISP:
        tokens = [token_pair[1] for token_pair in lex(text, CommonLispLexer())]
    elif detect_language(file_path) in [ENGLISH, FRENCH, GERMAN, CHINESE]:
        document = textacy.doc.Doc(text)
        tokens = [str(token) for token in document]

    return tokens


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
        text = text.replace(word, corrected_word)

    return text


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
    new_file_path = "".join(file_path.split(".")[:-1]) + "_processed" + extension
    if detect_language(file_path) in [ENGLISH, FRENCH, GERMAN, CHINESE]:
        processed_text = " ".join(tokens)
    else:
        processed_text = "".join(tokens)
    with open(new_file_path, "w") as processed_text_file:
        processed_text_file.write(processed_text)


def remove_excess_whitespace(text):
    text_cleaned = text
    for i, regex_char in enumerate(
            [r"\n+", r"\t+", r"\r+", r"\f+", r"\v+", r"[ ]+"]):
        if i == 0:
            text_cleaned = re.sub(regex_char, "\n", text_cleaned)
        elif i == 5:
            text_cleaned = re.sub(regex_char, " ", text_cleaned)
        else:
            text_cleaned = re.sub(regex_char, "", text_cleaned)

    return text_cleaned


def convert_spaces_to_tabs(text):
    return re.sub(r"[ ]{4}", "\t", text)


def convert_chinese_punctuation(text):
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
        "　": " ",
    }
    for punc_char in conversion_map:
        text = text.replace(punc_char, conversion_map[punc_char])

    return text


def extract_amazon_reviews(text, by_user=True):
    if by_user:
        reviews = {}
    else:
        reviews = []
    for line in text.split("\n"):
        review_dict = eval(line)
        if by_user:
            prev_reviews = reviews.get(review_dict["reviewerID"], [])
            reviews[review_dict["reviewerID"]] = (
                prev_reviews + [review_dict["reviewText"]])
        else:
            reviews.append(review_dict["reviewText"])

    return reviews


def extract_blogs(text, remove_hyperlinks=False):
    blogs = []
    for fragment in text.split("<post>"):
        blog = fragment.split("</post>")[0]
        blog = blog.replace("&nbsp;", "")
        if remove_hyperlinks:
            blog = blog.replace("urlLink", "")
        if "<date>" in blog or "</date>" in blog:
            raise ValueError("Not implemented correctly!")
        else:
            blogs.append(remove_excess_whitespace(blog))

    return blogs


def get_blog_author_id(file_name):
    return file_name.split(".")[0]


def clean_brown_corpus_text(text):
    # remove tags and excess whitespace
    text_cleaned = re.sub(r"/[^ \t\n\r\f\v]+", "", text)
    return remove_excess_whitespace(text_cleaned)


def extract_brown_corpus_categories(category_text):
    ids_by_category = {}
    for line in category_text.split("\n"):
        doc_id, category = line.strip().split(" ")
        prev_ids = ids_by_category.get(category, [])
        ids_by_category[category] = prev_ids + [doc_id]

    return ids_by_category


def extract_wiki_articles(text):
    articles = text.split("</article>")
    article_by_name = {}
    for article in articles:
        # Get name
        names = re.findall(r"<article name=\"(.+?)\">", article)
        if len(names) != 1:
            raise ValueError("More article tags than expected!")
        name = names[0]

        # Remove article tag
        article_cleaned = re.sub(r"<article name=\"(.+?)\">", "", article)

        # Remove 'footnote' material: notes, references, bibliography, etc.
        regex = (r"<h>\s*(see also|voir aussi|siehe auch|\\u53c2\\u89c1|note"
                 + r"|reference|bibliograph|literatur|einzelnachweise"
                 + r"|\\u53c2\\u8003\\u8457\\u4f5c|\\u76f8\\u95dc)\s*</h>")
        article_cleaned = re.split(regex, article_cleaned, flags=re.IGNORECASE)[0]

        # Remove heading tags
        article_cleaned = article_cleaned.replace("<h>", "").replace("</h>", "")
        article_cleaned = convert_chinese_punctuation(article_cleaned)

        article_by_name[name] = remove_excess_whitespace(article_cleaned)

    return article_by_name


def clean_gutenberg_text(text):
    text_cleaned = text.replace("[Illustration]", "")
    text_cleaned = re.sub(r"{.*?:.*?}", "", text_cleaned)   # Remove image 'tag'
    return remove_excess_whitespace(text_cleaned)


def get_name_gutenberg(file_name):
    return file_name.split("___")[0]


def clean_chambers_rostand_text(text):
    return remove_excess_whitespace(text)


def get_category_chambers_rostand(file_name):
    categories = {
        "c": "cultural", 
        "e": "editorial",
        "f": "finance",
        "i": "international news",
        "n": "national news",
        "s": "sports",
    }
    return categories[file_name.split("_")[2].lower()]


def clean_german_bible_text(text):
    return remove_excess_whitespace(text)


def clean_germanc_text(text):
    return remove_excess_whitespace(text)


def get_genre_name_germanc(file_name):
    genres = {
        "DRAM": "drama", 
        "HUMA": "humanities",
        "LETT": "letter"
        "LEGA": "legal",
        "NARR": "narrative",
        "NEWS": "newspaper",
        "SCIE": "science",
        "SERM": "sermon",
    }
    return genres[file_name.split("_")[0]]


def extract_parole_corpus_documents(text):
    text = BeautifulSoup(text).text  # Convert html encoding to utf-8
    if "<div1" in text:
        documents = re.split(r"<div1 type=.*?>", text, flags=re.IGNORECASE)[1:]
    else:
        documents = re.split(r"<body>", text, flags=re.IGNORECASE)[1:]

    # Remove tags + whitespace
    for i, document in enumerate(documents):
        documents[i] = remove_excess_whitespace(re.sub(r"<.*?>", "", document))

    return [document for document in documents if document.strip()]


def extract_lancaster_documents(text):
    documents = re.split(r"<file ID=.*?>", text)[1:]
    cleaned_documents = []
    for i, document in enumerate(documents):
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
                                tokens.append(re.sub(r"<.*?>", "", word))
                            if j != len(sentences) - 1:
                                tokens.append(". ")
                            else:
                                tokens.append(".\n")
            document = "".join(tokens)
            if document.strip():
                cleaned_documents.append(convert_chinese_punctuation(document))

    return cleaned_documents


