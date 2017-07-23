"""Functions for preparing text data of various formats for processing."""

import os
import re
import tarfile
import io
from time import time

import textacy
import cchardet
from bs4 import UnicodeDammit


BASE_DIR = os.path.join(os.path.join(os.pardir, "data"), "language_modeling")
if not os.path.exists(BASE_DIR):
    raise NotImplementedError("Can't work from current directory!")

# To mitigate file path character limit issue
BASE_DIR = ("C:\\Users\\lnabe\\Dropbox\\Artificial Intelligence and Robotics\\"
            "learning-language\\data\\language_modeling")

NATURAL_LANGUAGE = "natural_language"
PROGRAMMING_LANGUAGE = "programming_language"
MARKUP_LANGUAGE = "markup_language"

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

# English datasets
AMAZON_REVIEWS = "amazon_reviews"
BLOG_CORPUS = "blog_authorship_corpus"
BROWN_CORPUS = "brown_corpus"
ENGLISH_WIKI = "enwiki-20140707-corpus.xml"
GUTENBERG = "gutenberg"

# French datasets
CHAMBERS_ROSTAND_CORPUS = "chambers_rostand_journalistic_corpus"
FRENCH_WIKI = "frwiki-20140804-corpus.xml"
ORAL_NARRATIVE_CORPUS = "oral_narrative_corpus"
ABU_CORPUS = "abu_corpus"

# German datasets
GERMAN_BIBLE = "german_bible"
GERMAN_WIKI = "dewiki-20140725-corpus.xml"
GERMANC = "GerManC"
PAROLE_CORPUS = "parole_corpus"

# Chinese datasets
LANCASTER_CORPUS = "lancaster_mandarin_corpus"
CHINESE_WIKI = "zhwiki-20140804-corpus.xml"
LEIDEN_WEIBO_CORPUS = "leiden_weibo_corpus-messages"


def invert_mapping(mapping):
    inverse_mapping = {}
    for key, values in mapping.items():
        for value in values:
            inverse_mapping.setdefault(value, []).append(key)

    return inverse_mapping


def get_files_from_directory(root_path, extension=None):
    files = [file for file in os.listdir(root_path) 
             if os.path.isfile(os.path.join(root_path, file))]
    if extension is not None:
        files = [file for file in files 
                 if file.lower().endswith(extension.lower())]
    return files


def open_file(file_path, encoding=None):
    if not encoding:
        encoding = get_encoding(file_path)
    try:
        with open(file_path, "r", encoding=encoding) as text_file:
            text = text_file.read()
    except (UnicodeDecodeError, LookupError):
        try:
            with open(file_path, "r") as text_file:
                text = text_file.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as text_file:
                text = text_file.read()

    return text


def add_text_file(tar, string, file_name):
    string_file = io.BytesIO(string.encode("utf-8"))
    tarinfo = tarfile.TarInfo(file_name)
    tarinfo.size = len(string)
    tar.addfile(tarinfo, string_file)


def get_encoding(file_path):
    with open(file_path, "rb") as file:
        text = file.read()
    return cchardet.detect(text)["encoding"]


def detect_language(file_path, text=None, source_code=False):
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


def remove_excess_whitespace(text):
    text_cleaned = text.strip()
    for i, regex_char in enumerate(
            [r"\n?([ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*)+", r"\t+", 
             r"\r+", r"\f+", r"\v+", r"[ ]{2,}", r"(?<=\n)[ ]+"]):
        if i == 0:
            text_cleaned = re.sub(regex_char, "\n\n", text_cleaned)
        elif i == 5:
            text_cleaned = re.sub(regex_char, "  ", text_cleaned)
        else:
            text_cleaned = re.sub(regex_char, "", text_cleaned)

    return text_cleaned


def convert_spaces_to_tabs(code_text):
    return re.sub(r"[ ]{4}", "\t", code_text)


def remove_multiline_comments(code_text, language):
    if language == PYTHON:
        text_cleaned = re.sub(r"^\s*?\"\"\".*?\"\"\"\s*?\n?", "", 
                              code_text, flags=re.MULTILINE|re.DOTALL)
        text_cleaned = re.sub(r"^\s*?\'\'\'.*?\'\'\'\s*?\n?", "", 
                              text_cleaned, flags=re.MULTILINE|re.DOTALL)
        text_cleaned = re.sub(r"^\s*?#.*?\n", "", 
                              text_cleaned, flags=re.MULTILINE)
    elif language == C:
        text_cleaned = re.sub(r"^\s*?/\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*?\n?", "", 
                              code_text, flags=re.MULTILINE|re.DOTALL)
        text_cleaned = re.sub(r"^\s*?//.*?\n", "", 
                              text_cleaned, flags=re.MULTILINE)
    elif language == FORTRAN:
        text_cleaned = re.sub(r"^[\*cCdD]\s.*?\n", "", 
                              code_text, flags=re.MULTILINE)
        text_cleaned = re.sub(r"^\s*?!.*?\n", "",
                              text_cleaned, flags=re.MULTILINE)
    elif language == LISP:
        # Note: doesn't remove 'docstrings'
        text_cleaned = re.sub(r"^\s*?;.*?\n", "", 
                              code_text, flags=re.MULTILINE)
        text_cleaned_prev = text_cleaned
        while text_cleaned != text_cleaned_prev:
            text_cleaned_prev = text_cleaned
            text_cleaned = re.sub(r"^\s*?#\|(?:(?!#\|).)*?\|#\s*?\n?", "",
                                  text_cleaned, flags=re.MULTILINE|re.DOTALL)
    else:
        raise NotImplementedError("Can't remove comments for "
                                  "source code in that language!")

    return text_cleaned


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
        "＂": "\"",
        "　": " ",
    }
    for punc_char, converted_punc_char in conversion_map.items():
        text = text.replace(punc_char, converted_punc_char)

    return text


def extract_amazon_reviews(text, by_user=True):
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


def extract_blogs(text, remove_hyperlinks=True):
    blogs = []
    for fragment in text.split("<post>")[1:]:
        if not fragment.strip():
            continue
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
        if line.strip():
            doc_id, category = line.strip().split(" ")
            prev_ids = ids_by_category.get(category, [])
            ids_by_category[category] = prev_ids + [doc_id]

    return ids_by_category


def extract_wiki_articles(text):
    articles = text.split("</article>")
    article_by_name = {}
    for i, article in enumerate(articles):
        if i == len(articles) - 1:
            break  # Last element is likely not an article
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

        # Remove heading tags
        article_cleaned = article_cleaned.replace("<h>", "").replace("</h>", "")
        article_cleaned = convert_chinese_punctuation(article_cleaned)

        article_by_name[name] = (name + "\n\n" 
            + remove_excess_whitespace(article_cleaned))

    return article_by_name


def clean_gutenberg_text(text):
    text_cleaned = text.replace("[Illustration]", "")
    text_cleaned = re.sub(r"{.*?:.*?}", "", text_cleaned)   # Remove image 'tag'
    return remove_excess_whitespace(text_cleaned)


def get_author_title_gutenberg(file_name):
    return file_name.split("___")


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


def extract_oral_narrative(html_text):
    html_text = UnicodeDammit(html_text).unicode_markup.split("<body>")[1]
    html_text_cleaned = re.sub(r"\[.*?\]", "", html_text)   # Remove sounds
    text_cleaned = re.sub(r"<.*?>", "", html_text_cleaned)  # Remove tags
    text_cleaned = remove_excess_whitespace(text_cleaned)
    return re.sub(r"(?<!\n)[ \t\r\f\v]*\n[ \t\r\f\v]*(?!\n)", " ", text_cleaned)


def get_storyteller_oral_narrative(file_name):
    return file_name.split("_")[0]


def clean_abu_corpus_text(text):
    cleaned_text = re.split(r"-{25} DEBUT DU FICHIER .*? -{32}", text)[1]
    return remove_excess_whitespace(cleaned_text)


def get_author_genre_abu_text(text):
    author_names = re.findall(r"<IDENT_AUTEURS (.*?)>", text)
    genres = re.findall(r"<GENRE (.*?)>", text)
    if len(author_names) != 1:
        raise ValueError("No unique author name found!")
    elif len(genres) != 1:
        raise ValueError("No unique genre found!")
    else:
        return author_names[0], genres[0]


def clean_german_bible_text(text):
    return remove_excess_whitespace(text)


def clean_germanc_text(text):
    return remove_excess_whitespace(text)


def get_genre_name_germanc(file_name):
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
    return genres[file_name.split("_")[0]]


def extract_parole_corpus_documents(text):
    text = UnicodeDammit(text).unicode_markup
    if "<div1" in text:
        documents = re.split(r"<div1 type=.*?>", text, flags=re.IGNORECASE)[1:]
    else:
        documents = re.split(r"<body>", text, flags=re.IGNORECASE)[1:]

    # Remove tags + whitespace
    for i, document in enumerate(documents):
        documents[i] = remove_excess_whitespace(re.sub(r"<.*?>", "", document))

    return [document for document in documents if document.strip()]


def extract_lancaster_documents(text):
    domains = re.findall(r"<text ID=.*? TYPE=\"(.*?)\">", text)
    if len(domains) != 1:
        raise ValueError("No unique domain found!")
    else:
        domain = domains[0]
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
            if document.strip():
                cleaned_documents.append(convert_chinese_punctuation(document))

    return cleaned_documents, domain


def extract_leiden_weibo_messages(csv_text):
    messages_with_metadata = [line for line in csv_text.split("\n") 
                              if line.strip()]
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
            messages.append(convert_chinese_punctuation(message))

    return messages


def prepare_corpus_folder(corpus_name, root_path):
    root_path = "\\\\?\\" + root_path  # Mitigate character limit
    print("\nPreparing", corpus_name + "...")
    # Get file names
    if corpus_name in [GUTENBERG, CHAMBERS_ROSTAND_CORPUS, ABU_CORPUS, 
                       GERMAN_BIBLE, GERMANC, LEIDEN_WEIBO_CORPUS]:
        file_names = get_files_from_directory(root_path, ".txt")
    elif corpus_name == AMAZON_REVIEWS:
        file_names = get_files_from_directory(root_path, ".json")
    elif corpus_name in [BLOG_CORPUS, LANCASTER_CORPUS]:
        file_names = get_files_from_directory(root_path, ".xml")
    elif corpus_name == BROWN_CORPUS:
        with open(os.path.join(root_path, "cats.txt"), "r") as categories_file:
            categories_text = categories_file.read()
        categories = invert_mapping(
            extract_brown_corpus_categories(categories_text))
        for key, value in categories.items():
            categories[key] = value[0]  # Remove lists
        file_names = get_files_from_directory(root_path)
    elif corpus_name in [ENGLISH_WIKI, FRENCH_WIKI, GERMAN_WIKI, CHINESE_WIKI]:
        file_names = get_files_from_directory(root_path)
        file_name = [file_name for file_name in file_names 
                     if "corpus-processed.txt" in file_name][0]
    elif corpus_name == ORAL_NARRATIVE_CORPUS:
        file_names = get_files_from_directory(root_path, "-stripped.html")
    elif corpus_name == PAROLE_CORPUS:
        file_names = get_files_from_directory(root_path, ".sgm")

    # Open, process, and store dataset
    if corpus_name in [ENGLISH_WIKI, FRENCH_WIKI, GERMAN_WIKI, CHINESE_WIKI]:
        processed_file_name = file_name[:-4] + "_further_processed.tar"
        encoding = get_encoding(os.path.join(root_path, file_name))
        wiki_text = open_file(os.path.join(root_path, file_name))
        wiki_text = UnicodeDammit(wiki_text).unicode_markup
        articles_by_name = extract_wiki_articles(wiki_text)
        with tarfile.TarFile(
                os.path.join(root_path, processed_file_name), "w") as tar_file:
            for name, article in articles_by_name.items():
                article_file_name = name.replace(" ", "_") + ".txt"
                add_text_file(tar_file, article, article_file_name)
    else:
        for file_name in file_names:
            if ".".join(file_name.split(".")[:-1]).endswith("_processed"):
                continue
            text = open_file(os.path.join(root_path, file_name))
            text = UnicodeDammit(text).unicode_markup
            processed_file_root = None
            if corpus_name == AMAZON_REVIEWS:
                reviews_by_user = extract_amazon_reviews(text)
                processed_file_name = file_name[:-5] + "_processed.tar"
            elif corpus_name == BLOG_CORPUS:
                cleaned_texts = extract_blogs(text)
                processed_file_name = (get_blog_author_id(file_name)
                                       + "_blogs_processed.txt")
            elif corpus_name == BROWN_CORPUS:
                if not re.fullmatch(r"[a-zA-Z]{2}[0-9]{2}", file_name):
                    continue
                else:
                    cleaned_text = clean_brown_corpus_text(text)
                    processed_file_name = file_name + "_processed.txt"
                    processed_file_root = os.path.join(
                        root_path, categories[file_name])
            elif corpus_name == GUTENBERG:
                cleaned_text = clean_gutenberg_text(text)
                author, title = get_author_title_gutenberg(file_name)
                processed_file_name = title[:-4] + "_processed.txt"
                processed_file_root = os.path.join(root_path, author.lower())
            elif corpus_name == CHAMBERS_ROSTAND_CORPUS:
                cleaned_text = clean_chambers_rostand_text(text)
                category = get_category_chambers_rostand(file_name)
                processed_file_name = file_name[:-4] + "_processed.txt"
                processed_file_root = os.path.join(root_path, category)
            elif corpus_name == ORAL_NARRATIVE_CORPUS:
                cleaned_text = extract_oral_narrative(text)
                storyteller = get_storyteller_oral_narrative(file_name)
                processed_file_name = file_name[:-5] + "_processed.txt"
                processed_file_root = os.path.join(root_path, storyteller)
            elif corpus_name == ABU_CORPUS:
                cleaned_text = clean_abu_corpus_text(text)
                author, genre = get_author_genre_abu_text(text)
                processed_file_name = file_name[:-4] + "_processed.txt"
                processed_file_root = os.path.join(
                    root_path, os.path.join(genre, author))
            elif corpus_name == GERMAN_BIBLE:
                cleaned_text = clean_german_bible_text(text)
                processed_file_name = file_name[:-4] + "_processed.txt"
            elif corpus_name == GERMANC:
                cleaned_text = clean_germanc_text(text)
                genre = get_genre_name_germanc(file_name)
                processed_file_name = file_name[:-4] + "_processed.txt"
                processed_file_root = os.path.join(root_path, genre)
            elif corpus_name == PAROLE_CORPUS:
                cleaned_texts = extract_parole_corpus_documents(text)
                processed_file_name = file_name[:-4] + "_processed.txt"
            elif corpus_name == LANCASTER_CORPUS:
                cleaned_texts, domain = extract_lancaster_documents(text)
                processed_file_name = file_name[:-4] + "_processed.txt"
                processed_file_root = os.path.join(root_path, domain)
            elif corpus_name == LEIDEN_WEIBO_CORPUS:
                cleaned_texts = extract_leiden_weibo_messages(text)
                processed_file_name = file_name[:-4] + "_processed.txt"
            if processed_file_root is None:
                processed_file_path = os.path.join(root_path, processed_file_name)
            else:
                try:
                    os.makedirs(processed_file_root)
                except FileExistsError:
                    pass
                processed_file_path = os.path.join(processed_file_root, 
                                                   processed_file_name)
            if corpus_name == AMAZON_REVIEWS:
                with tarfile.TarFile(os.path.join(
                        root_path, processed_file_name), "w") as tar_file:
                    for user in reviews_by_user:
                        review_string = "\n\n".join(reviews_by_user[user])
                        user_file_name = str(user) + "_reviews_processed.txt"
                        add_text_file(tar_file, review_string, user_file_name)
            else:
                try:
                    with open(processed_file_path, 
                              "w", encoding="utf-8") as processed_file:
                        if corpus_name in [BLOG_CORPUS, PAROLE_CORPUS, 
                                           LANCASTER_CORPUS, LEIDEN_WEIBO_CORPUS]:
                            processed_file.write("\n\n".join(cleaned_texts))
                        else:
                            processed_file.write(cleaned_text)
                except FileNotFoundError as e:
                    print(str(e))


def prepare_source_code(file_path):
    file_path = "\\\\?\\" + file_path  # Mitigate character limit
    source_code = open_file(file_path)
    source_code = UnicodeDammit(source_code).unicode_markup
    language = detect_language(file_path, source_code=True)
    cleaned_code = remove_multiline_comments(source_code, language)
    cleaned_code = convert_spaces_to_tabs(cleaned_code)
    processed_file_path = (".".join(file_path.split(".")[:-1]) 
                           + "_processed." + file_path.split(".")[-1])
    with open(processed_file_path, "w", encoding="utf-8") as processed_source_file:
        processed_source_file.write(cleaned_code)


def prepare_datasets(root_path):
    for directory_path, directory_names, file_names in os.walk(root_path):
        if PROGRAMMING_LANGUAGE in directory_path:
            if directory_path.split("\\")[-1] == PROGRAMMING_LANGUAGE:
                print("\nPreparing programming language datasets...")
            for file_name in file_names:
                if ".".join(file_name.split(".")[:-1]).endswith("_processed"):
                    continue
                file_path = os.path.join(directory_path, file_name)
                for programming_language in [C, PYTHON, FORTRAN, LISP]:
                    if (programming_language in directory_path 
                            and detect_language(file_path, source_code=True) 
                                    == programming_language):
                        prepare_source_code(file_path)
        elif NATURAL_LANGUAGE in directory_path:
            continue  # Temporarily skip
            if ENGLISH in directory_path:
                continue  # Temporarily skip
                for dataset in [AMAZON_REVIEWS, BLOG_CORPUS, BROWN_CORPUS,
                                ENGLISH_WIKI, GUTENBERG]:
                    if dataset == ENGLISH_WIKI:
                        indicator_index = -1
                    elif dataset == GUTENBERG:
                        indicator_index = -3
                    else:
                        indicator_index = -2
                    if directory_path.split("\\")[indicator_index] == dataset:
                        if dataset == AMAZON_REVIEWS and "Books" in directory_path:
                            continue  # Temporarily skip this
                        prepare_corpus_folder(dataset, directory_path)
            elif FRENCH in directory_path:
                for dataset in [CHAMBERS_ROSTAND_CORPUS, FRENCH_WIKI, 
                                ORAL_NARRATIVE_CORPUS, ABU_CORPUS]:
                    if dataset != ORAL_NARRATIVE_CORPUS:
                        continue  # Temporarily skip
                    if dataset == CHAMBERS_ROSTAND_CORPUS:
                        indicator_index = -5
                        if "Plain text" not in directory_path:
                            continue
                    elif dataset == ORAL_NARRATIVE_CORPUS:
                        indicator_index = -3
                        if "HTML" not in directory_path:
                            continue
                    else:
                        indicator_index = -1
                    if directory_path.split("\\")[indicator_index] == dataset:
                        prepare_corpus_folder(dataset, directory_path)
            elif GERMAN in directory_path:
                for dataset in [GERMAN_BIBLE, GERMANC, GERMAN_WIKI, PAROLE_CORPUS]:
                    if dataset == GERMAN_BIBLE:
                        indicator_index = -3
                    elif dataset == GERMANC:
                        indicator_index = -3
                        if "RAW" not in directory_path:
                            continue
                    elif dataset == GERMAN_WIKI:
                        continue  # Temporarily skip
                        indicator_index = -1
                    elif dataset == PAROLE_CORPUS:
                        indicator_index = -5
                    if directory_path.split("\\")[indicator_index] == dataset:
                        prepare_corpus_folder(dataset, directory_path)
            elif CHINESE in directory_path:
                continue  # Temporarily skip
                for dataset in [LANCASTER_CORPUS, CHINESE_WIKI, LEIDEN_WEIBO_CORPUS]:
                    if dataset == LANCASTER_CORPUS:
                        indicator_index = -5
                        if "character" not in directory_path:
                            continue
                    else:
                        indicator_index = -1
                    if directory_path.split("\\")[indicator_index] == dataset:
                        prepare_corpus_folder(dataset, directory_path)


if __name__ == '__main__':
    start_time = time()
    print("\nstart time:", start_time)
    print("\nCleaning and preparing datasets...")
    prepare_datasets(BASE_DIR)
    print("\nFinished --- duration:", time() - start_time)