import argparse
import pickle
import re
import string
import sys
from os import listdir
from os.path import isfile, join
from urllib.parse import urlparse

import chardet
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')


def classify_content(content, subject_model, body_model):
    subject_model_decision = subject_model.predict([" ".join(content["subject"])])
    body_model_decision = body_model.predict([" ".join(content["body"])])
    url_body_model_decision = body_model.predict([" ".join(content["urls"])])

    decision = make_spam_decision(subject_model_decision[0], body_model_decision[0], url_body_model_decision[0])

    return decision


def make_spam_decision(entry1, entry2, entry3):
    if entry1 == "spam" or entry2 == "spam" or entry3 == "spam":
        return "inf"
    return "cln"


def pre_process_data(data):
    # Subject - body split + lowering
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": x["content"].split("\n")[0][8:].lower(),
                        "body": " ".join(x["content"].split("\n")[1:]).lower()
                    }, data))

    # URLs extraction
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": x["subject"],
                        "body": x["body"],
                        "urls": find_urls(x["body"])
                    }, data))

    # Extract text from HTML if it's the case
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": x["subject"],
                        "body": get_HTML_from_body(x),
                        "urls": x["urls"]
                    }, data))

    # Punctuation removal
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": re.sub('[' + string.punctuation + ']', ' ', x["subject"]),
                        "body": re.sub('[' + string.punctuation + ']', ' ', x["body"]),
                        "urls": x["urls"]
                    }, data))

    # Tokenize
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": nltk.word_tokenize(x["subject"]),
                        "body": nltk.word_tokenize(x["body"]),
                        "urls": x["urls"]
                    }, data))
    # Lemmatization
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": [nltk.WordNetLemmatizer().lemmatize(y) for y in x["subject"]],
                        "body": [nltk.WordNetLemmatizer().lemmatize(y) for y in x["body"]],
                        "urls": x["urls"]
                    }, data))

    # Stop words removal
    stop_words = set(stopwords.words('english'))
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": [y for y in x["subject"] if y not in stop_words],
                        "body": [y for y in x["body"] if y not in stop_words],
                        "urls": x["urls"]
                    }, data))

    # URL tokenization
    data = list(map(lambda x:
                    {
                        "id": x["id"],
                        "subject": x["subject"],
                        "body": x["body"],
                        "urls": get_urls_domains(x)
                    }, data))

    return data


def get_urls_domains(x):
    try:
        return " ".join([" ".join(urlparse(y).netloc.split(".")) for y in x["urls"]]).replace("com", "").replace("www",
                                                                                                                 ""),
    except:
        print("Couldn't parse URL domain " + str(x))
        return ""


def find_urls(string):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return url


def get_HTML_from_body(x):
    try:
        return BeautifulSoup(x["body"], 'html.parser').get_text()
    except:
        print("Couldn't extract the HTML text from " + x["body"])
        return x["body"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Luca Andrei Anti-Spam filter')

    parser.add_argument('-info', required=False, help="-info information_destination_file.txt")
    parser.add_argument('-scan', required=False, nargs=2, help="-scan test_lot_path result_destination")

    parsed_args = parser.parse_args(sys.argv[1:])

    if parsed_args.info is not None:
        f = open(parsed_args.info, "w")
        f.write("Anti-Spam filter\n")
        f.write("Luca Andrei\n")
        f.write("Nu lucrez la Bitdefender\n")
        f.write("beta")

    if parsed_args.scan is not None:
        subject_model = pickle.loads(open("subject.model", 'rb').read())
        body_model = pickle.loads(open("body.model", 'rb').read())

        lot_path = parsed_args.scan[0]
        destination_path = parsed_args.scan[1]

        files_to_be_scanned = [f for f in listdir(lot_path) if isfile(join(lot_path, f))]

        classification_results = []

        to_classify = []

        for file in files_to_be_scanned:
            with open(join(lot_path, file), 'r') as file_content:
                try:
                    try:
                        content = file_content.read()

                        to_classify.append({"id": file, "content": content})

                    except:
                        rawdata = open(join(lot_path, file), 'rb').read()
                        result = chardet.detect(rawdata)
                        charenc = result['encoding']

                        content_file = open(join(lot_path, file), 'r', encoding=charenc)

                        content = content_file.read()

                        to_classify.append({"id": file, "content": content})

                except:
                    classification_results.append(file + "|inf")

        f = open(destination_path, "w")

        pre_processed_to_classify = pre_process_data(to_classify)

        for content in pre_processed_to_classify:
            classification_results.append(content["id"] + "|" + classify_content(content, subject_model, body_model))

        for classification_result in classification_results:
            f.write(classification_result + "\n")
        f.close()