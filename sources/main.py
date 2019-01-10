import os
import re
import string
from nltk.stem import WordNetLemmatizer

import nltk
from bs4 import BeautifulSoup

from nltk.corpus import stopwords

import chardet

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(clean_lot_path, spam_lot_path):
    data = {"spam": [], "clean": []}

    for filename in os.listdir(clean_lot_path):
        with open(clean_lot_path + filename, 'r') as content_file:
            try:
                try:
                    content = content_file.read()
                    data["clean"].append({"id": filename, "content": content})
                except:
                    rawdata = open(clean_lot_path + filename, 'rb').read()
                    result = chardet.detect(rawdata)
                    charenc = result['encoding']
                    content_file = open(clean_lot_path + filename, 'r', encoding=charenc)

                    content = content_file.read()
                    data["clean"].append({"id": filename, "content": content})
            except:
                print("Couldn't parse file " + filename)

    for filename in os.listdir(spam_lot_path):
        with open(spam_lot_path + filename, 'r') as content_file:
            try:
                try:
                    content = content_file.read()
                    data["spam"].append({"id": filename, "content": content})
                except:
                    rawdata = open(spam_lot_path + filename, 'rb').read()
                    result = chardet.detect(rawdata)
                    charenc = result['encoding']
                    content_file = open(spam_lot_path + filename, 'r', encoding=charenc)

                    content = content_file.read()
                    data["spam"].append({"id": filename, "content": content})
            except:
                print("Couldn't parse file " + filename)

    spam_limit_index = int(len(data["spam"]) * 0.75)
    clean_limit_index = int(len(data["clean"]) * 0.75)

    train = {"spam": data["spam"][0:spam_limit_index], "clean": data["clean"][0:clean_limit_index]}
    test = {"spam": data["spam"][spam_limit_index:], "clean": data["clean"][clean_limit_index:]}

    return train, test


def find_urls(string):
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    return url


def pre_process_data(data):
    # Subject - body split + lowering
    data["spam"] = list(map(lambda x:
                                  {
                                      "id": x["id"],
                                      "subject": x["content"].split("\n")[0][8:].lower(),
                                      "body": " ".join(x["content"].split("\n")[1:]).lower()
                                  }, data["spam"]))
    data["clean"] = list(map(lambda x:
                                   {
                                       "id": x["id"],
                                       "subject": x["content"].split("\n")[0][8:].lower(),
                                       "body": " ".join(x["content"].split("\n")[0:]).lower()
                                   }, data["clean"]))

    # URLs extraction
    data["spam"] = list(map(lambda x:
                                  {
                                      "id": x["id"],
                                      "subject": x["subject"],
                                      "body": x["body"],
                                      "urls": find_urls(x["body"])
                                  }, data["spam"]))
    data["clean"] = list(map(lambda x:
                                   {
                                       "id": x["id"],
                                       "subject": x["subject"],
                                       "body": x["body"],
                                       "urls": find_urls(x["body"])
                                   }, data["clean"]))

    # Extract text from HTML if it's the case
    data["spam"] = list(map(lambda x:
                                  {
                                      "id": x["id"],
                                      "subject": x["subject"],
                                      "body": BeautifulSoup(x["body"], 'html.parser').get_text(),
                                      "urls": x["urls"]
                                  }, data["spam"]))
    data["clean"] = list(map(lambda x:
                                   {
                                       "id": x["id"],
                                       "subject": x["subject"],
                                       "body": BeautifulSoup(x["body"], 'html.parser').get_text(),
                                       "urls": x["urls"]
                                   }, data["clean"]))

    # Punctuation removal
    data["spam"] = list(map(lambda x:
                                  {
                                      "id": x["id"],
                                      "subject": re.sub('[' + string.punctuation + ']', ' ', x["subject"]),
                                      "body": re.sub('[' + string.punctuation + ']', ' ', x["body"]),
                                      "urls": x["urls"]
                                  }, data["spam"]))
    data["clean"] = list(map(lambda x:
                                   {
                                       "id": x["id"],
                                       "subject": re.sub('[' + string.punctuation + ']', ' ', x["subject"]),
                                       "body": re.sub('[' + string.punctuation + ']', ' ', x["body"]),
                                       "urls": x["urls"]
                                   }, data["clean"]))

    # Tokenize
    data["spam"] = list(map(lambda x:
                                  {
                                      "id": x["id"],
                                      "subject": nltk.word_tokenize(x["subject"]),
                                      "body": nltk.word_tokenize(x["body"]),
                                      "urls": x["urls"]
                                  }, data["spam"]))
    data["clean"] = list(map(lambda x:
                                   {
                                       "id": x["id"],
                                       "subject": nltk.word_tokenize(x["subject"]),
                                       "body": nltk.word_tokenize(x["body"]),
                                       "urls": x["urls"]
                                   }, data["clean"]))
    # Lemmatization
    data["spam"] = list(map(lambda x:
                                  {
                                      "id": x["id"],
                                      "subject": [WordNetLemmatizer().lemmatize(y) for y in x["subject"]],
                                      "body": [WordNetLemmatizer().lemmatize(y) for y in x["body"]],
                                      "urls": x["urls"]
                                  }, data["spam"]))
    data["clean"] = list(map(lambda x:
                                   {
                                       "id": x["id"],
                                       "subject": [WordNetLemmatizer().lemmatize(y) for y in x["subject"]],
                                       "body": [WordNetLemmatizer().lemmatize(y) for y in x["body"]],
                                       "urls": x["urls"]
                                   }, data["clean"]))

    # Stop words removal
    stop_words = set(stopwords.words('english'))
    data["spam"] = list(map(lambda x:
                                  {
                                      "id": x["id"],
                                      "subject": [y for y in x["subject"] if y not in stop_words],
                                      "body": [y for y in x["body"] if y not in stop_words],
                                      "urls": x["urls"]
                                  }, data["spam"]))
    data["clean"] = list(map(lambda x:
                                   {
                                       "id": x["id"],
                                       "subject": [y for y in x["subject"] if y not in stop_words],
                                       "body": [y for y in x["body"] if y not in stop_words],
                                       "urls": x["urls"]
                                   }, data["clean"]))
    return data


def train_model(train_data):
    pre_processed_train_data = pre_process_data(train_data)

    return pre_processed_train_data

    # for key, value in train_data.items():


if __name__ == '__main__':
    train_data, test_data = load_data("../data/Lot1-truncated/Clean/", "../data/Lot1-truncated/Spam/")

    model = train_model(train_data)
