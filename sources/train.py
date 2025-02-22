import os
import pickle
import re
import string
from urllib.parse import urlparse

import chardet
import nltk
import numpy
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')


def load_data(clean_lot_path_list, spam_lot_path_list):
    data = {"spam": [], "clean": []}

    for clean_lot_path in clean_lot_path_list:
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
                    print("Couldn't parse CLEAN file " + filename)

    for spam_lot_path in spam_lot_path_list:
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
                    print("Couldn't parse SPAM file " + filename)

    spam_limit_index = int(len(data["spam"]) * 0.9)
    clean_limit_index = int(len(data["clean"]) * 0.9)

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
                                "body": get_HTML_from_body(x),
                                "urls": x["urls"]
                            }, data["spam"]))
    data["clean"] = list(map(lambda x:
                             {
                                 "id": x["id"],
                                 "subject": x["subject"],
                                 "body": get_HTML_from_body(x),
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

    # URL tokenization
    data["spam"] = list(map(lambda x:
                            {
                                "id": x["id"],
                                "subject": x["subject"],
                                "body": x["body"],
                                "urls": get_urls_domains(x)
                            }, data["spam"]))
    data["clean"] = list(map(lambda x:
                             {
                                 "id": x["id"],
                                 "subject": x["subject"],
                                 "body": x["body"],
                                 "urls": get_urls_domains(x)
                             }, data["clean"]))

    return data


def get_urls_domains(x):
    try:
        return " ".join([" ".join(urlparse(y).netloc.split(".")) for y in x["urls"]]).replace("com", "").replace("www",
                                                                                                                 ""),
    except:
        print("Couldn't parse URL domain " + str(x))
        return ""


def get_HTML_from_body(x):
    try:
        return BeautifulSoup(x["body"], 'html.parser').get_text()
    except:
        print("Couldn't extract the HTML text from " + x["body"])
        return x["body"]


def get_URL_tokens(input):
    tokensBySlash = str(input.encode('utf-8')).split('/')  # get tokens after splitting by slash
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split('-')  # get tokens after splitting by dash
        tokensByDot = []
        for j in range(0, len(tokens)):
            tempTokens = str(tokens[j]).split('.')  # get tokens after splitting by dot
            tokensByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))  # remove redundant tokens
    if 'com' in allTokens:
        allTokens.remove('com')
    if 'http' in allTokens:
        allTokens.remove('http')
    if 'www' in allTokens:
        allTokens.remove('www')
    return allTokens


def train_model(train_data):
    pre_processed_train_data = pre_process_data(train_data)

    # Body and subject
    subject_X = ["", ""]
    body_X = ["", ""]
    url_X = ["", ""]

    for entry in pre_processed_train_data["clean"]:
        subject_X[0] += " ".join(entry["subject"])
        body_X[0] += " ".join(entry["body"])
        url_X[0] += " ".join(entry["urls"])

    for entry in pre_processed_train_data["spam"]:
        subject_X[1] += " ".join(entry["subject"])
        body_X[1] += " ".join(entry["body"])
        url_X[1] += " ".join(entry["urls"])

    Y = ["clean", "spam"]

    subject_model = Pipeline([
        ('vect', CountVectorizer(stop_words='english', lowercase=True)),  # pre_processing step
        ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),  # pre_processing step
        ('clf', MultinomialNB(alpha=1))  # prediction model
    ])

    body_model = Pipeline([
        ('vect', CountVectorizer(stop_words='english', lowercase=True)),  # pre_processing step
        ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),  # pre_processing step
        ('clf', MultinomialNB(alpha=1))  # prediction model
    ])

    url_model = Pipeline([
        ('vect', CountVectorizer(stop_words='english', lowercase=True)),  # pre_processing step
        ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),  # pre_processing step
        ('clf', MultinomialNB(alpha=1))  # prediction model
    ])

    subject_model.fit(subject_X, Y)
    body_model.fit(body_X, Y)
    url_model.fit(url_X, Y)

    return subject_model, body_model, url_model


def test_model(test_data, subject_model, body_model, url_model):
    pre_processed_test_data = pre_process_data(test_data)

    '''
    Subject Classifier
    '''
    clean_subjects = [" ".join(entry["subject"]) for entry in pre_processed_test_data["clean"]]
    spam_subjects = [" ".join(entry["subject"]) for entry in pre_processed_test_data["spam"]]

    predicted_clean_subjects = subject_model.predict(clean_subjects)
    unique, counts = numpy.unique(predicted_clean_subjects, return_counts=True)
    predicted_clean_subjects_counts = dict(zip(unique, counts))

    predicted_spam_subjects = subject_model.predict(spam_subjects)
    unique, counts = numpy.unique(predicted_spam_subjects, return_counts=True)
    predicted_spam_subjects_counts = dict(zip(unique, counts))

    generate_report(pre_processed_test_data, predicted_clean_subjects_counts, predicted_spam_subjects_counts, "subject")

    '''
    Body Classifier
    '''
    clean_bodies = [" ".join(entry["body"]) for entry in pre_processed_test_data["clean"]]
    spam_bodies = [" ".join(entry["body"]) for entry in pre_processed_test_data["spam"]]

    predicted_clean_bodies = body_model.predict(clean_bodies)
    unique, counts = numpy.unique(predicted_clean_bodies, return_counts=True)
    predicted_clean_bodies_counts = dict(zip(unique, counts))

    predicted_spam_bodies = body_model.predict(spam_bodies)
    unique, counts = numpy.unique(predicted_spam_bodies, return_counts=True)
    predicted_spam_bodies_counts = dict(zip(unique, counts))

    generate_report(pre_processed_test_data, predicted_clean_bodies_counts, predicted_spam_bodies_counts, "body")

    '''
    URL Classifier
    '''
    clean_urls = [" ".join(entry["urls"]) for entry in pre_processed_test_data["clean"]]
    spam_urls = [" ".join(entry["urls"]) for entry in pre_processed_test_data["spam"]]

    predicted_clean_urls = body_model.predict(clean_urls)
    unique, counts = numpy.unique(predicted_clean_urls, return_counts=True)
    predicted_clean_urls_counts = dict(zip(unique, counts))

    predicted_spam_urls = body_model.predict(spam_urls)
    unique, counts = numpy.unique(predicted_spam_urls, return_counts=True)
    predicted_spam_urls_counts = dict(zip(unique, counts))

    generate_report(pre_processed_test_data, predicted_clean_urls_counts, predicted_spam_urls_counts, "urls")

    '''
    Combined Classifier
    '''
    zipped_predicted_clean = list(zip(predicted_clean_subjects, predicted_clean_bodies, predicted_clean_urls))
    zipped_predicted_spam = list(zip(predicted_spam_subjects, predicted_spam_bodies, predicted_spam_urls))

    predicted_clean_combined = [make_spam_decision(entry1, entry2, entry3) for entry1, entry2, entry3 in
                                zipped_predicted_clean]
    unique, counts = numpy.unique(predicted_clean_combined, return_counts=True)
    predicted_clean_combined_counts = dict(zip(unique, counts))

    predicted_spam_combined = [make_spam_decision(entry1, entry2, entry3) for entry1, entry2, entry3 in
                               zipped_predicted_spam]
    unique, counts = numpy.unique(predicted_spam_combined, return_counts=True)
    predicted_spam_combined_counts = dict(zip(unique, counts))

    generate_report(pre_processed_test_data, predicted_clean_combined_counts, predicted_spam_combined_counts,
                    "combined")


def make_spam_decision(entry1, entry2, entry3):
    if entry1 == "spam" or entry2 == "spam" or entry3 == "spam":
        return "spam"
    return "clean"


def generate_report(pre_processed_test_data, predicted_clean_bodies_counts, predicted_spam_bodies_counts, type):
    # true positives
    try:
        TP = predicted_spam_bodies_counts["spam"]
    except:
        TP = 0
    # false positives
    try:
        FP = predicted_clean_bodies_counts["spam"]
    except:
        FP = 0
    # true negatives
    try:
        TN = predicted_clean_bodies_counts["clean"]
    except:
        TN = 0
    # false negatives
    try:
        FN = predicted_spam_bodies_counts["clean"]
    except:
        FN = 0

    # Scoring
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F_2_score = 2 * (precision * recall) / (precision + recall)

    # Report generation:
    print("====== Report based on the " + type + " classification model ======")
    print("Spams: " + str(len(pre_processed_test_data["spam"])))
    print("Clean: " + str(len(pre_processed_test_data["clean"])))
    print("-----------------")
    print("True positives: " + str(TP))
    print("False positives: " + str(FP))
    print("True negatives: " + str(TN))
    print("False negatives: " + str(FN))
    print("-----------------")
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-score: " + str(F_2_score))


if __name__ == '__main__':
    train_data, test_data = load_data(["../data/Lot2/Clean/", "../data/Lot1/Clean/"],
                                      ["../data/Lot2/Spam/", "../data/Lot1/Spam/"])

    # train_data, test_data = load_data(["../data/Lot1-truncated/Clean/"],
    #                                   ["../data/Lot1-truncated/Spam/"])

    subject_model, body_model, url_model = train_model(train_data)

    test_model(test_data, subject_model, body_model, url_model)

    subject_model_serialized = pickle.dumps(subject_model)
    body_model_serialized = pickle.dumps(body_model)

    subject_file = open("subject.model", "wb")
    subject_file.write(subject_model_serialized)
    subject_file.close()

    body_file = open("body.model", "wb")
    body_file.write(body_model_serialized)
    body_file.close()

    print("Finished the model writing...")

