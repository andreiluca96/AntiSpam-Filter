import os

import chardet


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


if __name__ == '__main__':
    train_data, test_data = load_data("../data/Lot1/Clean/", "../data/Lot1/Spam/")
