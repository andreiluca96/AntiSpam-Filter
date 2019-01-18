import argparse
import sys
from os import listdir
from os.path import isfile, join

import chardet


def classify_content(content):
    return "inf"


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
        lot_path = parsed_args.scan[0]
        destination_path = parsed_args.scan[1]

        files_to_be_scanned = [f for f in listdir(lot_path) if isfile(join(lot_path, f))]


        classification_results = []

        for file in files_to_be_scanned:
            with open(join(lot_path, file), 'r') as file_content:
                try:
                    try:
                        content = file_content.read()

                        classification_results.append(file + "|" + classify_content(content))

                    except:
                        rawdata = open(join(lot_path, file), 'rb').read()
                        result = chardet.detect(rawdata)
                        charenc = result['encoding']

                        content_file = open(join(lot_path, file), 'r', encoding=charenc)

                        content = content_file.read()
                        classification_results.append(file + "|" + classify_content(content))

                except:
                    print("Couldn't parse file " + file)

        f = open(destination_path, "w")

        for classification_result in classification_results:
            f.write(classification_result + "\n")