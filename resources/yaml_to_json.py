"""
Converting a YAML to JSON
"""

import os
import argparse
import yaml
import json


def parse_args():
    """ Passing file as an argument """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="File to convert", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        raise FileNotFoundError(f"Given file {args.file} does not exist...")
    extension = args.file.split(".")
    if len(extension) < 2:
        raise NameError(f"File {extension} has not extension in ['json', 'yaml', 'yml']")
    else:
        extension = extension[0]
    if extension not in ["json", "yaml", "yml"]:
        raise NameError(f"Extension of the given file {extension} not in ['json', 'yaml', 'yml']")
    return args.file


def press_yes_to_continue(message, key="y"):
    """ Asking the user for input to continut """
    if isinstance(message, (list, tuple)):
        for m in message:
            print(m)
    else:
        print(message)
    val = input(f"Press '{key}' to continue...")
    if(val != key):
        print("Exiting...")
        exit()
    return


def main(file):
    """ Main logic for conversion """
    extension = file.split(".")[-1]

    # converting json to yaml
    if extension in ["json"]:
        message = f"You are about to convert file {file} into YAML..."
        press_yes_to_continue(message=message)
        with open(file, 'r') as f:
            data = json.load(f)
        out_file = ".".join(file.split(".")[:-1]) + ".yaml"
        with open(out_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    # converting yaml to json
    elif extension in ["yaml", "yml"]:
        message = f"You are about to convert file {file} into json..."
        press_yes_to_continue(message=message)
        with open(file, 'r') as f:
            data = yaml.safe_load(f)
        out_file = ".".join(file.split(".")[:-1]) + ".json"
        with open(out_file, 'w') as f:
            json.dump(data, f)
    else:
        raise NameError(f"Extension of the given file {extension} not in ['json', 'yaml', 'yml']")

    print("Conversion finished successfully...")
    return


if __name__ == "__main__":
    file = parse_args()
    main(file=file)


#
