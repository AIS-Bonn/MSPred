"""
Computing processed metrics given results file.
For instance:
  - PCK@ --> AP,
  - PDJ@ --> AR,
  - Classwise IoU --> Moving, Static, Background
"""

import os
import json
import numpy as np

from lib.arguments import get_metrics_arguments
import lib.utils as utils

MOVING_ID = 22
STATIC_ID = range(1, 22)
BACKGROUND_ID = 0


def main(exp_dir, checkpoint_name):
    """ Main logic for processing results"""
    results_path = os.path.join(exp_dir, "results", f"results_{checkpoint_name}")

    results_files = sorted(os.listdir(results_path))
    results_files = [f for f in results_files if f[-5:] == ".json" and "processed_" not in f]

    for file in results_files:
        print(f"Processing file {file}...")
        fpath = os.path.join(results_path, file)
        with open(fpath, "r") as f:
            data = json.load(f)
        processed_data = process_data(data)

        outfile = os.path.join(results_path, f"processed_{file}")
        with open(outfile, "w") as f:
            json.dump(processed_data, f)
    return


def process_data(data):
    """ Processing results data and returning it as dictionary """
    processed_data = {}
    all_keys = list(data.keys())

    # processing classwise segmentation into moving, static and background
    for key in all_keys:
        if "classwise" in key:
            new_data = process_segmentation(data[key], key)
            processed_data = {**processed_data, **new_data}
        elif "framewise" in key:
            continue
        elif "pck" in key or "pdj" in key:
            factor = key.split("@")[-1]
            if factor == "0.2":
                processed_data[key] = data[key]
        else:
            processed_data[key] = data[key]

    # computing AR and AP
    pdjs = {}
    pcks = {}
    for key in all_keys:
        if "mean_pdj" in key:
            factor = key.split("@")[-1]
            pdjs[factor] = data[key]
        if "mean_pck" in key:
            factor = key.split("@")[-1]
            pcks[factor] = data[key]

    if len(pdjs) > 0:
        key = "AR" + ":".join(list(pdjs.keys()))
        processed_data[key] = np.mean(list(pdjs.values()))
    if len(pcks) > 0:
        key = "AP" + ":".join(list(pcks.keys()))
        processed_data[key] = np.mean(list(pcks.values()))

    return processed_data


def process_segmentation(data, metric):
    """ Aggregating segmentation results into moving, static, and background """
    processed_data = {}
    processed_data[f"{metric}_moving"] = data[MOVING_ID]
    processed_data[f"{metric}_background"] = data[BACKGROUND_ID]
    processed_data[f"{metric}_static"] = np.mean([data[i] for i in STATIC_ID])
    return processed_data


if __name__ == '__main__':
    utils.clear_cmd()
    exp_dir, checkpoint_name = get_metrics_arguments()
    main(exp_dir=exp_dir, checkpoint_name=checkpoint_name)
    print("Done")
