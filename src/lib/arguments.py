"""
Methods for processing command line arguments
"""

import os
import argparse
from configs import get_available_configs
from lib.utils import split_path

from CONFIG import CONFIG


def create_experiment_arguments():
    """
    Processing arguments for 01_*
    """
    configs = get_available_configs()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Directory where the experiment"
                        "folder will be created", required=True, default="test_dir")
    parser.add_argument("--name", help="Name to give to the experiment")
    parser.add_argument("--config", help=f"Name of the predetermined 'config' to use: {configs}")
    args = parser.parse_args()

    args.exp_directory = process_experiment_directory_argument(args.exp_directory, create=True)
    args.config = check_config(args.config)
    return args


def get_directory_argument():
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory", required=True)
    parser.add_argument("--checkpoint", help="Checkpoint with pretrained parameters to load", default=None)
    parser.add_argument("--resume_training", help="For resuming training", default=False, action='store_true')
    args = parser.parse_args()

    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.checkpoint = process_checkpoint(exp_directory, args.checkpoint)
    return exp_directory, args


def get_eval_arguments():
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory", required=True)
    parser.add_argument("--checkpoint", help="Checkpoint with pretrained parameters to load", default=None)
    parser.add_argument("--n_samples_per_seq", help="Number of samples to take from each sequence"
                        "during evaluation", default=1, type=int)
    args = parser.parse_args()

    exp_directory = process_experiment_directory_argument(args.exp_directory)
    args.checkpoint = process_checkpoint(exp_directory, args.checkpoint)
    return exp_directory, args


def get_metrics_arguments():
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory", required=True)
    parser.add_argument("--checkpoint_name", help="Name of the checkpoint that was evaluated. For "
                        "instance: 'checkpoint_epoch_100.pth'", required=True)
    args = parser.parse_args()

    exp_directory = process_experiment_directory_argument(args.exp_directory)
    checkpoint_name = process_checkpoint_name(exp_directory, args.checkpoint_name)
    return exp_directory, checkpoint_name


def get_fig_generator_arguments():
    """
    Processing arguments for main scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--exp_directory", help="Path to the experiment directory", required=True)
    parser.add_argument("--checkpoint", help="Checkpoint with pretrained parameters to load", default=None)
    parser.add_argument("--num_imgs", help="Number of images to save", default=5, type=int)
    args = parser.parse_args()

    exp_directory = process_experiment_directory_argument(args.exp_directory)
    checkpoint = process_checkpoint(exp_directory, args.checkpoint)
    if args.num_imgs <= 0:
        raise ValueError(f"'--num_imgs' must be > 0. Given value was {args.num_imgs}...")
    return exp_directory, checkpoint, args.num_imgs


def process_experiment_directory_argument(exp_directory, create=False):
    """
    Ensuring that the experiment directory argument exists
    and giving the full path if relative was detected
    """
    was_relative = False
    exp_path = CONFIG["paths"]["experiments_path"]
    split_exp_dir = split_path(exp_directory)
    if os.path.basename(exp_path) == split_exp_dir[0]:
        exp_directory = "/".join(split_exp_dir[1:])

    if(exp_path not in exp_directory):
        was_relative = True
        exp_directory = os.path.join(exp_path, exp_directory)

    # making sure experiment directory exists
    if(not os.path.exists(exp_directory) and create is False):
        print(f"ERROR! Experiment directorty {exp_directory} does not exist...")
        print(f"     The given path was: {exp_directory}")
        if(was_relative):
            print(f"     It was a relative path. The absolute would be: {exp_directory}")
        print("\n\n")
        exit()
    elif(not os.path.exists(exp_directory) and create is True):
        os.makedirs(exp_directory)

    return exp_directory


def check_config(config):
    """
    Making sure that the predetermined configuration file, if given, exists
    """
    if config is None or len(config) < 1:
        return None
    else:
        config_path = CONFIG["paths"]["configs_path"]
        if config[-5:] != ".json":
            config = f"{config}.json"
        if config is not None and not os.path.join(config_path, config):
            raise FileNotFoundError(f"Given config. {config} does not exist in configs. path {config_path}")
        return config


def process_checkpoint(exp_path, checkpoint):
    """ Making sure checkpoint exists """
    if checkpoint is not None:
        checkpoint_path = os.path.join(exp_path, "models", checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint {checkpoint} does not exist in exp {exp_path}")
    return checkpoint


def process_checkpoint_name(exp_directory, checkpoint_name):
    """" Making sure results for the given checkpoint name exist """
    results_path = os.path.join(exp_directory, "results", f"results_{checkpoint_name}")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results path {results_path} does not exist...")
    return checkpoint_name

#
