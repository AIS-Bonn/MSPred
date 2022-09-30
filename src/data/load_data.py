"""
Methods for loading data and fitting data_loaders
"""

from data import MovingMNIST, CustomMovingMNIST, KTH, SynpickMoving
from torch.utils.data import DataLoader

from CONFIG import DATASETS


def load_data(exp_params, split="train", transform=None):
    """
    Loading a dataset given the parameters

    Args:
    -----
    exp_params: dict
        Experiment parameters
    split: string
        dataset split to load
    """
    phase = "training" if split != "test" else "eval"
    n_frames = exp_params[phase]["context"] + exp_params[phase]["num_preds"]
    dataset_name = exp_params["dataset"]["dataset_name"]
    num_channels = exp_params["dataset"]["num_channels"]
    img_size = exp_params["dataset"]["img_size"]

    if dataset_name not in DATASETS:
        raise NotImplementedError(f"ERROR! '{dataset_name}' not in supported datasets {DATASETS}")

    if (dataset_name == "moving_mnist"):
        dataset = MovingMNIST(
                split=split,
                num_frames=n_frames,
                num_channels=num_channels,
                img_size=img_size
            )
    elif (dataset_name == "custom_moving_mnist"):
        dataset = CustomMovingMNIST(
                split=split,
                num_frames=n_frames,
                num_channels=num_channels,
                img_size=img_size
            )
    elif (dataset_name == "kth"):
        dataset = KTH(
                split=split,
                num_frames=n_frames,
                num_channels=num_channels,
                img_size=img_size
            )
    elif (dataset_name == "synpick"):
        dataset = SynpickMoving(
                split=split,
                num_frames=n_frames,
                img_size=img_size
            )
    return dataset


def build_data_loader(dataset, batch_size=8, shuffle=False):
    """
    Fitting a data loader for the given dataset

    Args:
    -----
    dataset: torch dataset
        Dataset (or dataset split) to fit to the DataLoader
    batch_size: integer
        number of elements per mini-batch
    shuffle: boolean
        If True, mini-batches are sampled randomly from the database
    """
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
