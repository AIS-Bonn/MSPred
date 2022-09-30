"""
Global configurations
"""

import os

# High level configurations, such as paths or random seed
CONFIG = {
    "random_seed": 13,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "paths": {
        "data_path": os.path.join(os.getcwd(), "..", "datasets"),
        "experiments_path": os.path.join(os.getcwd(), "experiments"),
        "resources_path": os.path.join(os.getcwd(), "resources"),
        "configs_path": os.path.join(os.getcwd(), "src", "configs"),
    }
}


# Supported datasets
DATASETS = ["moving_mnist", "custom_moving_mnist", "kth", "synpick"]

# Supported models and neural network modules
MODELS = ["deterministic", "ConvDeterministic", "LP", "ConvLP",
          "SpatioTempHierarchLSTM", "SpatioTempHierarchConvLSTM"]
ENCODERS = ["DCGAN", "VGG"]

# Supported losses and metrics
LOSSES = ["mse", "mae", "cross_entropy", "weighted_kpoint_loss"]
METRICS = [
        "mse", "mae", "psnr", "ssim", "lpips",
        "mpjpe", "pdj", "pck",
        "segmentation_accuracy", "iou"
    ]
METRIC_SETS = {
    "video_prediction": ["mse", "mae", "psnr", "ssim", "lpips"],
    "keypoint": ["mpjpe", "pdj@0.1", "pdj@0.2", "pdj@0.3", "pdj@0.4", "pdj@0.5",
                 "pck@0.1", "pck@0.2", "pck@0.3", "pck@0.4", "pck@0.5"],
    "segmentation": ["segmentation_accuracy", "iou"],
    "blob": ["mse", "mae"],
    "single_keypoint_metric": ["mpjpe"]
}


# Specific configurations and default values
DEFAULTS = {
    "dataset": {
        "dataset_name": "custom_moving_mnist",
        "img_size": 64,
        "num_channels": 3
    },
    "model": {  # model parameters
        "model_name": "SpatioTempHierarchConvLSTM",
        "enc_dec_type": "DCGAN",  # VGG
        "enc_dec": {
            "dim": 512,
            "num_filters": 64,
            "extra_deep": False
        },
        "LSTM": {
            "num_layers": 2,
            "hidden_dim": 256,
         },
        "LSTM_Prior": {
            "latent_dim": 10,
            "num_layers": 1,
            "hidden_dim": 64
        },
        "LSTM_Posterior": {
            "latent_dim": 10,
            "num_layers": 1,
            "hidden_dim": 64
        },
        "HierarchLSTM": {
            "num_layers": 4,
            "num_hierarch": 3,
            "periods": [1, 4, 8],
            "hidden_dim": 128,
            "aux_outputs": True,
            "stochastic": True,
            "ancestral_sampling": True
        },
        "autoreg_mode": "LAST_PRED_FEATS",  # "LAST_PRED_FRAME"
        "last_context_residuals": True
    },
    "loss": {
        "reconst_losses": ['mse', 'mse', 'mse'],  # reconstruction loss types (per level)
        "w_easy_kpts": 0.7,                       # weight of 'easy' keypoints in the weighted_kpoint_loss
        "beta": 1e-4,               # (final) weight of KL-loss term
        "beta0": 1e-4,              # initial weight of KL-loss term
        "beta_warmup_steps": 0,     # number of warmup epochs
        "alphas": [1.0, 0.5, 0.5],  # weight list for different level mse-loss terms (in case of aux_outputs)
    },
    "training": {  # training related parameters
        "num_iters": 500,       # number of optimization steps per training epoch (n mini-batches drawn)
        "num_epochs": 200,      # number of epochs to train for
        "save_frequency": 30,   # saving a checkpoint every x epochs
        "log_frequency": 250,   # logging stats after this amount of updates
        "batch_size": 8,
        "lr": 1e-3,
        "lr_factor": 0.1,       # factor by which lr is decayed in a scheduler
        "lr_warmup_steps": 10,     # number of epochs to warm-up the lr for
        "lr_decay_steps": [50, 100],  # number of epochs after which lr is decayed by lr_factor
        "momentum": 0,
        "nesterov": False,      # use Nesterov with SGD or not
        "optimizer": "adam",
        "scheduler": "warmup",
        "tf_epochs": 0,        # number of first epochs to use teacher-forcing
        "context": 5,          # number of past frames to condition on
        "num_preds": 5       # number of to be predicted frames
    },
    "eval": {
        "batch_size": 8,
        "context": 5,           # number of past frames to condition on at test-time
        "num_preds": 5,       # number of to be predicted frames at test-time
        "openloop_gen": False
    }
}
