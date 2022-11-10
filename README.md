# Multi-Scale Hierarchical Video Prediction


Official implementation of: **"Video Prediction at Multiple Spatio-Temporal Scales with	Hierarchical Recurrent Networks"** by Villar-Corrales et al. [[Paper]](http://www.angelvillarcorrales.com/templates/others/Publications/MSPred_BMVC_2022.pdf)  [[Project]](https://sites.google.com/view/mspred/home)




## Contents

 * [1. Requirements](#requirements)
 * [2. Directory Structure](#directory-structure)
 * [3. Models](#models)
 * [4. Quick Guide](#quick-guide)
 * [5. Contact and Citation](#contact-and-citation)


## Requirements

You can easily install all required packages and avoiding dependency issues by installing the ```conda``` environment file included in the repository. To do so, run the following command from the terminal:

```shell
$ conda env create -f environment.yml
$ conda activate MSPred
```

To obtain the datasets, follow the steps:

 - **MovingMNIST:** This dataset is generated on the fly using images from the popular MNIST dataset. The original MNIST is automatically downloaded using `torchvision`.

 - **KTH-Actions:** You can download and preprocess this dataset using the bash scripts in the `resources` directory:
```shell
 cd resources
 ./download_kth.sh $DATA_PATH
 ./convert_kth.sh $DATA_PATH
```
In case the download scripts are not working, please download the KTH-Actions dataset from [this shared directory](https://www.dropbox.com/sh/byp2c5s1q9d4uud/AACRJLBZTjc1c3kMS5IKDMa6a?dl=0)

 - **SynpickVP:** You can download this dataset from [this shared directory](https://www.dropbox.com/sh/byp2c5s1q9d4uud/AACRJLBZTjc1c3kMS5IKDMa6a?dl=0)


## Directory Structure

The following tree diagram displays the detailed directory structure of the project. Directory names and paths can be modified in the CONFIG File (`/src/CONFIG.py`).

```
MSPred
├── datasets/
|   ├── KTH-64/
|   ├── SYNPICK/
|   └── ...
├── experiments/
|   ├── exp_mmnist/
|   └── exp_kth/
├── resources/
|   └── ...
├── src/
|   └── ...
├── environment.yml
└── README.md
```


## Models

The following table contains links to pretrained MSPred models on the MovingMNIST, KTH-Actions and SynpickVP datasets. These models can be used to reproduce the results from our paper.

Additionally, we include links to training logs and Tensorboards, which show the training and validation progress of our model.

| Dataset  | Model & Config | Logs |
| ------------- | ------------- | ------------- |
| Moving MNIST | [Experiment](https://www.dropbox.com/sh/6euwvurae6y5j47/AACHnAYpQt8smKIKpwRqaz3Ra?dl=0) | [tboard](https://tensorboard.dev/experiment/h7YfwWyVSZy5N9nBOceyUA/#scalars) |
| KTH-Actions | [Experiment](https://www.dropbox.com/sh/qho15xgljhvzcta/AAAINQIv9hnp9TqJsS2zM57da?dl=0) | [tboard](https://tensorboard.dev/experiment/5aOLm5J4TgWnkZHzn0vNRw/#scalars) |
| SynpickVP | [Experiment](https://www.dropbox.com/sh/80yq2off5f0o25f/AADpQbu731yv8A94LiYQWRkpa?dl=0)  | [tboard](https://tensorboard.dev/experiment/M9GfRex7Qs6DYc3XeV06iA/#scalars) |


## Quick Guide

Follow this section for a quick guide on how to get started with this repository.

### Creating an Experiment

Our codebase is based is structured in an experiment-oriented manner.
Experiments contains an *experiment_params.json* with all the hyper-parameters and specifications needed to train and evaluate a model.

Creating an experiment automatically generates a directory in the specified EXP_DIRECTORY, containing a *JSON* file with the experiment parameters and sub-directories for the models, plot, and Tensorboard logs.


```shell
$ python src/01_create_experiment.py [-h] -d EXP_DIRECTORY [--name NAME] [--config CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Directory where the experiment folder will be created
  --name NAME           Name to give to the experiment
  --config CONFIG       Name of the predetermined 'config' to use
```


### Training and Evaluating a Model

Once the experiment is initialized and the experiment parameters are set to the desired values, a model can be trained following command:

```shell
$ CUDA_VISIBLE_DEVICES=0 python src/02_train.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT] [--resume_training]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Path to the experiment directory
  --checkpoint CHECKPOINT Checkpoint with pretrained parameters to load
  --resume_training     For resuming training

```

Model checkpoints, which are saved regularly during training, can be evaluated using the following command:

```shell
$ CUDA_VISIBLE_DEVICES=0 python src/3_evaluate.py [-h] -d EXP_DIRECTORY [--checkpoint CHECKPOINT]

optional arguments:
  -h, --help            show this help message and exit
  -d EXP_DIRECTORY, --exp_directory EXP_DIRECTORY Path to the experiment directory
  --checkpoint CHECKPOINT Checkpoint with pretrained parameters to load
```


##### Example


```shell
$ python src/01_create_experiment.py -d TestExp --name exp_mmnist --config mmnist
$ CUDA_VISIBLE_DEVICES=0 python src/02_train.py -d experiments/TestExp/exp_mmnist/
$ CUDA_VISIBLE_DEVICES=0 python src/03_evaluate.py -d experiments/TestExp/exp_mmnist/ --checkpoint checkpoint_epoch_final.pth

```


## Contact and Citation

This repository is maintained by [Angel Villar-Corrales](http://angelvillarcorrales.com/templates/home.php),

Please consider citing our paper if you find our work or our repository helpful.

```
@inproceedings{villar2022MSPred,
  title={MSPred: Video Prediction at Multiple Spatio-Temporal Scales with Hierarchical Recurrent Networks},
  author={Villar-Corrales, Angel and Karapetyan, Ani and Boltres, Andreas and Behnke, Sven},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2022}
}
```

In case of any questions or problems regarding the project or repository, do not hesitate to contact the authors at villar@ais.uni-bonn.de.
