"""
Setting up the model, optimizers, loss functions, loading/saving parameters, ...
"""

import os
import traceback
import torch

from lib.logger import log_function, print_
from lib.schedulers import WarmupStepLR, ExponentialLRSchedule
from lib.utils import create_directory
import models
from CONFIG import MODELS


@log_function
def setup_model(exp_params):
    """
    Loading the model given the model parameters stated in the exp_params file

    Args:
    -----
    model_params: dictionary
        model parameters sub-dictionary from the experiment parameters

    Returns:
    --------
    model: torch.nn.Module
        instanciated model given the parameters
    """
    model_params = exp_params["model"]
    model_params["num_channels"] = exp_params["dataset"]["num_channels"]
    img_size = exp_params["dataset"]["img_size"]
    if isinstance(img_size, int):
        img_size = (img_size, img_size)
    model_params["img_size"] = img_size
    model_name = model_params["model_type"]
    stochastic = model_params["HierarchLSTM"]["stochastic"]

    # Loading model specific parameters and specific modules
    if (model_name == "deterministic"):
        model = models.SVG_DET(model_params=model_params, linear=True)
    elif (model_name == "ConvDeterministic"):
        model = models.SVG_DET(model_params=model_params, linear=False)
    elif (model_name == "LP"):
        model = models.SVG_LP(model_params=model_params, linear=True)
    elif (model_name == "ConvLP"):
        model = models.SVG_LP(model_params=model_params, linear=False)
    elif (model_name == "SpatioTempHierarchLSTM"):
        model = models.SpatioTempHierarch(model_params=model_params, linear=True, stochastic=stochastic)
    elif (model_name == "SpatioTempHierarchConvLSTM"):
        model = models.SpatioTempHierarch(model_params=model_params, linear=False, stochastic=stochastic)
    else:
        raise ValueError(f"""Model '{model_name}' not recognized. Use one of the following: {MODELS}...""")

    # parameter intialization
    try:
        model.apply(models.init_weights)
        print_("Model parameters initialized correctly")
    except Exception:
        print_(f"Error during initialization of model of type {model_name}")
        pass

    return model


def emergency_save(f):
    """
    Decorator for saving a model in case of exception, either from code or triggered.
    Use for decorating the training loop:
        @setup_model.emergency_save
        def train_loop(self):
    """

    def try_call_except(*args, **kwargs):
        """ Wrapping function and saving checkpoint in case of exception """
        try:
            return f(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            print_("There has been an exception. Saving emergency checkpoint...")
            self_ = args[0]
            if hasattr(self_, "model") and hasattr(self_, "optimizer"):
                fname = f"emergency_checkpoint_epoch_{self_.epoch}.pth"
                save_checkpoint(
                    model=self_.model,
                    optimizer=self_.optimizer,
                    scheduler=self_.scheduler,
                    epoch=self_.epoch,
                    exp_path=self_.exp_path,
                    savedir="models",
                    savename=fname
                )
                print_(f"  --> Saved emergency checkpoint {fname}")
            message = traceback.format_exc()
            print_(message, message_type="error")
            exit()

    return try_call_except


@log_function
def save_checkpoint(model, optimizer, scheduler, epoch, exp_path, finished=False,
                    savedir="models", savename=None):
    """
    Saving a checkpoint in the models directory of the experiment. This checkpoint
    contains state_dicts for the mode, optimizer and lr_scheduler
    Args:
    -----
    model: torch Module
        model to be saved to a .pth file
    optimizer, scheduler: torch Optim
        modules corresponding to the parameter optimizer and lr-scheduler
    epoch: integer
        current epoch number
    exp_path: string
        path to the root directory of the experiment
    finished: boolean
        if True, current checkpoint corresponds to the finally trained model
    """

    if (savename is not None):
        checkpoint_name = savename
    elif (savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    create_directory(exp_path, savedir)
    savepath = os.path.join(exp_path, savedir, checkpoint_name)

    scheduler_data = "" if scheduler is None else scheduler.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data
            }, savepath)
    return


@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_location="cpu", **kwargs):
    """
    Loading a precomputed checkpoint: state_dicts for the mode, optimizer and lr_scheduler

    Args:
    -----
    checkpoint_path: string
        path to the .pth file containing the state dicts
    model: torch Module
        model for which the parameters are loaded
    only_model: boolean
        if True, only model state dictionary is loaded
    """

    if (checkpoint_path is None):
        return model

    checkpoint = torch.load(checkpoint_path,  map_location=map_location)
    # loading model parameters. Try catch is used to allow different dicts
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except IOError:
        model.load_state_dict(checkpoint)

    # returning only the model for transfer learning or returning also optimizer state
    if only_model:
        return model

    optimizer, scheduler = kwargs.get("optimizer", None), kwargs.get("scheduler", None)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint["epoch"]

    return model, optimizer, scheduler, epoch


@log_function
def setup_optimization(exp_params, model):
    """
    Initializing the optimizer object used to update the model parameters

    Args:
    -----
    exp_params: dictionary
        parameters corresponding to the different experiment
    model: nn.Module
        instanciated neural network model

    Returns:
    --------
    optimizer: Torch Optim object
        Initialized optimizer
    scheduler: Torch Optim object
        learning rate scheduler object used to decrease the lr after some epochs
    """

    # setting up optimizer and LR-scheduler
    optimizer = setup_optimizer(parameters=model.parameters(), exp_params=exp_params)
    scheduler = setup_scheduler(exp_params=exp_params, optimizer=optimizer)

    return optimizer, scheduler


def setup_optimizer(parameters, exp_params):
    """ Instanciating a new optimizer """
    lr = exp_params["training"]["lr"]
    momentum = exp_params["training"]["momentum"]
    optimizer = exp_params["training"]["optimizer"]
    nesterov = exp_params["training"]["nesterov"]

    # SGD-based optimizer
    if(optimizer == "adam"):
        print_("Setting up Adam optimizer:")
        print_(f"  --> LR: {lr}")
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif(optimizer == "adamw"):
        optimizer = torch.optim.AdamW(parameters, lr=lr)
        print_("Setting up AdamW optimizer:")
        print_(f"  --> LR: {lr}")
    else:
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=0.0005)
        print_("Setting up SGD optimizer:")
        print_(f"  --> LR: {lr}")
        print_(f"  --> Momentum: {momentum}")
        print_(f"  --> Nesterov: {nesterov}")
        print_("  --> Decay: 0.0005")

    return optimizer


def setup_scheduler(exp_params, optimizer):
    """ Instanciating a new scheduler """
    lr = exp_params["training"]["lr"]
    lr_factor = exp_params["training"]["lr_factor"]
    lr_warmup_steps = exp_params["training"]["lr_warmup_steps"]
    patience = exp_params["training"]["patience"]
    scheduler = exp_params["training"]["scheduler"]
    decay_steps = exp_params["training"]["lr_decay_steps"]

    if (scheduler == "multistep"):
        print_("Setting up Plateau LR-Scheduler:")
        print_(f"  --> Steps:   {decay_steps}")
        print_(f"  --> Factor:  {lr_factor}")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=decay_steps,
                gamma=lr_factor
            )
    elif (scheduler == "warmup"):
        print_("Setting up MultiStep/Warmup LR-Scheduler:")
        print_(f"  --> Warmup Steps:  {lr_warmup_steps}")
        print_(f"  --> Decay Steps:   {decay_steps}")
        print_(f"  --> Factor:        {lr_factor}")
        scheduler = WarmupStepLR(
                optimizer=optimizer,
                init_lr=lr,
                warmup_steps=lr_warmup_steps,
                decay_steps=decay_steps,
                gamma=lr_factor
            )
    elif(scheduler == "plateau"):
        print_("Setting up Plateau LR-Scheduler:")
        print_(f"  --> Patience: {patience}")
        print_(f"  --> Factor:   {lr_factor}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=patience,
                factor=lr_factor,
                min_lr=1e-8,
                mode="min",
                verbose=True
            )
    elif(scheduler == "step"):
        print_("Setting up Step LR-Scheduler")
        print_(f"  --> Step Size: {patience}")
        print_(f"  --> Factor:    {lr_factor}")
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                step_size=patience
            )
    elif(scheduler == "exponential"):
        print_("Setting up Exponential LR-Scheduler")
        print_(f"  --> Init LR: {lr}")
        print_(f"  --> Factor:  {lr_factor}")
        scheduler = ExponentialLRSchedule(
                optimizer=optimizer,
                init_lr=lr,
                gamma=lr_factor
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = None

    return scheduler


def update_scheduler(scheduler, exp_params, control_metric=None, iter=-1, end_epoch=False):
    """
    Updating the learning rate scheduler

    Args:
    -----
    scheduler: torch.optim
        scheduler to evaluate
    exp_params: dictionary
        dictionary containing the experiment parameters
    control_metric: float/torch Tensor
        Last computed validation metric.
        Needed for plateau scheduler
    iter: float
        number of optimization step.
        Needed for cyclic, cosine and exponential schedulers
    end_epoch: boolean
        True after finishing a validation epoch or certain number of iterations.
        Triggers schedulers such as plateau or fixed-step
    """
    scheduler_type = exp_params["training"]["scheduler"]
    if(scheduler_type == "plateau" and end_epoch):
        scheduler.step(control_metric)
    elif(scheduler_type in ["step", "multistep"] and end_epoch):
        scheduler.step()
    elif(scheduler_type == "warmup" and end_epoch):
        scheduler.step(iter)
    elif(scheduler_type == "exponential"):
        scheduler.step(iter)
    return


#
