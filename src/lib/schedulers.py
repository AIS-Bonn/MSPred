"""
Implementation of learning rate schedulers, early stopping and other utils
for improving optimization
"""

from lib.logger import print_
import models.model_utils as model_utils


class ExponentialLRSchedule:
    """
    Exponential LR Scheduler that decreases the learning rate by multiplying it
    by an exponentially decreasing decay factor:
        LR = LR * gamma ^ (step/total_steps)

    Args:
    -----
    optimizer: torch.optim
        Optimizer to schedule
    init_lr: float
        base learning rate to decrease with the exponential scheduler
    gamma: float
        exponential decay factor
    total_steps: int/float
        number of optimization steps to optimize for. Once this is reached,
        lr is not decreased anymore
    """

    def __init__(self, optimizer, init_lr, gamma=0.5, total_steps=100_000):
        """ Module initializer """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.gamma = gamma
        self.total_steps = total_steps
        return

    def update_lr(self, step):
        """ Computing exponential lr update """
        new_lr = self.init_lr * self.gamma ** (step / self.total_steps)
        return new_lr

    def step(self, iter):
        """ Scheduler step """
        if(iter < self.total_steps):
            for params in self.optimizer.param_groups:
                params["lr"] = self.update_lr(iter)
        elif(iter == self.total_steps):
            print_(f"Finished exponential decay due to reach of {self.total_steps} steps")
        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        return state_dict

    def load_state_dict(self, state_dict):
        """ Loading state dictinary """
        self.init_lr = state_dict["init_lr"]
        self.gamma = state_dict["gamma"]
        self.total_steps = state_dict["total_steps"]
        return


class WarmupStepLR:
    """
    Class for performing learning rate warm-ups or linearly decreasing the learning rate
    in a multi-step fashion.

    Args:
    -----
    init_lr: float
        initial learning rate
    warmup_steps: integer
        number of epochs to warmup for.
    decay_steps: integer
        number of epochs after which lr is decayed with factor gamma.
    gamma: float
        lr decay factor.
    """

    def __init__(self, optimizer, init_lr, warmup_steps, decay_steps, gamma):
        """ Initializer """
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.gamma = gamma
        for step in decay_steps:
            assert step > warmup_steps
        self.curr_lr = init_lr

    def _get_lr(self, iter):
        if (iter < self.warmup_steps):
            self.curr_lr = max(self.init_lr * (iter / self.warmup_steps), 1e-6)
        elif (iter in self.decay_steps):
            assert self.curr_lr != 0
            self.curr_lr = self.curr_lr * self.gamma
        return self.curr_lr

    def step(self, iter):
        """ Computing actual learning rate and updating optimizer """
        lr = self._get_lr(iter)
        for params in self.optimizer.param_groups:
            params["lr"] = lr
        return

    def state_dict(self):
        """ State dictionary """
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        return state_dict

    def load_state_dict(self, state_dict):
        """ Loading state dictinary """
        self.init_lr = state_dict["init_lr"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.decay_steps = state_dict["decay_steps"]
        self.gamma = state_dict["gamma"]
        self.curr_lr = state_dict["curr_lr"]
        return


class BetaScheduler:
    """
    Different scheduling functionalities for the scaling factor of the KL-Divergence

    Args:
    -----
    beta_init: float
        Initial value during scheduling
    beta_final: float
        Final value of the scheduling
    warmup_steps: integer
        Length (in batches) of the scheduling procedure
    warmup_type: string
        Type of scheduling to apply
    """

    def __init__(self, beta_init, beta_final, warmup_steps, warmup_type="linear"):
        """ Module initializer """
        self.beta_init = beta_init
        self.beta_final = beta_final
        self.warmup_steps = warmup_steps
        self.warmup_type = warmup_type
        self.curr_beta = beta_init

    def step(self, iter):
        """ Scheduling step """
        if self.warmup_type == "multistep":
            if iter < self.warmup_steps // 2:
                self.curr_beta = self.beta_init
            elif iter < self.warmup_steps:
                self.curr_beta = (self.beta_init + self.beta_final) / 2
            else:
                self.curr_beta = self.beta_final
        elif self.warmup_type == "step":
            if iter >= self.warmup_steps:
                self.curr_beta = self.beta_final
        else:
            assert self.warmup_type == "linear"
            if iter <= self.warmup_steps:
                self.curr_beta = self.beta_init + \
                    (iter-1) * (self.beta_final-self.beta_init) / (self.warmup_steps-1)
        return self.curr_beta


class EarlyStop:
    """
    Implementation of an early stop criterion

    Args:
    -----
    mode: string ['min', 'max']
        whether we validate based on maximizing or minmizing a metric
    delta: float
        threshold to consider improvements
    patience: integer
        number of epochs without improvement to trigger early stopping
    """

    def __init__(self, mode="min", delta=1e-6, patience=7):
        """ Early stopper initializer """
        assert mode in ["min", "max"]
        self.mode = mode
        self.delta = delta
        self.patience = patience
        self.counter = 0

        if(mode == "min"):
            self.best = 1e15
            self.criterion = lambda x: x < (self.best - self.min_delta)
        elif(mode == "max"):
            self.best = 1e-15
            self.criterion = lambda x: x < (self.best - self.min_delta)

        return

    def __call__(self, value):
        """
        Comparing current metric agains best past results and computing if we
        should early stop or not

        Args:
        -----
        value: float
            validation metric measured by the early stopping criterion

        Returns:
        --------
        stop_training: boolean
            If True, we should early stop. Otherwise, metric is still improving
        """
        are_we_better = self.criterion(value)
        if(are_we_better):
            self.counter = 0
            self.best = value
        else:
            self.counter = self.counter + 1

        stop_training = True if(self.counter >= self.patience) else False

        return stop_training


class Freezer:
    """
    Class for freezing and unfreezing nn.Module given number of epochs

    Args:
    -----
    module: nn.Module
        nn.Module to freeze or unfreeze
    frozen_epochs: integer
        Number of initial epochs in which the model is kept frozen
    """

    def __init__(self, module, frozen_epochs=0):
        """ Module initializer """
        self.module = module
        self.frozen_epochs = frozen_epochs
        self.is_frozen = False

    def __call__(self, epoch):
        """ """
        if epoch < self.frozen_epochs and self.is_frozen is False:
            print_(f"  --> Still in frozen epochs  {epoch} < {self.frozen_epochs}. Freezing module...")
            model_utils.freeze_params(self.module)
            self.is_frozen = True
        elif epoch >= self.frozen_epochs and self.is_frozen is True:
            print_(f"  --> Finished frozen epochs {epoch} = {self.frozen_epochs}. Unfreezing module...")
            model_utils.unfreeze_params(self.module)
            self.is_frozen = False
        return
