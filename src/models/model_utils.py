"""
Model utils
"""

from lib.logger import print_


def init_weights(m):
    """ Initializing model parameters """
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Conv3d') != -1 \
            or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        pass


def count_model_params(model, verbose=False):
    """Counting number of learnable parameters"""
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print_(f"  --> Number of learnable parameters: {num_params}")
    return num_params


def freeze_params(model):
    """Freezing model params to avoid updates in backward pass"""
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_params(model):
    """Unfreezing model params to allow for updates during backward pass"""
    for param in model.parameters():
        param.requires_grad = True
    return model
