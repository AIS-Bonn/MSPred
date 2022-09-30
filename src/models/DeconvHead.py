"""
Module for high-level decoders in the MSPRed model.
Inspired by:
    https://github.com/JimmySuen/integral-human-pose:
    --> pytorch_projects/common_pytorch/base_modules/deconv_head.py
"""

import torch.nn as nn

BN_TRACK_STATS = False


class DeconvHead(nn.Module):
    """
    Deconvolutional decoder head for high-level decoders in MSPred

    Args:
    -----
    in_channels: int
        Number of channels at the input of the head
    out_channels: int
        Number of output channels to predict
    num_filters: int
        number of channels in decoder-head hidden layers
    num_layers: int
        Number of convolutional layers in the decoder head
    period: int
        Firing period of the decoder head. It should match its corresponding RNN
    """

    def __init__(self, in_channels, out_channels, num_filters=256, num_layers=6, period=1):
        """ Module initializer """
        super().__init__()
        self.in_channels = in_channels
        self.period = period
        self.counter = 0
        self.last_output = None

        self.upc_layers = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.upc_layers.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=4, stride=2, padding=1,
                                   output_padding=0))
            self.upc_layers.append(nn.BatchNorm2d(num_filters, track_running_stats=BN_TRACK_STATS))
            self.upc_layers.append(nn.ReLU(inplace=True))

        # 1x1 convolution
        self.upc_layers.append(nn.Conv2d(num_filters, out_channels, kernel_size=1, padding=0))
        self.upc_layers.append(nn.Sigmoid())
        return

    def forward(self, x):
        """ forward pass"""
        should_fire = self.check_counters()
        if not should_fire:
            return self.last_output, should_fire
        if len(x.shape) == 2:
            x = x.view(-1, self.in_channels, 1, 1)
        for layer in self.upc_layers:
            x = layer(x)
        self.last_output = x
        return x, should_fire

    def check_counters(self):
        should_fire = (self.counter == 0)
        if(should_fire):
            self.reset_counter()
        else:
            self.counter = self.counter - 1
        return should_fire

    def reset_counter(self):
        self.counter = self.period - 1
        return

    def init_counter(self):
        self.counter = 0
        return
