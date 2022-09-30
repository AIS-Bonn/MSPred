"""
VGG Encoder and Decoder, but stopped after Conv_3.
These modules are meant to be used along with the ConvLSTMs.
"""

import torch
import torch.nn as nn

BN_TRACK_STATS = False


class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 3, 1, 1),
                nn.BatchNorm2d(nout, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True)
                )

    def forward(self, input_):
        return self.main(input_)


class encoder(nn.Module):
    def __init__(self, dim, nc=1, extra_deep=False):
        super(encoder, self).__init__()
        self.num_blocks = 4
        self.dim = dim
        # 64 x 64
        n_layers = 3 if extra_deep else 2
        self.c1 = nn.Sequential(
                vgg_layer(nc, 64),
                *[vgg_layer(64, 64) for i in range(n_layers-1)]
            )
        # 32 x 32
        n_layers = 4 if extra_deep else 2
        self.c2 = nn.Sequential(
                vgg_layer(64, 128),
                *[vgg_layer(128, 128) for i in range(n_layers-1)]
            )
        # 16 x 16
        n_layers = 5 if extra_deep else 3
        self.c3 = nn.Sequential(
                vgg_layer(128, 256),
                *[vgg_layer(256, 256) for i in range(n_layers-1)]
            )
        # 8 x 8
        self.out_conv = nn.Conv2d(in_channels=256, out_channels=dim, kernel_size=3, padding=1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        return

    def get_spatial_dims(self, img_size, level=-1):
        if level == -1:
            level = self.num_blocks - 1
        assert level in range(self.num_blocks)
        H, W = img_size
        scale = 2**level
        return (H//scale, W//scale)

    def forward(self, input_):
        h1 = self.c1(input_)  # (3, 64, 64) -> (64, 64, 64)
        h2 = self.c2(self.mp(h1))  # (64, 64, 64) -> (128, 32, 32)
        h3 = self.c3(self.mp(h2))  # (128, 32, 32) -> (256, 16, 16)
        h4 = self.out_conv(self.mp(h3))  # (256, 16, 16) -> (dim, 8, 8)
        return h4, [h1, h2, h3]


class decoder(nn.Module):
    def __init__(self, dim, nc=1, extra_deep=False):
        super(decoder, self).__init__()
        self.dim = dim
        self.in_conv = nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=1)

        # 16 x 16
        n_layers = 5 if extra_deep else 3
        self.upc1 = nn.Sequential(
                vgg_layer(256*2, 256),
                *[vgg_layer(256, 256) for i in range(n_layers-2)],
                vgg_layer(256, 128)
            )
        # 32 x 32
        n_layers = 4 if extra_deep else 2
        self.upc2 = nn.Sequential(
                vgg_layer(128*2, 128),
                *[vgg_layer(128, 128) for i in range(n_layers-2)],
                vgg_layer(128, 64)
            )
        # 64 x 64
        n_layers = 3 if extra_deep else 2
        self.upc3 = nn.Sequential(
                vgg_layer(64*2, 64),
                *[vgg_layer(64, 64) for i in range(n_layers-2)],
                nn.ConvTranspose2d(64, nc, 3, 1, 1),
                nn.Sigmoid()
            )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input_):
        vec, skip = input_
        d1 = self.in_conv(vec)  # (dim, 8, 8) -> (256, 8, 8)
        up1 = self.up(d1)  # (256, 8, 8) -> (256, 16, 16)
        d2 = self.upc1(torch.cat([up1, skip[-1]], 1))  # (256, 16, 16) -> (128, 16, 16)
        up2 = self.up(d2)  # (128, 16, 16) -> (128, 32, 32)
        d3 = self.upc2(torch.cat([up2, skip[-2]], 1))  # (128, 32, 32) -> (64, 32, 32)
        up3 = self.up(d3)  # (64, 32, 32) -> (64, 64, 64)
        output = self.upc3(torch.cat([up3, skip[-3]], 1))  # (64, 64, 64) -> (3, 64, 64)
        return output, [d1, d2, d3]
