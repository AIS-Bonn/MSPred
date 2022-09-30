"""
DCGAN-64 Encoder and Decoder, but stopped after Conv_3.
These modules are meant to be used along with the ConvLSTMs.
"""

import torch
import torch.nn as nn

BN_TRACK_STATS = False

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input_):
        return self.main(input_)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout, track_running_stats=BN_TRACK_STATS),
                nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, input_):
        return self.main(input_)


class encoder(nn.Module):
    def __init__(self, dim, nc=1, nf=64, extra_deep=False):
        super().__init__()
        self.num_blocks = 4
        self.dim = dim
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = nn.Sequential(
                nn.Conv2d(nf * 4, dim, 4, 2, 1),
                nn.BatchNorm2d(dim, track_running_stats=BN_TRACK_STATS),
                nn.Tanh()
            )
        # state size. (dim) x 4 x 4

    def get_spatial_dims(self, img_size, level=-1):
        if level == -1:
            level = self.num_blocks - 1
        assert level in range(self.num_blocks)
        H, W = img_size
        scale = 2**(level+1)
        return (H//scale, W//scale)

    def forward(self, input_):
        h1 = self.c1(input_)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        return h4, [h1, h2, h3]


class decoder(nn.Module):
    def __init__(self, dim, nc=1, nf=64, extra_deep=False):
        super().__init__()
        self.dim = dim
        # state size. (dim) x 4 x 4
        self.upc1 = dcgan_upconv(dim, nf * 4)
        # state size. (nf * 2) x 8 x 8
        self.upc2 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc3 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf*2) x 32 x 32
        self.upc4 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Sigmoid()
            )
        # state size. (nf) x 64 x 64
        return

    def forward(self, input_):
        vec, skips = input_
        d1 = self.upc1(vec)
        d2 = self.upc2(torch.cat([d1, skips[-1]], 1))
        d3 = self.upc3(torch.cat([d2, skips[-2]], 1))
        output = self.upc4(torch.cat([d3, skips[-3]], 1))
        return output, [d1, d2, d3]

#
