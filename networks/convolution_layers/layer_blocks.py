import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1,
                 weight_norm='none'):
        super().__init__()

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None

        # initialize convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.Sigmoid()

        if self.weight_norm:
            self.conv1 = self.weight_norm(self.conv1)
            self.conv2 = self.weight_norm(self.conv2)

    def forward(self, xin):
        x1 = self.conv1(xin)
        x1 = self.norm1(x1)
        x1 = self.activation1(x1)
        x2 = self.conv2(xin)
        x2 = self.norm2(x2)
        x2 = self.activation2(x2)
        x = x1 * x2
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super().__init__()
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(x.shape[2] // 2, x.shape[2] // 2))
        out = self.conv_out(x)
        return out


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x_up = self.pixel_shuffle(x)
        out = self.conv_out(x_up)
        return out
