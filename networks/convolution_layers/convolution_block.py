import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, groups=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='reflect', use_bias=False, transpose=False):
        super(Conv2dBlock, self).__init__()

        self.use_bias = use_bias
        self.transpose = transpose
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zeros':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_channels
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=False)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        self.g_norm = nn.InstanceNorm2d(1, track_running_stats=False)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'swish':
            self.activation = nn.SiLU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'htanh':
            self.activation = nn.Hardtanh(min_val=0, max_val=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        # if transpose:
        # self.tconv = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
        #                        output_padding=conv_padding, dilation=dilation, groups=groups)
        # )
        self.tconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, stride=1, padding=(1, 1),
                      dilation=1),
            nn.PixelShuffle(upscale_factor=2)
        )
        # self.tconv = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(1, 1),
        #               dilation=(1, 1))
        # )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=(conv_padding, conv_padding), dilation=(dilation, dilation), groups=groups),
        )
        # self.g_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1,
        #               padding=0, dilation=(1, 1), groups=groups)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, xin):
        if self.transpose:
            if self.pad:
                x = self.tconv(self.pad(xin))
            else:
                x = self.tconv(xin)
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
            return x
        else:
            if self.pad:
                x = self.conv(self.pad(xin))
            else:
                x = self.conv(xin)
            # x = x * torch.sigmoid(self.g_conv(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x
