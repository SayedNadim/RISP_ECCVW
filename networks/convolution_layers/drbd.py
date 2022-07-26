import torch
from torch import nn


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class PA(nn.Module):
    '''PA is pixel attention'''

    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out


class PCA(nn.Module):

    def __init__(self, nChannels):
        super().__init__()
        self.ca_layer = CALayer(nChannels, 16)
        self.pa_layer = PA(nChannels)
        self.conv_1x1 = nn.Conv2d(nChannels, nChannels, 1)

    def forward(self, x):
        x_ca = self.ca_layer(x)
        x_pa = self.pa_layer(x_ca)
        x = self.conv_1x1(x) + x_pa
        return x


class make_dense(nn.Module):

    def __init__(self, nChannels, growthRate, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nChannels, growthRate, kernel_size=3, stride=1,
                      padding=1,
                      bias=True, dilation=1),
            # nn.BatchNorm2d(growthRate),
            nn.LeakyReLU(),
            nn.Conv2d(growthRate, growthRate, kernel_size=3, stride=1,
                      padding=1,
                      bias=True, dilation=1),
            # nn.BatchNorm2d(growthRate),
            nn.LeakyReLU(),
            nn.Conv2d(growthRate, growthRate, kernel_size=3, stride=1,
                      padding=1,
                      bias=True, dilation=1)
        )

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


class DRDB(nn.Module):

    def __init__(self, nChannels, denseLayer, growthRate):
        super(DRDB, self).__init__()
        num_channels = nChannels
        modules = []
        for i in range(denseLayer):
            modules.append(
                make_dense(num_channels, growthRate)
            )
            num_channels += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.pca = PCA(nChannels)
        self.conv_1x1 = nn.Conv2d(num_channels, nChannels, kernel_size=1, padding=0, bias=True)
        self.out_1x1 = nn.Conv2d(num_channels, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        out = self.pca(out)
        out = out + x
        return out
