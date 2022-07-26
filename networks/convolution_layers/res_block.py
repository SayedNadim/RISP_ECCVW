from torch import nn

class ResNetBlock(nn.Module):
    """ResNet block"""

    def __init__(self, dim, use_dropout=False, use_bias=True):
        super(ResNetBlock, self).__init__()

        sequence = list()

        sequence += [
            nn.Conv2d(dim,
                        dim,
                        kernel_size=3,
                        stride=1,
                        padding=1),
        ]

        if use_dropout:
            sequence += [nn.Dropout(0.5)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = x + self.model(x)
        return out