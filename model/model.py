import torch
from torch import nn
from networks.convolution_layers.convolution_block import Conv2dBlock
from networks.convolution_layers.drbd import DRDB


def pack_raw_tensor(im):
    """
    Pack RAW image from (h,w,1) to (h/2 , w/2, 4)
    """
    img_shape = im.shape
    H = img_shape[2]
    W = img_shape[3]
    ## R G G B
    out = torch.stack((im[:, 0, 0:H:2, 0:W:2],
                       im[:, 1, 0:H:2, 1:W:2],
                       im[:, 1, 1:H:2, 0:W:2],
                       im[:, 2, 1:H:2, 1:W:2]), dim=1)
    out = out.to(im.device)
    return out


# main model
class RISP(nn.Module):
    def __init__(self, nFeat):
        super().__init__()

        module_head = [Conv2dBlock(3, nFeat, kernel_size=3, stride=1, padding=1, dilation=1)]
        module_head += [Conv2dBlock(nFeat, nFeat, kernel_size=3, stride=1, padding=1, dilation=1)]

        modules_body = [DRDB(nFeat, 3, nFeat) for _ in range(6)]

        modules_tail = [
            Conv2dBlock(nFeat, 3, kernel_size=3, stride=1, padding=1, dilation=1, norm='none', activation='none')]

        self.head = nn.Sequential(*module_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, xin):
        x = self.head(xin)
        x = self.body(x) + x
        x = self.tail(x)
        x = pack_raw_tensor(x)
        x = torch.clip(x, 0, 1)
        return x


if __name__ == '__main__':
    height, width = 504, 504
    iim = torch.rand(1, 3, height, width).cuda()
    net = RISP(64).cuda()
    out_n = net(iim)
    print(out_n.shape, out_n.max(), out_n.min())
