import kornia
import torch
from torch import nn


def unpack_raw_tensor(im):
    """
    Unpack RAW image from (h,w,4) to (h*2 , w*2, 1)
    """
    b, chan, h, w = im.shape
    H, W = h * 2, w * 2
    img2 = torch.zeros(b, H, W)
    img2[:, 0:H:2, 0:W:2] = im[:, 0, :, :]
    img2[:, 0:H:2, 1:W:2] = im[:, 1, :, :]
    img2[:, 1:H:2, 0:W:2] = im[:, 2, :, :]
    img2[:, 1:H:2, 1:W:2] = im[:, 3, :, :]
    img2 = img2.unsqueeze(1)
    img2 = img2.to(im.device)
    return img2


def loss(refined, gt):
    loss_pixel = nn.L1Loss()(refined, gt)  # F.l1
    loss_ssim = kornia.losses.ssim_loss(refined, gt, window_size=11)
    return loss_pixel + loss_ssim
