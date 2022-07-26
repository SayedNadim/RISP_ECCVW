import random

import numpy as np
import cv2
import imageio
import rawpy
import torch



def extract_bayer_channels(raw):
    ch_B = raw[1::2, 1::2]
    ch_Gb = raw[0::2, 1::2]
    ch_R = raw[0::2, 0::2]
    ch_Gr = raw[1::2, 0::2]

    return ch_R, ch_Gr, ch_B, ch_Gb


def load_rawpy(raw_file):
    raw = rawpy.imread(raw_file)
    raw_image = raw.raw_image
    return raw_image


def load_img(filename, debug=False, norm=True, resize=None):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if norm:
        img = img / 255.
        img = img.astype(np.float32)
    if debug:
        print(img.shape, img.dtype, img.min(), img.max())

    if resize:
        img = cv2.resize(img, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)

    return img


def save_rgb(img, filename):
    if np.max(img) <= 1:
        img = img * 255

    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filename, img)


def load_raw_png(raw, debug=False):
    '''
    Load RAW images from the ZurichRAW2RGB Dataset
    Reference: https://github.com/aiff22/PyNET-PyTorch/blob/master/dng_to_png.py
    by Andrey Ignatov.

    inputs:
     - raw: filename to the raw image saved as '.png'
    returns:
     - RAW_norm: normalized float32 4-channel raw image with bayer pattern RGGB.
    '''

    assert '.png' in raw
    raw = np.asarray(imageio.imread((raw)))
    ch_R, ch_Gr, ch_B, ch_Gb = extract_bayer_channels(raw)

    RAW_combined = np.dstack((ch_R, ch_Gr, ch_Gb, ch_B))
    RAW_norm = RAW_combined.astype(np.float32) / (4 * 255)
    RAW_norm = np.clip(RAW_norm, 0, 1)

    if debug:
        print(RAW_norm.shape, RAW_norm.dtype, RAW_norm.min(), RAW_norm.max())

    # raw as (h,w,1) in RGBG domain! do not use
    raw_unpack = raw.astype(np.float32) / (4 * 255)
    raw_unpack = np.expand_dims(raw_unpack, axis=-1)

    return RAW_norm


def load_raw(raw, max_val=2 ** 10):
    raw = np.load(raw) / max_val
    return raw.astype(np.float32)


########## RAW image manipulation

def unpack_raw(im):
    """
    Unpack RAW image from (h,w,4) to (h*2 , w*2, 1)
    """
    h, w, chan = im.shape
    H, W = h * 2, w * 2
    img2 = np.zeros((H, W))
    img2[0:H:2, 0:W:2] = im[:, :, 0]
    img2[0:H:2, 1:W:2] = im[:, :, 1]
    img2[1:H:2, 0:W:2] = im[:, :, 2]
    img2[1:H:2, 1:W:2] = im[:, :, 3]
    img2 = np.squeeze(img2)
    img2 = np.expand_dims(img2, axis=-1)
    return img2


def pack_raw(im):
    """
    Pack RAW image from (h,w,1) to (h/2 , w/2, 4)
    """
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]
    ## R G G B
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :],
                          im[1:H:2, 1:W:2, :]), axis=2)
    return out


########## VISUALIZATION

def demosaic(raw):
    """Simple demosaicing to visualize RAW images
    Inputs:
     - raw: (h,w,4) RAW RGGB image normalized [0..1] as float32
    Returns:
     - Simple Avg. Green Demosaiced RAW image with shape (h*2, w*2, 3)
    """

    assert raw.shape[-1] == 4
    shape = raw.shape

    red = raw[:, :, 0]
    green_red = raw[:, :, 1]
    green_blue = raw[:, :, 2]
    blue = raw[:, :, 3]
    avg_green = (green_red + green_blue) / 2
    image = np.stack((red, avg_green, blue), axis=-1)
    image = cv2.resize(image, (shape[1] * 2, shape[0] * 2))
    return image


def mosaic(rgb):
    """Extracts RGGB Bayer planes from an RGB image."""

    assert rgb.shape[-1] == 3
    shape = rgb.shape

    red = rgb[0::2, 0::2, 0]
    green_red = rgb[0::2, 1::2, 1]
    green_blue = rgb[1::2, 0::2, 1]
    blue = rgb[1::2, 1::2, 2]

    image = np.stack((red, green_red, green_blue, blue), axis=-1)
    return image


def gamma_compression(image):
    """Converts from linear to gamma space."""
    return np.maximum(image, 1e-8) ** (1.0 / 2.2)


def tonemap(image):
    """Simple S-curved global tonemap"""
    return (3 * (image ** 2)) - (2 * (image ** 3))


def postprocess_raw(raw):
    """Simple post-processing to visualize demosaic RAW imgaes
    Input:  (h,w,3) RAW image normalized
    Output: (h,w,3) post-processed RAW image
    """
    raw = gamma_compression(raw)
    raw = tonemap(raw)
    raw = np.clip(raw, 0, 1)
    return raw


def demosaic_tensor(image):
    # Generate single channel input for opencv
    im_sc = torch.zeros(image.shape[0], 1, image.shape[-2] * 2, image.shape[-1] * 2)
    im_sc[:, 0, ::2, ::2] = image[:, 0, :, :]
    im_sc[:, 0, ::2, 1::2] = image[:, 1, :, :]
    im_sc[:, 0, 1::2, ::2] = image[:, 2, :, :]
    im_sc[:, 0, 1::2, 1::2] = image[:, 3, :, :]
    return im_sc


def get_random_crop(raw, rgb, crop_height, crop_width):

    max_x = raw.shape[1] - crop_width
    max_y = raw.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop_raw = raw[y: y + crop_height, x: x + crop_width,:]
    crop_rgb = rgb[y: y + crop_height, x: x + crop_width,:]

    return crop_raw, crop_rgb

