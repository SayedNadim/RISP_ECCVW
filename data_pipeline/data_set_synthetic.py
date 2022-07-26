import sys
from glob import glob

import torchvision.transforms
from torch.utils.data import Dataset
import random

from .data_utils import *


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, config, debug=False):
        self.config = config
        self.root = config['train_data_loader']['args']['dataset_path']
        self.train_raw_samples, self.train_rgb_samples = self.get_filenames()
        self.debug = debug
        if self.debug:
            self.rgbs = self.train_rgb_samples[:100]
            self.raws = self.train_raw_samples[:100]

    def get_filenames(self):
        train_rgbs = sorted(glob(self.root + '/train/*.jpg'))
        assert len(train_raws) == len(train_rgbs)
        print(f'Training samples: {len(train_raws)}')
        return train_raws, train_rgbs

    def __len__(self):
        return len(self.train_raw_samples)

    def __getitem__(self, idx):
        rgb = load_img(self.train_rgb_samples[idx], norm=False)
        raw = load_raw(self.train_raw_samples[idx])
        # raw = unpack_raw(raw)
        # raw, rgb = get_random_crop(raw, rgb, 256, 256)

        if 0.3 < random.random() < 0.5:
            rgb = cv2.flip(rgb, 0)
            raw = cv2.flip(raw, 0)
        if 0.3 > random.random() < 0.2:
            rgb = cv2.flip(rgb, 1)
            raw = cv2.flip(raw, 1)
        if random.random() < 0.2:
            rgb = cv2.flip(rgb, -1)
            raw = cv2.flip(raw, -1)
        #
        # raw = np.expand_dims(raw, axis=-1)
        # raw = pack_raw(raw)
        # raw = np.squeeze(raw)
        # raw = raw.astype(np.float32)

        rgb = torchvision.transforms.ToTensor()(rgb)
        raw = torchvision.transforms.ToTensor()(raw)

        return {"rgb": rgb, "raw": raw}


class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, config, debug=False):
        self.config = config
        self.root = config['valid_data_loader']['args']['dataset_path']
        self.valid_raw_samples, self.valid_rgb_samples = self.get_filenames()
        self.debug = debug
        if self.debug:
            self.rgbs = self.valid_rgb_samples[:100]
            self.raws = self.valid_raw_samples[:100]

    def get_filenames(self):
        valid_raws = sorted(glob(self.root + '/validation/*.npy'))
        valid_rgbs = sorted(glob(self.root + '/validation/*.jpg'))
        assert len(valid_raws) == len(valid_rgbs)
        print(f'Validation samples: {len(valid_raws)}')
        return valid_raws, valid_rgbs

    def __len__(self):
        return len(self.valid_raw_samples)

    def __getitem__(self, idx):
        rgb = load_img(self.valid_rgb_samples[idx], norm=False)
        raw = load_raw(self.valid_raw_samples[idx])

        rgb = torchvision.transforms.ToTensor()(rgb)
        raw = torchvision.transforms.ToTensor()(raw)

        return {"rgb": rgb, "raw": raw}


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, config, debug=False):
        self.config = config
        self.root = config['test_data_loader']['args']['dataset_path']
        self.test_rgb_samples = self.get_filenames()
        self.debug = debug
        if self.debug:
            self.rgbs = self.test_rgb_samples[:100]

    def get_filenames(self):
        test_rgbs = sorted(glob(self.root + '/val_rgb/*.jpg'))
        print(f'Test samples: {len(test_rgbs)}')
        return test_rgbs

    def __len__(self):
        return len(self.test_rgb_samples)

    def __getitem__(self, idx):
        rgb = load_img(self.test_rgb_samples[idx], norm=False)

        rgb = torchvision.transforms.ToTensor()(rgb)

        return {"rgb": rgb, 'idx': self.test_rgb_samples[idx]}
