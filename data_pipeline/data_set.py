from glob import glob
from torch.utils.data import Dataset
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
        train_raws = sorted(glob(self.root + '/train/*.npy'))
        train_rgbs = sorted(glob(self.root + '/train/*.jpg'))
        assert len(train_raws) == len(train_rgbs)
        print(f'Training samples: {len(train_raws)}')
        return train_raws, train_rgbs

    def __len__(self):
        return len(self.train_raw_samples)

    def __getitem__(self, idx):
        rgb = load_img(self.train_rgb_samples[idx], norm=True)
        raw = load_raw(self.train_raw_samples[idx])
        # raw = unpack_raw(raw)
        # raw, rgb = get_random_crop(raw, rgb, 256, 256)

        choices = ['vflip', 'hflip', 'vhflip', 'rotate', 'no_aug']
        aug_choice = random.choice(choices)

        if aug_choice == 'hflip':
            rgb = cv2.flip(rgb, 1)
            raw = cv2.flip(raw, 1)
        elif aug_choice == 'vflip':
            rgb = cv2.flip(rgb, 0)
            raw = cv2.flip(raw, 0)
        elif aug_choice == 'vhflip':
            rgb = cv2.flip(rgb, -1)
            raw = cv2.flip(raw, -1)
        elif aug_choice == 'rotate':
            rotation_choice = ['90, 180', '270']
            rot_aug_choice = random.choice(rotation_choice)
            if rot_aug_choice == '90':
                rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                raw = cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE)
            elif rot_aug_choice == '180':
                rgb = cv2.rotate(rgb, cv2.ROTATE_180)
                raw = cv2.rotate(raw, cv2.ROTATE_180)
            elif rot_aug_choice == '270':
                rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                raw = cv2.rotate(raw, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif aug_choice == 'no_aug':
            rgb = rgb
            raw = raw

        raw = raw.astype(np.float32)
        rgb = rgb.astype(np.float32)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        raw = torch.from_numpy(raw).permute(2, 0, 1)

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
        rgb = load_img(self.valid_rgb_samples[idx], norm=True)
        raw = load_raw(self.valid_raw_samples[idx])

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)
        raw = torch.from_numpy(raw).permute(2, 0, 1)

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
        test_rgbs = sorted(glob(self.root + '/test_rgb/*.jpg'))
        print(f'Test samples: {len(test_rgbs)}')
        return test_rgbs

    def __len__(self):
        return len(self.test_rgb_samples)

    def __getitem__(self, idx):
        rgb = load_img(self.test_rgb_samples[idx], norm=True)

        rgb = torch.from_numpy(rgb).permute(2, 0, 1)

        return {"rgb": rgb, 'idx': self.test_rgb_samples[idx]}
