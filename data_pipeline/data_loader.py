from torch.utils.data import DataLoader
from . import data_set


class TrainDataLoader(DataLoader):
    """
    Custom data loader
    """

    def __init__(self, config):
        self.root = config['train_data_loader']['args']['dataset_path']
        self.shuffle = config['train_data_loader']['args']['shuffle']
        self.debug = config['train_data_loader']['args']['debug']
        self.batch_size = config['train_data_loader']['args']['batch_size']
        self.num_workers = config['train_data_loader']['args']['num_workers']
        self.dataset = data_set.TrainDataset(config=config, debug=self.debug)
        super().__init__(dataset=self.dataset, shuffle=self.shuffle, batch_size=self.batch_size,
                         num_workers=self.num_workers, drop_last=True)


class ValidationDataLoader(DataLoader):
    """
    Custom data loader
    """

    def __init__(self, config):
        self.root = config['valid_data_loader']['args']['dataset_path']
        self.shuffle = config['valid_data_loader']['args']['shuffle']
        self.debug = config['valid_data_loader']['args']['debug']
        self.batch_size = config['valid_data_loader']['args']['batch_size']
        self.num_workers = config['valid_data_loader']['args']['num_workers']
        self.dataset = data_set.ValidationDataset(config=config, debug=self.debug)
        super().__init__(dataset=self.dataset, shuffle=self.shuffle, batch_size=self.batch_size,
                         num_workers=self.num_workers, drop_last=True)


class TestDataLoader(DataLoader):
    """
    Custom data loader
    """

    def __init__(self, config):
        self.root = config['test_data_loader']['args']['dataset_path']
        self.shuffle = config['test_data_loader']['args']['shuffle']
        self.batch_size = config['test_data_loader']['args']['batch_size']
        self.debug = config['test_data_loader']['args']['debug']
        self.num_workers = config['test_data_loader']['args']['num_workers']
        self.dataset = data_set.TestDataset(config=config, debug=self.debug)
        super().__init__(dataset=self.dataset, shuffle=self.shuffle, batch_size=self.batch_size,
                         num_workers=self.num_workers, drop_last=False)
