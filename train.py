import logging
import os
import yaml
import argparse
import torch
from trainer.trainer import Trainer
from data_pipeline import data_loader as module_data
from model import loss as module_loss
from model import metric as module_metric
from model import model as module_arch

# viewing model summary


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if torch.cuda.is_available():
    print("Using cuda device: {}".format(torch.cuda.current_device()))
else:
    print("No cuda device found!")


def main(config, resume):
    if not os.path.exists(config['trainer']['log_save_dir']):
        os.makedirs(config['trainer']['log_save_dir'])
    if not os.path.isfile(config['trainer']['log_save_dir'] + '/{}.log'.format(config['name'])):
        os.mknod(config['trainer']['log_save_dir'] + '/{}.log'.format(config['name']))

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=config['trainer']['log_save_dir'] + '/{}.log'.format(config['name']),
                        filemode='a')

    # setup data_pipeline instances
    # ok
    train_data_loader_class = getattr(module_data, config['train_data_loader']['type'])
    train_data_loader = train_data_loader_class(config)

    # ok
    valid_data_loader_class = getattr(module_data, config['valid_data_loader']['type'])
    valid_data_loader = valid_data_loader_class(config)

    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # summary(model, (3,504,504))

    # get function handles of loss and metrics

    loss = {k: getattr(module_loss, v) for k, v in config['loss'].items()}
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer for models
    model_trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_class = getattr(torch.optim, config['optimizer']['type'])
    optimizer = dict()
    optimizer['model'] = optimizer_class(model_trainable_params, **config['optimizer']['args'])

    # start to train the network
    trainer = Trainer(config,
                      model,
                      loss,
                      metrics,
                      optimizer,
                      resume,
                      train_data_loader,
                      valid_data_loader)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RevIspNet')
    parser.add_argument('-c', '--config', default='configs/train_config.yaml', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume',
                        default='saved/checkpoints/s_7_v3/0724_165155/model_best.pth',
                        type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--device', default='None', type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = yaml.safe_load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])
    elif args.resume:
        # load config from checkpoint if new config file is not given.
        # Use '--config' and '--resume' together to fine-tune trained model with changed configurations.
        config = torch.load(args.resume)['config']
    else:
        raise AssertionError("Configuration file need to be specified. Add '-c train_config.yaml', for example.")

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    main(config, args.resume)
