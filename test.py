"""
Test file for the network
"""
import os
import argparse
import sys
import time

import cv2
import torch
from torchvision import utils as vutils

from tqdm import tqdm
import numpy as np
import yaml

import warnings
import shutil

warnings.filterwarnings("ignore")

import data_pipeline.data_loader as module_data
from model import model as module_arch


def main(config, resume):
    if not os.path.exists(config['trainer']['submission_folder']):
        os.makedirs(config['trainer']['submission_folder'], exist_ok=True)
    # load checkpoint
    checkpoint = torch.load(resume, map_location='cuda:{}'.format(config['use_gpu']))

    # setup data_pipeline instances
    data_loader_class = getattr(module_data, config['test_data_loader']['type'])
    test_data_loader = data_loader_class(config)

    # build colorization model architecture
    model_class = getattr(module_arch, config['model']['type'])
    model = model_class(**config['model']['args'])

    # prepare model for testing
    device = torch.device('cuda:{}'.format(config['use_gpu']) if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.load_state_dict(checkpoint['model'])

    model.eval()
    cnt = 0
    runtime = []
    psnr_val = []
    ssim_val = []
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(test_data_loader, ascii=True)):
            # get data and send them to GPU
            rgb_batch = sample['rgb'].to(device)
            rgb_name = sample['idx'][0].split('/')[-1].replace('.jpg', '')

            st = time.time()
            # get G's output
            recon_raw = model(rgb_batch)
            tt = time.time() - st
            runtime.append(tt)
            recon_raw = recon_raw[0].detach().cpu().permute(1, 2, 0).numpy()
            ## save as np.uint16
            assert recon_raw.shape[-1] == 4
            recon_raw = np.clip((recon_raw * 2 ** 10), 0, 2 ** 10)
            recon_raw = recon_raw.astype(np.uint16)
            np.save(config['trainer']['submission_folder'] + '/' + rgb_name + '.npy', recon_raw)
            cnt += 1
    info_lines = ['runtime per frame [s] : {}'.format(np.mean(runtime)), 'CPU[1] / GPU[0] : 0',
                  'Extra Data [1] / No Extra Data [0] : 0', 'Other description : v1 submission']
    with open(config['trainer']['submission_folder'] + '/' + 'readme.txt', 'w') as f:
        for line in info_lines:
            f.write(line)
            f.write('\n')
    # shutil.make_archive('submission', 'zip', config['trainer']['submission_folder'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RevIspNet')

    parser.add_argument('-c', '--config', default='configs/test_config.yaml', type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume',
                        default='saved/checkpoints/s_7_v3/0624_210946_best/model_best.pth',
                        type=str, help='path to latest checkpoint (default: None)')
    parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = parser.parse_args()

    if args.config:
        # load config file
        with open(args.config) as handle:
            config = yaml.safe_load(handle)
        # setting path to save trained models and log files
        path = os.path.join(config['trainer']['save_dir'], config['name'])

    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # from utils.util import denormalize

    main(config, args.resume)
