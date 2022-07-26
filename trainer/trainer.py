import numpy as np
import torch
from base.base_trainer import BaseTrainer
from tqdm import tqdm
from model.metric import ssim, CPSNR

from data_pipeline.data_utils import demosaic_tensor


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, config,
                 model,
                 loss,
                 metrics,
                 optimizer,
                 resume,
                 data_loader,
                 valid_data_loader=None,
                 train_logger=None,
                 render=True
                 ):
        super(Trainer, self).__init__(
            config,
            model,
            loss,
            metrics,
            optimizer,
            resume,
            train_logger,
            render)

        self.render = render
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = 1  # int(np.sqrt(data_pipeline.batch_size))

    def _eval_metrics(self, output, target):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """

        loss_lambda = self.config['others']['loss_lambda']
        # set models to train mode
        self.model.train()

        total_model_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, sample in (enumerate(tqdm(self.data_loader))):
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.optimizer.zero_grad()

            # get data and send them to GPU
            rgb, gt_raw = sample['rgb'].to(self.device), sample['raw'].to(self.device)

            # get G's output
            pred_raw = self.model(rgb)
            pixel_loss = self.loss(pred_raw, gt_raw)

            model_loss = pixel_loss * loss_lambda

            total_model_loss += model_loss.item()
            model_loss.backward()
            self.optimizer.step()
            total_model_loss += model_loss.item()

            self.writer.add_scalar('training/loss_pixel', pixel_loss.item())

            if batch_idx % self.config['trainer']['tensorboard_disp_freq'] == 0:
                pred_raw_vis = demosaic_tensor(pred_raw)
                gt_raw_vis = demosaic_tensor(gt_raw)
                disp_images = torch.cat([pred_raw_vis, gt_raw_vis], dim=3)
                self.writer.add_image('Training_images', disp_images.cpu())

            # # add histogram of model parameters to the tensorboard
            # for name, p in self.model.named_parameters():
            #     self.writer.add_histogram('training_' + name, p, bins='auto')

            # calculate the metrics
            total_metrics += self._eval_metrics(pred_raw, gt_raw)

            if self.verbosity >= 2:
                ssim_val = ssim(pred_raw, gt_raw)
                psnr_val = CPSNR(pred_raw, gt_raw)
                print(
                    ' ---> Training - Epoch: {} [{}/{} ({:.0f}%)] model_loss: {:.6f}, CPSNR: {:.6f}, SSIM: {:.6f}'.format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        len(self.data_loader) * self.data_loader.batch_size,
                        100.0 * batch_idx / len(self.data_loader),
                        model_loss.item(),  # it's a tensor, so we call .item() method
                        psnr_val,
                        ssim_val
                    )
                )
        log = {
            'model_loss': total_model_loss / len(self.data_loader),
            'metrics': (total_metrics / len(self.data_loader)).tolist()
        }

        if epoch % self.config['trainer']['validation_epoch'] == 0:
            if self.do_validation:
                print(" # ++++++++++++++++++++++ # Validation # ++++++++++++++++++++++ #")
                val_log = self._valid_epoch(epoch)
                log = {**log, **val_log}

        self.lr_scheduler.step()
        print("Current Learning Rate -> ", self.lr_scheduler.get_last_lr())

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """

        loss_lambda = self.config['others']['loss_lambda']

        self.model.eval()

        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        validation_psnr = 0
        validation_ssim = 0

        print("Validation on {} instances....".format(len(self.valid_data_loader)))
        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(self.valid_data_loader)):
                # get data and send them to GPU
                rgb, gt_raw = sample['rgb'].to(self.device), sample['raw'].to(self.device)

                # get G's output
                pred_raw = self.model(rgb)
                pred_raw = torch.clip(pred_raw, 0, 1)
                pixel_loss = self.loss(pred_raw, gt_raw)

                model_loss = pixel_loss * loss_lambda

                total_val_loss += model_loss.item()

                # calculate the metrics
                total_val_metrics += self._eval_metrics(pred_raw, gt_raw)

                self.writer.add_scalar('validation/loss', model_loss.item())

                validation_ssim += (ssim(pred_raw, gt_raw))
                validation_psnr += (CPSNR(pred_raw, gt_raw))
                pred_raw_vis = demosaic_tensor(pred_raw)
                gt_raw_vis = demosaic_tensor(gt_raw)
                disp_images = torch.cat([pred_raw_vis, gt_raw_vis], dim=3)
                self.writer.add_image('Validation_images', disp_images.cpu())
        print("-" * 80)
        print("| Validation: SSIM: {:0.6f}, PSNR: {:0.6f} |".format(
            validation_ssim / len(self.valid_data_loader),
            validation_psnr / len(self.valid_data_loader)
        )
        )
        print("-" * 80)

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram('validation_' + name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_metrics': (total_val_metrics / len(self.valid_data_loader)).tolist()
        }
