import logging
from typing import Dict, Tuple, Any

import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from pytorch_lightning.core import LightningModule
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class LitClassifier(LightningModule):

    def __init__(
        self, cfg: Dict[str, Any], **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = instantiate(self.cfg.model)
        self.loss = instantiate(self.cfg.loss)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_train = self.loss(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('train_loss', loss_train, on_step=True, on_epoch=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_val = self.loss(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss_val, on_epoch=True)
        self.log('val_acc1', acc1, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_epoch=True)
        return loss_val

    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def test_step(self, batch, batch_idx):
        images, target = batch
        output = self(images)
        loss_test = self.loss(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('test_loss', loss_test, on_epoch=True)
        self.log('test_acc1', acc1, prog_bar=True, on_epoch=True)
        self.log('test_acc5', acc5, on_epoch=True)
        return loss_test

    def _log_metrics(self):
        if self.trainer.is_global_zero:
            str_metrics = ''
            for key, val in self.trainer.logged_metrics.items():
                str_metrics += f'\n\t{key}: {val}'
            logger.info(str_metrics)

    def on_validation_end(self):
        self._log_metrics()

    def on_test_end(self):
        self._log_metrics()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, self.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer)
        return [optimizer], [scheduler]
