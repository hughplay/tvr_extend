import json
import logging
from typing import Dict, Any

from pytorch_lightning.core import LightningModule
from hydra.utils import instantiate


logger = logging.getLogger(__name__)


class LitReasoner(LightningModule):

    def __init__(
        self, cfg: Dict[str, Any], **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.model = instantiate(self.cfg.model)
        self.evaluator = instantiate(self.cfg.evaluator)
        self.r_train = instantiate(self.cfg.recorder)
        self.r_val = instantiate(self.cfg.recorder)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optim, self.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer)
        return [optimizer], [scheduler]

    def forward(self, *x):
        return self.model(*x)

    def _parse_data(self, batch, training=False):
        init = batch['init'].to(self.device).float().div(255)
        fin = batch['fin'].to(self.device).float().div(255)
        init_desc = batch['init_desc'].to(self.device).float()
        obj_target, pair_target = [x.to(self.device) for x in batch['target']]
        obj_target_vec = batch['obj_target_vec'].to(self.device)

        if 'basic' in self.cfg.dataset.name:
            options = batch['options'].to(self.device)
            if training:
                inputs = (init, fin, init_desc, obj_target_vec)
            else:
                inputs = (init, fin, init_desc)
            targets = (obj_target, pair_target, options)
            return inputs, targets
        else:
            fin_desc = batch['fin_desc'].to(self.device).float()
            if training:
                inputs = (init, fin, init_desc, obj_target_vec, pair_target)
            else:
                inputs = (init, fin, init_desc)
            targets = (init_desc, fin_desc, obj_target, pair_target)
            return inputs, targets

    def training_step(self, batch, batch_idx):
        inputs, targets = self._parse_data(batch, training=True)
        obj_choice, pair_choice = self(*inputs)
        loss, info = self.evaluator.evaluate(
            obj_choice, pair_choice, *targets, detail=False)
        self.r_train.batch_update(info)
        for key, val in self.r_train.state.items():
            prog_bar = True if key in ['acc'] else False
            key = f'train/{key}' if key != 'acc' else 'train_acc'
            self.log(key, val, on_step=True, prog_bar=prog_bar)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = self._parse_data(batch, training=False)
        obj_choice, pair_choice = self(*inputs)
        val_loss, info = self.evaluator.evaluate(
            obj_choice, pair_choice, *targets, detail=False)
        self.r_val.batch_update(info)
        for key, val in self.r_val.state.items():
            prog_bar = True if key in ['acc'] else False
            key = f'val/{key}' if key != 'acc' else 'val_acc'
            self.log(key, val, on_epoch=True, prog_bar=prog_bar)
        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, targets = self._parse_data(batch, training=False)
        obj_choice, pair_choice = self(*inputs)
        test_loss, info = self.evaluator.evaluate(
            obj_choice, pair_choice, *targets, detail=True)
        self.r_test.batch_update(info)
        for key, val in self.r_test.state.items():
            prog_bar = True if key in ['acc'] else False
            key = f'test/{key}'
            self.log(key, val, on_epoch=True, prog_bar=prog_bar)
        return test_loss

    def on_train_epoch_start(self):
        self.r_train.epoch_start()

    def on_validation_epoch_start(self):
        self.r_val.epoch_start()

    def on_test_epoch_start(self):
        self.r_test = instantiate(self.cfg.recorder)
        self.r_test.epoch_start()

    def on_test_epoch_end(self):
        self.r_test.epoch_end()
        if self.trainer.is_global_zero:
            logger.info(f'[Test] {self.r_test.brief}')
            with open('log/results.json', 'w') as f:
                json.dump(self.r_test.detail, f, indent=2)

    def on_validation_epoch_end(self):
        self._log_metrics()

    def on_test_epoch_end(self):
        self._log_metrics()

    def _log_metrics(self):
        if self.trainer.is_global_zero:
            str_metrics = ''
            for key, val in self.trainer.logged_metrics.items():
                str_metrics += f'\n\t{key}: {val}'
            logger.info(str_metrics)
    