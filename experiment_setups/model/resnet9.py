from typing import List
from torch.functional import F
from experiment_setups.model.rensen9_backbone import ResNet9Backbone
from data_engine.model import HighLossModule

import torch
import numpy as np


class ResNet9(HighLossModule):
    def __init__(self, batch_size):
        super().__init__(F.cross_entropy)
        self.model = ResNet9Backbone()
        self.batch_size = batch_size

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=1,
            weight_decay=0.0005 * self.batch_size,
            momentum=0.9,
            nesterov=True
        )

        # this throws warning when using mixed precision
        # related to https://github.com/PyTorchLightning/pytorch-lightning/issues/5558
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, self._calculate_learning_rate),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _calculate_learning_rate(self, step: int) -> float:
        lr = np.interp(self._scale_step(step), [0, 5, 0], [0, .4, 0])
        return (lr / self.batch_size)[0]

    def _scale_step(self, step) -> List[float]:
        return [step / self._train_number_of_batches]

    @property
    def _train_number_of_batches(self):
        dataloader = self._train_dataloader
        return 1 if dataloader is None else len(dataloader)

    @property
    def _train_dataloader(self):
        return self.trainer.train_dataloader

