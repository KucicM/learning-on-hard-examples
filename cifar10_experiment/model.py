from typing import Dict, List
from torch import nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import numpy as np
from cifar10_experiment.config import ResNet9Config, ConvWithBatchNormConfig, ResidualBlockConfig, OptimizerConfig

import logging
LOGGER = logging.getLogger(__name__)


class ResNet9(pl.LightningModule):
    def __init__(self, net_config: ResNet9Config, optimizer_config: OptimizerConfig, batch_size: int) -> None:
        super().__init__()

        self.model = nn.Sequential(
            ConvWithBatchNorm(net_config.l0_conv_with_bn_config),

            ConvWithBatchNorm(net_config.l1_conv_with_bn_config),
            nn.MaxPool2d(**net_config.l1_max_pool_params),
            ResidualBlock(net_config.l1_residual_config),

            ConvWithBatchNorm(net_config.l2_conv_with_bn_config),
            nn.MaxPool2d(**net_config.l2_max_pool_params),

            ConvWithBatchNorm(net_config.l3_conv_with_bn_config),
            nn.MaxPool2d(**net_config.l3_max_pool_params),
            ResidualBlock(net_config.l3_residual_config),

            nn.MaxPool2d(**net_config.final_max_pool_params),
            nn.Flatten(),
            nn.Linear(**net_config.linear_params),
            Mul(**net_config.scalar),
        )
        print(net_config)

        self._optimizer_config = optimizer_config
        self._batch_size = batch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> Dict:
        x, y = batch
        out = self.model(x)
        loss = F.cross_entropy(out, y, reduction="none")

        return {"loss": loss.sum(), "costs": loss}

    def test_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        out = self.model(x)

        loss = F.cross_entropy(out, y, reduction="none")
        _, predictions = torch.max(out.data, 1)
        accuracy = torch.sum(y == predictions).item() / len(y)

        self.log("test_loss", loss)
        self.log("accuracy", accuracy, prog_bar=True, enable_graph=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), **self._sgd_params)

        # this throws warning when using mixed precision
        # related to https://github.com/PyTorchLightning/pytorch-lightning/issues/5558
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, self._calculate_learning_rate),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def _calculate_learning_rate(self, step: int) -> float:
        lr = np.interp(self._scale_step(step), self._lr_scheduler_epoch_range, self._lr_scheduler_value_range)
        return (lr / self._batch_size)[0]

    @property
    def _train_number_of_batches(self):
        dataloader = self._train_dataloader
        return 1 if dataloader is None else len(dataloader)

    @property
    def _train_dataloader(self):
        return self.trainer.train_dataloader

    @property
    def _sgd_params(self) -> Dict:
        params = self._optimizer_config.sgd_params
        params["weight_decay"] *= self._batch_size
        return params

    @property
    def _lr_scheduler_epoch_range(self) -> List[int]:
        epoch_range = self._optimizer_config.lr_scheduler_epochs
        if epoch_range[-1] == -1:
            epoch_range[-1] = self._num_of_epochs
        return epoch_range

    @property
    def _lr_scheduler_value_range(self) -> List[float]:
        return self._optimizer_config.lr_scheduler_values

    def _scale_step(self, step) -> List[float]:
        step /= self._train_number_of_batches
        return [step]

    @property
    def _num_of_epochs(self) -> int:
        return self.trainer.max_epochs

    def configure_callbacks(self):
        return []


class ResidualBlock(nn.Module):
    """
    A residual block as defined by He et al.
    """
    def __init__(self, config: ResidualBlockConfig) -> None:
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Identity(),
            ConvWithBatchNorm(config.conv_with_bn_config),
            ConvWithBatchNorm(config.conv_with_bn_config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).add(x)


class ConvWithBatchNorm(nn.Module):
    def __init__(self, config: ConvWithBatchNormConfig) -> None:
        super(ConvWithBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(**config.conv_params),
            nn.BatchNorm2d(**config.batch_norm_params),
            nn.ReLU(**config.relu_params),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Mul(nn.Module):
    def __init__(self, weight: float) -> None:
        super(Mul, self).__init__()
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight
