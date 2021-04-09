from cifar10_experiment import config as exp_config
from torchvision.datasets import CIFAR10
from torchvision import transforms
from high_cost_data_engine.datamodule import HighCostDataModule
from high_cost_data_engine import config as hc_config
from typing import Dict, Tuple

import torch


class CIFAR10DataModule(HighCostDataModule):
    def __init__(self, data_config: exp_config.DatasetConfig, dataloader_config: hc_config.DataLoader):
        super().__init__(data_config, dataloader_config, CIFAR10)
        self._data_config = data_config

        self.train_transforms = transforms.Compose([
            transforms.Pad(**self._pad_params),
            transforms.ToTensor(),
            transforms.Normalize(**self._norm_params),
            transforms.RandomCrop(**self._crop_params),
            transforms.RandomHorizontalFlip(),
            Cutout(**self._cutout_params)
        ])

        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**self._norm_params)
        ])

    @property
    def _pad_params(self) -> Dict:
        return self._data_config.pad_params

    @property
    def _norm_params(self) -> Dict:
        return self._data_config.normalization_values

    @property
    def _crop_params(self) -> Dict:
        return self._data_config.crop_params

    @property
    def _cutout_params(self) -> Dict:
        return self._data_config.cutout_params


class Cutout:
    """
    Simplified Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, size=8):
        assert size > 0
        self.size = size

    def __call__(self, tensor):
        _, height, width = tensor.shape
        x, y = self._get_random_xy(height, width)

        cutout = torch.ones(height, width)
        cutout[x:x+self.size, y:y+self.size] = 0

        return tensor * cutout

    def _get_random_xy(self, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(low=0, high=(height - self.size), size=(1,))
        y = torch.randint(low=0, high=(width - self.size), size=(1,))
        return x, y
