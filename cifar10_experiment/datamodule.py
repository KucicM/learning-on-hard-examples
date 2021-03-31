from cifar10_experiment import config
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from high_cost_data_engine.dataloader import HighCostDataLoader
from typing import Optional, Union, List, Dict, Tuple
import pytorch_lightning as pl
import torch


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_config: config.DatasetConfig, dataloader_config: config.DataLoaderConfig) -> None:
        super().__init__()
        self._data_config = data_config
        self._dataloader_config = dataloader_config

        self._train_set = ...
        self._test_set = ...

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

    def prepare_data(self) -> None:
        # probably do basic transformations and cache it
        CIFAR10(self.data_path, train=True, download=True)
        CIFAR10(self.data_path, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self._train_set = CIFAR10(self.data_path, train=True, transform=self._train_transforms)  # cache this shit

        if stage == "test":
            self._test_set = CIFAR10(self.data_path, train=False, transform=self._test_transforms)

    def train_dataloader(self) -> DataLoader:
        return HighCostDataLoader(self._train_set, **self._train_dataloader_params)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._test_set, **self._test_dataloader_params)

    @property
    def data_path(self):
        return self._data_config.data_path

    @property
    def batch_size(self) -> int:
        return self._dataloader_config.batch_size

    @property
    def _preprocessing_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Pad(**self._pad_params),
            transforms.ToTensor(),
            transforms.Normalize(**self._norm_params)
        ])

    @property
    def _pad_params(self) -> Dict:
        return self._data_config.pad_params

    @property
    def _norm_params(self) -> Dict:
        return self._data_config.norm_params

    @property
    def _crop_params(self) -> Dict:
        return self._data_config.crop_params

    @property
    def _cutout_params(self) -> Dict:
        return self._data_config.cutout_params

    @property
    def _train_dataloader_params(self) -> Dict:
        return self._dataloader_config.train_params

    @property
    def _test_dataloader_params(self) -> Dict:
        return self._dataloader_config.test_params


class Cutout:
    """
    Simplified Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, size=8):
        """
        Parameters
        ----------
        size : int
            The size of the cutout
        """
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
