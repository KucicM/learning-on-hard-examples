from high_cost_data_engine import config
from high_cost_data_engine.sampler import HighCostSampler
from torch.utils.data import DataLoader
from typing import Union, List, Optional, Dict

import pytorch_lightning as pl


class HighCostDataModule(pl.LightningDataModule):
    _train_set = None
    _test_set = None
    __sampler: HighCostSampler = None

    def __init__(self, data_config: config.Dataset, dataloader_config: config.DataLoader, dataset) -> None:
        super().__init__()
        self._data_config = data_config
        self._dataloader_config = dataloader_config
        self._data = dataset

    def prepare_data(self) -> None:
        self._data(self._data_path, train=True, download=True)
        self._data(self._data_path, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self._train_set = self._data(self._data_path, train=True, transform=self.train_transforms)
            self._test_set = self._data(self._data_path, train=False, transform=self.test_transforms)
        elif stage == "test" and self._test_set is None:
            self._test_set = self._data(self._data_path, train=False, transform=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_set, **self._train_dataloader_params, sampler=self._sampler)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_dataloader = self.test_dataloader()

        if self._use_selective_backprop:
            val_dataloader = DataLoader(self._train_set, **self._val_dataloader_params, sampler=self._sampler)
            return [val_dataloader, test_dataloader]

        return test_dataloader

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self._test_set, **self._test_dataloader_params)

    @property
    def _sampler(self) -> HighCostSampler:
        if self.__sampler is None:
            self.__sampler = HighCostSampler(self._train_set)
        return self.__sampler

    @property
    def _data_path(self):
        return self._data_config.data_path

    @property
    def _use_selective_backprop(self) -> bool:
        return self._data_config.use_selective_backprop

    @property
    def _train_dataloader_params(self) -> Dict:
        return self._dataloader_config.train_params

    @property
    def _test_dataloader_params(self) -> Dict:
        return self._dataloader_config.test_params

    @property
    def _val_dataloader_params(self) -> Dict:
        return self._dataloader_config.val_params
