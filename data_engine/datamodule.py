from typing import Optional, List, Union
from data_engine.sampler import HighLossSampler
from torch.utils.data import DataLoader

import pytorch_lightning as pl


class HighLossDataModule(pl.LightningDataModule):
    _train_set = None
    _test_set = None
    _sampler = None

    def __init__(
            self,
            dataset,
            sampler: HighLossSampler,
            train_dataloader,
            val_dataloader,
            test_dataloader
    ) -> None:

        super().__init__()
        self._dataset = dataset
        self._sampler = sampler
        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def prepare_data(self) -> None:
        self._dataset(train=True, download=True)
        self._dataset(train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self._train_set = self._dataset(train=True, transform=self.train_transforms)
            self._sampler.datasource = self._train_set
            self._test_set = self._dataset(train=False, transform=self.test_transforms)

        elif stage == "test" and self._test_set is None:
            self._test_set = self._dataset(train=False, transform=self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader(dataset=self._train_set)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self._val_dataloader is None:
            return self.test_dataloader()

        test_dataloader = self.test_dataloader()
        val_dataloader = self._val_dataloader(dataset=self._train_set)

        return [test_dataloader, val_dataloader]

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._test_dataloader(dataset=self._test_set)
