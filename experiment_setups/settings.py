import numpy as np
from data_engine.sampler import HighLossSampler
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from argparse import Namespace
from experiment_setups.dataset.cifar10 import CIFAR10Impl
from experiment_setups.model.resnet9 import ResNet9
from data_engine.update_callback import HighLossUpdateCallback
from data_engine.datamodule import HighLossDataModule
from functools import partial

import pytorch_lightning as pl


class ExperimentSettings:
    _sampler = None
    train_batch_size = 512
    _allowed_stale = None
    _std = None

    def __init__(self, args: Namespace):
        self._args = args

    def start(self):
        if self.random_seed is not None:
            pl.seed_everything(self.random_seed)  # wait for new version for workers=True

        self.trainer.fit(self.model, self.datamodule)

    @property
    def datamodule(self) -> HighLossDataModule:
        return CIFAR10Impl(
            self.shuffle_proportion,
            self.sampler,
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader
        )

    @property
    def shuffle_proportion(self) -> float:
        return self._args.shuffle_proportion

    @property
    def sampler(self) -> HighLossSampler:
        if self._sampler is None:
            self._sampler = HighLossSampler(
                database_host=self.database_host,
                allowed_stale=self.allowed_stale,
                cutoff=self.cdf_cutoff,
                robust_std=self.robust_std
            )
        return self._sampler

    @property
    def database_host(self) -> str:
        return ":memory:"

    @property
    def allowed_stale(self) -> int:
        if self._allowed_stale is not None:
            return self._allowed_stale

        allowed_stale = self._args.allowed_stale

        if type(allowed_stale) == list:
            low, high = allowed_stale
            allowed_stale = np.random.randint(low, high, size=100)[self.run]

        self._allowed_stale = allowed_stale
        return allowed_stale

    @property
    def cdf_cutoff(self) -> float:
        return self._args.cutoff

    @property
    def robust_std(self) -> float:
        if self._std is not None:
            return self._std
        std = self._args.std

        if type(std) == list:
            std = np.random.uniform(*std, size=100)[self.run]

        self._std = std
        return std

    @property
    def train_dataloader(self) -> partial:
        return partial(DataLoader, num_workers=12, batch_size=self.train_batch_size)

    @property
    def val_dataloader(self):
        if self.allowed_stale is None:
            return None
        return partial(DataLoader, num_workers=12, batch_size=512)

    @property
    def test_dataloader(self) -> partial:
        return partial(DataLoader, num_workers=12, batch_size=512)

    @property
    def run(self) -> int:
        return self._args.run

    @property
    def trainer(self) -> pl.Trainer:
        return pl.Trainer(
            gpus=1,
            max_epochs=1,
            precision=16,
            num_sanity_val_steps=0,
            weights_summary="full",
            log_gpu_memory="all",
            deterministic=True,
            log_every_n_steps=1,
            profiler=self.profiler,
            callbacks=self.callback,
            logger=self.logger
        )

    @property
    def profiler(self):
        return self._args.profiler

    @property
    def callback(self):
        return [HighLossUpdateCallback(self.sampler)]

    @property
    def logger(self):
        # todo name should have params
        return CSVLogger(save_dir="logs", name=self.experiment_name)

    @property
    def model(self):
        return ResNet9(self.train_batch_size)

    @property
    def random_seed(self) -> int:
        return self._args.seed

    @property
    def experiment_name(self) -> str:

        if self.allowed_stale is None:
            experiment = "traditional"
        elif self.robust_std is None:
            experiment = "selective"
        else:
            experiment = "robust"

        return f"{experiment}_{self.shuffle_proportion}_{self.run}_{self.allowed_stale}_{self.robust_std}"
