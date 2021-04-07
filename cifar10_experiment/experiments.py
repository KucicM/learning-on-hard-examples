from typing import Dict
from enum import Enum
from cifar10_experiment import config
from cifar10_experiment.datamodule import CIFAR10DataModule
from cifar10_experiment.model import ResNet9
from high_cost_data_engine import utils
import pytorch_lightning as pl
import logging
LOGGER = logging.getLogger(__name__)


class ExperimentSetup:
    _model = None
    _datamodule = None
    __model_config: config.ResNet9Config = None
    __dataset_config: config.DatasetConfig = None
    __dataloader_config: config.DataLoaderConfig = None
    __optimizer_config: config.OptimizerConfig = None

    def __init__(self, baseline_config_path: str, overwrite_config_path: str = None) -> None:
        baseline = utils.load_yml(baseline_config_path)
        overwrite = utils.load_yml(overwrite_config_path)
        self._config = utils.resolve_overwrites(baseline, overwrite)

    @property
    def trainer_params(self) -> Dict:
        return self._config["trainer"]

    @property
    def model(self) -> pl.LightningModule:
        if self._model is not None:
            return self._model

        self._model = ResNet9(self._model_config, self._optimizer_config, self._batch_size)
        LOGGER.info(self._model)
        return self._model

    @property
    def datamodule(self) -> pl.LightningDataModule:
        if self._datamodule is not None:
            return self._datamodule

        self._datamodule = CIFAR10DataModule(self._dataset_config, self._dataloader_config)
        return self._datamodule

    @property
    def _model_config(self) -> config.ResNet9Config:
        if self.__model_config is not None:
            return self.__model_config

        self.__model_config = config.ResNet9Config(self._config)
        return self.__model_config

    @property
    def _optimizer_config(self) -> config.OptimizerConfig:
        if self.__optimizer_config is not None:
            return self.__optimizer_config

        self.__optimizer_config = config.OptimizerConfig(self._config)
        return self.__optimizer_config

    @property
    def _dataset_config(self) -> config.DatasetConfig:
        if self.__dataset_config is not None:
            return self.__dataset_config

        self.__dataset_config = config.DatasetConfig(self._config)
        return self.__dataset_config

    @property
    def _dataloader_config(self) -> config.DataLoaderConfig:
        if self.__dataloader_config is not None:
            return self.__dataloader_config

        self.__dataloader_config = config.DataLoaderConfig(self._config)
        return self.__dataloader_config

    @property
    def _batch_size(self) -> int:
        return self._dataloader_config.batch_size


class Experiments(Enum):
    cifar_10_baseline = ExperimentSetup("cifar10_experiment/baseline_configuration.yml")
    cifar_10_biggest_losers = ExperimentSetup("cifar10_experiment/baseline_configuration.yml",
                                              "cifar10_experiment/biggest_losers_configuration.yml")

    def __call__(self, *args, **kwargs) -> ExperimentSetup:
        return self.value
