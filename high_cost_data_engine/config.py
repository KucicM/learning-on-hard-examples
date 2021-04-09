from dataclasses import dataclass
from typing import Dict


@dataclass(init=False)
class Dataset:
    data_path: str
    normalization_values: Dict
    use_selective_backprop: bool

    def __init__(self, config: Dict) -> None:
        dataset_config = config["dataset"]
        self.data_path = dataset_config["data_path"]
        self.normalization_values = dataset_config["normalize"]
        self.use_selective_backprop = dataset_config.get("use_selective_backprop", False)


@dataclass(init=False)
class DataLoader:
    train_params: Dict
    test_params: Dict
    val_params: Dict

    def __init__(self, config: Dict):
        dataloaders = config["dataloaders"]
        self.train_params = dataloaders["train"]
        self.test_params = dataloaders["test"]
        self.val_params = dataloaders.get("val", None)

    @property
    def train_batch_size(self) -> int:
        return self.train_params["batch_size"]
