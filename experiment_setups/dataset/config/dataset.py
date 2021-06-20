from dataclasses import dataclass
from typing import Dict
from high_cost_data_engine import config


@dataclass(init=False)
class DatasetConfig(config.Dataset):
    pad_params: Dict
    crop_params: Dict
    cutout_params: Dict

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        transforms = config["data_transformations"]
        self.pad_params = transforms["pad"]
        self.crop_params = transforms["crop"]
        self.cutout_params = transforms["cutout"]
