from typing import Dict


class TransformConfig:
    def __init__(self, config: Dict):
        self._config = config

    @property
    def pad_params(self) -> Dict:
        return self._config.get("pad")

    @property
    def norm_params(self) -> Dict:
        return self._config.get("normalize")

    @property
    def crop_params(self) -> Dict:
        return self._config.get("crop")

    @property
    def cutout_params(self) -> Dict:
        return self._config.get("cutout")
