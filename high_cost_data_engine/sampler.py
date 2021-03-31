import logging
from abc import ABC
from pytorch_lightning import LightningModule, Callback
from torch.utils.data import Sampler, SequentialSampler
from typing import Optional, Sized, Any
from high_cost_data_engine.repository import HighCostRepository
LOGGER = logging.getLogger(__name__)


class HighCostSampler(Sampler, Callback, ABC):
    def __init__(self, data_source: Optional[Sized], *args, **kwargs):
        super().__init__(data_source)
        self._cost_provider = HighCostRepository()
        self._inner = SequentialSampler(data_source)

    def update_costs(self):
        LOGGER.info("Testing")

    def __iter__(self):
        return iter(self._inner)

    def __len__(self):
        return len(self._inner)
