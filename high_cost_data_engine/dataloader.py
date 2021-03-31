from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from high_cost_data_engine.sampler import HighCostSampler
import logging
LOGGER = logging.getLogger(__name__)


class HighCostDataLoader(DataLoader):
    def __init__(self, dataset: Dataset, num_workers: int, batch_size: int, **kwargs) -> None:

        sampler = HighCostSampler(dataset)
        super(HighCostDataLoader, self).__init__(
            dataset, num_workers=num_workers, batch_size=batch_size, sampler=sampler, **kwargs)
