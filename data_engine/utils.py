from typing import Dict, Optional, List
from functools import lru_cache
from itertools import chain
import torch
import yaml


def convert_list_of_tuples_to_list(func):
    def inner(*args, **kwargs) -> List:
        values = func(*args, **kwargs)
        return list(chain.from_iterable(values))
    return inner


@lru_cache
def load_yml(file_path: Optional[str]) -> Optional[Dict]:
    if file_path is None:
        return None

    with open(file_path) as f:
        return yaml.safe_load(f)


def random_shuffle_portion_of_1d_tensor(tensor: torch.Tensor, proportion: float):
    assert 0 <= proportion <= 1, f"Illegal portion size of: {proportion}"
    if proportion == 0:
        return tensor

    shuffle_size = int(tensor.shape[0] * proportion)
    shuffled_indices = torch.randperm(tensor.shape[0])
    selected_proportion = shuffled_indices[:shuffle_size]
    shuffled_proportion = torch.randperm(shuffle_size)
    tensor[selected_proportion] = tensor[shuffled_proportion]
