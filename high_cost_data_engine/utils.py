from typing import Dict, Optional, List
from functools import lru_cache
from itertools import chain
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


def resolve_overwrites(baseline: Dict, overwrite: Optional[Dict]) -> Dict:
    """
    marge baseline and overwrites and use overwrite if item exits in both.
    marge is done only on root level of configuration
    e.g.
    >>> baseline = {"a": 1, "b": 2, "c": {"a": 10, "b": 20}}
    >>> overwrite = {"a": 0, "b": 2, "c": {"a": 11, "c": 31}, "d": 3}
    >>> result = resolve_overwrites(baseline, overwrite)
    >>> result == {"a": 0, "b": 2, "c": {"a": 11, "c": 31}, "d": 3}
    true
    """
    if overwrite is None:
        return baseline

    return {**baseline, **overwrite}
