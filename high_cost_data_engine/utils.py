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
    e.g.
    >>> baseline = {"a": 1, "b": 2, "c": {"a": 10, "b": 20}}
    >>> overwrite = {"a": 0, "b": 2, "c": {"a": 11, "c": 31}, "d": 3}
    >>> result = resolve_overwrites(baseline, overwrite)
    >>> result == {"a": 0, "b": 2, "c": {"a": 11, "b": 20, "c": 31}, "d": 3}
    true
    """
    if overwrite is None:
        return baseline

    for k, v in baseline.items():
        if isinstance(v, dict) and k in overwrite:
            overwrite[k] = resolve_overwrites(v, overwrite[k])

    return {**baseline, **overwrite}


def rename_columns_in_csv(log_dir: str, metrics_name: str = "metrics") -> None:
    file_name = f"{log_dir}/{metrics_name}.csv"
    with open(file_name) as f:
        lines = f.readlines()

    old_names = lines[0].strip().split(",")
    new_names = [replace_name(name) for name in old_names]
    new_line = f"{','.join(new_names)}\n"
    lines[0] = new_line

    with open(file_name, "w") as f:
        f.writelines(lines)


def replace_name(name):
    if name in {"epoch", "step"}:
        return name

    typ = "train" if name.endswith("0") else "test"
    name = name.split("/")[0]
    return f"{typ}_{name}"
