from typing import List
from argparse import ArgumentParser
from experiment_setups.settings import ExperimentSettings
from logging.config import fileConfig
import logging

fileConfig("logging_config.ini", disable_existing_loggers=False)
LOGGER = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, default="traditional", help="Which method to run")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--shuffle-proportion", type=float, default=0,
                        help="What proportion of labels will be shuffled")

    parser.add_argument("--allowed-stale", type=List[int], default=None)
    parser.add_argument("--cutoff", type=float, default=0.3)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--std", type=List[float], default=None)

    parser.add_argument("--profiler", type=str, default=None, help="Available advanced, simple, pytorch")

    args = parser.parse_args()
    LOGGER.info(f"Running with arguments: {args}")

    settings = ExperimentSettings(args)
    settings.start()
