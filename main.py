from argparse import ArgumentParser
from cifar10_experiment.experiments import Experiments
import pytorch_lightning as pl


if __name__ == "__main__":
    from logging.config import fileConfig
    import logging
    fileConfig("logging_config.ini", disable_existing_loggers=False)
    LOGGER = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default="baseline",
                        help="Which experiment to run")

    args = parser.parse_args()

    configuration = None
    if args.experiment == "baseline":
        configuration = Experiments.cifar_10_baseline()

    if configuration is None:
        LOGGER.error(f"{args.experiment} does not exist")
        exit(1)

    trainer = pl.Trainer(**configuration.trainer_params)
    trainer.fit(configuration.model, configuration.datamodule)
    trainer.test(configuration.model, datamodule=configuration.datamodule)
