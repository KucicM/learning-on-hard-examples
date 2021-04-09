from argparse import ArgumentParser
from cifar10_experiment.experiments import Experiments
import pytorch_lightning as pl
import time
from pytorch_lightning.loggers import TensorBoardLogger


def run_experiment(config, exp_name):
    logger = TensorBoardLogger("logs", name=exp_name, log_graph=True)

    st = time.time()
    trainer = pl.Trainer(**config.trainer_params, logger=logger)
    model = config.model
    trainer.fit(model, config.datamodule)
    LOGGER.info(f"END TIME {time.time() - st:.3f} seconds")


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
    elif args.experiment == "selective-backprop":
        configuration = Experiments.cifar_10_selective_backprop()
    else:
        LOGGER.error(f"{args.experiment} does not exist")
        exit(1)

    run_experiment(configuration, args.experiment)
