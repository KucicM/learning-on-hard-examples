from typing import Any
from data_engine.sampler import HighLossSampler
import pytorch_lightning as pl


class HighLossUpdateCallback(pl.Callback):
    def __init__(self, sampler: HighLossSampler):
        self._sampler = sampler

    def on_validation_start(self, trainer, pl_module: pl.LightningModule) -> None:
        self._sampler.inference_mode = False

    def on_validation_end(self, trainer, pl_module: pl.LightningModule) -> None:
        self._sampler.inference_mode = True

    def on_validation_batch_end(
        self, trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if batch_idx == 1:
            losses = outputs.tolist()
            self._sampler.update_losses(losses)
