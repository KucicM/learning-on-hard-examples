from torch.functional import F

import pytorch_lightning as pl
import torch


class HighCostModule(pl.LightningModule):
    _loss_fun: F = ...

    def __init__(self, loss_function: F) -> None:
        super().__init__()
        self._loss_fun = loss_function

    def validation_step(self, batch, _: int, datalaoder_idx: int = None) -> None:
        x, y = batch
        out = self.model(x)

        loss = self._loss_fun(out, y, reduction="none")
        loss_sum = loss.sum().item()
        _, predictions = torch.max(out.data, 1)
        accuracy = torch.sum(y == predictions).item() / len(y)

        if datalaoder_idx == 0:
            self.log("train_loss", loss_sum, on_step=True, add_dataloader_idx=False)
            self.log("train_accuracy", accuracy, on_step=True, add_dataloader_idx=False)
        else:
            self.log("test_loss", loss_sum, on_step=True, add_dataloader_idx=False)
            self.log("test_accuracy", accuracy, on_step=True, add_dataloader_idx=False)

    def training_step(self, batch, _: int) -> torch.Tensor:
        x, y = batch
        out = self.model(x)
        return self._loss_fun(out, y, reduction="sum")
