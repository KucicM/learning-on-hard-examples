from typing import Tuple

import torch


class Cutout:
    """
    Simplified Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, size=8):
        assert size > 0
        self.size = size

    def __call__(self, tensor):
        _, height, width = tensor.shape
        x, y = self._get_random_xy(height, width)

        cutout = torch.ones(height, width)
        cutout[x:x+self.size, y:y+self.size] = 0

        return tensor * cutout

    def _get_random_xy(self, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(low=0, high=(height - self.size), size=(1,))
        y = torch.randint(low=0, high=(width - self.size), size=(1,))
        return x, y
