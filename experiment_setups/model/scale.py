from torch import nn, Tensor


class Mul(nn.Module):
    def __init__(self, weight: float) -> None:
        super(Mul, self).__init__()
        self.weight = weight

    def __call__(self, x: Tensor) -> Tensor:
        return x * self.weight
