import torch
from torch import nn


class Resnet9(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
                ConvWithBatchNorm(in_channels=3, out_channels=64),
                ConvWithBatchNorm(in_channels=64, out_channels=128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualBlock(n_ch=128),
                ConvWithBatchNorm(in_channels=128, out_channels=256),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ConvWithBatchNorm(in_channels=256, out_channels=512),
                nn.MaxPool2d(kernel_size=2, stride=2),
                ResidualBlock(n_ch=512),
                nn.MaxPool2d(kernel_size=4),
                nn.Flatten(),
                nn.Linear(in_features=512, out_features=10, bias=False),
                Mul(0.125),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ConvWithBatchNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvWithBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_ch: int) -> None:
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Identity(),
            ConvWithBatchNorm(in_channels=n_ch, out_channels=n_ch),
            ConvWithBatchNorm(in_channels=n_ch, out_channels=n_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).add(x)


class Mul(nn.Module):
    def __init__(self, weight: float) -> None:
        super(Mul, self).__init__()
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class StepOptimizer():
    def __init__(self, weights, optimizer, **optim_params) -> None:
        self.params = optim_params
        self.step_count = 0
        self.optim = optimizer(weights, **self.update())

    def update(self):
        return {k: v(self.step_count) if callable(v) else v for k, v in self.params.items()}

    def step(self):
        self.step_count += 1
        self.optim.param_groups[0].update(**self.update())
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()
