from torch import nn, Tensor
from experiment_setups.model.convolution_with_batch_norm import ConvWithBatchNorm
from experiment_setups.model.residual import ResidualBlock
from experiment_setups.model.scale import Mul


class ResNet9Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            ConvWithBatchNorm(in_channels=3, out_channels=64),
            ConvWithBatchNorm(in_channels=64, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(num_channels=128),
            ConvWithBatchNorm(in_channels=128, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvWithBatchNorm(in_channels=256, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(num_channels=512),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=10, bias=False),
            Mul(0.125),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
