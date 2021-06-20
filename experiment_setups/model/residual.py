from torch import nn, Tensor
from experiment_setups.model.convolution_with_batch_norm import ConvWithBatchNorm


class ResidualBlock(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Identity(),
            ConvWithBatchNorm(in_channels=num_channels, out_channels=num_channels),
            ConvWithBatchNorm(in_channels=num_channels, out_channels=num_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x).add(x)
