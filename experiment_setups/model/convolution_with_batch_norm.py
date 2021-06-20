from torch import nn, Tensor


class ConvWithBatchNorm(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ConvWithBatchNorm, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
