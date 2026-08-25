import torch
import torch.nn as nn

from wholebody.core.registry import NECKS
from wholebody.models.necks.base import BaseNeck


@NECKS.register("IdentityNeck")
class IdentityNeck(BaseNeck):
    """Pass-through identity neck."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@NECKS.register("ConvNeck")
class ConvNeck(BaseNeck):
    """Single convolution projection neck."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
