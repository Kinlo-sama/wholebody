from typing import List, Tuple
import torch
import torch.nn as nn

from wholebody.core.registry import BACKBONES
from wholebody.models.backbones.base import BaseBackbone


@BACKBONES.register("SimpleCNN")
class SimpleCNN(BaseBackbone):
    """Lightweight convolutional backbone for testing, MVP validation, and fast baselines.
    
    Extracts deep 2D feature maps from input images.
    """

    def __init__(
        self,
        in_channels: int = 3,
        stage_channels: Tuple[int, ...] = (64, 128, 256, 256),
        freeze: bool = False,
    ) -> None:
        super().__init__(freeze=freeze)
        self.in_channels = in_channels
        self.stage_channels = stage_channels

        layers: List[nn.Module] = []
        curr_in = in_channels
        for out_c in stage_channels:
            layers.extend([
                nn.Conv2d(curr_in, out_c, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            curr_in = out_c

        self.features = nn.Sequential(*layers)
        self.out_channels = stage_channels[-1]

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.freeze and self.training:
            self.eval()
        return self.features(x)
