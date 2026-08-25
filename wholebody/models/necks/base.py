from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseNeck(nn.Module, ABC):
    """Abstract Base Class for feature pyramid and multi-scale neck modules."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
