from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
import torch
import torch.nn as nn


class BaseBackbone(nn.Module, ABC):
    """Abstract Base Class for visual feature extractor backbones."""

    def __init__(self, freeze: bool = False) -> None:
        super().__init__()
        self.freeze = freeze

    def init_weights(self) -> None:
        """Initialize backbone layer weights."""
        pass

    def forward_frozen(self) -> None:
        """Helper to enforce evaluation mode if frozen."""
        if self.freeze:
            self.eval()
            for p in self.parameters():
                p.requires_grad = False
