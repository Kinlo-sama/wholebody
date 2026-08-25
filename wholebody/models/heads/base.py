from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from wholebody.structures.data_sample import PoseDataSample


class BaseHead(nn.Module, ABC):
    """Abstract Base Class for Pose Estimation Prediction Heads."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, feats: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute forward pass from neck/backbone features."""
        pass

    @abstractmethod
    def loss(
        self,
        feats: torch.Tensor,
        batch_data_samples: List[PoseDataSample],
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        pass

    @abstractmethod
    def predict(
        self,
        feats: torch.Tensor,
        batch_data_samples: List[PoseDataSample],
    ) -> List[PoseDataSample]:
        """Decode predictions and populate pred_instances in data_samples."""
        pass
