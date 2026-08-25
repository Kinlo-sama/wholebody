from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from wholebody.core.registry import LOSSES
from wholebody.models.losses.base import BaseLoss


@LOSSES.register("SmoothL1Loss")
class SmoothL1Loss(BaseLoss):
    """Smooth L1 (Huber) regression loss."""

    def __init__(self, beta: float = 1.0, loss_weight: float = 1.0) -> None:
        super().__init__(loss_weight=loss_weight)
        self.beta = beta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        diff = F.smooth_l1_loss(pred, target, beta=self.beta, reduction="none")
        if target_weight is not None:
            if target_weight.ndim < diff.ndim:
                target_weight = target_weight.unsqueeze(-1)
            diff = diff * target_weight
            loss = diff.sum() / target_weight.sum().clamp(min=1.0)
        else:
            loss = diff.mean()
        return loss * self.loss_weight
