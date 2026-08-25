from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from wholebody.core.registry import LOSSES
from wholebody.models.losses.base import BaseLoss


@LOSSES.register("KeypointMSELoss")
class KeypointMSELoss(BaseLoss):
    """Mean Squared Error Loss for 2D heatmaps with visibility weight masking."""

    def __init__(self, use_target_weight: bool = True, loss_weight: float = 1.0) -> None:
        super().__init__(loss_weight=loss_weight)
        self.use_target_weight = use_target_weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_weight: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute weighted heatmap MSE.
        
        Args:
            pred: (B, K, H, W) float tensor
            target: (B, K, H, W) float tensor
            target_weight: (B, K) float tensor
        """
        batch_size, num_kpts = pred.shape[:2]
        pred_flat = pred.reshape((batch_size, num_kpts, -1))
        target_flat = target.reshape((batch_size, num_kpts, -1))

        diff = (pred_flat - target_flat) ** 2

        if self.use_target_weight and target_weight is not None:
            # target_weight: (B, K) -> (B, K, 1)
            weight = target_weight.unsqueeze(-1)
            diff = diff * weight
            valid_kpts = weight.sum().clamp(min=1.0)
            loss = diff.sum() / (2.0 * valid_kpts * pred.shape[2] * pred.shape[3])
        else:
            loss = 0.5 * diff.mean()

        return loss * self.loss_weight
