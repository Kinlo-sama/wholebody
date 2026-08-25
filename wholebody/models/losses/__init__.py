from wholebody.models.losses.base import BaseLoss, CombinedLoss
from wholebody.models.losses.mse_loss import KeypointMSELoss
from wholebody.models.losses.smooth_l1 import SmoothL1Loss

__all__ = [
    "BaseLoss",
    "CombinedLoss",
    "KeypointMSELoss",
    "SmoothL1Loss",
]
