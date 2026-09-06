from wholebody.models.losses.base import BaseLoss, CombinedLoss
from wholebody.models.losses.mse_loss import KeypointMSELoss
from wholebody.models.losses.smooth_l1 import SmoothL1Loss

__all__ = [
    "BaseLoss",
    "CombinedLoss",
    "KeypointMSELoss",
    "SmoothL1Loss",
]
from .fea_loss import FeaLoss
from .kd_loss import KDLoss
from .kl_discret_loss import KLDiscretLoss

__all__.extend(["FeaLoss", "KDLoss", "KLDiscretLoss"])
