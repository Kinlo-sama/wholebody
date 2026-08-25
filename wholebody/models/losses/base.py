from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from wholebody.core.registry import LOSSES


class BaseLoss(nn.Module, ABC):
    """Abstract Base Class for keypoint and heatmap loss functions."""

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        pass


@LOSSES.register("CombinedLoss")
class CombinedLoss(BaseLoss):
    """Loss module combining multiple sub-losses with individual weighting."""

    def __init__(self, losses: Dict[str, Union[Dict[str, Any], BaseLoss]]) -> None:
        super().__init__(loss_weight=1.0)
        self.loss_modules = nn.ModuleDict()
        for name, cfg_or_module in losses.items():
            if isinstance(cfg_or_module, dict):
                self.loss_modules[name] = LOSSES.build(cfg_or_module)
            else:
                self.loss_modules[name] = cfg_or_module

    def forward(
        self,
        pred: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: Union[torch.Tensor, Dict[str, torch.Tensor]],
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        loss_dict: Dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else next(iter(pred.values())).device)

        for name, module in self.loss_modules.items():
            p = pred[name] if isinstance(pred, dict) and name in pred else pred
            t = target[name] if isinstance(target, dict) and name in target else target
            sub_loss = module(p, t, **kwargs) * module.loss_weight
            loss_dict[name] = sub_loss
            total_loss = total_loss + sub_loss

        loss_dict["loss_total"] = total_loss
        return loss_dict
