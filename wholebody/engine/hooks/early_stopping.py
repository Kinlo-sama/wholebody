from typing import Any, Dict

from wholebody.core.registry import HOOKS
from wholebody.engine.hooks.base import BaseHook
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.engine.hooks.early_stopping")


@HOOKS.register("EarlyStoppingHook")
class EarlyStoppingHook(BaseHook):
    """Halts training if validation metric does not improve after patience epochs."""

    def __init__(self, metric: str = "AP", patience: int = 10, rule: str = "greater") -> None:
        self.metric = metric
        self.patience = patience
        self.rule = rule
        self.best_score = float("-inf") if rule == "greater" else float("inf")
        self.wait = 0

    def after_val_epoch(self, trainer: Any, metrics: Dict[str, float]) -> None:
        if self.metric not in metrics:
            return

        current = metrics[self.metric]
        improved = (current > self.best_score) if self.rule == "greater" else (current < self.best_score)

        if improved:
            self.best_score = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logger.warning(f"Early stopping triggered! No improvement in {self.metric} for {self.patience} epochs.")
                trainer.stop_training = True
