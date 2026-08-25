import time
from typing import Any, Dict, Optional

from wholebody.core.registry import HOOKS
from wholebody.engine.hooks.base import BaseHook
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.engine.hooks.logger")


@HOOKS.register("TextLoggerHook")
class TextLoggerHook(BaseHook):
    """Logs iteration losses, time, ETA, and memory usage to console."""

    def __init__(self, interval: int = 10) -> None:
        self.interval = interval
        self.iter_start_time: float = 0.0

    def before_train_epoch(self, trainer: Any) -> None:
        self.epoch_start_time = time.time()

    def before_train_iter(self, trainer: Any, batch_idx: int, data: Dict[str, Any]) -> None:
        self.iter_start_time = time.time()

    def after_train_iter(
        self,
        trainer: Any,
        batch_idx: int,
        data: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> None:
        if batch_idx % self.interval != 0 and batch_idx != trainer.max_iters_per_epoch - 1:
            return

        iter_time = time.time() - self.iter_start_time
        lr = trainer.optimizer.param_groups[0]["lr"]

        loss_str = " | ".join([f"{k}: {v.item():.4f}" if hasattr(v, "item") else f"{k}: {v:.4f}" for k, v in outputs.items()])
        logger.info(
            f"Epoch [{trainer.current_epoch}/{trainer.max_epochs}] "
            f"Iter [{batch_idx + 1}/{trainer.max_iters_per_epoch}] "
            f"lr: {lr:.6f} | time: {iter_time:.3f}s | {loss_str}"
        )

    def after_val_epoch(self, trainer: Any, metrics: Dict[str, float]) -> None:
        metrics_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        logger.info(f"Validation Epoch [{trainer.current_epoch}/{trainer.max_epochs}] -> {metrics_str}")
