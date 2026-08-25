from pathlib import Path
from typing import Any, Dict, Optional

from wholebody.core.registry import HOOKS
from wholebody.engine.checkpointer import CheckpointManager
from wholebody.engine.hooks.base import BaseHook
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.engine.hooks.checkpoint")


@HOOKS.register("CheckpointHook")
class CheckpointHook(BaseHook):
    """Automatically save latest, periodic, and best checkpoints."""

    def __init__(
        self,
        interval: int = 1,
        save_best: Optional[str] = "AP",
        rule: str = "greater",
        max_keep_ckpts: int = 3,
    ) -> None:
        self.interval = interval
        self.save_best_metric = save_best
        self.rule = rule
        self.max_keep_ckpts = max_keep_ckpts
        self.best_score = float("-inf") if rule == "greater" else float("inf")

    def after_train_epoch(self, trainer: Any) -> None:
        # Save latest checkpoint
        ckpt_dir = Path(trainer.work_dir) / "checkpoints"
        latest_path = ckpt_dir / "latest.pth"
        CheckpointManager.save_checkpoint(
            filepath=latest_path,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=trainer.current_epoch,
            iteration=trainer.current_iter,
            meta=trainer.device_manager.get_hardware_info(),
        )

        if trainer.current_epoch % self.interval == 0:
            epoch_path = ckpt_dir / f"epoch_{trainer.current_epoch}.pth"
            CheckpointManager.save_checkpoint(
                filepath=epoch_path,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=trainer.current_epoch,
                iteration=trainer.current_iter,
                meta=trainer.device_manager.get_hardware_info(),
            )

    def after_val_epoch(self, trainer: Any, metrics: Dict[str, float]) -> None:
        if self.save_best_metric is None or self.save_best_metric not in metrics:
            return

        current_score = metrics[self.save_best_metric]
        is_best = (
            (current_score > self.best_score)
            if self.rule == "greater"
            else (current_score < self.best_score)
        )

        if is_best:
            self.best_score = current_score
            ckpt_dir = Path(trainer.work_dir) / "checkpoints"
            best_path = ckpt_dir / "best.pth"
            CheckpointManager.save_checkpoint(
                filepath=best_path,
                model=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                epoch=trainer.current_epoch,
                iteration=trainer.current_iter,
                metrics=metrics,
                meta=trainer.device_manager.get_hardware_info(),
            )
            logger.info(f"New best {self.save_best_metric}: {current_score:.4f} -> Saved to {best_path}")
