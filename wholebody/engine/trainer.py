import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wholebody.core.device import DeviceManager
from wholebody.core.registry import HOOKS
from wholebody.engine.evaluator import Evaluator
from wholebody.engine.hooks.base import BaseHook
from wholebody.structures.data_sample import PoseDataSample
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.engine.trainer")


class Trainer:
    """Unified, research-first Training Engine for Whole-Body Pose Estimation.
    
    Supports:
      - Apple Silicon (MPS), CUDA, and CPU acceleration
      - AMP mixed precision autocast
      - Dynamic hook lifecycle events
      - Periodic evaluation and early stopping
      - Gradient clipping and accumulation
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        evaluator: Optional[Evaluator] = None,
        device_manager: Optional[DeviceManager] = None,
        hooks: Optional[List[Union[Dict[str, Any], BaseHook]]] = None,
        work_dir: Union[str, Path] = "./work_dirs/exp",
        max_epochs: int = 20,
        val_interval: int = 1,
        grad_clip: Optional[float] = 10.0,
    ) -> None:
        self.device_manager = device_manager or DeviceManager()
        self.model = self.device_manager.to_device(model)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_epochs = max_epochs
        self.val_interval = val_interval
        self.grad_clip = grad_clip

        self.current_epoch = 0
        self.current_iter = 0
        self.max_iters_per_epoch = len(train_dataloader)
        self.stop_training = False

        # Build hooks
        self.hooks: List[BaseHook] = []
        if hooks:
            for h in hooks:
                if isinstance(h, dict):
                    self.hooks.append(HOOKS.build(h))
                elif isinstance(h, BaseHook):
                    self.hooks.append(h)
                else:
                    raise TypeError(f"Invalid hook: {h}")

        # Save experiment environment metadata
        self._save_metadata()

    def _save_metadata(self) -> None:
        meta = {
            "max_epochs": self.max_epochs,
            "val_interval": self.val_interval,
            "hardware": self.device_manager.get_hardware_info(),
        }
        meta_file = self.work_dir / "experiment_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _call_hook(self, fn_name: str, *args, **kwargs) -> None:
        for hook in self.hooks:
            fn = getattr(hook, fn_name, None)
            if callable(fn):
                fn(self, *args, **kwargs)

    def train(self) -> None:
        """Execute the entire multi-epoch training loop."""
        logger.info(f"Starting training for {self.max_epochs} epochs on {self.device_manager.get_device()}...")
        self._call_hook("before_train")

        for epoch in range(1, self.max_epochs + 1):
            if self.stop_training:
                break

            self.current_epoch = epoch
            self._call_hook("before_train_epoch")

            # Training epoch
            self.model.train()
            for batch_idx, batch in enumerate(self.train_dataloader):
                self._call_hook("before_train_iter", batch_idx, batch)

                inputs = self.device_manager.to_device(batch["inputs"])
                data_samples: List[PoseDataSample] = batch["data_samples"]

                if self.optimizer is not None:
                    self.optimizer.zero_grad()

                with self.device_manager.autocast():
                    loss_dict = self.model(
                        inputs=inputs,
                        data_samples=data_samples,
                        mode="train",
                    )
                    # Aggregate total loss
                    total_loss = sum(v for v in loss_dict.values())

                total_loss.backward()

                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                if self.optimizer is not None:
                    self.optimizer.step()

                self.current_iter += 1
                loss_dict["total_loss"] = total_loss.detach()
                self._call_hook("after_train_iter", batch_idx, batch, loss_dict)

            if self.scheduler is not None:
                self.scheduler.step()

            self._call_hook("after_train_epoch")

            # Validation epoch
            if self.val_dataloader is not None and self.evaluator is not None and (epoch % self.val_interval == 0 or epoch == self.max_epochs):
                self._call_hook("before_val_epoch")
                val_metrics = self.evaluator.evaluate(
                    model=self.model,
                    dataloader=self.val_dataloader,
                    device_manager=self.device_manager,
                )
                self._call_hook("after_val_epoch", val_metrics)

        self._call_hook("after_train")
        logger.info("Training completed successfully!")
