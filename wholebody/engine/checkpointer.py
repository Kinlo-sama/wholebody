import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from wholebody.utils.logger import get_logger
from wholebody.utils.model_utils import load_partial_state_dict

logger = get_logger("wholebody.engine.checkpointer")


class CheckpointManager:
    """Manages cross-platform checkpoint serialization and safe loading."""

    @staticmethod
    def save_checkpoint(
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        iteration: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save complete training state and hardware metadata into a checkpoint file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: Dict[str, Any] = {
            "epoch": epoch,
            "iteration": iteration,
            "state_dict": model.state_dict(),
            "metrics": metrics or {},
            "meta": meta or {},
        }

        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()

        torch.save(checkpoint, str(filepath))
        logger.info(f"Saved checkpoint to: {filepath}")
        return str(filepath)

    @staticmethod
    def load_checkpoint(
        filepath: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        strict: bool = False,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Load a checkpoint safely across different accelerators (MPS/CUDA/CPU)."""
        filepath = Path(filepath).resolve()
        if not filepath.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        # Resolve target device for map_location
        map_loc = str(device) if not isinstance(device, torch.device) else str(device)
        try:
            checkpoint = torch.load(str(filepath), map_location=map_loc, weights_only=False)
        except TypeError:
            checkpoint = torch.load(str(filepath), map_location=map_loc)

        state_dict = checkpoint.get("state_dict", checkpoint)
        matched, missing, unexpected = load_partial_state_dict(
            model=model,
            state_dict=state_dict,
            strict=strict,
            logger=logger,
        )

        if optimizer is not None and "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
                logger.info("Restored optimizer state from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")

        if scheduler is not None and "scheduler" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
                logger.info("Restored scheduler state from checkpoint.")
            except Exception as e:
                logger.warning(f"Could not restore scheduler state: {e}")

        return {
            "epoch": checkpoint.get("epoch", 0),
            "iteration": checkpoint.get("iteration", 0),
            "metrics": checkpoint.get("metrics", {}),
            "meta": checkpoint.get("meta", {}),
            "matched_keys": matched,
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
