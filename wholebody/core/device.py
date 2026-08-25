import os
import platform
from contextlib import contextmanager
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.core.device")


class DeviceManager:
    """Centralized hardware and device abstraction manager.
    
    Supports first-class Apple Silicon (MPS), NVIDIA CUDA, and CPU backends.
    Handles device placement, mixed precision (AMP), memory monitoring,
    and graceful fallback without scattering hardware checks across the codebase.
    """

    def __init__(
        self,
        device: str = "auto",
        precision: str = "fp32",
        allow_cpu_fallback: bool = True,
    ) -> None:
        self.device_str = device.lower()
        self.precision_str = precision.lower()
        self.allow_cpu_fallback = allow_cpu_fallback
        self.torch_device = self._resolve_device(self.device_str)
        self._setup_mps_fallback()
        self._validate_precision()

    def _resolve_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            if torch.cuda.is_available():
                resolved = torch.device("cuda:0")
                logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
                return resolved
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved = torch.device("mps")
                logger.info("Auto-selected Apple Silicon MPS device")
                return resolved
            else:
                resolved = torch.device("cpu")
                logger.info("Auto-selected CPU device (no GPU / MPS detected)")
                return resolved

        elif requested_device.startswith("cuda"):
            if not torch.cuda.is_available():
                if self.allow_cpu_fallback:
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                    return torch.device("cpu")
                else:
                    raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
            return torch.device(requested_device)

        elif requested_device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                if self.allow_cpu_fallback:
                    logger.warning("MPS requested but not available on this system. Falling back to CPU.")
                    return torch.device("cpu")
                else:
                    raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")
            return torch.device("mps")

        elif requested_device == "cpu":
            return torch.device("cpu")

        else:
            raise ValueError(f"Unknown device specification: '{requested_device}'. Choose from: 'auto', 'cuda', 'mps', 'cpu'.")

    def _setup_mps_fallback(self) -> None:
        """Configure PyTorch environment variable for MPS fallback if allowed."""
        if self.torch_device.type == "mps" and self.allow_cpu_fallback:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif self.torch_device.type == "mps" and not self.allow_cpu_fallback:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    def _validate_precision(self) -> None:
        """Ensure the chosen precision is safe for the active device."""
        if self.precision_str == "fp16":
            if self.torch_device.type == "cpu":
                logger.warning("FP16 is not fully optimized for CPU training. Using FP32 or BF16 is recommended.")
        elif self.precision_str == "bf16":
            if self.torch_device.type == "mps":
                # BF16 on MPS in PyTorch has partial operator coverage
                logger.warning("BF16 on Apple Silicon MPS might have unsupported ops. Consider FP16 or FP32.")

    def get_device(self) -> torch.device:
        """Get the resolved torch.device instance."""
        return self.torch_device

    def to_device(self, obj: Any) -> Any:
        """Recursively move tensors, modules, or collections to the active device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(self.torch_device)
        elif isinstance(obj, nn.Module):
            return obj.to(self.torch_device)
        elif isinstance(obj, dict):
            return {k: self.to_device(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.to_device(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.to_device(v) for v in obj)
        elif hasattr(obj, "to"):
            try:
                return obj.to(self.torch_device)
            except Exception:
                return obj
        return obj

    @contextmanager
    def autocast(self):
        """Context manager for automatic mixed precision (AMP) across CUDA, MPS, and CPU."""
        if self.precision_str == "fp32":
            yield
            return

        device_type = self.torch_device.type
        amp_dtype = torch.float16 if self.precision_str == "fp16" else torch.bfloat16

        # PyTorch supports device_type='mps' in torch.autocast from 2.0+
        if device_type in ("cuda", "mps", "cpu"):
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                yield
        else:
            yield

    def empty_cache(self) -> None:
        """Safely release unused cached memory on active accelerator."""
        if self.torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.torch_device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def synchronize(self) -> None:
        """Wait for all kernels in current device stream to finish."""
        if self.torch_device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif self.torch_device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass

    def get_hardware_info(self) -> Dict[str, Any]:
        """Return comprehensive metadata regarding system hardware and active backend."""
        info: Dict[str, Any] = {
            "device": str(self.torch_device),
            "backend": self.torch_device.type,
            "precision": self.precision_str,
            "allow_cpu_fallback": self.allow_cpu_fallback,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "pytorch_version": torch.__version__,
            "python_version": platform.python_version(),
        }

        if self.torch_device.type == "cuda" and torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(self.torch_device)
            info["cuda_count"] = torch.cuda.device_count()
            info["cuda_capability"] = torch.cuda.get_device_capability(self.torch_device)

        elif self.torch_device.type == "mps":
            info["mps_is_built"] = torch.backends.mps.is_built()
            info["mps_is_available"] = torch.backends.mps.is_available()
            info["mps_allocated_memory_bytes"] = (
                torch.mps.current_allocated_memory()
                if hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory")
                else None
            )

        return info

    def __repr__(self) -> str:
        return f"DeviceManager(device='{self.torch_device}', precision='{self.precision_str}', fallback={self.allow_cpu_fallback})"
