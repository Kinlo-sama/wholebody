import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seed across Python, NumPy, PyTorch (CPU, CUDA, MPS).
    
    Args:
        seed: Random seed integer.
        deterministic: If True, enables deterministic operations when available.
            Note: On Apple Silicon (MPS), deterministic operations are limited and
            some non-deterministic fallbacks may still occur.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
