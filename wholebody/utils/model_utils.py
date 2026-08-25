from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


def freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a PyTorch module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a PyTorch module."""
    for param in module.parameters():
        param.requires_grad = True
    module.train()


def count_parameters(module: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters in a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def load_partial_state_dict(
    model: nn.Module,
    state_dict: Dict[str, Any],
    strict: bool = False,
    ignore_shape_mismatch: bool = True,
    logger: Optional[Any] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Robustly load a state dict into a model, supporting fine-tuning and partial loading.
    
    Handles:
      - Missing keys (parameters in model not in checkpoint)
      - Unexpected keys (parameters in checkpoint not in model)
      - Shape mismatches (e.g. changing num_keypoints from 17 to 133 in a head)
    
    Returns:
        Tuple of (matched_keys, missing_keys, unexpected_keys)
    """
    model_state = model.state_dict()
    matched_keys: List[str] = []
    missing_keys: List[str] = []
    unexpected_keys: List[str] = []
    mismatched_shape_keys: List[str] = []

    clean_state_dict: Dict[str, Any] = {}

    for key, param in state_dict.items():
        # Handle possible 'module.' prefix from DDP
        clean_key = key.replace("module.", "") if key.startswith("module.") else key
        if clean_key in model_state:
            if model_state[clean_key].shape == param.shape:
                clean_state_dict[clean_key] = param
                matched_keys.append(clean_key)
            else:
                mismatched_shape_keys.append(
                    f"{clean_key} (expected {tuple(model_state[clean_key].shape)}, got {tuple(param.shape)})"
                )
                if not ignore_shape_mismatch:
                    raise RuntimeError(f"Shape mismatch for key {clean_key}: {model_state[clean_key].shape} vs {param.shape}")
        else:
            unexpected_keys.append(clean_key)

    for key in model_state.keys():
        if key not in clean_state_dict:
            missing_keys.append(key)

    if strict and (missing_keys or unexpected_keys or mismatched_shape_keys):
        raise RuntimeError(
            f"Strict loading failed! Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}, Mismatched: {len(mismatched_shape_keys)}"
        )

    model.load_state_dict(clean_state_dict, strict=False)

    if logger is not None:
        logger.info(f"Loaded {len(matched_keys)} / {len(model_state)} model weights successfully.")
        if mismatched_shape_keys:
            logger.warning(f"Skipped {len(mismatched_shape_keys)} keys due to shape mismatch: {mismatched_shape_keys}")
        if missing_keys:
            logger.info(f"Uninitialized / missing keys in model ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            logger.debug(f"Unexpected keys in checkpoint ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")

    return matched_keys, missing_keys, unexpected_keys
