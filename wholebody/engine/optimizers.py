from typing import Any, Dict, Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)

from wholebody.core.registry import OPTIMIZERS, SCHEDULERS


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> optim.Optimizer:
    """Build PyTorch optimizer from config dictionary."""
    cfg = dict(cfg)
    opt_type = cfg.pop("type", "AdamW")
    lr = cfg.pop("lr", 1e-3)
    weight_decay = cfg.pop("weight_decay", 1e-4)

    # Separate decay and no-decay parameters (norms & biases)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if opt_type == "AdamW":
        return optim.AdamW(param_groups, lr=lr, **cfg)
    elif opt_type == "Adam":
        return optim.Adam(param_groups, lr=lr, **cfg)
    elif opt_type == "SGD":
        return optim.SGD(param_groups, lr=lr, **cfg)
    else:
        # Fallback to PyTorch optimizer class
        opt_cls = getattr(optim, opt_type)
        return opt_cls(param_groups, lr=lr, **cfg)


def build_scheduler(
    optimizer: optim.Optimizer,
    cfg: Dict[str, Any],
    total_epochs: int = 100,
) -> _LRScheduler:
    """Build PyTorch learning rate scheduler from config."""
    cfg = dict(cfg)
    sched_type = cfg.pop("type", "CosineAnnealingLR")
    warmup_epochs = cfg.pop("warmup_epochs", 0)

    main_scheduler: _LRScheduler
    if sched_type == "CosineAnnealingLR":
        t_max = cfg.pop("T_max", total_epochs - warmup_epochs)
        eta_min = cfg.pop("eta_min", 1e-6)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, t_max), eta_min=eta_min, **cfg)
    elif sched_type == "MultiStepLR":
        milestones = cfg.pop("milestones", [int(total_epochs * 0.7), int(total_epochs * 0.9)])
        gamma = cfg.pop("gamma", 0.1)
        main_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma, **cfg)
    else:
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    if warmup_epochs > 0:
        warmup_sched = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup_sched, main_scheduler], milestones=[warmup_epochs])

    return main_scheduler
