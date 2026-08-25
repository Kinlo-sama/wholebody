import argparse
import sys
from pathlib import Path
from torch.utils.data import DataLoader

from wholebody.core.config import Config
from wholebody.core.device import DeviceManager
from wholebody.core.registry import DATASETS, MODELS
from wholebody.datasets.base import BasePoseDataset
from wholebody.engine.evaluator import Evaluator
from wholebody.engine.optimizers import build_optimizer, build_scheduler
from wholebody.engine.trainer import Trainer
from wholebody.utils.logger import get_logger
from wholebody.utils.seed import seed_everything

logger = get_logger("wholebody.tools.train")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Whole-Body Pose Estimation Model")
    parser.add_argument("config", type=str, help="Path to experiment configuration YAML file")
    parser.add_argument("--work-dir", type=str, default=None, help="Directory to save logs and checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to train on ('auto', 'mps', 'cuda', 'cpu')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--override", nargs="*", default=[], help="CLI overrides (e.g. model.head.num_keypoints=133)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load and merge config
    cfg = Config.from_file(args.config)
    if args.override:
        cfg.merge_from_cli_args(args.override)

    if args.work_dir is not None:
        cfg.training.work_dir = args.work_dir
    if args.device is not None:
        cfg.runtime.device = args.device
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    # Setup seed
    seed = cfg.runtime.get("seed", 42)
    deterministic = cfg.runtime.get("deterministic", False)
    seed_everything(seed=seed, deterministic=deterministic)

    # Setup DeviceManager
    device_manager = DeviceManager(
        device=cfg.runtime.get("device", "auto"),
        precision=cfg.runtime.get("precision", "fp32"),
        allow_cpu_fallback=cfg.runtime.get("allow_cpu_fallback", True),
    )

    logger.info(f"Loaded config from: {args.config}")
    logger.info(f"Target Work Dir: {cfg.training.work_dir}")
    logger.info(f"Hardware Info: {device_manager.get_hardware_info()}")

    # Build Training Dataset & DataLoader
    train_dataset = DATASETS.build(cfg.data.train.dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train.dataloader.get("batch_size", 8),
        shuffle=cfg.data.train.dataloader.get("shuffle", True),
        num_workers=cfg.data.train.dataloader.get("num_workers", 0),
        collate_fn=BasePoseDataset.collate_fn,
    )

    # Build Validation Dataset & DataLoader if present
    val_loader = None
    evaluator = None
    if "val" in cfg.data and cfg.data.val is not None:
        val_dataset = DATASETS.build(cfg.data.val.dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.data.val.dataloader.get("batch_size", 8),
            shuffle=False,
            num_workers=cfg.data.val.dataloader.get("num_workers", 0),
            collate_fn=BasePoseDataset.collate_fn,
        )
        if "evaluation" in cfg and "metrics" in cfg.evaluation:
            evaluator = Evaluator(metrics=cfg.evaluation.metrics)

    # Build Model
    model = MODELS.build(cfg.model)

    # Build Optimizer & Scheduler
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_scheduler(optimizer, cfg.scheduler, total_epochs=cfg.training.epochs)

    # Build Trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluator=evaluator,
        device_manager=device_manager,
        hooks=cfg.get("hooks", []),
        work_dir=cfg.training.work_dir,
        max_epochs=cfg.training.epochs,
        val_interval=cfg.training.get("val_interval", 1),
        grad_clip=cfg.training.get("grad_clip", 10.0),
    )

    # Save resolved full configuration
    full_cfg_path = Path(cfg.training.work_dir) / "config.yaml"
    cfg.dump(full_cfg_path)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
