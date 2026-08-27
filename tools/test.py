import argparse
import torch
from torch.utils.data import DataLoader

from wholebody.core.config import Config
from wholebody.core.device import DeviceManager
from wholebody.core.registry import DATASETS, METRICS
from wholebody.engine.evaluator import Evaluator
from wholebody.inference.api import init_model
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.tools.test")

def parse_args():
    parser = argparse.ArgumentParser(description="Test Pose Estimation Model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'mps', 'cuda', 'cpu')")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.from_file(args.config)
    device_manager = DeviceManager(device=args.device)

    # 1. Initialize Model
    model = init_model(config=cfg, checkpoint=args.checkpoint, device=args.device)
    logger.info("Model initialized and checkpoint loaded.")

    # 2. Build Dataset & DataLoader
    val_dataset_cfg = cfg.data.val.dataset
    val_dataset = DATASETS.build(val_dataset_cfg)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val.dataloader.get("batch_size", 32),
        shuffle=False,
        num_workers=cfg.data.val.dataloader.get("num_workers", 2),
        collate_fn=val_dataset.collate_fn,
    )

    # 3. Build Evaluator and Metrics
    metrics_cfg = cfg.evaluation.metrics
    metrics = [METRICS.build(m_cfg) for m_cfg in metrics_cfg]
    
    evaluator = Evaluator(metrics=metrics)

    # 4. Run Evaluation
    logger.info("Starting evaluation...")
    final_metrics = evaluator.evaluate(
        model=model,
        dataloader=val_loader,
        device_manager=device_manager,
    )
    logger.info(f"Evaluation Results:\n{final_metrics}")

if __name__ == "__main__":
    main()
