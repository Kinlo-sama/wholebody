import sys
from wholebody.core.config import Config
from wholebody.core.registry import DATASETS, METRICS
from wholebody.inference.api import init_model
from torch.utils.data import DataLoader
from wholebody.datasets.coco_wholebody import collate_fn
import torch

def main():
    cfg = Config.from_file("configs/experiments/rtmw-l_384x288.yaml")
    
    # Subset dataset
    val_dataset = DATASETS.build(cfg.val_dataloader['dataset'])
    val_dataset.data_list = val_dataset.data_list[:1000] # only 1000 images
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.val_dataloader['batch_size'],
        num_workers=0,
        collate_fn=collate_fn
    )
    
    model = init_model(cfg, "weights/rtmw-l_384x288_fixed.pth")
    metric = METRICS.build(cfg.val_evaluator['metrics'][0])
    
    for i, batch in enumerate(val_loader):
        inputs = batch["inputs"].to(model.device_manager.get_device())
        data_samples = batch["data_samples"]
        with torch.no_grad():
            pred = model(inputs=inputs, data_samples=data_samples, mode="predict")
        metric.process(pred)
        print(f"Batch {i}/{len(val_loader)}", end='\r')
    print()
    res = metric.compute_metrics()
    print("Fast Eval Results:", res)

if __name__ == "__main__":
    main()
