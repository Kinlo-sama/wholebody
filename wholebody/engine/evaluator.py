from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wholebody.core.device import DeviceManager
from wholebody.core.registry import METRICS
from wholebody.evaluation.metrics import BaseMetric
from wholebody.structures.data_sample import PoseDataSample
from wholebody.utils.logger import get_logger

logger = get_logger("wholebody.engine.evaluator")


class Evaluator:
    """Unified evaluator orchestrating metric computation on evaluation datasets."""

    def __init__(self, metrics: List[Union[Dict[str, Any], BaseMetric]]) -> None:
        self.metrics: List[BaseMetric] = []
        for m in metrics:
            if isinstance(m, dict):
                self.metrics.append(METRICS.build(m))
            elif isinstance(m, BaseMetric):
                self.metrics.append(m)
            else:
                raise TypeError(f"Invalid metric item: {m}")

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

    def process(self, data_samples: List[PoseDataSample]) -> None:
        for m in self.metrics:
            m.process(data_samples)

    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device_manager: DeviceManager,
    ) -> Dict[str, float]:
        """Run full evaluation pass over dataloader."""
        self.reset()
        model.eval()
        device = device_manager.get_device()

        import time
        import datetime

        with torch.no_grad():
            total_batches = len(dataloader)
            
            batch_times = []
            data_times = []
            end_time = time.perf_counter()
            
            for batch_idx, batch in enumerate(dataloader):
                data_time = time.perf_counter() - end_time
                data_times.append(data_time)
                
                inputs = device_manager.to_device(batch["inputs"])
                data_samples: List[PoseDataSample] = batch["data_samples"]

                with device_manager.autocast():
                    pred_samples: List[PoseDataSample] = model(
                        inputs=inputs,
                        data_samples=data_samples,
                        mode="predict",
                    )

                self.process(pred_samples)
                
                batch_time = time.perf_counter() - end_time
                batch_times.append(batch_time)
                
                # Progress logging!
                if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
                    avg_time = sum(batch_times[-50:]) / len(batch_times[-50:])
                    avg_data_time = sum(data_times[-50:]) / len(data_times[-50:])
                    remaining_batches = total_batches - (batch_idx + 1)
                    eta_seconds = int(avg_time * remaining_batches)
                    eta_str = str(datetime.timedelta(seconds=eta_seconds))
                    
                    mem_str = ""
                    if torch.cuda.is_available():
                        mem_MB = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
                        mem_str = f"  memory: {mem_MB}"
                        
                    logger.info(
                        f"Epoch(test) [{batch_idx + 1:4d}/{total_batches}]    "
                        f"eta: {eta_str}  time: {avg_time:.4f}  "
                        f"data_time: {avg_data_time:.4f}{mem_str}"
                    )
                
                end_time = time.perf_counter()

        all_results: Dict[str, float] = {}
        for m in self.metrics:
            res = m.compute_metrics()
            all_results.update(res)

        return all_results
