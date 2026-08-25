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

        with torch.no_grad():
            for batch in dataloader:
                inputs = device_manager.to_device(batch["inputs"])
                data_samples: List[PoseDataSample] = batch["data_samples"]

                with device_manager.autocast():
                    pred_samples: List[PoseDataSample] = model(
                        inputs=inputs,
                        data_samples=data_samples,
                        mode="predict",
                    )

                self.process(pred_samples)

        all_results: Dict[str, float] = {}
        for m in self.metrics:
            res = m.compute_metrics()
            all_results.update(res)

        return all_results
