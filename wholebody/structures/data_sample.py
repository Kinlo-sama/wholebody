from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union
import numpy as np
import torch


@dataclass
class BoundingBox:
    """Bounding box container in [x1, y1, x2, y2] format."""
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0], dtype=np.float32)

    @property
    def scale(self) -> np.ndarray:
        """Returns standard 200px normalized scale vector [w, h] / 200."""
        return np.array([self.width / 200.0, self.height / 200.0], dtype=np.float32)

    def as_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


class InstanceData:
    """Container for instances (ground truth or predictions) in a sample."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, key: str, value: Any) -> None:
        self.__dict__[key] = value

    def __getattr__(self, key: str) -> Any:
        return self.__dict__.get(key, None)

    def to(self, device: torch.device) -> "InstanceData":
        """Move all contained tensors to target device."""
        new_inst = InstanceData()
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                new_inst.__dict__[k] = v.to(device)
            else:
                new_inst.__dict__[k] = v
        return new_inst

    def cpu(self) -> "InstanceData":
        return self.to(torch.device("cpu"))

    def numpy(self) -> "InstanceData":
        """Convert all contained torch.Tensors to numpy arrays."""
        new_inst = InstanceData()
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                new_inst.__dict__[k] = v.detach().cpu().numpy()
            else:
                new_inst.__dict__[k] = v
        return new_inst

    def __repr__(self) -> str:
        items = {k: type(v).__name__ if not isinstance(v, (int, float, str)) else v for k, v in self.__dict__.items()}
        return f"InstanceData({items})"


class PoseDataSample:
    """Standard unified data structure passing through the entire pipeline.
    
    Contains:
      - metainfo: image paths, transforms, original shapes
      - gt_instances: ground truth keypoints, bboxes, weights
      - pred_instances: model predictions (keypoints in original coordinates, scores)
    """

    def __init__(
        self,
        metainfo: Optional[Dict[str, Any]] = None,
        gt_instances: Optional[InstanceData] = None,
        pred_instances: Optional[InstanceData] = None,
    ) -> None:
        self.metainfo: Dict[str, Any] = metainfo or {}
        self.gt_instances: InstanceData = gt_instances or InstanceData()
        self.pred_instances: InstanceData = pred_instances or InstanceData()

    def set_metainfo(self, info: Dict[str, Any]) -> None:
        self.metainfo.update(info)

    def to(self, device: torch.device) -> "PoseDataSample":
        return PoseDataSample(
            metainfo=self.metainfo,
            gt_instances=self.gt_instances.to(device) if self.gt_instances else None,
            pred_instances=self.pred_instances.to(device) if self.pred_instances else None,
        )

    def cpu(self) -> "PoseDataSample":
        return self.to(torch.device("cpu"))

    def numpy(self) -> "PoseDataSample":
        return PoseDataSample(
            metainfo=self.metainfo,
            gt_instances=self.gt_instances.numpy() if self.gt_instances else None,
            pred_instances=self.pred_instances.numpy() if self.pred_instances else None,
        )

    def __repr__(self) -> str:
        return f"PoseDataSample(meta_keys={list(self.metainfo.keys())}, gt={self.gt_instances}, pred={self.pred_instances})"
