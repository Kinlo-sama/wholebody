from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from wholebody.core.registry import METRICS
from wholebody.structures.data_sample import PoseDataSample


class BaseMetric(ABC):
    """Abstract Base Class for pose estimation evaluation metrics."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.reset()

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def process(self, data_samples: List[PoseDataSample]) -> None:
        pass

    @abstractmethod
    def compute_metrics(self) -> Dict[str, float]:
        pass


@METRICS.register("PCKMetric")
class PCKMetric(BaseMetric):
    """Percentage of Correct Keypoints (PCK) metric.
    
    A predicted keypoint is correct if normalized Euclidean distance <= threshold.
    """

    def __init__(self, threshold: float = 0.05, norm_factor: str = "bbox") -> None:
        super().__init__(name="PCK")
        self.threshold = threshold
        self.norm_factor = norm_factor
        self.reset()

    def reset(self) -> None:
        self.correct_counts = 0
        self.total_counts = 0

    def process(self, data_samples: List[PoseDataSample]) -> None:
        for sample in data_samples:
            if not sample.pred_instances or not sample.gt_instances:
                continue

            pred_kpts = sample.pred_instances.keypoints
            gt_kpts = sample.gt_instances.keypoints
            gt_vis = sample.gt_instances.keypoints_visible

            if isinstance(pred_kpts, torch.Tensor):
                pred_kpts = pred_kpts.detach().cpu().numpy()
            if isinstance(gt_kpts, torch.Tensor):
                gt_kpts = gt_kpts.detach().cpu().numpy()
            if isinstance(gt_vis, torch.Tensor):
                gt_vis = gt_vis.detach().cpu().numpy()

            if pred_kpts.ndim == 3:
                pred_kpts = pred_kpts[0]
            if gt_kpts.ndim == 3:
                gt_kpts = gt_kpts[0]
            if gt_vis is not None and gt_vis.ndim == 2:
                gt_vis = gt_vis[0]

            # Euclidean distances (K,)
            dists = np.linalg.norm(pred_kpts - gt_kpts, axis=-1)

            # Normalization scale (diagonal of bbox or 200px)
            if "scale" in sample.metainfo:
                scale_val = np.linalg.norm(sample.metainfo["scale"] * 200.0)
            else:
                scale_val = 200.0
            scale_val = max(1e-4, float(scale_val))

            norm_dists = dists / scale_val

            if gt_vis is not None:
                mask = gt_vis > 0
                correct = (norm_dists[mask] <= self.threshold).sum()
                total = mask.sum()
            else:
                correct = (norm_dists <= self.threshold).sum()
                total = len(norm_dists)

            self.correct_counts += int(correct)
            self.total_counts += int(total)

    def compute_metrics(self) -> Dict[str, float]:
        pck = (self.correct_counts / max(1, self.total_counts)) * 100.0
        return {f"PCK@{self.threshold}": round(float(pck), 2)}


@METRICS.register("MPJPEMetric")
class MPJPEMetric(BaseMetric):
    """Mean Per Joint Position Error (MPJPE) in pixels."""

    def __init__(self) -> None:
        super().__init__(name="MPJPE")
        self.reset()

    def reset(self) -> None:
        self.total_error = 0.0
        self.total_kpts = 0

    def process(self, data_samples: List[PoseDataSample]) -> None:
        for sample in data_samples:
            if not sample.pred_instances or not sample.gt_instances:
                continue

            pred_kpts = sample.pred_instances.keypoints
            gt_kpts = sample.gt_instances.keypoints
            gt_vis = sample.gt_instances.keypoints_visible

            if isinstance(pred_kpts, torch.Tensor):
                pred_kpts = pred_kpts.detach().cpu().numpy()
            if isinstance(gt_kpts, torch.Tensor):
                gt_kpts = gt_kpts.detach().cpu().numpy()
            if isinstance(gt_vis, torch.Tensor):
                gt_vis = gt_vis.detach().cpu().numpy()

            if pred_kpts.ndim == 3: pred_kpts = pred_kpts[0]
            if gt_kpts.ndim == 3: gt_kpts = gt_kpts[0]
            if gt_vis is not None and gt_vis.ndim == 2: gt_vis = gt_vis[0]

            dists = np.linalg.norm(pred_kpts - gt_kpts, axis=-1)

            if gt_vis is not None:
                mask = gt_vis > 0
                self.total_error += float(dists[mask].sum())
                self.total_kpts += int(mask.sum())
            else:
                self.total_error += float(dists.sum())
                self.total_kpts += len(dists)

    def compute_metrics(self) -> Dict[str, float]:
        mpjpe = self.total_error / max(1, self.total_kpts)
        return {"MPJPE": round(float(mpjpe), 2)}


@METRICS.register("OKSMetric")
class OKSMetric(BaseMetric):
    """Object Keypoint Similarity (OKS) AP evaluation."""

    def __init__(self, sigmas: Optional[np.ndarray] = None) -> None:
        super().__init__(name="OKS")
        self.sigmas = sigmas
        self.reset()

    def reset(self) -> None:
        self.oks_scores: List[float] = []

    def process(self, data_samples: List[PoseDataSample]) -> None:
        for sample in data_samples:
            if not sample.pred_instances or not sample.gt_instances:
                continue

            pred_kpts = sample.pred_instances.keypoints
            gt_kpts = sample.gt_instances.keypoints
            gt_vis = sample.gt_instances.keypoints_visible

            if isinstance(pred_kpts, torch.Tensor): pred_kpts = pred_kpts.detach().cpu().numpy()
            if isinstance(gt_kpts, torch.Tensor): gt_kpts = gt_kpts.detach().cpu().numpy()
            if isinstance(gt_vis, torch.Tensor): gt_vis = gt_vis.detach().cpu().numpy()

            if pred_kpts.ndim == 3: pred_kpts = pred_kpts[0]
            if gt_kpts.ndim == 3: gt_kpts = gt_kpts[0]
            if gt_vis is not None and gt_vis.ndim == 2: gt_vis = gt_vis[0]

            # Scale area calculation
            if "scale" in sample.metainfo:
                scale_v = sample.metainfo["scale"] * 200.0
                area = float(scale_v[0] * scale_v[1])
            else:
                area = 200.0 * 200.0
            area = max(1.0, area)

            sigmas = sample.metainfo.get("sigmas", self.sigmas)
            if sigmas is None:
                sigmas = np.ones((gt_kpts.shape[0],), dtype=np.float32) * 0.05
            elif not isinstance(sigmas, np.ndarray):
                sigmas = np.array(sigmas, dtype=np.float32)

            dists_sq = np.sum((pred_kpts - gt_kpts) ** 2, axis=-1)
            vars_v = (sigmas * 2.0) ** 2
            e = dists_sq / (2.0 * vars_v * area + 1e-8)

            if gt_vis is not None:
                mask = gt_vis > 0
                if mask.sum() > 0:
                    oks = np.mean(np.exp(-e[mask]))
                else:
                    oks = 0.0
            else:
                oks = float(np.mean(np.exp(-e)))

            self.oks_scores.append(float(oks))

    def compute_metrics(self) -> Dict[str, float]:
        if not self.oks_scores:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

        oks_arr = np.array(self.oks_scores)
        ap50 = float(np.mean(oks_arr >= 0.50)) * 100.0
        ap75 = float(np.mean(oks_arr >= 0.75)) * 100.0
        # AP across thresholds 0.50:0.05:0.95
        thresholds = np.linspace(0.50, 0.95, 10)
        aps = [np.mean(oks_arr >= t) for t in thresholds]
        ap = float(np.mean(aps)) * 100.0

        return {
            "AP": round(ap, 2),
            "AP50": round(ap50, 2),
            "AP75": round(ap75, 2),
        }
