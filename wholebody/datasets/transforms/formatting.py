from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from wholebody.core.registry import TRANSFORMS
from wholebody.datasets.transforms.base import BaseTransform
from wholebody.structures.data_sample import InstanceData, PoseDataSample


@TRANSFORMS.register("Normalize")
class Normalize(BaseTransform):
    """Normalize image pixel values by subtracting mean and dividing by std."""

    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"].astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        img = (img - self.mean) / self.std
        results["img"] = img
        return results


@TRANSFORMS.register("ToTensor")
class ToTensor(BaseTransform):
    """Convert numpy HWC image array to PyTorch CHW float tensor."""

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results["img"]
        if img.ndim == 2:
            img = img[:, :, None]
        # HWC to CHW
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        results["img"] = torch.from_numpy(img).float()
        return results


@TRANSFORMS.register("PackPoseInputs")
class PackPoseInputs(BaseTransform):
    """Package transformed image tensor and build standard PoseDataSample."""

    def __init__(
        self,
        meta_keys: Tuple[str, ...] = (
            "img_id", "img_path", "img_shape", "input_size", "center", "scale",
            "warp_mat", "warp_mat_inv", "rotation", "flipped", "raw_keypoints"
        ),
    ) -> None:
        self.meta_keys = meta_keys

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        packed_results: Dict[str, Any] = {}

        if "img" in results:
            img = results["img"]
            if isinstance(img, np.ndarray):
                if img.ndim == 3:
                    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
                elif img.ndim == 2:
                    img = torch.from_numpy(img[None, :, :]).float()
            packed_results["inputs"] = img

        data_sample = PoseDataSample()
        gt_instances = InstanceData()

        # Transfer ground truth tensors
        if "heatmaps" in results:
            gt_instances.heatmaps = torch.from_numpy(results["heatmaps"]).float()
        if "keypoint_weights" in results:
            gt_instances.keypoint_weights = torch.from_numpy(results["keypoint_weights"]).float()
        if "keypoints" in results:
            gt_instances.keypoints = torch.from_numpy(results["keypoints"]).float()
        if "keypoints_visible" in results:
            gt_instances.keypoints_visible = torch.from_numpy(results["keypoints_visible"]).float()
        if "target_coords" in results:
            gt_instances.target_coords = torch.from_numpy(results["target_coords"]).float()
        if "bboxes" in results:
            gt_instances.bboxes = torch.from_numpy(results["bboxes"]).float()

        data_sample.gt_instances = gt_instances

        # Metainfo
        metainfo: Dict[str, Any] = {}
        for k in self.meta_keys:
            if k in results:
                metainfo[k] = results[k]
        data_sample.metainfo = metainfo

        packed_results["data_samples"] = data_sample
        return packed_results
