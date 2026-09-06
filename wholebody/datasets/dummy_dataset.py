from typing import Any, Dict
import torch
import numpy as np
from torch.utils.data import Dataset
from wholebody.core.registry import DATASETS
from wholebody.structures.data_sample import PoseDataSample, InstanceData

@DATASETS.register("DummyPoseDataset")
class DummyPoseDataset(Dataset):
    def __init__(
        self,
        length: int = 100,
        image_size: tuple = (256, 192),
        num_keypoints: int = 133,
        pipeline=None,
        **kwargs
    ):
        super().__init__()
        self.length = length
        self.image_size = image_size
        self.num_keypoints = num_keypoints
        # Mock codec directly internally for testing if needed,
        # but normally pipelines would do this. 
        # Here we directly yield what the collate_fn expects.

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Generate fake image tensor
        H, W = self.image_size
        image = torch.randn(3, H, W)
        
        # Generate fake data sample with GT targets
        sample = PoseDataSample()
        sample.gt_instances = InstanceData()
        
        # Random keypoints within image bounds
        keypoints = torch.rand(self.num_keypoints, 2)
        keypoints[:, 0] *= W
        keypoints[:, 1] *= H
        
        weights = torch.ones(self.num_keypoints)
        
        sample.gt_instances.keypoints = keypoints
        sample.gt_instances.keypoint_weights = weights
        
        return {
            "inputs": image,
            "data_samples": sample
        }
