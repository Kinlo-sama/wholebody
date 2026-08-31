from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from wholebody.core.registry import BACKBONES, HEADS, MODELS, NECKS
from wholebody.models.backbones.base import BaseBackbone
from wholebody.models.heads.base import BaseHead
from wholebody.models.necks.base import BaseNeck
from wholebody.structures.data_sample import PoseDataSample


class BasePoseEstimator(nn.Module, ABC):
    """Abstract Base Class for all Human Pose and Whole-Body Estimation Models.
    
    Enforces unified interfaces for:
      - training: forward(mode='train')
      - prediction/inference: forward(mode='predict')
      - raw tensor computation: forward(mode='tensor')
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        data_samples: Optional[List[PoseDataSample]] = None,
        mode: str = "tensor",
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], List[PoseDataSample]]:
        if mode == "train":
            return self.forward_train(inputs, data_samples)
        elif mode in ("predict", "test"):
            return self.forward_predict(inputs, data_samples)
        elif mode == "tensor":
            return self.forward_tensor(inputs)
        else:
            raise ValueError(f"Invalid mode '{mode}'. Choose from: 'train', 'predict', 'test', 'tensor'.")

    @abstractmethod
    def forward_train(
        self,
        inputs: torch.Tensor,
        data_samples: List[PoseDataSample],
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses."""
        pass

    @abstractmethod
    def forward_predict(
        self,
        inputs: torch.Tensor,
        data_samples: List[PoseDataSample],
    ) -> List[PoseDataSample]:
        """Run inference and populate predictions in data_samples."""
        pass

    @abstractmethod
    def forward_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run direct tensor forward pass (useful for ONNX/export)."""
        pass


@MODELS.register("TopDownPoseEstimator")
class TopDownPoseEstimator(BasePoseEstimator):
    """Modular Top-Down Whole-Body Pose Estimator.
    
    Composed of decoupled:
      - Backbone: feature extractor (e.g. SimpleCNN, HRNet, ViT)
      - Neck (optional): feature aggregator (e.g. Identity, FPN)
      - Head: keypoint predictor (e.g. HeatmapHead, SimCCHead)
    """

    def __init__(
        self,
        backbone: Union[Dict[str, Any], BaseBackbone],
        head: Union[Dict[str, Any], BaseHead],
        neck: Optional[Union[Dict[str, Any], BaseNeck]] = None,
        pretrained: Optional[str] = None,
        freeze_backbone: bool = False,
        test_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.test_cfg = test_cfg or {}

        # Build backbone
        if isinstance(backbone, dict):
            self.backbone: BaseBackbone = BACKBONES.build(backbone)
        else:
            self.backbone = backbone

        # Build neck
        if neck is not None:
            if isinstance(neck, dict):
                self.neck: Optional[BaseNeck] = NECKS.build(neck)
            else:
                self.neck = neck
        else:
            self.neck = None

        # Build head
        if isinstance(head, dict):
            self.head: BaseHead = HEADS.build(head)
        else:
            self.head = head

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        if pretrained:
            from wholebody.utils.model_utils import load_partial_state_dict
            try:
                checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint = torch.load(pretrained, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            load_partial_state_dict(self, state_dict, strict=False)

    def extract_feat(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract features through Backbone and Neck."""
        x = self.backbone(inputs)
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        feats = self.extract_feat(inputs)
        return self.head.forward(feats)

    def forward_train(
        self,
        inputs: torch.Tensor,
        data_samples: List[PoseDataSample],
    ) -> Dict[str, torch.Tensor]:
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def forward_predict(
        self,
        inputs: torch.Tensor,
        data_samples: List[PoseDataSample],
    ) -> List[PoseDataSample]:
        
        # 1. Forward original
        feats = self.extract_feat(inputs)
        heatmaps = self.head.forward(feats)
        
        # 2. Test-Time Augmentation (flip_test)
        if self.test_cfg.get("flip_test", False):
            inputs_flipped = torch.flip(inputs, dims=[3])
            feats_flipped = self.extract_feat(inputs_flipped)
            heatmaps_flipped = self.head.forward(feats_flipped)
            
            from wholebody.core.registry import KEYPOINT_SPECS
            spec = KEYPOINT_SPECS.get("coco_wholebody_133")
            flip_indices = torch.tensor(spec.flip_indices, device=inputs.device)

            if isinstance(heatmaps_flipped, tuple):
                # SimCC TTA
                pred_x, pred_y = heatmaps_flipped
                pred_x_flipped = torch.flip(pred_x, dims=[2]) # Flip X bins
                
                # Reorder channels
                pred_x_flipped = pred_x_flipped.index_select(1, flip_indices)
                pred_y_flipped = pred_y.index_select(1, flip_indices)
                
                pred_x = (heatmaps[0] + pred_x_flipped) / 2.0
                pred_y = (heatmaps[1] + pred_y_flipped) / 2.0
                heatmaps = (pred_x, pred_y)
            else:
                # Heatmap TTA
                heatmaps_flipped = torch.flip(heatmaps_flipped, dims=[3])
                heatmaps_flipped = heatmaps_flipped.index_select(1, flip_indices)
                if self.test_cfg.get("shift_heatmap", True):
                    heatmaps_flipped[..., 1:] = heatmaps_flipped[..., :-1].clone()
                heatmaps = (heatmaps + heatmaps_flipped) / 2.0
            
        # 3. Decode
        metainfo_list = [s.metainfo for s in data_samples]
        pred_coords, pred_scores = self.head.codec.decode(
            encoded=heatmaps,
            metainfo=metainfo_list,
        )

        from wholebody.structures.data_sample import InstanceData
        for b, sample in enumerate(data_samples):
            pred_instances = InstanceData()
            pred_instances.keypoints = torch.from_numpy(pred_coords[b]).float()
            pred_instances.keypoint_scores = torch.from_numpy(pred_scores[b]).float()
            sample.pred_instances = pred_instances

        return data_samples
