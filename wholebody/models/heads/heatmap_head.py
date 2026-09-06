from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from wholebody.codecs.base import BaseCodec
from wholebody.core.registry import CODECS, HEADS, LOSSES
from wholebody.models.heads.base import BaseHead
from wholebody.models.losses.base import BaseLoss
from wholebody.structures.data_sample import InstanceData, PoseDataSample


@HEADS.register("HeatmapHead")
class HeatmapHead(BaseHead):
    """Gaussian Heatmap Prediction Head.
    
    Supports arbitrary number of keypoints (17, 133, custom), decoupled via Codec.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_keypoints: int = 133,
        deconv_out_channels: Optional[Tuple[int, ...]] = None,
        loss: Optional[Union[Dict[str, Any], BaseLoss]] = None,
        codec: Optional[Union[Dict[str, Any], BaseCodec]] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        # Build deconvolution / upsampling layers if specified
        layers: List[nn.Module] = []
        curr_in = in_channels
        if deconv_out_channels is not None:
            for out_c in deconv_out_channels:
                layers.extend([
                    nn.ConvTranspose2d(curr_in, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(out_c),
                    nn.ReLU(inplace=True),
                ])
                curr_in = out_c

        self.deconv_layers = nn.Sequential(*layers) if layers else nn.Identity()
        self.final_layer = nn.Conv2d(curr_in, num_keypoints, kernel_size=1, stride=1, padding=0)

        # Loss
        if loss is None:
            from wholebody.models.losses.mse_loss import KeypointMSELoss
            self.loss_module: BaseLoss = KeypointMSELoss()
        elif isinstance(loss, dict):
            self.loss_module = LOSSES.build(loss)
        else:
            self.loss_module = loss

        # Codec
        if codec is None:
            from wholebody.codecs.heatmap import MSRAHeatmapCodec
            self.codec: BaseCodec = MSRAHeatmapCodec()
        elif isinstance(codec, dict):
            self.codec = CODECS.build(codec)
        else:
            self.codec = codec

    def forward(self, feats: Union[torch.Tensor, Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Compute heatmap logits."""
        if isinstance(feats, tuple) or isinstance(feats, list):
            feats = feats[-1]
        x = self.deconv_layers(feats)
        heatmaps = self.final_layer(x)
        return heatmaps

    def loss(
        self,
        feats: torch.Tensor,
        batch_data_samples: List[PoseDataSample],
    ) -> Dict[str, torch.Tensor]:
        """Calculate loss against ground truth heatmaps in batch_data_samples."""
        pred_heatmaps = self.forward(feats)

        gt_heatmaps_list = [s.gt_instances.heatmaps for s in batch_data_samples]
        gt_weights_list = [s.gt_instances.keypoint_weights for s in batch_data_samples]

        gt_heatmaps = torch.stack(gt_heatmaps_list, dim=0).to(pred_heatmaps.device)
        gt_weights = torch.stack(gt_weights_list, dim=0).to(pred_heatmaps.device)

        loss_kpt = self.loss_module(
            pred=pred_heatmaps,
            target=gt_heatmaps,
            target_weight=gt_weights,
        )

        return {"loss_kpt": loss_kpt}

    def predict(
        self,
        feats: torch.Tensor,
        batch_data_samples: List[PoseDataSample],
    ) -> List[PoseDataSample]:
        """Decode predicted heatmaps back to keypoints in original image coordinate space."""
        pred_heatmaps = self.forward(feats)
        metainfo_list = [s.metainfo for s in batch_data_samples]

        # Decode coordinates and confidence scores
        pred_coords, pred_scores = self.codec.decode(
            encoded=pred_heatmaps,
            metainfo=metainfo_list,
        )

        for b, sample in enumerate(batch_data_samples):
            pred_instances = InstanceData()
            pred_instances.keypoints = torch.from_numpy(pred_coords[b]).float()
            pred_instances.keypoint_scores = torch.from_numpy(pred_scores[b]).float()
            sample.pred_instances = pred_instances

        return batch_data_samples
