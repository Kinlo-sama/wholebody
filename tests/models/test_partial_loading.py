import unittest
import torch
from wholebody.models.base import TopDownPoseEstimator
from wholebody.models.backbones.simple_cnn import SimpleCNN
from wholebody.models.heads.heatmap_head import HeatmapHead
from wholebody.utils.model_utils import load_partial_state_dict, freeze_module


class TestPartialLoading(unittest.TestCase):

    def test_partial_loading_head_replacement(self):
        # Create model with 17 keypoints
        model_17 = TopDownPoseEstimator(
            backbone=SimpleCNN(in_channels=3, stage_channels=(32, 64)),
            head=HeatmapHead(in_channels=64, num_keypoints=17),
        )
        state_17 = model_17.state_dict()

        # Create new model with 133 keypoints
        model_133 = TopDownPoseEstimator(
            backbone=SimpleCNN(in_channels=3, stage_channels=(32, 64)),
            head=HeatmapHead(in_channels=64, num_keypoints=133),
        )

        # Freeze backbone
        freeze_module(model_133.backbone)
        for p in model_133.backbone.parameters():
            self.assertFalse(p.requires_grad)

        # Partial load 17 checkpoint into 133 model
        matched, missing, unexpected = load_partial_state_dict(
            model=model_133,
            state_dict=state_17,
            strict=False,
            ignore_shape_mismatch=True,
        )

        # Backbone should match
        self.assertGreater(len(matched), 0)
        # Head final layer (due to 17 vs 133 shape mismatch) should be left uninitialized / missing
        self.assertIn("head.final_layer.weight", missing)


if __name__ == "__main__":
    unittest.main()
