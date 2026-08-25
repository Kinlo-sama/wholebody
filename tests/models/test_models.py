import unittest
import torch
from wholebody.core.registry import MODELS
from wholebody.models.base import TopDownPoseEstimator
from wholebody.models.backbones.simple_cnn import SimpleCNN
from wholebody.models.heads.heatmap_head import HeatmapHead
from wholebody.structures.data_sample import InstanceData, PoseDataSample


class TestModels(unittest.TestCase):

    def test_topdown_pose_estimator_train_and_predict(self):
        backbone = SimpleCNN(in_channels=3, stage_channels=(32, 64, 128, 256))
        head = HeatmapHead(in_channels=256, num_keypoints=17)
        model = TopDownPoseEstimator(backbone=backbone, head=head)

        inputs = torch.randn(2, 3, 256, 192)

        # Prepare dummy samples
        samples = []
        for _ in range(2):
            s = PoseDataSample()
            s.gt_instances.heatmaps = torch.zeros(17, 16, 12)
            s.gt_instances.keypoint_weights = torch.ones(17)
            samples.append(s)

        # Test forward train
        loss_dict = model(inputs, samples, mode="train")
        self.assertIn("loss_kpt", loss_dict)
        self.assertTrue(loss_dict["loss_kpt"].requires_grad)

        # Test forward predict
        pred_samples = model(inputs, samples, mode="predict")
        self.assertEqual(len(pred_samples), 2)
        self.assertEqual(pred_samples[0].pred_instances.keypoints.shape, (17, 2))


if __name__ == "__main__":
    unittest.main()
