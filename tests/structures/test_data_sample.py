import unittest
import numpy as np
import torch
from wholebody.structures.data_sample import BoundingBox, InstanceData, PoseDataSample


class TestDataSample(unittest.TestCase):

    def test_bounding_box(self):
        bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=220.0)
        self.assertEqual(bbox.width, 100.0)
        self.assertEqual(bbox.height, 200.0)
        np.testing.assert_allclose(bbox.center, [60.0, 120.0])
        np.testing.assert_allclose(bbox.scale, [0.5, 1.0])

    def test_pose_data_sample_to_device(self):
        sample = PoseDataSample()
        sample.gt_instances.keypoints = torch.randn(17, 2)
        sample.set_metainfo({"img_id": 1})

        sample_cpu = sample.cpu()
        self.assertEqual(sample_cpu.gt_instances.keypoints.device.type, "cpu")

        sample_np = sample.numpy()
        self.assertIsInstance(sample_np.gt_instances.keypoints, np.ndarray)


if __name__ == "__main__":
    unittest.main()
