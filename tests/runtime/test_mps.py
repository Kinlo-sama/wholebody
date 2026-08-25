import tempfile
import unittest
from pathlib import Path
import torch
import torch.optim as optim

from wholebody.core.device import DeviceManager
from wholebody.engine.checkpointer import CheckpointManager
from wholebody.models.base import TopDownPoseEstimator
from wholebody.models.backbones.simple_cnn import SimpleCNN
from wholebody.models.heads.heatmap_head import HeatmapHead
from wholebody.structures.data_sample import InstanceData, PoseDataSample


class TestAppleSiliconMPS(unittest.TestCase):
    """Rigorous Apple Silicon / MPS Hardware Acceleration Test Suite.
    
    Verifies all 10 criteria specified in Section 50 of the architectural specification:
      1. PyTorch detects MPS.
      2. Tensor can be created in MPS.
      3. Model moves to MPS.
      4. Forward works on MPS.
      5. Backward works on MPS.
      6. Optimizer works on MPS.
      7. Checkpoint can be saved.
      8. Checkpoint can be loaded again.
      9. Inference works on MPS.
      10. Same checkpoint can be loaded on CPU.
    """

    def setUp(self):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            self.skipTest("Apple Silicon MPS is not available on this environment.")

    def test_mps_full_lifecycle(self):
        # 1. PyTorch detects MPS
        self.assertTrue(torch.backends.mps.is_available())
        self.assertTrue(torch.backends.mps.is_built())

        # 2. Tensor created in MPS
        t = torch.ones((2, 3), device="mps")
        self.assertEqual(t.device.type, "mps")

        # 3. Model moves to MPS
        device_manager = DeviceManager(device="mps")
        model = TopDownPoseEstimator(
            backbone=SimpleCNN(in_channels=3, stage_channels=(16, 32)),
            head=HeatmapHead(
                in_channels=32,
                num_keypoints=133,
                codec={"type": "MSRAHeatmapCodec", "input_size": (64, 48), "heatmap_size": (16, 12)},
            ),
        )
        model = device_manager.to_device(model)
        self.assertEqual(next(model.parameters()).device.type, "mps")

        # 4. Forward works on MPS
        inputs = torch.randn(2, 3, 64, 48, device="mps")
        samples = []
        for _ in range(2):
            s = PoseDataSample()
            s.gt_instances.heatmaps = torch.zeros(133, 16, 12, device="mps")
            s.gt_instances.keypoint_weights = torch.ones(133, device="mps")
            samples.append(s)

        with device_manager.autocast():
            loss_dict = model(inputs, samples, mode="train")
            total_loss = sum(v for v in loss_dict.values())

        self.assertIn("loss_kpt", loss_dict)
        self.assertTrue(total_loss.requires_grad)

        # 5. Backward works on MPS
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        total_loss.backward()

        # 6. Optimizer works on MPS
        optimizer.step()

        # 7. Checkpoint saves
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "mps_test_checkpoint.pth"
            CheckpointManager.save_checkpoint(
                filepath=ckpt_path,
                model=model,
                optimizer=optimizer,
                epoch=1,
                meta=device_manager.get_hardware_info(),
            )
            self.assertTrue(ckpt_path.is_file())

            # 8. Checkpoint loads back into MPS model
            new_mps_model = TopDownPoseEstimator(
                backbone=SimpleCNN(in_channels=3, stage_channels=(16, 32)),
                head=HeatmapHead(
                    in_channels=32,
                    num_keypoints=133,
                    codec={"type": "MSRAHeatmapCodec", "input_size": (64, 48), "heatmap_size": (16, 12)},
                ),
            )
            new_mps_model = device_manager.to_device(new_mps_model)
            CheckpointManager.load_checkpoint(ckpt_path, new_mps_model, device="mps")

            # 9. Inference works on MPS
            new_mps_model.eval()
            with torch.no_grad():
                pred_samples = new_mps_model(inputs, samples, mode="predict")
            self.assertEqual(len(pred_samples), 2)
            self.assertEqual(pred_samples[0].pred_instances.keypoints.shape, (133, 2))

            # 10. Same checkpoint loads cleanly on CPU
            cpu_device_mgr = DeviceManager(device="cpu")
            cpu_model = TopDownPoseEstimator(
                backbone=SimpleCNN(in_channels=3, stage_channels=(16, 32)),
                head=HeatmapHead(
                    in_channels=32,
                    num_keypoints=133,
                    codec={"type": "MSRAHeatmapCodec", "input_size": (64, 48), "heatmap_size": (16, 12)},
                ),
            )
            cpu_model = cpu_device_mgr.to_device(cpu_model)
            CheckpointManager.load_checkpoint(ckpt_path, cpu_model, device="cpu")
            self.assertEqual(next(cpu_model.parameters()).device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
