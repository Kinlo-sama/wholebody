import tempfile
import unittest
from torch.utils.data import DataLoader
import torch.optim as optim

from wholebody.core.device import DeviceManager
from wholebody.datasets.synthetic import SyntheticWholeBodyDataset
from wholebody.datasets.transforms.affine import TopDownAffine
from wholebody.datasets.transforms.formatting import Normalize, PackPoseInputs
from wholebody.datasets.transforms.target_gen import GenerateTarget
from wholebody.datasets.base import BasePoseDataset
from wholebody.engine.trainer import Trainer
from wholebody.models.base import TopDownPoseEstimator
from wholebody.models.backbones.simple_cnn import SimpleCNN
from wholebody.models.heads.heatmap_head import HeatmapHead


class TestTrainer(unittest.TestCase):

    def test_single_epoch_training_cpu(self):
        pipeline = [
            TopDownAffine(input_size=(64, 48)),
            GenerateTarget(codec={"type": "MSRAHeatmapCodec", "input_size": (64, 48), "heatmap_size": (16, 12)}),
            Normalize(),
            PackPoseInputs(),
        ]
        dataset = SyntheticWholeBodyDataset(
            num_samples=4,
            img_size=(128, 128),
            keypoint_spec="coco_17",
            pipeline=pipeline,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=BasePoseDataset.collate_fn)

        model = TopDownPoseEstimator(
            backbone=SimpleCNN(in_channels=3, stage_channels=(16, 32)),
            head=HeatmapHead(
                in_channels=32,
                num_keypoints=17,
                codec={"type": "MSRAHeatmapCodec", "input_size": (64, 48), "heatmap_size": (16, 12)},
            ),
        )
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                train_dataloader=loader,
                optimizer=optimizer,
                device_manager=DeviceManager(device="cpu"),
                work_dir=tmpdir,
                max_epochs=1,
            )
            trainer.train()
            self.assertEqual(trainer.current_epoch, 1)


if __name__ == "__main__":
    unittest.main()
