import tempfile
import unittest
from pathlib import Path
from wholebody.core.config import Config


class TestConfig(unittest.TestCase):

    def test_dot_access_and_merge(self):
        cfg = Config({"model": {"type": "SimpleCNN", "channels": 64}})
        self.assertEqual(cfg.model.type, "SimpleCNN")
        self.assertEqual(cfg.model.channels, 64)

        cfg.merge_from_cli_args(["model.channels=128", "training.epochs=50"])
        self.assertEqual(cfg.model.channels, 128)
        self.assertEqual(cfg.training.epochs, 50)

    def test_inheritance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_p = Path(tmpdir) / "base.yaml"
            child_p = Path(tmpdir) / "child.yaml"

            with open(base_p, "w") as f:
                f.write("model:\n  backbone: resnet\n  layers: 50\n")

            with open(child_p, "w") as f:
                f.write(f"_base_: [base.yaml]\nmodel:\n  layers: 101\n")

            child_cfg = Config.from_file(child_p)
            self.assertEqual(child_cfg.model.backbone, "resnet")
            self.assertEqual(child_cfg.model.layers, 101)


if __name__ == "__main__":
    unittest.main()
