import unittest
import torch
import torch.nn as nn
from wholebody.core.device import DeviceManager


class TestDeviceManager(unittest.TestCase):

    def test_auto_device_detection(self):
        dm = DeviceManager(device="auto")
        device = dm.get_device()
        self.assertIsInstance(device, torch.device)

    def test_to_device_tensor_and_module(self):
        dm = DeviceManager(device="cpu")
        tensor = torch.randn(2, 3)
        module = nn.Linear(3, 2)

        dev_tensor = dm.to_device(tensor)
        dev_module = dm.to_device(module)

        self.assertEqual(dev_tensor.device.type, "cpu")
        self.assertEqual(next(dev_module.parameters()).device.type, "cpu")

    def test_hardware_info(self):
        dm = DeviceManager(device="auto")
        info = dm.get_hardware_info()
        self.assertIn("device", info)
        self.assertIn("pytorch_version", info)
        self.assertIn("platform", info)


if __name__ == "__main__":
    unittest.main()
