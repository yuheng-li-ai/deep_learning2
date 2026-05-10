import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGG_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(VGG_ROOT))

from models.vgg import VGG_A_Light
from train_cifar import build_optimizer, resolve_device


class TrainCifarTests(unittest.TestCase):
    def test_resolve_device_accepts_explicit_cpu(self):
        device = resolve_device("cpu")

        self.assertEqual(device.type, "cpu")

    def test_build_optimizer_supports_required_choices(self):
        for optimizer_name in ["adam", "adamw", "sgd"]:
            with self.subTest(optimizer_name=optimizer_name):
                model = VGG_A_Light()

                optimizer = build_optimizer(
                    optimizer_name, model, lr=1e-3, weight_decay=1e-4
                )

                self.assertIsInstance(optimizer, torch.optim.Optimizer)


if __name__ == "__main__":
    unittest.main()
