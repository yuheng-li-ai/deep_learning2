import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGG_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(VGG_ROOT))

from models.vgg import VGG_A_Light
from train_cifar import build_optimizer, resolve_device, save_metrics


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

    def test_save_metrics_writes_step_losses_when_provided(self):
        metrics = {
            "epochs": [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "train_accuracy": 0.5,
                    "val_loss": 1.1,
                    "val_accuracy": 0.4,
                    "elapsed_seconds": 0.1,
                }
            ]
        }
        step_losses = [
            {"step": 1, "epoch": 1, "batch": 1, "train_loss": 2.0},
            {"step": 2, "epoch": 1, "batch": 2, "train_loss": 1.5},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            save_metrics(output_dir, metrics, step_losses=step_losses)

            step_loss_path = output_dir / "step_losses.csv"
            self.assertTrue(step_loss_path.exists())
            self.assertIn("step,epoch,batch,train_loss", step_loss_path.read_text())


if __name__ == "__main__":
    unittest.main()
