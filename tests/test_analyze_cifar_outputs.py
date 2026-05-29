import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGG_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(VGG_ROOT))

from analyze_cifar_outputs import CIFAR10_CLASSES, architecture_rows, prepare_output_dir
from models.vgg import VGG_A_Light


class AnalyzeCifarOutputsTests(unittest.TestCase):
    def test_class_names_cover_cifar10(self):
        self.assertEqual(len(CIFAR10_CLASSES), 10)
        self.assertIn("airplane", CIFAR10_CLASSES)
        self.assertIn("truck", CIFAR10_CLASSES)

    def test_architecture_rows_collect_leaf_layers(self):
        model = VGG_A_Light()
        rows = architecture_rows(model)

        self.assertTrue(any(row["type"] == "Conv2d" for row in rows))
        self.assertTrue(any(row["type"] == "Linear" for row in rows))
        self.assertTrue(all("output_shape" in row for row in rows))
        self.assertGreater(sum(row["parameters"] for row in rows), 0)

    def test_architecture_rows_preserve_training_mode(self):
        model = VGG_A_Light()
        model.train()

        architecture_rows(model)

        self.assertTrue(model.training)

    def test_prepare_output_dir_refuses_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            (output_dir / "analysis_summary.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_output_dir(output_dir)


if __name__ == "__main__":
    unittest.main()
