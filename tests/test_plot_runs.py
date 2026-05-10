import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGG_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(VGG_ROOT))

from plot_runs import plot_runs, summarize_run


class PlotRunsTests(unittest.TestCase):
    def test_summarize_run_reports_best_epoch_and_error(self):
        metrics = {
            "parameters": 123,
            "epochs": [
                {
                    "epoch": 1,
                    "train_accuracy": 0.5,
                    "val_accuracy": 0.6,
                    "val_loss": 1.2,
                },
                {
                    "epoch": 2,
                    "train_accuracy": 0.8,
                    "val_accuracy": 0.75,
                    "val_loss": 0.9,
                },
            ],
        }

        summary = summarize_run("toy", metrics)

        self.assertEqual(summary["name"], "toy")
        self.assertEqual(summary["parameters"], 123)
        self.assertEqual(summary["best_epoch"], 2)
        self.assertAlmostEqual(summary["best_val_accuracy"], 0.75)
        self.assertAlmostEqual(summary["best_val_error"], 0.25)

    def test_plot_runs_writes_png(self):
        metrics = {
            "epochs": [
                {
                    "epoch": 1,
                    "train_loss": 1.0,
                    "val_loss": 1.1,
                    "train_accuracy": 0.5,
                    "val_accuracy": 0.45,
                },
                {
                    "epoch": 2,
                    "train_loss": 0.8,
                    "val_loss": 0.9,
                    "train_accuracy": 0.7,
                    "val_accuracy": 0.65,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "plot.png"

            plot_runs([("toy", metrics)], output_path, "Toy Plot")

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
