import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGG_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(VGG_ROOT))

from plot_loss_landscape import compute_envelope, load_step_losses, parse_run_arg


class PlotLossLandscapeTests(unittest.TestCase):
    def test_parse_run_arg_requires_group_and_name(self):
        group, name, path = parse_run_arg("bn:lr1e-3=reports/run")

        self.assertEqual(group, "bn")
        self.assertEqual(name, "lr1e-3")
        self.assertEqual(path, Path("reports/run"))

    def test_compute_envelope_uses_common_step_length(self):
        envelope = compute_envelope([[3.0, 2.0, 1.0], [2.5, 2.2]])

        np.testing.assert_array_equal(envelope["steps"], np.array([1, 2]))
        np.testing.assert_allclose(envelope["min_curve"], np.array([2.5, 2.0]))
        np.testing.assert_allclose(envelope["max_curve"], np.array([3.0, 2.2]))
        np.testing.assert_allclose(envelope["mean_curve"], np.array([2.75, 2.1]))

    def test_load_step_losses_accepts_run_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            with (run_dir / "step_losses.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["step", "epoch", "batch", "train_loss"]
                )
                writer.writeheader()
                writer.writerow(
                    {"step": 1, "epoch": 1, "batch": 1, "train_loss": 2.0}
                )

            self.assertEqual(load_step_losses(run_dir), [2.0])


if __name__ == "__main__":
    unittest.main()
