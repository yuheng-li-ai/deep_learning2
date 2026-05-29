import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGG_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(VGG_ROOT))

from plot_gradient_smoothness import parse_run_arg


class PlotGradientSmoothnessTests(unittest.TestCase):
    def test_parse_run_arg_accepts_label_model_checkpoint(self):
        label, model_name, checkpoint = parse_run_arg("Plain=vgg_a=run/best.pt")

        self.assertEqual(label, "Plain")
        self.assertEqual(model_name, "vgg_a")
        self.assertEqual(checkpoint, Path("run/best.pt"))

    def test_parse_run_arg_rejects_unknown_model(self):
        with self.assertRaises(Exception):
            parse_run_arg("Plain=unknown=run/best.pt")


if __name__ == "__main__":
    unittest.main()
