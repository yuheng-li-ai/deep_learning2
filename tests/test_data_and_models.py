import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VGG_ROOT = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
sys.path.insert(0, str(VGG_ROOT))

from data.loaders import PartialDataset
from models.vgg import (
    VGG_A,
    VGG_A_BatchNorm,
    VGG_A_BatchNorm_GELU,
    VGG_A_BatchNorm_LeakyReLU,
    get_number_of_parameters,
)


class ToyDataset(Dataset):
    def __init__(self):
        self.items = [(torch.tensor([idx]), idx) for idx in range(5)]

    def __getitem__(self, index):
        return self.items[index]

    def __len__(self):
        return len(self.items)


class DataLoaderTests(unittest.TestCase):
    def test_partial_dataset_indexes_wrapped_dataset(self):
        dataset = PartialDataset(ToyDataset(), n_items=3)

        self.assertEqual(len(dataset), 3)
        item, label = dataset[2]

        self.assertEqual(item.item(), 2)
        self.assertEqual(label, 2)


class ModelTests(unittest.TestCase):
    def test_vgg_a_forward_shape(self):
        model = VGG_A(init_weights=False)
        x = torch.randn(2, 3, 32, 32)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertGreater(get_number_of_parameters(model), 0)

    def test_vgg_a_batchnorm_forward_shape_and_layers(self):
        model = VGG_A_BatchNorm(init_weights=False)
        x = torch.randn(2, 3, 32, 32)

        logits = model(x)
        batchnorm_layers = [
            module for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d)
        ]

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertGreaterEqual(len(batchnorm_layers), 1)

    def test_vgg_a_batchnorm_leaky_relu_forward_shape_and_layers(self):
        model = VGG_A_BatchNorm_LeakyReLU(init_weights=False)
        x = torch.randn(2, 3, 32, 32)

        logits = model(x)
        activation_layers = [
            module for module in model.modules() if isinstance(module, torch.nn.LeakyReLU)
        ]

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertGreaterEqual(len(activation_layers), 1)

    def test_vgg_a_batchnorm_gelu_forward_shape_and_layers(self):
        model = VGG_A_BatchNorm_GELU(init_weights=False)
        x = torch.randn(2, 3, 32, 32)

        logits = model(x)
        activation_layers = [
            module for module in model.modules() if isinstance(module, torch.nn.GELU)
        ]

        self.assertEqual(tuple(logits.shape), (2, 10))
        self.assertGreaterEqual(len(activation_layers), 1)


if __name__ == "__main__":
    unittest.main()
