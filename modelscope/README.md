---
license: Apache License 2.0
tags:
  - computer-vision
  - image-classification
  - cifar10
  - pytorch
  - batch-normalization
tasks:
  - image-classification
frameworks:
  - PyTorch
---

# Deep Learning Project 2 Final Model

This repository contains the final trained checkpoint for the Neural Network and Deep Learning Project 2 CIFAR-10 classification report.

## Model

The uploaded checkpoint is the best-performing model from the experiment suite:

- Architecture: VGG-A with BatchNorm and GELU activations
- Dataset: CIFAR-10
- Input size: 32 x 32 RGB images
- Number of classes: 10
- Parameters: 9,756,426
- Framework: PyTorch

## Training Configuration

- Epochs: 20
- Batch size: 128
- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 0.0
- Loss: CrossEntropyLoss
- Seed: 2020
- Data preprocessing: ToTensor, then Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
- Training split: full CIFAR-10 training split
- Evaluation split: full CIFAR-10 test split

## Result

- Best test accuracy: 0.8396
- Best test error: 0.1604
- Best epoch: 19
- Final test accuracy: 0.8373
- Final test loss: 0.7710

## Files

- `vgg_a_bn_gelu_adam_lr1e-3_best.pt`: trained PyTorch checkpoint.
- `vgg_a_bn_gelu_adam_lr1e-3_metrics.json`: full per-epoch metrics and training configuration.

## Loading the Checkpoint

The checkpoint stores a dictionary with `model_state_dict`, `config`, `epoch`, and `val_accuracy`.

```python
import torch
from codes.VGG_BatchNorm.models.vgg import VGG_A_BatchNorm_GELU

checkpoint = torch.load("vgg_a_bn_gelu_adam_lr1e-3_best.pt", map_location="cpu")
model = VGG_A_BatchNorm_GELU()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The complete code, experiment logs, figures, and final report source are available at:

https://github.com/yuheng-li-ai/deep_learning2
