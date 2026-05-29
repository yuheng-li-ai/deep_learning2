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

This repository contains the trained checkpoints for the Neural Network and Deep Learning Project 2 CIFAR-10 classification report. The final selected model is stored separately from the auxiliary ablation checkpoints.

## Final Selected Model

The final model used for the report conclusion is:

- Architecture: VGG-A with BatchNorm and GELU activations
- File: `final/vgg_a_bn_gelu_adam_lr1e-3_best.pt`
- Metrics: `final/vgg_a_bn_gelu_adam_lr1e-3_metrics.json`
- Dataset: CIFAR-10
- Input size: 32 x 32 RGB images
- Number of classes: 10
- Parameters: 9,756,426
- Framework: PyTorch

## Final Model Training Configuration

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

## Final Model Result

- Best test accuracy: 0.8396
- Best test error: 0.1604
- Best epoch: 19
- Final test accuracy: 0.8373
- Final test loss: 0.7710

## Ablation Checkpoints

The `ablations/` directory contains auxiliary checkpoints used for the experimental comparisons in the report:

- `vgg_a_adam_lr1e-3_best.pt`: plain VGG-A baseline.
- `vgg_a_bn_adam_lr1e-3_best.pt`: VGG-A with BatchNorm and ReLU.
- `vgg_a_dropout_adam_lr1e-3_best.pt`: VGG-A with classifier Dropout.
- `vgg_a_light_adam_lr1e-3_best.pt`: reduced-width VGG-A-Light.
- `vgg_a_bn_adamw_lr1e-3_wd1e-4_best.pt`: VGG-A-BN trained with AdamW and weight decay.
- `vgg_a_bn_leaky_relu_adam_lr1e-3_best.pt`: VGG-A-BN with LeakyReLU.
Each ablation checkpoint has a matching `*_metrics.json` file with the training configuration and per-epoch metrics.

Loss-landscape learning-rate sweep checkpoints are not uploaded because they are auxiliary trajectory-analysis runs. Their metrics, step-loss curves, figures, and source code are available in the GitHub repository. The strict-audit regularization sweep outputs are stored under `reports/runs/strict_audit/` in the GitHub repository.

## Loading the Final Checkpoint

The checkpoint stores a dictionary with `model_state_dict`, `config`, `epoch`, and `val_accuracy`.

```python
import torch
from codes.VGG_BatchNorm.models.vgg import VGG_A_BatchNorm_GELU

checkpoint = torch.load(
    "final/vgg_a_bn_gelu_adam_lr1e-3_best.pt",
    map_location="cpu",
)
model = VGG_A_BatchNorm_GELU()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

The complete code, experiment logs, figures, and final report source are available at:

https://github.com/yuheng-li-ai/deep_learning2
