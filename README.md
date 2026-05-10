# Deep Learning Project 2

This repository contains the working code and report draft for Project 2 of
"Neural Network and Deep Learning".

## Repository Notes

The execution environment contains a read-only empty `.git/` directory. The
usable Git metadata for this project is stored in `.git-real/`, so local Git
commands in this workspace use:

```bash
git --git-dir=.git-real --work-tree=. status
```

The GitHub repository is:

```text
https://github.com/yuheng-li-ai/deep_learning2
```

## Environment Baseline

- Python: 3.12.9
- PyTorch: 2.6.0+cu126
- Torchvision: 0.21.0+cu126
- GPU checked on this machine: 3 x NVIDIA RTX A6000

During the first environment check, GPU 2 was the mostly free GPU. For manual
training, prefer `--device cuda:2` when it is still available.

## Tests

Run the lightweight non-training checks with:

```bash
python -m unittest discover -s tests
python -m compileall codes/VGG_BatchNorm tests
```

## Manual Training

Training should be launched manually from your terminal. The training script
uses `tqdm` progress bars for the epoch loop, the train batches, and the
validation batches.

Small smoke run:

```bash
mkdir -p reports/runs

python codes/VGG_BatchNorm/train_cifar.py \
  --model vgg_a_light \
  --device cuda:2 \
  --epochs 1 \
  --batch-size 128 \
  --n-train-items 512 \
  --n-val-items 256 \
  --output-dir reports/runs/smoke_vgg_a_light
```

Baseline VGG-A run:

```bash
mkdir -p reports/runs

python codes/VGG_BatchNorm/train_cifar.py \
  --model vgg_a \
  --device cuda:2 \
  --epochs 20 \
  --batch-size 128 \
  --optimizer adam \
  --lr 1e-3 \
  --output-dir reports/runs/vgg_a_adam_lr1e-3
```

VGG-A with BatchNorm run:

```bash
mkdir -p reports/runs

python codes/VGG_BatchNorm/train_cifar.py \
  --model vgg_a_bn \
  --device cuda:2 \
  --epochs 20 \
  --batch-size 128 \
  --optimizer adam \
  --lr 1e-3 \
  --output-dir reports/runs/vgg_a_bn_adam_lr1e-3
```

Each run writes:

- `metrics.json`
- `metrics.csv`
- `best.pt`

Model weights are ignored by Git and should later be uploaded to external
storage for the final PDF report.
