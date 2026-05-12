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

VGG-A with BatchNorm and LeakyReLU run:

```bash
mkdir -p reports/runs

nohup python codes/VGG_BatchNorm/train_cifar.py \
  --model vgg_a_bn_leaky_relu \
  --device cuda:2 \
  --epochs 20 \
  --batch-size 128 \
  --optimizer adam \
  --lr 1e-3 \
  --output-dir reports/runs/vgg_a_bn_leaky_relu_adam_lr1e-3 \
  > reports/runs/vgg_a_bn_leaky_relu_adam_lr1e-3.log 2>&1 &
```

VGG-A with BatchNorm and GELU run:

```bash
mkdir -p reports/runs

nohup python codes/VGG_BatchNorm/train_cifar.py \
  --model vgg_a_bn_gelu \
  --device cuda:2 \
  --epochs 20 \
  --batch-size 128 \
  --optimizer adam \
  --lr 1e-3 \
  --output-dir reports/runs/vgg_a_bn_gelu_adam_lr1e-3 \
  > reports/runs/vgg_a_bn_gelu_adam_lr1e-3.log 2>&1 &
```

Each run writes:

- `metrics.json`
- `metrics.csv`
- `best.pt`

When `--record-step-losses` is set, the run also writes:

- `step_losses.csv`

Model weights are ignored by Git and should later be uploaded to external
storage for the final PDF report.

## Plotting Runs

Generate a single-run or multi-run comparison plot from saved `metrics.json`
files:

```bash
python codes/VGG_BatchNorm/plot_runs.py \
  --run vgg_a=reports/runs/vgg_a_adam_lr1e-3 \
  --run vgg_a_bn=reports/runs/vgg_a_bn_adam_lr1e-3 \
  --output reports/figures/vgg_a_vs_bn_adam_lr1e-3.png \
  --title "VGG-A vs VGG-A-BN, Adam lr=1e-3"
```

The script prints a compact JSON summary for each run, including best epoch,
best validation accuracy, best validation error, final training accuracy, and
final validation accuracy.

## Loss Landscape Runs

The loss-landscape requirement needs per-training-step losses across several
learning rates, with and without BatchNorm. Run these manually from the
terminal; this launches the eight runs sequentially inside one `nohup` job:

```bash
mkdir -p reports/runs/loss_landscape

nohup sh -c '
for model in vgg_a vgg_a_bn; do
  for lr in 1e-4 5e-4 1e-3 2e-3; do
    out=reports/runs/loss_landscape/${model}_adam_lr${lr}_steps
    python codes/VGG_BatchNorm/train_cifar.py \
      --model ${model} \
      --device cuda:2 \
      --epochs 20 \
      --batch-size 128 \
      --optimizer adam \
      --lr ${lr} \
      --output-dir ${out} \
      --record-step-losses \
      > ${out}.log 2>&1
  done
done
' > reports/runs/loss_landscape/all.log 2>&1 &
```

After all eight runs finish, generate the BN vs. no-BN loss envelope plot:

```bash
python codes/VGG_BatchNorm/plot_loss_landscape.py \
  --run no_bn:1e-4=reports/runs/loss_landscape/vgg_a_adam_lr1e-4_steps \
  --run no_bn:5e-4=reports/runs/loss_landscape/vgg_a_adam_lr5e-4_steps \
  --run no_bn:1e-3=reports/runs/loss_landscape/vgg_a_adam_lr1e-3_steps \
  --run no_bn:2e-3=reports/runs/loss_landscape/vgg_a_adam_lr2e-3_steps \
  --run bn:1e-4=reports/runs/loss_landscape/vgg_a_bn_adam_lr1e-4_steps \
  --run bn:5e-4=reports/runs/loss_landscape/vgg_a_bn_adam_lr5e-4_steps \
  --run bn:1e-3=reports/runs/loss_landscape/vgg_a_bn_adam_lr1e-3_steps \
  --run bn:2e-3=reports/runs/loss_landscape/vgg_a_bn_adam_lr2e-3_steps \
  --output reports/figures/vgg_a_loss_landscape_bn_vs_no_bn.png \
  --summary reports/figures/vgg_a_loss_landscape_bn_vs_no_bn_summary.json \
  --title "VGG-A Loss Landscape: BatchNorm vs No BatchNorm"
```
