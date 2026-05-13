# Project 2 Working Draft

This document is the running English draft for Project 2 of "Neural Network and Deep Learning". It records the assignment requirements, engineering plan, experiment protocol, results, and report-writing notes. Every completed experiment should append its configuration, metrics, figure paths, interpretation, and failure notes here. The final LaTeX report and PDF will be written from this draft.

## 0. Baseline Status

- Assignment PDF: `project_2_2026.pdf`, 6 pages, created on 2026-05-08.
- GitHub repository: `https://github.com/yuheng-li-ai/deep_learning2`.
- Local Git note: the environment provides a read-only empty `.git/` directory, so the real local repository metadata is stored in `.git-real/` and Git commands must use `--git-dir=.git-real --work-tree=.`.
- Main starter code directory: `codes/VGG_BatchNorm/`.
- Existing figures: `pic/CIFAR-10.png`, `pic/vgg.png`, `pic/loss_landscape.png`, `pic/adam.png`, `pic/dessilbi.png`.
- Existing bibliography: `bib.bib`.

Environment check on 2026-05-10:

- Python: 3.12.9.
- PyTorch: 2.6.0+cu126.
- Torchvision: 0.21.0+cu126.
- System GPU status from `nvidia-smi`: 3 x NVIDIA RTX A6000, driver 550.54.14, system CUDA 12.4.
- PyTorch CUDA under normal sandbox: unavailable, with `Can't initialize NVML`.
- PyTorch CUDA outside sandbox: available, 3 GPUs visible.
- Tiny CUDA matmul on `cuda:2`: passed. GPU 2 was essentially free during the check.

## 1. Submission Requirements

- Deadline: 23:59, 2026-06-14.
- Submission platform: elearning.
- Final submission artifact: one PDF report.
- The PDF report must include:
  - Student name.
  - Student ID.
  - GitHub link to the code.
  - Dataset link.
  - Link to trained model weights.
- Missing code link or model-weight link will lead to a score penalty.
- Dataset and model weights may be uploaded to Google Drive or another net-disk platform.
- Late penalty: 10% score reduction for each delayed week.
- The write-up must document the experiments and main findings, not only final numbers.

## 2. Score Structure

- Task 1: Train a Network on CIFAR-10, 60%.
- Task 2: Batch Normalization, 30%.
- The remaining implicit score depends on report quality, completeness, reproducibility, links, and clarity of interpretation.

## 3. Task 1: CIFAR-10 Classification Requirements

Goal: train neural networks on CIFAR-10, optimize classification performance, report the best test error, and describe the network structure that achieved it.

CIFAR-10 facts:

- 60,000 RGB images.
- Image size: 32 x 32.
- 10 classes: airplane, car, bird, cat, deer, dog, frog, horse, ship, truck.
- 6,000 images per class.

The network must contain all of the following components, worth 16%:

- Fully connected layer.
- 2D convolutional layer.
- 2D pooling layer.
- Activation functions.

The network must contain at least one of the following components, worth 8%:

- Batch normalization.
- Dropout.
- Residual connection.
- Other reasonable component.

The optimization study must try all of the following strategies, worth 8%:

- Different numbers of neurons or filters.
- Different loss functions, including different regularization settings.
- Different activation functions.

The project must also choose at least one of the following optimization strategies, worth 8%:

- Try different optimizers using `torch.optim`.
- Implement an optimizer for a network containing the required basic components, while still using `torch.optim` to optimize the full model.
- Implement an optimizer for the full model from scratch.

The report must reveal insights about the network, worth 8%. Possible directions:

- Filter visualization.
- Loss landscape.
- Network interpretation.
- Other meaningful visualizations of the model or training process.

Task 1 scoring emphasis:

- Classification performance, especially test error.
- Number of parameters.
- Network structure.
- Training speed.
- For similar performance, the grader may compare total parameters, structure, and whether new optimization algorithms are used.
- Insightful learned-model or training-process visualizations are valuable.
- Reporting multiple networks is allowed.
- Directly using public models without modification may be penalized.

## 4. Task 2: Batch Normalization Requirements

Goal: test the effectiveness of batch normalization during training, then investigate how batch normalization helps optimization.

Experimental setup:

- Dataset: CIFAR-10 image classification.
- Architecture: similar to VGG-A, with smaller linear layers because the input is 32 x 32 x 3 rather than 224 x 224 x 3.
- Starter code: PyTorch.
- Partial datasets may be used for faster preliminary experiments via `n_items`.

### 4.1 VGG-A With and Without BN

This part is worth 15%.

Requirements:

- Understand the starter code instead of treating it as a black box.
- Train the baseline `VGG_A` first.
- Implement `VGG_A_BatchNorm` or an equivalent BN variant by adding BN layers to the original network.
- Compare the performance and training characteristics of VGG-A with and without BN.
- Visualize training results for both models.
- Extending or modifying the starter code for clearer experimental evidence is encouraged.

Recommended records:

- Train loss per epoch.
- Train accuracy per epoch.
- Validation/test accuracy per epoch.
- Best test error and corresponding epoch.
- Training time.
- Parameter count.
- Seed, batch size, optimizer, learning rate, scheduler, and data augmentation.

### 4.2 How BN Helps Optimization

This part is worth 15%.

The PDF asks us to analyze BN from the optimization-landscape perspective. It explicitly mentions:

- Loss landscape or variation of the loss value.
- Gradient predictiveness or change of the loss gradient.
- Maximum difference in gradient over distance.

Minimum required loss-landscape experiment from the PDF:

1. Choose a list of learning rates as different step sizes, for example `[1e-3, 2e-3, 1e-4, 5e-4]`.
2. Train models with those learning rates and save training losses for every step.
3. Maintain `max_curve` and `min_curve`: for the same training step, take the maximum and minimum losses across all learning-rate runs.
4. Plot the curves and use `matplotlib.pyplot.fill_between` to fill the region between them.
5. Repeat the same method for VGG-A with BN and without BN.
6. Plot the BN and no-BN comparisons in the same figure.

The report must explain:

- The selected learning rates.
- The reproducibility setup.
- The final comparison plot.
- Whether BN makes the loss landscape smoother and what evidence supports that claim.

## 5. Starter Code Findings

Files inspected:

- `codes/VGG_BatchNorm/VGG_Loss_Landscape.py`
  - Starter training and visualization script for the BN/loss-landscape task.
  - Contains placeholders for sample inspection, accuracy computation, loss/gradient recording, validation, min/max curves, and plotting.
  - Hard-codes `cuda:3` and calls `torch.cuda.get_device_name(3)`, but the machine has only GPU indices 0, 1, and 2. This must be fixed before running training.
  - Imports `VGG_A_BatchNorm`, but that class is not implemented yet.
- `codes/VGG_BatchNorm/models/vgg.py`
  - Provides `VGG_A`, `VGG_A_Light`, and `VGG_A_Dropout`.
  - `VGG_A` already includes convolution, pooling, activations, and fully connected layers.
  - Imports `init_weights_` from `codes_for_pj.utils.nn`, but the current project tree does not contain `codes_for_pj`; the import path must be fixed.
- `codes/VGG_BatchNorm/data/loaders.py`
  - Loads CIFAR-10 through `torchvision.datasets.CIFAR10`.
  - Supports `n_items` for partial-dataset experiments.
  - `PartialDataset.__getitem__` lacks an index argument, so partial datasets will fail until fixed.
- `codes/VGG_BatchNorm/utils/nn.py`
  - Provides initialization for Conv2d, BatchNorm, and Linear layers.

## 6. Execution Plan

### Phase 1: Git, Environment, and Project Baseline

- Initialize usable Git metadata through `.git-real/`.
- Connect to `https://github.com/yuheng-li-ai/deep_learning2`.
- Commit the original materials plus this English draft.
- Keep generated caches, datasets, model weights, and local ECC files out of Git.
- Record environment versions and GPU availability.

### Phase 2: Make the Starter Code Runnable

- Fix import paths.
- Fix `PartialDataset.__getitem__`.
- Replace hard-coded device selection with robust GPU/CPU selection.
- Complete training, validation, accuracy, metric saving, and figure saving.
- Run a small smoke test with a partial dataset and a small epoch count.

### Phase 3: Task 1 Main CIFAR-10 Experiments

Minimum experiment matrix:

- Baseline CNN or VGG-A.
- Filter/neuron ablation with at least 2 to 3 width or classifier-size variants.
- Loss and regularization ablation, such as CrossEntropy, weight decay, and optionally label smoothing.
- Activation ablation, such as ReLU, LeakyReLU, and GELU or SiLU.
- Optimizer comparison, such as SGD with momentum, Adam, and AdamW.
- At least one structural enhancement: BN, Dropout, or residual connection.

Each experiment should record:

- Commit hash.
- Model and configuration.
- Seed, epochs, batch size, optimizer, learning rate, scheduler, weight decay, activation, loss, and augmentation.
- Parameter count.
- Runtime.
- Best validation/test accuracy.
- Best test error.
- Figure paths.
- Key observations.

### Phase 4: VGG-A BN Comparison

- Implement `VGG_A_BatchNorm`.
- Train VGG-A without BN and VGG-A with BN under matched settings.
- Generate comparison plots:
  - Train loss curve.
  - Train/test accuracy curve.
  - Convergence speed.
  - Final and best test error.
- Explain the effect of BN on convergence, stability, and generalization.

### Phase 5: BN Optimization-Landscape Study

- Choose learning rates such as `1e-4`, `5e-4`, `1e-3`, and `2e-3`.
- Run no-BN and BN models across the same learning-rate list.
- Save per-step losses.
- Generate min/max curves and filled landscape plots.
- Optional extensions:
  - Record gradient norms.
  - Compute gradient cosine similarity between nearby steps.
  - Estimate gradient predictiveness.
  - Estimate maximum gradient difference over comparable distances.

### Phase 6: Final Report

- Convert this draft into an English LaTeX report.
- Suggested report sections:
  - Introduction.
  - CIFAR-10 model design and optimization.
  - Batch normalization comparison.
  - Optimization landscape analysis.
  - Discussion and limitations.
  - Conclusion.
  - References.
- Insert the GitHub link, dataset link, and model-weight link.
- Export the final PDF.
- Final checklist:
  - PDF opens correctly.
  - Student name and ID are included.
  - All figures are readable.
  - GitHub link is accessible.
  - Dataset and model-weight links are accessible.
  - Key results can be traced back to this draft and Git commits.

## 7. Candidate Extension Directions

High-priority directions tied closely to scoring:

- Accuracy-parameter-speed trade-off among small VGG, standard VGG-A, BN, Dropout, and residual variants.
- Data augmentation: random crop, random horizontal flip, Cutout, MixUp, or CutMix.
- Regularization: weight decay, Dropout, label smoothing.
- Learning-rate schedules: StepLR, cosine annealing, warmup.
- Optimizers: SGD with momentum, Adam, AdamW.
- Filter visualization of the first convolutional layer.
- Feature-map visualization for selected layers.
- Confusion matrix and class-level error analysis.

Research-oriented directions for stronger discussion:

- BN stability under high learning rates.
- Gradient norm and gradient cosine-similarity changes with BN.
- Whether BN narrows the loss envelope across learning rates.
- Interaction between BN and Dropout.
- Relative benefit of residual connections and BN on small CIFAR-10 networks.
- Wall-clock time and epoch count required to reach a fixed accuracy threshold.

Riskier directions to attempt only if time allows:

- Implementing a custom optimizer.
- Implementing layers or a full training framework from scratch.
- DessiLBI or structural sparsity experiments.
- Large-scale model search.

## 8. Experiment Log Template

Append each experiment using this format:

```markdown
### Experiment YYYY-MM-DD-NN: <short name>

- Commit:
- Code changes:
- Dataset:
- Model:
- Parameters:
- Seed:
- Device:
- Epochs:
- Batch size:
- Optimizer:
- Learning rate:
- Scheduler:
- Weight decay / regularization:
- Activation:
- Loss:
- Augmentation:
- Runtime:
- Best train accuracy:
- Best validation/test accuracy:
- Best test error:
- Saved weights:
- Figures:
- Raw metrics:
- Result summary:
- Interpretation:
- Problems / next action:
```

## 9. Open Items

- Student name and ID will be filled in near final report writing.
- The final model-weight hosting location is not decided yet.
- The final dataset link can use the official CIFAR-10 website or a net-disk mirror.
- Training commands that need CUDA should run outside the normal sandbox because PyTorch CUDA is unavailable inside the sandbox.

## 10. Implementation Log

### 2026-05-10: Git and Environment Baseline

- Commit: `f1809b3`.
- Summary:
  - Established the initial Git baseline and pushed it to GitHub.
  - Converted this draft to English.
  - Added `.gitignore` rules for local caches, dataset archives, model weights, and local ECC files.
  - Verified that PyTorch CUDA works outside the sandbox.
- Environment:
  - Python 3.12.9.
  - PyTorch 2.6.0+cu126.
  - Torchvision 0.21.0+cu126.
  - 3 x NVIDIA RTX A6000.
  - Tiny CUDA matmul passed on `cuda:2`.

### 2026-05-10: Phase 2 Starter-Code Readiness

- Commit: `7b03fa3`.
- Code changes:
  - Fixed `PartialDataset` indexing.
  - Fixed the VGG utility import path.
  - Implemented `VGG_A_BatchNorm`.
  - Added `codes/VGG_BatchNorm/train_cifar.py` as a manual training entry point.
  - Added `tqdm` progress bars for epoch, train-batch, and validation-batch loops.
  - Added lightweight `unittest` coverage for data wrapping, model forward passes, and training utility functions.
  - Added `README.md` with test and manual training commands.
- Verification:
  - `python -m unittest discover -s tests`: passed.
  - `python -m compileall codes/VGG_BatchNorm tests`: passed.
- Training status:
  - No formal training was run by the agent.
  - Training is intended to be launched manually from the user's terminal.
  - Recommended manual device argument: `--device cuda:2`, if GPU 2 is still free.

### 2026-05-10: Manual Training Command Fix

- Commit: `c5aa353`.
- Issue:
  - A manual `nohup` training command failed before Python started because shell redirection tried to create `reports/runs/vgg_a_adam_lr1e-3.log`, but the parent directory did not exist yet.
- Fix:
  - Added `reports/runs/.gitkeep` so the default run-log parent directory exists after clone.
  - Updated README commands to create `reports/runs` before launching training.
- Training status:
  - The failed command did not run training.
  - The corrected command can be relaunched manually by the user.

## 11. Experiment Results

### Experiment 2026-05-10-01: VGG-A Baseline with Adam

- Commit before training: `c5aa353`.
- Code state:
  - Training entry point: `codes/VGG_BatchNorm/train_cifar.py`.
  - Model: `VGG_A`.
  - Progress display: `tqdm` epoch, train-batch, and validation-batch progress bars.
- Dataset:
  - CIFAR-10 train split and test split loaded by `torchvision.datasets.CIFAR10`.
  - `n_train_items = -1`, `n_val_items = -1`, so the full train/test splits were used.
- Configuration:
  - Device: `cuda:2`.
  - Epochs: 20.
  - Batch size: 128.
  - Optimizer: Adam.
  - Learning rate: 1e-3.
  - Weight decay: 0.0.
  - Loss: CrossEntropyLoss.
  - Seed: 2020.
  - Data preprocessing: ToTensor and Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
- Model size:
  - Parameters: 9,750,922.
- Runtime:
  - Total recorded epoch time: 145.52 seconds.
  - Epoch 1 took 62.32 seconds, likely due to dataset preparation and initial runtime overhead.
  - Later epochs were approximately 4.3 to 4.5 seconds each.
- Best result:
  - Best validation/test accuracy: 0.7651.
  - Best validation/test error: 0.2349.
  - Best epoch: 18.
  - Best checkpoint: `reports/runs/vgg_a_adam_lr1e-3/best.pt`.
- Final epoch:
  - Train loss: 0.0977.
  - Train accuracy: 0.9680.
  - Validation/test loss: 1.4421.
  - Validation/test accuracy: 0.7411.
- Raw metrics:
  - `reports/runs/vgg_a_adam_lr1e-3/metrics.json`.
  - `reports/runs/vgg_a_adam_lr1e-3/metrics.csv`.
- Figure:
  - `reports/figures/vgg_a_adam_lr1e-3_curves.png`.
- Interpretation:
  - The baseline VGG-A model reaches usable CIFAR-10 performance, with the best validation/test accuracy of 76.51%.
  - The gap between final train accuracy (96.80%) and final validation/test accuracy (74.11%) indicates strong overfitting.
  - Validation accuracy improves quickly through the first 8 to 12 epochs, then fluctuates while validation loss increases.
  - This run is a useful no-BN baseline for later BN, regularization, activation, optimizer, and width comparisons.
- Next action:
  - Run `VGG_A_BatchNorm` with the same optimizer, learning rate, batch size, seed, and epoch count.
  - Compare convergence speed, best validation/test accuracy, final overfitting gap, and loss-curve stability.

### 2026-05-10: Non-Training Comparison Tooling

- Commit: `671a8f6`.
- Code changes:
  - Added `codes/VGG_BatchNorm/plot_runs.py`.
  - Added unit tests for run summarization and plot-file creation.
  - Updated README with a reusable command for comparing saved `metrics.json` files.
- Verification:
  - Generated `reports/figures/vgg_a_adam_lr1e-3_summary.png` from the completed VGG-A baseline metrics.
  - `python -m unittest discover -s tests`: passed.
  - `python -m compileall codes/VGG_BatchNorm tests`: passed.
- Purpose:
  - Once the user manually trains `VGG_A_BatchNorm`, the saved metrics can be plotted against the no-BN baseline without modifying training code.

### Experiment 2026-05-12-01: VGG-A with BatchNorm, Adam

- Commit before training: `671a8f6`.
- Code state:
  - Training entry point: `codes/VGG_BatchNorm/train_cifar.py`.
  - Model: `VGG_A_BatchNorm`.
  - Comparison tooling: `codes/VGG_BatchNorm/plot_runs.py`.
- Dataset:
  - CIFAR-10 train split and test split loaded by `torchvision.datasets.CIFAR10`.
  - `n_train_items = -1`, `n_val_items = -1`, so the full train/test splits were used.
- Configuration:
  - Device: `cuda:2`.
  - Epochs: 20.
  - Batch size: 128.
  - Optimizer: Adam.
  - Learning rate: 1e-3.
  - Weight decay: 0.0.
  - Loss: CrossEntropyLoss.
  - Seed: 2020.
  - Data preprocessing: ToTensor and Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
- Model size:
  - Parameters: 9,756,426.
  - Parameter increase over no-BN VGG-A: 5,504 parameters.
- Runtime:
  - Total recorded epoch time: 104.39 seconds.
  - Epoch 1 took 7.52 seconds.
  - Later epochs were approximately 5.0 to 5.3 seconds each.
- Best result:
  - Best validation/test accuracy: 0.8324.
  - Best validation/test error: 0.1676.
  - Best epoch: 12.
  - Best checkpoint: `reports/runs/vgg_a_bn_adam_lr1e-3/best.pt`.
- Final epoch:
  - Train loss: 0.0416.
  - Train accuracy: 0.9862.
  - Validation/test loss: 0.9035.
  - Validation/test accuracy: 0.8257.
- Raw metrics:
  - `reports/runs/vgg_a_bn_adam_lr1e-3/metrics.json`.
  - `reports/runs/vgg_a_bn_adam_lr1e-3/metrics.csv`.
- Figures:
  - `reports/figures/vgg_a_bn_adam_lr1e-3_summary.png`.
  - `reports/figures/vgg_a_vs_bn_adam_lr1e-3.png`.
- Comparison with no-BN VGG-A:
  - No-BN best validation/test accuracy: 0.7651 at epoch 18.
  - BN best validation/test accuracy: 0.8324 at epoch 12.
  - Absolute accuracy gain: 6.73 percentage points.
  - Absolute test-error reduction: 6.73 percentage points.
  - BN reaches strong validation accuracy earlier, improving convergence behavior under the same Adam learning rate and batch size.
  - BN also keeps the final validation loss lower than the no-BN model: 0.9035 vs. 1.4421.
- Interpretation:
  - BatchNorm substantially improves both accuracy and convergence speed in this matched VGG-A setting.
  - The no-BN model overfits strongly after the early/middle epochs, while the BN model reaches a higher best validation accuracy and ends with a much stronger final validation accuracy.
  - The train-validation gap remains visible for BN, so BN is not enough by itself to remove overfitting; additional regularization or data augmentation should still be tested.
- Next action:
  - Run a parameter/width ablation such as `VGG_A_Light`.
  - Run a regularization ablation such as `VGG_A_Dropout`.
  - Run an optimizer comparison for `VGG_A_BatchNorm`, for example AdamW or SGD with momentum.

### Experiment 2026-05-12-02: VGG-A with Dropout, Adam

- Commit before training: `01a3436`.
- Code state:
  - Training entry point: `codes/VGG_BatchNorm/train_cifar.py`.
  - Model: `VGG_A_Dropout`.
  - Dropout placement: classifier dropout before the first two linear layers.
- Dataset:
  - CIFAR-10 train split and test split loaded by `torchvision.datasets.CIFAR10`.
  - `n_train_items = -1`, `n_val_items = -1`, so the full train/test splits were used.
- Configuration:
  - Device: `cuda:2`.
  - Epochs: 20.
  - Batch size: 128.
  - Optimizer: Adam.
  - Learning rate: 1e-3.
  - Weight decay: 0.0.
  - Loss: CrossEntropyLoss.
  - Seed: 2020.
  - Data preprocessing: ToTensor and Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
- Model size:
  - Parameters: 9,750,922.
- Runtime:
  - Total recorded epoch time: 89.38 seconds.
  - Later epochs were approximately 4.3 to 4.7 seconds each.
- Best result:
  - Best validation/test accuracy: 0.7419.
  - Best validation/test error: 0.2581.
  - Best epoch: 12.
  - Best checkpoint: `reports/runs/vgg_a_dropout_adam_lr1e-3/best.pt`.
- Final epoch:
  - Train loss: 0.1410.
  - Train accuracy: 0.9571.
  - Validation/test loss: 1.3755.
  - Validation/test accuracy: 0.7348.
- Raw metrics:
  - `reports/runs/vgg_a_dropout_adam_lr1e-3/metrics.json`.
  - `reports/runs/vgg_a_dropout_adam_lr1e-3/metrics.csv`.
- Figures:
  - `reports/figures/vgg_a_dropout_adam_lr1e-3_summary.png`.
  - `reports/figures/vgg_a_bn_dropout_adam_lr1e-3.png`.
- Comparison with structural variants:
  - VGG-A best validation/test accuracy: 0.7651.
  - VGG-A-BN best validation/test accuracy: 0.8324.
  - VGG-A-Dropout best validation/test accuracy: 0.7419.
  - Dropout under this configuration is 2.32 percentage points below the no-BN VGG-A baseline and 9.05 percentage points below the BN variant.
- Interpretation:
  - Classifier-only Dropout with the default probability does not improve this VGG-A training setup.
  - It slightly reduces final train accuracy compared with no-Dropout VGG-A, but it does not raise validation accuracy.
  - This suggests that the main bottleneck in this configuration is not solved by classifier Dropout alone. BN gives a much stronger gain under the same optimizer, learning rate, seed, and epoch budget.
- Next action:
  - Run `VGG_A_Light` to satisfy a clear parameter/filter/neurons ablation.
  - Then run an optimizer comparison, preferably `VGG_A_BatchNorm` with AdamW or SGD with momentum.

### Experiment 2026-05-12-03: VGG-A-Light Width/Parameter Ablation, Adam

- Commit before training: `3af08ab`.
- Code state:
  - Training entry point: `codes/VGG_BatchNorm/train_cifar.py`.
  - Model: `VGG_A_Light`.
  - Architecture change: smaller convolutional width and a smaller classifier compared with full VGG-A.
- Dataset:
  - CIFAR-10 train split and test split loaded by `torchvision.datasets.CIFAR10`.
  - `n_train_items = -1`, `n_val_items = -1`, so the full train/test splits were used.
- Configuration:
  - Device: `cuda:2`.
  - Epochs: 20.
  - Batch size: 128.
  - Optimizer: Adam.
  - Learning rate: 1e-3.
  - Weight decay: 0.0.
  - Loss: CrossEntropyLoss.
  - Seed: 2020.
  - Data preprocessing: ToTensor and Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
- Model size:
  - Parameters: 285,162.
  - Relative size: about 2.92% of full VGG-A parameters.
  - Parameter reduction: about 97.08% fewer parameters than full VGG-A.
- Runtime:
  - Total recorded epoch time: 56.06 seconds.
  - Later epochs were approximately 2.7 to 2.9 seconds each.
- Best result:
  - Best validation/test accuracy: 0.7027.
  - Best validation/test error: 0.2973.
  - Best epoch: 9.
  - Best checkpoint: `reports/runs/vgg_a_light_adam_lr1e-3/best.pt`.
- Final epoch:
  - Train loss: 0.2069.
  - Train accuracy: 0.9271.
  - Validation/test loss: 1.4801.
  - Validation/test accuracy: 0.6818.
- Raw metrics:
  - `reports/runs/vgg_a_light_adam_lr1e-3/metrics.json`.
  - `reports/runs/vgg_a_light_adam_lr1e-3/metrics.csv`.
- Figures:
  - `reports/figures/vgg_a_light_adam_lr1e-3_summary.png`.
  - `reports/figures/vgg_a_variants_adam_lr1e-3.png`.
- Comparison with full VGG-A variants:
  - VGG-A-Light best validation/test accuracy: 0.7027.
  - Full VGG-A best validation/test accuracy: 0.7651.
  - VGG-A-BN best validation/test accuracy: 0.8324.
  - VGG-A-Dropout best validation/test accuracy: 0.7419.
  - VGG-A-Light is much smaller and faster, but loses 6.24 percentage points against full VGG-A and 12.97 percentage points against VGG-A-BN.
- Interpretation:
  - Reducing width and classifier size gives a large speed and parameter-count benefit, but the capacity loss is clearly visible on CIFAR-10 accuracy.
  - The light model still overfits after its best epoch, so capacity reduction alone does not fully solve generalization.
  - This experiment satisfies the filters/neurons ablation requirement and supports the report discussion about accuracy-parameter-speed trade-offs.
- Next action:
  - Run an optimizer comparison on the best current structure, `VGG_A_BatchNorm`, using AdamW or SGD with momentum.
  - Then prepare loss-landscape experiments across multiple learning rates for with-BN vs without-BN.

### Experiment 2026-05-12-04: VGG-A-BN Optimizer Comparison with AdamW

- Commit before training: `2689627`.
- Code state:
  - Training entry point: `codes/VGG_BatchNorm/train_cifar.py`.
  - Model: `VGG_A_BatchNorm`.
  - Optimizer under test: AdamW with decoupled weight decay.
- Dataset:
  - CIFAR-10 train split and test split loaded by `torchvision.datasets.CIFAR10`.
  - `n_train_items = -1`, `n_val_items = -1`, so the full train/test splits were used.
- Configuration:
  - Device: `cuda:2`.
  - Epochs: 20.
  - Batch size: 128.
  - Optimizer: AdamW.
  - Learning rate: 1e-3.
  - Weight decay: 1e-4.
  - Loss: CrossEntropyLoss.
  - Seed: 2020.
  - Data preprocessing: ToTensor and Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
- Model size:
  - Parameters: 9,756,426.
- Runtime:
  - Total recorded epoch time: 102.32 seconds.
  - Later epochs were approximately 5.0 to 5.2 seconds each.
- Best result:
  - Best validation/test accuracy: 0.8302.
  - Best validation/test error: 0.1698.
  - Best epoch: 20.
  - Best checkpoint: `reports/runs/vgg_a_bn_adamw_lr1e-3_wd1e-4/best.pt`.
- Final epoch:
  - Train loss: 0.0414.
  - Train accuracy: 0.9871.
  - Validation/test loss: 0.8285.
  - Validation/test accuracy: 0.8302.
- Raw metrics:
  - `reports/runs/vgg_a_bn_adamw_lr1e-3_wd1e-4/metrics.json`.
  - `reports/runs/vgg_a_bn_adamw_lr1e-3_wd1e-4/metrics.csv`.
- Figures:
  - `reports/figures/vgg_a_bn_adamw_lr1e-3_wd1e-4_summary.png`.
  - `reports/figures/vgg_a_bn_optimizer_adam_vs_adamw.png`.
- Comparison with VGG-A-BN + Adam:
  - Adam best validation/test accuracy: 0.8324 at epoch 12.
  - AdamW best validation/test accuracy: 0.8302 at epoch 20.
  - AdamW is 0.22 percentage points lower in best accuracy, but its final validation/test accuracy is higher: 0.8302 vs. 0.8257.
  - AdamW also has lower final validation loss: 0.8285 vs. 0.9035.
- Interpretation:
  - Adam and AdamW perform very similarly for VGG-A-BN under this 20-epoch setting.
  - Adam reaches a slightly higher peak earlier, while AdamW with weight decay gives a smoother final state and the best value at the final epoch.
  - This experiment satisfies the optimizer and regularization comparison requirement.
- Next action:
  - Add activation-function variants so the report can explicitly compare activations.
  - Prepare loss-landscape experiments across learning rates for VGG-A with and without BN.

### 2026-05-12: Activation Variant Support

- Commit: `9041dba`.
- Code changes:
  - Added reusable activation construction in `codes/VGG_BatchNorm/models/vgg.py`.
  - Added `VGG_A_BatchNorm_LeakyReLU`.
  - Added `VGG_A_BatchNorm_GELU`.
  - Added training registry keys:
    - `vgg_a_bn_leaky_relu`.
    - `vgg_a_bn_gelu`.
  - Added unit tests to verify both activation variants can run a forward pass and contain the intended activation layer.
  - Added README commands for manual LeakyReLU and GELU training.
- Verification:
  - `python -m unittest discover -s tests`: passed.
  - `python -m compileall codes/VGG_BatchNorm tests`: passed.
  - `python codes/VGG_BatchNorm/train_cifar.py --help` lists both new model choices.
- Training status:
  - No activation-variant training was run by the agent.
  - The user manually ran both activation-variant trainings after this code support was committed.

### Experiment 2026-05-12-05: VGG-A-BN Activation Comparison

- Commit before training: `9041dba`.
- Code state:
  - Training entry point: `codes/VGG_BatchNorm/train_cifar.py`.
  - Baseline activation: ReLU in `VGG_A_BatchNorm`.
  - Activation variants:
    - `VGG_A_BatchNorm_LeakyReLU`.
    - `VGG_A_BatchNorm_GELU`.
- Dataset:
  - CIFAR-10 train split and test split loaded by `torchvision.datasets.CIFAR10`.
  - `n_train_items = -1`, `n_val_items = -1`, so the full train/test splits were used.
- Shared configuration:
  - Device: `cuda:2`.
  - Epochs: 20.
  - Batch size: 128.
  - Optimizer: Adam.
  - Learning rate: 1e-3.
  - Weight decay: 0.0.
  - Loss: CrossEntropyLoss.
  - Seed: 2020.
  - Data preprocessing: ToTensor and Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
- Model size:
  - ReLU parameters: 9,756,426.
  - LeakyReLU parameters: 9,756,426.
  - GELU parameters: 9,756,426.
- Runs:
  - ReLU baseline: `reports/runs/vgg_a_bn_adam_lr1e-3`.
  - LeakyReLU: `reports/runs/vgg_a_bn_leaky_relu_adam_lr1e-3`.
  - GELU: `reports/runs/vgg_a_bn_gelu_adam_lr1e-3`.
- Best results:
  - ReLU best validation/test accuracy: 0.8324, error: 0.1676, best epoch: 12.
  - LeakyReLU best validation/test accuracy: 0.8254, error: 0.1746, best epoch: 18.
  - GELU best validation/test accuracy: 0.8396, error: 0.1604, best epoch: 19.
- Final epoch results:
  - ReLU final train accuracy: 0.9862, final validation/test loss: 0.9035, final validation/test accuracy: 0.8257.
  - LeakyReLU final train accuracy: 0.9840, final validation/test loss: 0.8619, final validation/test accuracy: 0.8194.
  - GELU final train accuracy: 0.9891, final validation/test loss: 0.7710, final validation/test accuracy: 0.8373.
- Runtime:
  - LeakyReLU total recorded epoch time: 99.21 seconds.
  - GELU total recorded epoch time: 98.03 seconds.
- Raw metrics:
  - `reports/runs/vgg_a_bn_leaky_relu_adam_lr1e-3/metrics.json`.
  - `reports/runs/vgg_a_bn_leaky_relu_adam_lr1e-3/metrics.csv`.
  - `reports/runs/vgg_a_bn_gelu_adam_lr1e-3/metrics.json`.
  - `reports/runs/vgg_a_bn_gelu_adam_lr1e-3/metrics.csv`.
- Figures:
  - `reports/figures/vgg_a_bn_leaky_relu_adam_lr1e-3_summary.png`.
  - `reports/figures/vgg_a_bn_gelu_adam_lr1e-3_summary.png`.
  - `reports/figures/vgg_a_bn_activation_comparison_adam_lr1e-3.png`.
- Interpretation:
  - GELU is the best activation in this comparison, improving the best validation/test accuracy by 0.72 percentage points over the ReLU BN baseline and by 1.42 percentage points over LeakyReLU.
  - GELU also has the strongest final validation/test accuracy and lowest final validation/test loss, suggesting that the smoother activation is helpful in this setting.
  - LeakyReLU does not improve peak accuracy over ReLU, although its final validation loss is lower than the ReLU baseline.
  - This experiment satisfies the activation-function comparison requirement.
- Next action:
  - Implement loss-landscape logging for multiple learning rates with and without BatchNorm.
  - Run the loss-landscape experiments manually from the terminal, then use their saved loss curves for the final report.

### 2026-05-12: Loss-Landscape Logging Support

- Commit: current change set; final hash is recorded in Git history.
- Purpose:
  - Prepare the required BatchNorm optimization-landscape experiment without running formal training inside the agent session.
  - Save per-training-step loss values so the max/min loss envelope can be computed across learning rates.
- Code changes:
  - Added `--record-step-losses` to `codes/VGG_BatchNorm/train_cifar.py`.
  - When enabled, training writes `step_losses.csv` with columns:
    - `step`.
    - `epoch`.
    - `batch`.
    - `train_loss`.
  - Added `codes/VGG_BatchNorm/plot_loss_landscape.py`.
  - The plotting script accepts multiple `GROUP:NAME=PATH` runs, computes per-step `min_curve`, `max_curve`, and `mean_curve` for each group, and fills the min-max region with `matplotlib.pyplot.fill_between`.
  - Added unit tests for step-loss persistence and envelope computation.
- Planned manual runs:
  - Models:
    - `vgg_a` for no BatchNorm.
    - `vgg_a_bn` for BatchNorm.
  - Learning rates:
    - `1e-4`.
    - `5e-4`.
    - `1e-3`.
    - `2e-3`.
  - Shared settings:
    - Device: `cuda:2`.
    - Epochs: 20.
    - Batch size: 128.
    - Optimizer: Adam.
    - Weight decay: 0.0.
    - Seed: 2020.
    - Full CIFAR-10 train/test splits.
- Planned output:
  - Per-run directories under `reports/runs/loss_landscape/`.
  - Each run will include `metrics.json`, `metrics.csv`, `step_losses.csv`, and local ignored `best.pt`.
  - Final plot: `reports/figures/vgg_a_loss_landscape_bn_vs_no_bn.png`.
  - Final summary: `reports/figures/vgg_a_loss_landscape_bn_vs_no_bn_summary.json`.
- Manual command location:
  - The exact `nohup` command and final plotting command are documented in `README.md`.

### Experiment 2026-05-13-06: Loss Landscape Across Learning Rates

- Commit before training: `6bf2657`.
- Purpose:
  - Analyze whether BatchNorm reduces the variation of the training loss across different learning-rate step sizes.
  - This directly addresses the project requirement to compare BN and no-BN loss landscapes using per-step loss curves.
- Models:
  - No BatchNorm: `VGG_A`.
  - BatchNorm: `VGG_A_BatchNorm`.
- Dataset:
  - CIFAR-10 train split and test split loaded by `torchvision.datasets.CIFAR10`.
  - `n_train_items = -1`, `n_val_items = -1`, so the full train/test splits were used.
- Shared configuration:
  - Device: `cuda:2`.
  - Epochs: 20.
  - Batch size: 128.
  - Optimizer: Adam.
  - Weight decay: 0.0.
  - Loss: CrossEntropyLoss.
  - Seed: 2020.
  - Data preprocessing: ToTensor and Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]).
  - Step-loss logging: enabled with `--record-step-losses`.
- Learning rates:
  - `1e-4`.
  - `5e-4`.
  - `1e-3`.
  - `2e-3`.
- Completion check:
  - All 8 runs finished.
  - Each run contains 20 epoch records and 7,820 per-step loss records.
- No-BN results:
  - `1e-4`: best accuracy 0.7659 at epoch 17; final accuracy 0.7526; final validation loss 1.3084; total epoch time 88.22 seconds.
  - `5e-4`: best accuracy 0.7840 at epoch 14; final accuracy 0.7801; final validation loss 1.2380; total epoch time 88.40 seconds.
  - `1e-3`: best accuracy 0.7651 at epoch 18; final accuracy 0.7411; final validation loss 1.4421; total epoch time 88.70 seconds.
  - `2e-3`: best accuracy 0.7289 at epoch 16; final accuracy 0.7179; final validation loss 1.1336; total epoch time 88.68 seconds.
- BN results:
  - `1e-4`: best accuracy 0.7453 at epoch 18; final accuracy 0.7203; final validation loss 1.6119; total epoch time 101.69 seconds.
  - `5e-4`: best accuracy 0.8244 at epoch 20; final accuracy 0.8244; final validation loss 0.8311; total epoch time 101.87 seconds.
  - `1e-3`: best accuracy 0.8324 at epoch 12; final accuracy 0.8257; final validation loss 0.9035; total epoch time 101.46 seconds.
  - `2e-3`: best accuracy 0.8305 at epoch 16; final accuracy 0.8192; final validation loss 0.8693; total epoch time 100.90 seconds.
- Envelope summary:
  - No-BN mean envelope width: 0.4084.
  - BN mean envelope width: 0.2147.
  - No-BN final loss range: 0.0576 to 0.2764.
  - BN final loss range: 0.0306 to 0.0551.
  - No-BN maximum envelope width: 2.7904.
  - BN maximum envelope width: 3.5587.
- Raw metrics and step losses:
  - `reports/runs/loss_landscape/vgg_a_adam_lr1e-4_steps/`.
  - `reports/runs/loss_landscape/vgg_a_adam_lr5e-4_steps/`.
  - `reports/runs/loss_landscape/vgg_a_adam_lr1e-3_steps/`.
  - `reports/runs/loss_landscape/vgg_a_adam_lr2e-3_steps/`.
  - `reports/runs/loss_landscape/vgg_a_bn_adam_lr1e-4_steps/`.
  - `reports/runs/loss_landscape/vgg_a_bn_adam_lr5e-4_steps/`.
  - `reports/runs/loss_landscape/vgg_a_bn_adam_lr1e-3_steps/`.
  - `reports/runs/loss_landscape/vgg_a_bn_adam_lr2e-3_steps/`.
- Figures and summaries:
  - `reports/figures/vgg_a_loss_landscape_bn_vs_no_bn.png`.
  - `reports/figures/vgg_a_loss_landscape_bn_vs_no_bn_summary.json`.
- Interpretation:
  - BatchNorm narrows the average per-step loss envelope substantially: 0.2147 vs. 0.4084, about a 47.4% reduction relative to the no-BN envelope.
  - The final loss range is also much narrower with BN: 0.0245 vs. 0.2188 for no-BN.
  - The largest instantaneous envelope width is higher for BN, likely because the very small learning rate `1e-4` converges slowly for the BN model while the larger learning rates improve quickly; this creates a transient spread early in training.
  - Despite that transient maximum, the average and final envelopes support the conclusion that BN makes optimization less sensitive to the tested learning-rate choices after the early phase.
  - The best BN runs at `1e-3` and `2e-3` both reach about 83% validation accuracy, while the no-BN model degrades at `2e-3`; this further supports the claim that BN improves optimization stability at larger step sizes.
- Next action:
  - Commit the loss-landscape metrics, step losses, figure, summary, and updated draft.
  - Use this section as the evidence base for the final LaTeX report's BN optimization analysis.

### 2026-05-13: Local 3D Loss-Surface Visualization

- Commit: `d1734cf feat: add 3d loss surface visualization`.
- Purpose:
  - Add an intuitive 3D visualization comparing the naive VGG-A checkpoint and the BatchNorm VGG-A checkpoint.
  - This is a qualitative local surface plot, not the same measurement as the learning-rate envelope experiment above.
- Method:
  - Added `codes/VGG_BatchNorm/plot_3d_loss_surface.py`.
  - Loaded the trained checkpoints:
    - Naive: `reports/runs/vgg_a_adam_lr1e-3/best.pt`.
    - BatchNorm: `reports/runs/vgg_a_bn_adam_lr1e-3/best.pt`.
  - Sampled two random normalized parameter-space directions around each checkpoint.
  - Evaluated validation cross-entropy on a 2D grid around each checkpoint.
  - Rendered the two loss surfaces as side-by-side 3D plots.
- Configuration:
  - Device used for plotting: `cuda:2`.
  - Grid size: 13 by 13.
  - Radius: 0.35.
  - Validation subset: first 1,024 CIFAR-10 test examples.
  - Batch size: 128.
  - Seed: 2020.
- Output:
  - Figure: `reports/figures/vgg_a_3d_loss_surface_naive_vs_bn.png`.
  - Summary: `reports/figures/vgg_a_3d_loss_surface_naive_vs_bn_summary.json`.
- Numeric summary:
  - Naive center loss: 1.2258.
  - Naive min loss: 1.2150.
  - Naive max loss: 2.8070.
  - Naive mean surface loss: 1.7244.
  - BatchNorm center loss: 0.7133.
  - BatchNorm min loss: 0.7074.
  - BatchNorm max loss: 4.2101.
  - BatchNorm mean surface loss: 1.7018.
- Interpretation:
  - The center loss is lower for the BN checkpoint on the sampled validation subset, matching the stronger validation performance observed in the main experiments.
  - Because the plotted directions are random and the subset is small, the 3D surface should be presented as a qualitative illustration rather than a definitive sharpness metric.
  - The earlier learning-rate envelope remains the stronger evidence for reduced step-size sensitivity.

### 2026-05-13: ModelScope Checkpoint Packaging

- GitHub commit for the local model-card draft: `49bb97c docs: add modelscope model card`.
- GitHub commit for the report link update: `1952fb3 docs: add modelscope weights link to report`.
- ModelScope repository:
  - Summary page: `https://www.modelscope.cn/models/yuhengli/deep-learning2-final-model/summary`.
  - Files page: `https://www.modelscope.cn/models/yuhengli/deep-learning2-final-model/files`.
- Uploaded final selected model:
  - `final/vgg_a_bn_gelu_adam_lr1e-3_best.pt`.
  - `final/vgg_a_bn_gelu_adam_lr1e-3_metrics.json`.
- Uploaded ablation checkpoints:
  - `ablations/vgg_a_adam_lr1e-3_best.pt`.
  - `ablations/vgg_a_bn_adam_lr1e-3_best.pt`.
  - `ablations/vgg_a_dropout_adam_lr1e-3_best.pt`.
  - `ablations/vgg_a_light_adam_lr1e-3_best.pt`.
  - `ablations/vgg_a_bn_adamw_lr1e-3_wd1e-4_best.pt`.
  - `ablations/vgg_a_bn_leaky_relu_adam_lr1e-3_best.pt`.
  - Matching `*_metrics.json` files were uploaded for every ablation checkpoint.
- Excluded files:
  - Loss-landscape learning-rate sweep checkpoints were not uploaded because they are auxiliary trajectory-analysis checkpoints.
  - Their metrics, step-loss records, figures, and scripts remain in the GitHub repository.
- Remote cleanup:
  - The first upload placed a duplicate final checkpoint and metrics file at the ModelScope repository root.
  - A clean ModelScope Git clone was used to remove those root-level duplicates.
  - ModelScope cleanup commit: `52c44c4 Remove duplicate root checkpoint files`.
- Verification:
  - The final ModelScope file tree contains only `final/`, `ablations/`, `README.md`, `configuration.json`, and `.gitattributes`.
  - No root-level checkpoint duplicate remains.
- Interpretation:
  - The ModelScope upload now separates the report's final model from auxiliary experimental checkpoints.
  - The organization makes the final deliverable unambiguous while still preserving the major ablation evidence needed to reproduce the report tables.
