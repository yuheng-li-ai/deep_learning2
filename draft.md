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
