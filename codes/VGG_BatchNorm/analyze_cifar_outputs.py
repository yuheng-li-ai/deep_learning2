import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data.loaders import get_cifar_loader
from models.vgg import (
    VGG_A,
    VGG_A_BatchNorm,
    VGG_A_BatchNorm_GELU,
    VGG_A_BatchNorm_LeakyReLU,
    VGG_A_Dropout,
    VGG_A_Light,
    get_number_of_parameters,
)


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


MODEL_REGISTRY = {
    "vgg_a": VGG_A,
    "vgg_a_bn": VGG_A_BatchNorm,
    "vgg_a_bn_leaky_relu": VGG_A_BatchNorm_LeakyReLU,
    "vgg_a_bn_gelu": VGG_A_BatchNorm_GELU,
    "vgg_a_dropout": VGG_A_Dropout,
    "vgg_a_light": VGG_A_Light,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CIFAR-10 report artifacts for trained VGG models."
    )
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="codes/VGG_BatchNorm/data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-eval-items", type=int, default=-1)
    parser.add_argument("--sample-items", type=int, default=30)
    return parser.parse_args()


def resolve_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def denormalize(image):
    return np.clip(np.transpose(image.numpy(), (1, 2, 0)) * 0.5 + 0.5, 0.0, 1.0)


def save_sample_grid(data_loader, output_path, max_items):
    images, labels = next(iter(data_loader))
    count = min(max_items, images.size(0))
    cols = 10
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
    axes = np.atleast_2d(axes)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < count:
            ax.imshow(denormalize(images[index].cpu()))
            ax.set_title(CIFAR10_CLASSES[int(labels[index])], fontsize=7)
    fig.tight_layout(pad=0.15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def load_model(model_name, checkpoint_path, device):
    model = MODEL_REGISTRY[model_name]().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def evaluate_confusion(model, data_loader, device):
    confusion = torch.zeros(10, 10, dtype=torch.long)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_examples = 0
    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        predictions = logits.argmax(dim=1)
        for target, predicted in zip(y.cpu(), predictions.cpu()):
            confusion[int(target), int(predicted)] += 1
        total_examples += y.numel()

    correct = confusion.diag()
    class_totals = confusion.sum(dim=1).clamp_min(1)
    per_class_accuracy = correct.float() / class_totals.float()
    return {
        "confusion": confusion.numpy(),
        "accuracy": correct.sum().item() / max(total_examples, 1),
        "loss": total_loss / max(total_examples, 1),
        "per_class_accuracy": per_class_accuracy.numpy(),
    }


def save_confusion_matrix(confusion, output_path):
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            value = int(confusion[i, j])
            color = "white" if value > confusion.max() * 0.55 else "black"
            ax.text(j, i, value, ha="center", va="center", fontsize=6, color=color)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_filter_grid(model, output_path, max_filters=32):
    first_conv = next(module for module in model.modules() if isinstance(module, nn.Conv2d))
    weights = first_conv.weight.detach().cpu()
    count = min(max_filters, weights.size(0))
    cols = 8
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.0, rows * 1.0))
    axes = np.atleast_2d(axes)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < count:
            filt = weights[index]
            filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-12)
            ax.imshow(np.transpose(filt.numpy(), (1, 2, 0)))
    fig.tight_layout(pad=0.08)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def architecture_rows(model, input_shape=(1, 3, 32, 32)):
    rows = []
    hooks = []

    def hook(name, module):
        def capture(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                shape = list(output.shape)
            else:
                shape = [list(item.shape) for item in output]
            params = sum(parameter.numel() for parameter in module.parameters(recurse=False))
            rows.append(
                {
                    "layer": name,
                    "type": module.__class__.__name__,
                    "output_shape": str(shape),
                    "parameters": params,
                }
            )

        return capture

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(hook(name, module)))

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        model(torch.zeros(input_shape, device=device))
    if was_training:
        model.train()
    for handle in hooks:
        handle.remove()
    return rows


def save_architecture(rows, csv_path, json_path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["layer", "type", "output_shape", "parameters"]
        )
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    sample_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=max(args.sample_items, 30),
        train=True,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=max(args.sample_items, 30),
    )
    eval_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.n_eval_items,
    )

    model = load_model(args.model, args.checkpoint, device)
    save_sample_grid(sample_loader, output_dir / "cifar10_sample_grid.png", args.sample_items)
    save_filter_grid(model, output_dir / "first_layer_filters.png")

    evaluation = evaluate_confusion(model, eval_loader, device)
    save_confusion_matrix(evaluation["confusion"], output_dir / "confusion_matrix.png")

    architecture = architecture_rows(model)
    save_architecture(
        architecture,
        output_dir / "architecture_table.csv",
        output_dir / "architecture_table.json",
    )

    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "parameters": get_number_of_parameters(model),
        "accuracy": evaluation["accuracy"],
        "loss": evaluation["loss"],
        "classes": CIFAR10_CLASSES,
        "per_class_accuracy": {
            label: float(value)
            for label, value in zip(CIFAR10_CLASSES, evaluation["per_class_accuracy"])
        },
        "confusion_matrix": evaluation["confusion"].tolist(),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
