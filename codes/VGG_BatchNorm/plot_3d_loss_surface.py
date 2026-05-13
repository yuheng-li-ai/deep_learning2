import argparse
import json
import random
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

mpl.use("Agg")
import matplotlib.pyplot as plt

from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm
from train_cifar import resolve_device, set_random_seeds


MODEL_REGISTRY = {
    "vgg_a": VGG_A,
    "vgg_a_bn": VGG_A_BatchNorm,
}


def parse_surface_arg(value):
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("Surface must use LABEL:MODEL=CHECKPOINT format.")
    label_model, checkpoint = value.split("=", 1)
    label, model_name = label_model.split(":", 1)
    label = label.strip()
    model_name = model_name.strip()
    if not label or model_name not in MODEL_REGISTRY:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise argparse.ArgumentTypeError(f"Model must be one of: {choices}.")
    return label, model_name, Path(checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a 3D loss surface around trained VGG checkpoints."
    )
    parser.add_argument(
        "--surface",
        action="append",
        required=True,
        type=parse_surface_arg,
        help="Surface in LABEL:MODEL=CHECKPOINT format.",
    )
    parser.add_argument("--data-root", default="codes/VGG_BatchNorm/data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-val-items", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", help="Optional JSON summary output path.")
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint


def parameter_items(model):
    return [(name, parameter) for name, parameter in model.named_parameters()]


def clone_parameters(model):
    return {name: parameter.detach().clone() for name, parameter in parameter_items(model)}


def make_direction_like(base_parameters):
    direction = {}
    for name, parameter in base_parameters.items():
        random_tensor = torch.randn_like(parameter)
        random_norm = random_tensor.norm()
        parameter_norm = parameter.norm()
        if random_norm > 0 and parameter_norm > 0:
            random_tensor = random_tensor * (parameter_norm / random_norm)
        direction[name] = random_tensor
    return direction


@torch.no_grad()
def set_surface_parameters(model, base_parameters, direction_x, direction_y, alpha, beta):
    for name, parameter in parameter_items(model):
        parameter.copy_(
            base_parameters[name] + alpha * direction_x[name] + beta * direction_y[name]
        )


@torch.no_grad()
def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for x, y in data_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
    return total_loss / total_examples


def compute_surface(model, data_loader, criterion, device, grid, label):
    base_parameters = clone_parameters(model)
    direction_x = make_direction_like(base_parameters)
    direction_y = make_direction_like(base_parameters)
    losses = np.zeros((len(grid), len(grid)), dtype=np.float64)

    for i, alpha in enumerate(tqdm(grid, desc=f"{label} x-axis", unit="x")):
        for j, beta in enumerate(grid):
            set_surface_parameters(
                model, base_parameters, direction_x, direction_y, alpha, beta
            )
            losses[i, j] = evaluate_loss(model, data_loader, criterion, device)

    set_surface_parameters(model, base_parameters, direction_x, direction_y, 0.0, 0.0)
    return losses


def summarize_surface(label, model_name, checkpoint_path, grid, losses):
    center_index = len(grid) // 2
    return {
        "label": label,
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "grid_size": int(len(grid)),
        "center_loss": float(losses[center_index, center_index]),
        "min_loss": float(losses.min()),
        "max_loss": float(losses.max()),
        "mean_loss": float(losses.mean()),
    }


def plot_surfaces(surfaces, grid, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_values, y_values = np.meshgrid(grid, grid, indexing="ij")
    fig = plt.figure(figsize=(6.4 * len(surfaces), 5.2))
    for index, (label, losses) in enumerate(surfaces, start=1):
        axis = fig.add_subplot(1, len(surfaces), index, projection="3d")
        axis.plot_surface(
            x_values,
            y_values,
            losses,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.95,
        )
        axis.set_title(label)
        axis.set_xlabel("Direction x")
        axis.set_ylabel("Direction y")
        axis.set_zlabel("Cross entropy")
        axis.view_init(elev=30, azim=-125)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if args.grid_size < 3 or args.grid_size % 2 == 0:
        raise ValueError("--grid-size must be an odd integer >= 3.")

    device = resolve_device(args.device)
    set_random_seeds(args.seed, device)
    random.seed(args.seed)

    val_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.n_val_items,
    )
    criterion = nn.CrossEntropyLoss()
    grid = np.linspace(-args.radius, args.radius, args.grid_size)

    plotted_surfaces = []
    summaries = []
    for label, model_name, checkpoint_path in args.surface:
        model = MODEL_REGISTRY[model_name]().to(device)
        load_checkpoint(model, checkpoint_path, device)
        losses = compute_surface(model, val_loader, criterion, device, grid, label)
        plotted_surfaces.append((label, losses))
        summaries.append(summarize_surface(label, model_name, checkpoint_path, grid, losses))

    plot_surfaces(plotted_surfaces, grid, args.output)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "device": str(device),
                    "n_val_items": args.n_val_items,
                    "radius": args.radius,
                    "surfaces": summaries,
                },
                f,
                indent=2,
            )
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
