import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import (
    VGG_A,
    VGG_A_BatchNorm,
    VGG_A_BatchNorm_GELU,
    VGG_A_BatchNorm_LeakyReLU,
    VGG_A_Dropout,
    VGG_A_Light,
)


MODEL_REGISTRY = {
    "vgg_a": VGG_A,
    "vgg_a_bn": VGG_A_BatchNorm,
    "vgg_a_bn_leaky_relu": VGG_A_BatchNorm_LeakyReLU,
    "vgg_a_bn_gelu": VGG_A_BatchNorm_GELU,
    "vgg_a_dropout": VGG_A_Dropout,
    "vgg_a_light": VGG_A_Light,
}


def parse_run_arg(value):
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Run must use LABEL=MODEL=CHECKPOINT format."
        )
    label, model_name, checkpoint = [part.strip() for part in parts]
    if not label or not model_name or not checkpoint:
        raise argparse.ArgumentTypeError("Run label, model, and checkpoint are required.")
    if model_name not in MODEL_REGISTRY:
        raise argparse.ArgumentTypeError(f"Unsupported model: {model_name}")
    return label, model_name, Path(checkpoint)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure first-order loss predictiveness and gradient variation."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run in LABEL=MODEL=CHECKPOINT format.",
    )
    parser.add_argument("--data-root", default="codes/VGG_BatchNorm/data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-val-items", type=int, default=512)
    parser.add_argument(
        "--step-size",
        action="append",
        type=float,
        default=None,
        help="Gradient-step size to evaluate. May be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--summary", required=True, help="Output JSON summary path.")
    return parser.parse_args()


def resolve_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model(model_name, checkpoint_path, device):
    model = MODEL_REGISTRY[model_name]().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_validation_batch(data_loader, device, limit):
    xs = []
    ys = []
    total = 0
    for x, y in data_loader:
        remaining = limit - total
        if remaining <= 0:
            break
        xs.append(x[:remaining])
        ys.append(y[:remaining])
        total += min(x.size(0), remaining)
    if not xs:
        raise ValueError("Validation loader did not produce any examples.")
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def loss_and_gradient(model, criterion, x, y):
    model.zero_grad(set_to_none=True)
    loss = criterion(model(x), y)
    loss.backward()
    grad = parameters_to_vector(
        [parameter.grad.detach() for parameter in model.parameters()]
    )
    return loss.detach(), grad


def evaluate_run(model, criterion, x, y, step_sizes):
    base_parameters = parameters_to_vector(model.parameters()).detach().clone()
    base_loss, base_gradient = loss_and_gradient(model, criterion, x, y)
    gradient_sq_norm = torch.dot(base_gradient, base_gradient).item()
    gradient_norm = gradient_sq_norm ** 0.5

    rows = []
    for step_size in tqdm(step_sizes, desc="gradient steps", leave=False):
        candidate_parameters = base_parameters - step_size * base_gradient
        vector_to_parameters(candidate_parameters, model.parameters())
        candidate_loss, candidate_gradient = loss_and_gradient(model, criterion, x, y)

        distance = step_size * gradient_norm
        predicted_loss = base_loss.item() - step_size * gradient_sq_norm
        prediction_error = abs(candidate_loss.item() - predicted_loss)
        gradient_change = torch.linalg.vector_norm(
            candidate_gradient - base_gradient
        ).item()
        gradient_difference_over_distance = gradient_change / max(distance, 1e-12)

        rows.append(
            {
                "step_size": step_size,
                "distance": distance,
                "base_loss": base_loss.item(),
                "actual_loss": candidate_loss.item(),
                "predicted_loss": predicted_loss,
                "prediction_error": prediction_error,
                "gradient_change_norm": gradient_change,
                "gradient_difference_over_distance": gradient_difference_over_distance,
            }
        )

    vector_to_parameters(base_parameters, model.parameters())
    return {
        "base_loss": base_loss.item(),
        "base_gradient_norm": gradient_norm,
        "mean_prediction_error": sum(row["prediction_error"] for row in rows)
        / len(rows),
        "max_prediction_error": max(row["prediction_error"] for row in rows),
        "max_gradient_difference_over_distance": max(
            row["gradient_difference_over_distance"] for row in rows
        ),
        "rows": rows,
    }


def plot_summary(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    for label, summary in results.items():
        rows = summary["rows"]
        step_sizes = [row["step_size"] for row in rows]
        axes[0].plot(
            step_sizes,
            [row["prediction_error"] for row in rows],
            marker="o",
            label=label,
        )
        axes[1].plot(
            step_sizes,
            [row["gradient_difference_over_distance"] for row in rows],
            marker="o",
            label=label,
        )

    axes[0].set_title("First-order loss prediction error")
    axes[0].set_xlabel("Step size along -gradient")
    axes[0].set_ylabel("|actual loss - linear prediction|")
    axes[0].set_xscale("log")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Gradient variation per distance")
    axes[1].set_xlabel("Step size along -gradient")
    axes[1].set_ylabel(r"$\|\nabla L(w')-\nabla L(w)\| / \|w'-w\|$")
    axes[1].set_xscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    step_sizes = args.step_size or [1e-4, 5e-4, 1e-3, 2e-3]
    device = resolve_device(args.device)
    criterion = nn.CrossEntropyLoss()
    val_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.n_val_items,
    )
    x, y = load_validation_batch(val_loader, device, args.n_val_items)

    results = {}
    for label, model_name, checkpoint_path in args.run:
        model = load_model(model_name, checkpoint_path, device)
        results[label] = evaluate_run(model, criterion, x, y, step_sizes)

    plot_summary(results, args.output)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": str(device),
        "n_val_items": int(x.size(0)),
        "step_sizes": step_sizes,
        "runs": results,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
