import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def parse_run_arg(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must use NAME=PATH format.")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("Run name cannot be empty.")
    metrics_path = Path(path)
    if metrics_path.is_dir():
        metrics_path = metrics_path / "metrics.json"
    return name.strip(), metrics_path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CIFAR training run metrics.")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run in NAME=PATH format. PATH can be a run directory or metrics.json.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument(
        "--title",
        default="CIFAR-10 Training Comparison",
        help="Figure title.",
    )
    return parser.parse_args()


def load_metrics(metrics_path):
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    if "epochs" not in metrics or not metrics["epochs"]:
        raise ValueError(f"No epoch metrics found in {metrics_path}")
    return metrics


def summarize_run(name, metrics):
    best_epoch = max(metrics["epochs"], key=lambda row: row["val_accuracy"])
    final_epoch = metrics["epochs"][-1]
    return {
        "name": name,
        "parameters": metrics.get("parameters"),
        "best_epoch": best_epoch["epoch"],
        "best_val_accuracy": best_epoch["val_accuracy"],
        "best_val_error": 1.0 - best_epoch["val_accuracy"],
        "final_train_accuracy": final_epoch["train_accuracy"],
        "final_val_accuracy": final_epoch["val_accuracy"],
        "final_val_loss": final_epoch["val_loss"],
    }


def plot_runs(runs, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, metrics in runs:
        epochs = [row["epoch"] for row in metrics["epochs"]]
        train_loss = [row["train_loss"] for row in metrics["epochs"]]
        val_loss = [row["val_loss"] for row in metrics["epochs"]]
        train_accuracy = [row["train_accuracy"] for row in metrics["epochs"]]
        val_accuracy = [row["val_accuracy"] for row in metrics["epochs"]]

        axes[0].plot(epochs, train_loss, linestyle="--", label=f"{name} train")
        axes[0].plot(epochs, val_loss, label=f"{name} val")
        axes[1].plot(epochs, train_accuracy, linestyle="--", label=f"{name} train")
        axes[1].plot(epochs, val_accuracy, label=f"{name} val")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    runs = [(name, load_metrics(path)) for name, path in args.run]
    plot_runs(runs, args.output, args.title)
    for name, metrics in runs:
        summary = summarize_run(name, metrics)
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
