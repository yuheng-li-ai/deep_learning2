import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from plot_runs import load_metrics, parse_run_arg, summarize_run


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot and summarize loss/regularization ablation runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run in NAME=PATH format. PATH can be a run directory or metrics.json.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--title", default="Loss and Regularization Ablation")
    return parser.parse_args()


def plot_regularization(runs, output_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for name, metrics in runs:
        epochs = [row["epoch"] for row in metrics["epochs"]]
        val_loss = [row["val_loss"] for row in metrics["epochs"]]
        val_accuracy = [row["val_accuracy"] for row in metrics["epochs"]]
        axes[0].plot(epochs, val_loss, marker="o", markersize=2, label=name)
        axes[1].plot(epochs, val_accuracy, marker="o", markersize=2, label=name)

    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    runs = [(name, load_metrics(path)) for name, path in args.run]
    plot_regularization(runs, args.output, args.title)
    rows = [summarize_run(name, metrics) for name, metrics in runs]
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(json.dumps(rows, sort_keys=True))


if __name__ == "__main__":
    main()
