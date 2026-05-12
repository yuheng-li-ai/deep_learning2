import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


def parse_run_arg(value):
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise argparse.ArgumentTypeError("Run must use GROUP:NAME=PATH format.")
    label, path = value.split("=", 1)
    group, name = label.split(":", 1)
    group = group.strip()
    name = name.strip()
    if not group or not name:
        raise argparse.ArgumentTypeError("Group and run name cannot be empty.")
    return group, name, Path(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot min/max per-step loss envelopes across learning-rate runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=parse_run_arg,
        help="Run in GROUP:NAME=PATH format. PATH may be a run dir or step_losses.csv.",
    )
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument(
        "--summary",
        help="Optional JSON path for envelope summary statistics.",
    )
    parser.add_argument(
        "--title",
        default="VGG-A Loss Landscape Across Learning Rates",
        help="Figure title.",
    )
    return parser.parse_args()


def resolve_step_loss_path(path):
    if path.is_dir():
        return path / "step_losses.csv"
    return path


def load_step_losses(path):
    step_loss_path = resolve_step_loss_path(path)
    losses = []
    with step_loss_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "train_loss" not in reader.fieldnames:
            raise ValueError(f"No train_loss column found in {step_loss_path}")
        for row in reader:
            losses.append(float(row["train_loss"]))
    if not losses:
        raise ValueError(f"No step losses found in {step_loss_path}")
    return losses


def compute_envelope(loss_runs):
    if not loss_runs:
        raise ValueError("At least one loss run is required.")
    min_length = min(len(losses) for losses in loss_runs)
    if min_length == 0:
        raise ValueError("Loss runs cannot be empty.")

    matrix = np.asarray([losses[:min_length] for losses in loss_runs], dtype=float)
    return {
        "steps": np.arange(1, min_length + 1),
        "min_curve": matrix.min(axis=0),
        "max_curve": matrix.max(axis=0),
        "mean_curve": matrix.mean(axis=0),
    }


def summarize_group(group_name, run_names, envelope):
    width = envelope["max_curve"] - envelope["min_curve"]
    return {
        "group": group_name,
        "runs": run_names,
        "steps": int(len(envelope["steps"])),
        "mean_envelope_width": float(width.mean()),
        "max_envelope_width": float(width.max()),
        "final_min_loss": float(envelope["min_curve"][-1]),
        "final_max_loss": float(envelope["max_curve"][-1]),
        "final_mean_loss": float(envelope["mean_curve"][-1]),
    }


def plot_envelopes(grouped_runs, output_path, title):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    summaries = []
    for group_name, runs in grouped_runs.items():
        run_names = [name for name, _ in runs]
        loss_runs = [losses for _, losses in runs]
        envelope = compute_envelope(loss_runs)
        steps = envelope["steps"]

        ax.fill_between(
            steps,
            envelope["min_curve"],
            envelope["max_curve"],
            alpha=0.18,
            label=f"{group_name} min-max",
        )
        ax.plot(steps, envelope["mean_curve"], linewidth=1.6, label=f"{group_name} mean")
        ax.plot(steps, envelope["min_curve"], linewidth=0.9, linestyle="--")
        ax.plot(steps, envelope["max_curve"], linewidth=0.9, linestyle="--")
        summaries.append(summarize_group(group_name, run_names, envelope))

    ax.set_title(title)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross entropy loss")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return summaries


def main():
    args = parse_args()
    grouped_runs = {}
    for group, name, path in args.run:
        grouped_runs.setdefault(group, []).append((name, load_step_losses(path)))

    summaries = plot_envelopes(grouped_runs, args.output, args.title)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
