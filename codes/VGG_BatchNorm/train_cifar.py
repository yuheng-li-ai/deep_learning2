import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from data.loaders import get_cifar_loader
from models.vgg import (
    VGG_A,
    VGG_A_BatchNorm,
    VGG_A_Dropout,
    VGG_A_Light,
    get_number_of_parameters,
)


MODEL_REGISTRY = {
    "vgg_a": VGG_A,
    "vgg_a_bn": VGG_A_BatchNorm,
    "vgg_a_dropout": VGG_A_Dropout,
    "vgg_a_light": VGG_A_Light,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG variants on CIFAR-10.")
    parser.add_argument("--model", choices=MODEL_REGISTRY.keys(), default="vgg_a")
    parser.add_argument("--data-root", default="codes/VGG_BatchNorm/data")
    parser.add_argument("--output-dir", default="reports/runs/manual")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--n-train-items", type=int, default=-1)
    parser.add_argument("--n-val-items", type=int, default=-1)
    return parser.parse_args()


def resolve_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def set_random_seeds(seed_value, device):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(name, model, lr, weight_decay):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    raise ValueError(f"Unsupported optimizer: {name}")


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, desc):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(data_loader, desc=desc, unit="batch", leave=False)
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)
        predictions = logits.argmax(dim=1)

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == y).sum().item()
        total_examples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            acc=f"{total_correct / total_examples:.4f}",
        )

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def train_one_epoch(model, data_loader, criterion, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    progress = tqdm(
        data_loader,
        desc=f"epoch {epoch}/{epochs} train",
        unit="batch",
        leave=False,
    )
    for x, y in progress:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == y).sum().item()
        total_examples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_examples:.4f}",
            acc=f"{total_correct / total_examples:.4f}",
        )

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


def save_metrics(output_dir, metrics):
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for row in metrics["epochs"]:
            writer.writerow(row)


def run_training(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    set_random_seeds(args.seed, device)

    train_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=True,
        shuffle=True,
        num_workers=args.num_workers,
        n_items=args.n_train_items,
    )
    val_loader = get_cifar_loader(
        root=args.data_root,
        batch_size=args.batch_size,
        train=False,
        shuffle=False,
        num_workers=args.num_workers,
        n_items=args.n_val_items,
    )

    model = MODEL_REGISTRY[args.model]().to(device)
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    metrics = {
        "config": vars(args),
        "device": str(device),
        "parameters": get_number_of_parameters(model),
        "epochs": [],
        "best_val_accuracy": 0.0,
        "best_epoch": 0,
        "best_checkpoint": str(output_dir / "best.pt"),
    }

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="epochs", unit="epoch")
    for epoch in epoch_bar:
        start = time.perf_counter()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, desc=f"epoch {epoch}/{args.epochs} val"
        )
        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "elapsed_seconds": elapsed,
        }
        metrics["epochs"].append(row)

        if val_metrics["accuracy"] > metrics["best_val_accuracy"]:
            metrics["best_val_accuracy"] = val_metrics["accuracy"]
            metrics["best_epoch"] = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "val_accuracy": val_metrics["accuracy"],
                },
                output_dir / "best.pt",
            )

        save_metrics(output_dir, metrics)
        epoch_bar.set_postfix(
            train_acc=f"{train_metrics['accuracy']:.4f}",
            val_acc=f"{val_metrics['accuracy']:.4f}",
            best=f"{metrics['best_val_accuracy']:.4f}",
        )

    return metrics


if __name__ == "__main__":
    run_training(parse_args())
