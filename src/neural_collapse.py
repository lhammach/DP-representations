#!/usr/bin/env python
"""
neural_collapse.py
==================
Measure the four Neural Collapse (NC1–NC4) properties described in:
  Papyan et al., "Prevalence of Neural Collapse during the Terminal Phase
  of Deep Learning Training", PNAS 2020.

Usage:
    # Single checkpoint
    python neural_collapse.py \\
        --ckpt ../networks/grad_checkpoint/baseline_resnet18_..._seed1.pth \\
        --data-root ../cifar10 \\
        --experiment grad_checkpoint

    # Compare baseline vs DP (and optionally no-clip / no-noise) at epoch 100
    python neural_collapse.py \\
        --ckpt \\
            ../networks/grad_checkpoint/baseline_resnet18_..._seed1.pth \\
            ../networks/grad_checkpoint/dp_resnet18_..._seed1.pth \\
        --labels Baseline DP \\
        --experiment grad_checkpoint

    # Evolution over epochs (pass checkpoints in epoch order)
    python neural_collapse.py \\
        --ckpt $(ls ../networks/grad_checkpoint/dp_resnet18_..._ep*.pth | sort -V) \\
               ../networks/grad_checkpoint/dp_resnet18_....pth \\
        --labels $(seq 5 5 100 | xargs printf "ep%d ") \\
        --experiment grad_checkpoint \\
        --plot-over-epochs

    # Full comparison: baseline, DP, no-clip, no-noise (over epochs)
    python neural_collapse.py \\
        --ckpt $(ls ../networks/.../baseline_..._ep*.pth | sort -V) \\
               ../networks/.../baseline_....pth \\
        --ckpt $(ls ../networks/.../dp_..._ep*.pth | sort -V) \\
               ../networks/.../dp_....pth \\
        --group-labels Baseline DP \\
        --experiment grad_checkpoint \\
        --plot-over-epochs

NC metrics are computed on the TRAINING set (as in the original paper),
since NC is a training-phase phenomenon. The penultimate layer (before the
classification head) is used as the representation space.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3",
          "#937860", "#da8bc3", "#8c8c8c", "#ccb974", "#64b5cd"]


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_penultimate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
    layer_index: int = -1,
    correct_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from a chosen Linear layer.

    Args:
        layer_index: which Linear layer to hook, counting from the end.
            -1 = last Linear (penultimate representation, default)
            -2 = second-to-last Linear
        correct_only: if True, only keep examples the model classifies
            correctly. Gives a cleaner NC signal at the cost of excluding
            misclassified examples (whose representations pull class means
            away from the true geometry). The original Papyan paper uses
            all training examples; correct_only is an optional refinement.
    """
    model.eval()
    captured: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    preds_list: list[torch.Tensor] = []

    linear_layers: list[nn.Linear] = [
        m for m in model.modules() if isinstance(m, nn.Linear)
    ]
    if not linear_layers:
        raise RuntimeError("No nn.Linear found in model.")

    chosen = linear_layers[layer_index]
    logger.info("Hooking Linear layer at index %d (out=%d, in=%d)%s",
                layer_index, chosen.out_features, chosen.in_features,
                " [correct-only]" if correct_only else "")

    def _hook(module, input, output):
        captured.append(input[0].detach().cpu())

    handle = chosen.register_forward_hook(_hook)

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Extracting features", leave=False):
            images = images.to(device)
            logits = model(images)
            preds_list.append(logits.argmax(dim=1).cpu())
            labels_list.append(targets)

    handle.remove()

    features = torch.cat(captured, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).numpy()
    preds = torch.cat(preds_list, dim=0).numpy()

    if correct_only:
        mask = (preds == labels)
        n_total = len(labels)
        n_correct = int(mask.sum())
        features = features[mask]
        labels = labels[mask]
        logger.info("correct_only: %d/%d examples kept (%.1f%%)",
                    n_correct, n_total, 100 * n_correct / n_total)

    logger.info("Extracted features: shape=%s", features.shape)
    return features, labels


def log_per_class_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[int, dict]:
    """Compute and log per-class accuracy + number of correctly classified examples.

    Uses the classifier weight matrix W to predict classes (argmax of W @ h).
    Returns a dict {class_id: {n_total, n_correct, accuracy}}.
    """
    logits = features @ W.T        # (N, C)
    preds = np.argmax(logits, axis=1)

    names = class_names or [str(c) for c in range(num_classes)]
    per_class: dict[int, dict] = {}

    logger.info("Per-class accuracy:")
    logger.info("  %-12s  %6s  %7s  %8s", "Class", "Total", "Correct", "Accuracy")
    logger.info("  " + "-" * 38)

    for c in range(num_classes):
        mask_c = labels == c
        n_total = int(mask_c.sum())
        n_correct = int(((preds == c) & mask_c).sum())
        acc = n_correct / n_total if n_total > 0 else 0.0
        per_class[c] = {"class": names[c], "n_total": n_total,
                        "n_correct": n_correct, "accuracy": acc}
        logger.info("  %-12s  %6d  %7d  %7.1f%%",
                    names[c], n_total, n_correct, 100 * acc)

    overall = int((preds == labels).sum())
    logger.info("  " + "-" * 38)
    logger.info("  %-12s  %6d  %7d  %7.1f%%",
                "TOTAL", len(labels), overall, 100 * overall / len(labels))
    return per_class

def compute_nc_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    b: np.ndarray | None,
    num_classes: int = 10,
) -> dict[str, float]:
    """Compute NC1–NC4 metrics.

    Args:
        features   : (N, d) penultimate-layer representations
        labels     : (N,)   true labels
        W          : (C, d) weight matrix of the classification head
        b          : (C,)   bias of the classification head (or None)
        num_classes: number of classes C

    Returns:
        dict with keys: nc1, nc2, nc3, nc4,
                        nc2_cosines_mean, nc2_cosines_std,
                        nc1_within_class_var, nc1_between_class_var
    """
    C = num_classes
    d = features.shape[1]

    # ── Class means and global mean ──
    mu_c = np.zeros((C, d), dtype=np.float64)
    n_c = np.zeros(C, dtype=int)
    for c in range(C):
        mask = labels == c
        n_c[c] = mask.sum()
        if n_c[c] > 0:
            mu_c[c] = features[mask].mean(axis=0)
    mu_G = mu_c.mean(axis=0)   # (d,)

    # ── NC1: within-class variability collapse ──
    # Sigma_W: mean of within-class covariance traces
    # Sigma_B: between-class covariance trace
    sigma_W_trace = 0.0
    for c in range(C):
        mask = labels == c
        if n_c[c] < 2:
            continue
        diff = features[mask] - mu_c[c]   # (n_c, d)
        sigma_W_trace += np.sum(diff ** 2) / n_c[c]   # tr(Sigma_W^(c))
    sigma_W_trace /= C

    mu_centered = mu_c - mu_G   # (C, d)
    sigma_B_trace = np.sum(mu_centered ** 2) / C

    nc1 = sigma_W_trace / (sigma_B_trace + 1e-10)

    # ── NC2: equiangularity and equinorm of class means ──
    # Target cosine between different classes: -1/(C-1)
    norms = np.linalg.norm(mu_centered, axis=1, keepdims=True) + 1e-10
    mu_normed = mu_centered / norms   # (C, d)
    cosines = mu_normed @ mu_normed.T  # (C, C)

    target = -1.0 / (C - 1)
    # Deviation from ideal ETF (equiangular tight frame)
    ideal = np.full((C, C), target)
    np.fill_diagonal(ideal, 1.0)
    nc2 = np.linalg.norm(cosines - ideal, ord='fro')

    # Summary stats on off-diagonal cosines
    off_diag = cosines[~np.eye(C, dtype=bool)]
    nc2_cosines_mean = float(off_diag.mean())
    nc2_cosines_std = float(off_diag.std())

    # ── NC3: self-duality (alignment of W with class means) ──
    # W: (C, d), mu_centered: (C, d)
    W_norm = W / (np.linalg.norm(W, ord='fro') + 1e-10)
    M_norm = mu_centered / (np.linalg.norm(mu_centered, ord='fro') + 1e-10)
    nc3 = float(np.linalg.norm(W_norm - M_norm, ord='fro'))

    # ── NC4: convergence to simplex ETF classifier ──
    # Compare argmax(W h) vs argmax(mu_c^T h) for each example
    logits_W = features @ W.T      # (N, C)  classifier prediction
    logits_M = features @ mu_centered.T   # (N, C) nearest-centroid prediction
    pred_W = np.argmax(logits_W, axis=1)
    pred_M = np.argmax(logits_M, axis=1)
    nc4 = float((pred_W != pred_M).mean())

    return {
        "nc1": float(nc1),
        "nc2": float(nc2),
        "nc2_cosines_mean": nc2_cosines_mean,
        "nc2_cosines_std": nc2_cosines_std,
        "nc3": nc3,
        "nc4": nc4,
        "nc1_within_class_var": float(sigma_W_trace),
        "nc1_between_class_var": float(sigma_B_trace),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _load_model_and_head(
    ckpt_path: str, num_classes: int, device: torch.device
) -> tuple[nn.Module, np.ndarray, np.ndarray | None, int]:
    """Load model, extract W and b from the classification head, return epoch."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from checkpoint import load_checkpoint
    from model import build_model

    ckpt = load_checkpoint(ckpt_path, map_location="cpu")

    json_path = Path(ckpt_path).with_suffix(".json")
    model_name = "resnet18"
    cifar_stem = False
    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f)
        model_name = meta.get("model", "resnet18")
        cifar_stem = meta.get("cifar_stem", False)
    else:
        model_name = ckpt.get("model", "resnet18")
        cifar_stem = ckpt.get("cifar_stem", False)

    epoch = ckpt.get("epoch", 0)
    model = build_model(model_name, num_classes=num_classes,
                        cifar_stem=cifar_stem).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Extract W and b from the last Linear layer
    last_linear: nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    W = last_linear.weight.detach().cpu().numpy()   # (C, d)
    b = (last_linear.bias.detach().cpu().numpy()
         if last_linear.bias is not None else None)

    return model, W, b, int(epoch)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_nc_bars(
    results: list[dict],
    labels: list[str],
    out_dir: Path,
    title: str = "",
    model_tag: str = "",
) -> None:
    """Bar chart comparing NC1–NC4 across multiple checkpoints."""
    metrics = ["nc1", "nc2", "nc3", "nc4"]
    ylabels = [
        r"NC1  $\Sigma_W / \Sigma_B$  (↓ = collapse)",
        r"NC2  ETF deviation  (↓ = equiangular)",
        r"NC3  $||W/||W|| - \tilde{M}/||\tilde{M}|||_F$  (↓ = self-dual)",
        r"NC4  classifier disagreement  (↓ = simplex ETF)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(labels))
    for ax, metric, ylabel in zip(axes, metrics, ylabels):
        vals = [r[metric] for r in results]
        bars = ax.bar(x, vals,
                      color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
                      alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(metric.upper(), fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + max(vals) * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(title or "Neural Collapse metrics", fontsize=11, y=1.01)
    plt.tight_layout()
    tag = f"_{model_tag}" if model_tag else ""
    _save(fig, out_dir / f"nc_bars{tag}_{_ts()}.png")


def plot_nc_over_epochs(
    epoch_results: list[list[dict]],
    group_labels: list[str],
    epochs: list[list[int]],
    out_dir: Path,
    title: str = "",
    model_tag: str = "",
) -> None:
    """Line plot of NC1–NC4 over training epochs for multiple groups."""
    metrics = ["nc1", "nc2", "nc3", "nc4"]
    ylabels = [
        r"NC1  $\Sigma_W / \Sigma_B$",
        r"NC2  ETF deviation",
        r"NC3  $||W/||W|| - \tilde{M}/||\tilde{M}|||_F$",
        r"NC4  classifier disagreement",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes_flat = axes.flatten()

    for ax, metric, ylabel in zip(axes_flat, metrics, ylabels):
        for i, (results, label, eps) in enumerate(
            zip(epoch_results, group_labels, epochs)
        ):
            vals = [r[metric] for r in results]
            ax.plot(eps, vals, "o-", color=COLORS[i % len(COLORS)],
                    linewidth=2, label=label, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(metric.upper(), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle(title or "Neural Collapse over training epochs", fontsize=12)
    plt.tight_layout()
    tag = f"_{model_tag}" if model_tag else ""
    _save(fig, out_dir / f"nc_over_epochs{tag}_{_ts()}.png")


def plot_nc2_cosines(
    results: list[dict],
    labels: list[str],
    out_dir: Path,
    num_classes: int = 10,
    title: str = "",
    model_tag: str = "",
) -> None:
    """Boxplot of off-diagonal cosines between class means (NC2 detail)."""
    # We need to re-run for the cosine distributions — store them during compute
    # This plot uses nc2_cosines_mean and nc2_cosines_std
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))
    x = np.arange(len(labels))
    means = [r["nc2_cosines_mean"] for r in results]
    stds = [r["nc2_cosines_std"] for r in results]
    target = -1.0 / (num_classes - 1)

    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
           color=[COLORS[i % len(COLORS)] for i in range(len(labels))],
           edgecolor="white", label="Mean ± std of off-diagonal cosines")
    ax.axhline(target, color="red", linestyle="--", linewidth=1.5,
               label=f"ETF target = {target:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$\cos(\tilde\mu_c, \tilde\mu_{c'})$  for $c \neq c'$")
    ax.set_title("NC2 detail — off-diagonal cosines between class means\n"
                 "(ETF = all equal to -1/(C-1))")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    tag = f"_{model_tag}" if model_tag else ""
    _save(fig, out_dir / f"nc2_cosines{tag}_{_ts()}.png")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Neural Collapse (NC1–NC4) from saved checkpoints."
    )
    parser.add_argument("--experiment", default="default",
                        help="Experiment name — results go to results/<experiment>/")
    parser.add_argument("--results-dir", default="../results")
    parser.add_argument(
        "--ckpt", nargs="+", required=True,
        help="One or more checkpoint .pth files to analyze. "
             "For --plot-over-epochs, pass checkpoints in epoch order. "
             "For multi-group comparison, use --group-sizes to split them.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Label for each checkpoint (default: auto from filename).",
    )
    parser.add_argument(
        "--group-labels", nargs="+", default=None,
        help="Labels for groups of checkpoints (used with --group-sizes). "
             "If set, enables multi-group over-epoch plots.",
    )
    parser.add_argument(
        "--group-sizes", nargs="+", type=int, default=None,
        help="Number of checkpoints per group (must sum to len(--ckpt)). "
             "E.g. --group-sizes 20 20 means two groups of 20 epochs each.",
    )
    parser.add_argument(
        "--plot-over-epochs", action="store_true",
        help="Plot NC metrics as a function of epoch rather than as bars.",
    )
    parser.add_argument("--data-root", default="../cifar10")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--split", default="train", choices=["train", "test"],
        help="Dataset split to use (default: train, as in the original paper).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layer-index", type=int, default=-1,
        help=(
            "Which Linear layer to hook for feature extraction, counting from the end. "
            "-1 (default) = last Linear = penultimate representation (just before classifier). "
            "-2 = second-to-last Linear (one layer earlier). "
            "For ResNet18: -1 = fc (512→10). "
            "For WideResNet: -1 = linear (256→10). "
            "For ViT-S: -1 = mlp_head.1 (512→10), -2 = mlp_head SPT linear (240→512)."
        ),
    )
    parser.add_argument(
        "--class-names", nargs="+", default=None,
        help="Class names in label order (default: CIFAR-10 names). "
             "E.g. --class-names airplane automobile bird cat deer dog frog horse ship truck",
    )
    parser.add_argument(
        "--correct-only", action="store_true",
        help="Only use correctly classified examples for NC metrics. "
             "Gives a cleaner signal but excludes misclassified examples. "
             "Default: use all examples (as in the original Papyan paper).",
    )
    parser.add_argument("--title", default="")

    args = parser.parse_args()
    out_dir = Path(args.results_dir) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data import load_cifar10, set_seed
    from torch.utils.data import DataLoader

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s | split: %s", device, args.split)

    train_loader_full, test_loader_full, train_dataset, test_dataset = load_cifar10(
        args.data_root, batch_size=256, shuffle_train=False, drop_last_train=False,
    )
    dataset = train_dataset if args.split == "train" else test_dataset
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    # Process all checkpoints
    all_results: list[dict] = []
    all_epochs: list[int] = []
    auto_labels: list[str] = []

    for ckpt_path in args.ckpt:
        logger.info("=== %s ===", Path(ckpt_path).name)
        model, W, b, epoch = _load_model_and_head(ckpt_path, args.num_classes, device)
        features, labels_arr = extract_penultimate(
            model, loader, device, args.num_classes,
            layer_index=args.layer_index,
            correct_only=args.correct_only,
        )
        metrics = compute_nc_metrics(features, labels_arr, W, b, args.num_classes)
        metrics["epoch"] = epoch
        metrics["checkpoint"] = str(ckpt_path)

        # Per-class accuracy (always computed, logged when --correct-only or --verbose)
        class_names = args.class_names or [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ][:args.num_classes]
        per_class = log_per_class_accuracy(
            features, labels_arr, W, args.num_classes, class_names
        )
        metrics["per_class_accuracy"] = per_class
        all_results.append(metrics)
        all_epochs.append(epoch)
        auto_labels.append(Path(ckpt_path).stem[:40])
        logger.info(
            "NC1=%.4f | NC2=%.4f | NC3=%.4f | NC4=%.4f",
            metrics["nc1"], metrics["nc2"], metrics["nc3"], metrics["nc4"],
        )

    labels = args.labels or auto_labels

    # Infer model name from first checkpoint JSON
    first_json = Path(args.ckpt[0]).with_suffix(".json")
    model_tag = "unknown"
    if first_json.exists():
        with open(first_json) as f:
            model_tag = json.load(f).get("model", "unknown")
    if args.layer_index != -1:
        model_tag = f"{model_tag}_layer{args.layer_index}"
    if args.correct_only:
        model_tag = f"{model_tag}_correctonly"

    # Save JSON
    json_out = out_dir / f"nc_metrics_{model_tag}_{_ts()}.json"
    with open(json_out, "w") as f:
        json.dump({"results": all_results, "labels": labels,
                   "model": model_tag}, f, indent=2)
    logger.info("Metrics saved: %s", json_out)

    title = args.title or f"Neural Collapse — {args.experiment} ({model_tag})"

    if args.plot_over_epochs and args.group_labels and args.group_sizes:
        groups: list[list[dict]] = []
        group_epochs: list[list[int]] = []
        idx = 0
        for size in args.group_sizes:
            groups.append(all_results[idx:idx+size])
            group_epochs.append(all_epochs[idx:idx+size])
            idx += size
        plot_nc_over_epochs(groups, args.group_labels, group_epochs,
                            out_dir, title, model_tag)
    elif args.plot_over_epochs:
        plot_nc_over_epochs([all_results], [args.experiment], [all_epochs],
                            out_dir, title, model_tag)
    else:
        plot_nc_bars(all_results, labels, out_dir, title, model_tag)
        plot_nc2_cosines(all_results, labels, out_dir, args.num_classes,
                         title, model_tag)


if __name__ == "__main__":
    main()