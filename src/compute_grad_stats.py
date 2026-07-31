#!/usr/bin/env python
"""
compute_grad_stats.py
======================
Single script for per-sample gradient norm analysis: compute from checkpoints,
then plot from the saved JSON files.

Two subcommands:

    compute   Load one or several checkpoints, run a clean backward pass on
              the training set (or a subset), and save raw per-sample norms
              to a JSON file in results/<experiment>/.

    plot      Load previously computed JSON files and produce figures
              (clip fraction, stage norms, layer distribution, ...) in
              results/<experiment>/.

Why compute outside training?
    Computing grad_sample during Opacus training is fragile (virtual steps,
    grad_sample lifetime, BatchMemoryManager interactions). Here we do a plain
    backward pass, one example at a time, completely independently of Opacus:

        optimizer.zero_grad()
        loss = CrossEntropy(model(x_i), y_i)
        loss.backward()
        g^(i) = {param.grad for each param}

    This gives exact per-sample gradients. We compute:
      - global norm  : ||g^(i)||_2 = sqrt( sum_l ||g_l^(i)||_F^2 )   [all params]
      - layer norm   : ||g_l^(i)||_2 = ||g_l^(i)||_F                 [one layer]
      - stage norm   : ||g_s^(i)||_2 = sqrt( sum_{l in s} ... )       [one stage]
      - RMS norm     : ||g||_RMS = ||g||_2 / sqrt(d)                  [scale-free]

    RMS normalization divides by sqrt(d), the square root of the number of
    parameters. This makes cross-layer comparison fair: a raw L2 norm grows
    as sigma*sqrt(d) for random gradients, so larger layers always look bigger.
    The RMS norm is O(sigma) regardless of d.

Examples:
    # 1. Compute norms for several checkpoints (e.g. epochs 25, 50, 75, 100)
    python compute_grad_stats.py compute \\
        --experiment optimal \\
        --ckpt ../networks/optimal/dp_..._ep25.pth \\
               ../networks/optimal/dp_..._ep50.pth \\
               ../networks/optimal/dp_..._ep75.pth \\
               ../networks/optimal/dp_....pth \\
        --data-root ../cifar10

    # 2. Quick estimate on 5000 examples to tune C before a full run
    python compute_grad_stats.py compute \\
        --experiment optimal \\
        --ckpt ../networks/optimal/dp_..._ep50.pth \\
        --data-root ../cifar10 \\
        --max-examples 5000

    # 3. What-if: what fraction would be clipped with C=0.5 instead of C=1.0?
    python compute_grad_stats.py compute \\
        --experiment optimal \\
        --ckpt ../networks/optimal/dp_..._ep50.pth \\
        --data-root ../cifar10 --C 0.5

    # 4. Plot clip fraction + stage norms from a previously computed JSON
    python compute_grad_stats.py plot \\
        --experiment optimal \\
        --json ../results/optimal/grad_norms_dp_..._ep50_<ts>.json \\
        --plots clip-fraction stage-norms

    # 5. Compare two checkpoints (e.g. baseline vs DP) on layer RMS norms
    python compute_grad_stats.py plot \\
        --experiment optimal \\
        --json ../results/optimal/grad_norms_baseline_..._ep50_<ts>.json \\
               ../results/optimal/grad_norms_dp_..._ep50_<ts>.json \\
        --labels Baseline DP \\
        --plots layer-norms

    # 6. Evolution of clip fraction across epochs (pass JSON files in epoch order)
    python compute_grad_stats.py plot \\
        --experiment optimal \\
        --json ../results/optimal/grad_norms_dp_..._ep25_<ts>.json \\
               ../results/optimal/grad_norms_dp_..._ep50_<ts>.json \\
               ../results/optimal/grad_norms_dp_..._ep75_<ts>.json \\
               ../results/optimal/grad_norms_dp_...._<ts>.json \\
        --plots clip-fraction-over-epochs
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STAGE_ORDER = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
STAGE_COLORS = {
    "conv1": "#4c72b0", "layer1": "#dd8452", "layer2": "#55a868",
    "layer3": "#c44e52", "layer4": "#8172b3", "fc": "#937860",
}

CONV_LAYER_NAMES: frozenset[str] = frozenset([
    "conv1.weight",
    "layer1.0.conv1.weight", "layer1.0.conv2.weight",
    "layer1.1.conv1.weight", "layer1.1.conv2.weight",
    "layer2.0.conv1.weight", "layer2.0.conv2.weight",
    "layer2.1.conv1.weight", "layer2.1.conv2.weight",
    "layer3.0.conv1.weight", "layer3.0.conv2.weight",
    "layer3.1.conv1.weight", "layer3.1.conv2.weight",
    "layer4.0.conv1.weight", "layer4.0.conv2.weight",
    "layer4.1.conv1.weight", "layer4.1.conv2.weight",
    "fc.weight",
])

STAGE_ASSIGNMENT: dict[str, str] = {
    "conv1.weight": "conv1", "fc.weight": "fc",
    **{k: f"layer{k[5]}" for k in CONV_LAYER_NAMES if k.startswith("layer")},
}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


def _summarize(arr: np.ndarray, C: float | None = None) -> dict[str, float]:
    s: dict[str, float] = {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)), "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)), "p90": float(np.percentile(arr, 90)),
        "min": float(arr.min()), "max": float(arr.max()), "n": int(len(arr)),
    }
    if C is not None:
        s["frac_clipped"] = float((arr > C).mean())
    return s


# ─────────────────────────────────────────────
# COMPUTE subcommand
# ─────────────────────────────────────────────

def _get_layer_config(model_name: str) -> tuple[frozenset[str], dict[str, str], list[str], dict[str, str]]:
    """Return (layer_names, stage_assignment, stage_order, stage_colors) for a model."""
    name = (model_name or "resnet18").lower()

    if name in ("vit-s", "vit-t"):
        # ViT-S: 6 transformer blocks × (attention Q/K/V + MLP first linear) + head
        depth = 6 if name == "vit-s" else 4
        layer_names = frozenset(
            [f"transformer.layers.{i}.0.fn.to_qkv.weight" for i in range(depth)]
            + [f"transformer.layers.{i}.0.fn.to_out.0.weight" for i in range(depth)]
            + [f"transformer.layers.{i}.1.fn.net.0.weight" for i in range(depth)]
            + [f"transformer.layers.{i}.1.fn.net.3.weight" for i in range(depth)]
            + ["mlp_head.1.weight"]
        )
        stage_assignment = {}
        for i in range(depth):
            stage_assignment[f"transformer.layers.{i}.0.fn.to_qkv.weight"] = f"block{i}"
            stage_assignment[f"transformer.layers.{i}.0.fn.to_out.0.weight"] = f"block{i}"
            stage_assignment[f"transformer.layers.{i}.1.fn.net.0.weight"] = f"block{i}"
            stage_assignment[f"transformer.layers.{i}.1.fn.net.3.weight"] = f"block{i}"
        stage_assignment["mlp_head.1.weight"] = "head"
        stage_order = [f"block{i}" for i in range(depth)] + ["head"]
        colors = plt.cm.tab10(np.linspace(0, 0.9, len(stage_order)))
        stage_colors = {s: f"#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}"
                        for s, c in zip(stage_order, colors)}
        return layer_names, stage_assignment, stage_order, stage_colors

    elif "wideresnet" in name:
        layer_names = frozenset([
            "conv1.weight",
            "layer1.0.conv1.weight", "layer1.0.conv2.weight",
            "layer1.1.conv1.weight", "layer1.1.conv2.weight",
            "layer2.0.conv1.weight", "layer2.0.conv2.weight",
            "layer2.1.conv1.weight", "layer2.1.conv2.weight",
            "layer3.0.conv1.weight", "layer3.0.conv2.weight",
            "layer3.1.conv1.weight", "layer3.1.conv2.weight",
            "linear.weight",
        ])
        stage_assignment = {
            "conv1.weight": "conv1", "linear.weight": "fc",
            **{k: f"layer{k[5]}" for k in layer_names if k.startswith("layer")},
        }
        stage_order = ["conv1", "layer1", "layer2", "layer3", "fc"]
        stage_colors = {
            "conv1": "#4c72b0", "layer1": "#dd8452", "layer2": "#55a868",
            "layer3": "#c44e52", "fc": "#937860",
        }
        return layer_names, stage_assignment, stage_order, stage_colors

    else:  # resnet18 (default)
        return CONV_LAYER_NAMES, STAGE_ASSIGNMENT, STAGE_ORDER, STAGE_COLORS


def cmd_compute(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from checkpoint import load_checkpoint
    from data import load_cifar10, set_seed
    from model import build_model
    from tqdm.auto import tqdm

    out_dir = Path(args.results_dir) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    for ckpt_path in args.ckpt:
        logger.info("=== %s ===", Path(ckpt_path).name)

        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        cifar_stem = ckpt.get("cifar_stem", False)
        C = args.C or ckpt.get("max_grad_norm") or ckpt.get("max_grad_norm_C")
        epoch = ckpt.get("epoch", "?")

        # Read model name from JSON sidecar or checkpoint payload
        json_path = Path(ckpt_path).with_suffix(".json")
        model_name = "resnet18"
        if json_path.exists():
            import json as _json
            with open(json_path) as f:
                meta = _json.load(f)
            model_name = meta.get("model", "resnet18")
            if not cifar_stem:
                cifar_stem = meta.get("cifar_stem", False)
        else:
            model_name = ckpt.get("model", "resnet18")

        logger.info("epoch=%s | model=%s | cifar_stem=%s | C=%s",
                    epoch, model_name, cifar_stem, C)

        # Get layer/stage config for this architecture
        layer_names, stage_assignment, stage_order, stage_colors = _get_layer_config(model_name)

        model = build_model(model_name, num_classes=args.num_classes,
                            cifar_stem=cifar_stem).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.train()

        set_seed(args.seed)
        _, _, train_dataset, _ = load_cifar10(
            args.data_root, batch_size=64,
            shuffle_train=False, drop_last_train=False,
        )
        loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        param_dims = {n: int(p.numel()) for n, p in model.named_parameters()}
        total_d = sum(param_dims.values())
        sqrt_total_d = total_d ** 0.5

        global_norms: list[float] = []
        global_norms_rms: list[float] = []
        layer_norms: dict[str, list[float]] = {k: [] for k in layer_names}
        layer_norms_rms: dict[str, list[float]] = {k: [] for k in layer_names}
        stage_norms: dict[str, list[float]] = {s: [] for s in stage_order}

        n = 0
        for images, targets in tqdm(loader, desc="grad norms"):
            for i in range(len(images)):
                if args.max_examples and n >= args.max_examples:
                    break
                x = images[i:i+1].to(device)
                y = targets[i:i+1].to(device)

                model.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()

                global_sq = 0.0
                stage_sq: dict[str, float] = {s: 0.0 for s in stage_order}

                for name, param in model.named_parameters():
                    if param.grad is None or name not in layer_names:
                        continue
                    g_sq = param.grad.detach().norm(2).item() ** 2
                    d = param_dims[name]
                    layer_norms[name].append(g_sq ** 0.5)
                    layer_norms_rms[name].append(g_sq ** 0.5 / d ** 0.5)
                    global_sq += g_sq
                    stage = stage_assignment.get(name)
                    if stage:
                        stage_sq[stage] += g_sq

                g_norm = global_sq ** 0.5
                global_norms.append(g_norm)
                global_norms_rms.append(g_norm / sqrt_total_d)
                for s in stage_order:
                    stage_norms[s].append(stage_sq[s] ** 0.5)
                n += 1

            if args.max_examples and n >= args.max_examples:
                break

        logger.info("Processed %d examples.", n)

        arr = np.array(global_norms)
        global_summary = _summarize(arr, C)
        if C:
            logger.info("Clip fraction (C=%.2f): %.1f%%",
                        C, global_summary.get("frac_clipped", 0) * 100)
        logger.info("Global norm: median=%.4f | p90=%.4f",
                    global_summary["median"], global_summary["p90"])

        out_path = out_dir / f"grad_norms_{Path(ckpt_path).stem}_{_ts()}.json"
        with open(out_path, "w") as f:
            json.dump({
                "checkpoint": str(ckpt_path),
                "model": model_name,
                "epoch": epoch, "cifar_stem": cifar_stem,
                "C": C, "n_examples": n,
                "param_dims": {k: param_dims[k] for k in layer_names if k in param_dims},
                "total_params": total_d,
                "global_summary": global_summary,
                "layer_summaries": {k: _summarize(np.array(v), C)
                                    for k, v in layer_norms.items() if v},
                "layer_rms_summaries": {k: _summarize(np.array(v))
                                        for k, v in layer_norms_rms.items() if v},
                "stage_summaries": {s: _summarize(np.array(v), C)
                                    for s, v in stage_norms.items() if v},
                "stage_rms_summaries": {
                    s: _summarize(
                        np.array(stage_norms[s]) / max((sum(
                            param_dims.get(k, 0) for k in layer_names
                            if stage_assignment.get(k) == s
                        )) ** 0.5, 1.0)
                    )
                    for s in stage_order if stage_norms[s]
                },
                "global_norms": global_norms,
                "global_norms_rms": global_norms_rms,
                "layer_norms": {k: v for k, v in layer_norms.items() if v},
                "layer_norms_rms": {k: v for k, v in layer_norms_rms.items() if v},
                "stage_norms": {s: v for s, v in stage_norms.items() if v},
                "stage_norms_rms": {
                    s: (np.array(stage_norms[s]) / max((sum(
                        param_dims.get(k, 0) for k in layer_names
                        if stage_assignment.get(k) == s
                    )) ** 0.5, 1.0)).tolist()
                    for s in stage_order if stage_norms[s]
                },
            }, f, indent=2)
        logger.info("Saved: %s", out_path)


# ─────────────────────────────────────────────
# PLOT subcommand — individual plot functions
# ─────────────────────────────────────────────

def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_clip_fraction(data: dict, out_dir: Path, label: str = "") -> None:
    """Distribution of per-sample global norms + clipping threshold + median."""
    norms = np.array(data["global_norms"])
    C = data.get("C")
    epoch = data.get("epoch", "?")
    n = data["n_examples"]
    gs = data.get("global_summary", {})
    median = gs.get("median", float(np.median(norms)))
    frac = gs.get("frac_clipped")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Panel 1: histogram + C line + median line
    ax1.hist(norms, bins=60, color="#4c72b0", alpha=0.8, edgecolor="none")
    if C:
        ax1.axvline(C, color="red", linestyle="--", linewidth=2, label=f"C = {C}")
    ax1.axvline(median, color="orange", linestyle="-", linewidth=2,
                label=f"Median = {median:.3f}")
    ax1.set_xlabel(r"$||g^{(i)}||_2$  (global per-sample norm)")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"Per-sample gradient norm distribution\n"
        f"{label or Path(data['checkpoint']).stem[:50]}  (epoch {epoch}, n={n})"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Panel 2: horizontal bar chart of summary stats + C + median
    stats = ["p10", "p25", "median", "p75", "p90", "mean"]
    vals = [gs.get(s, float("nan")) for s in stats]
    bar_colors = ["#aec7e8", "#aec7e8", "#f28e2b", "#aec7e8", "#aec7e8", "#1f77b4"]
    ax2.barh(stats, vals, color=bar_colors, alpha=0.85, edgecolor="white")
    if C:
        ax2.axvline(C, color="red", linestyle="--", linewidth=2, label=f"C = {C}")
    ax2.axvline(median, color="orange", linestyle="-", linewidth=1.5,
                label=f"Median = {median:.3f}")
    # Annotate values on bars
    for stat, v in zip(stats, vals):
        if not np.isnan(v):
            ax2.text(v + max(vals) * 0.01, stats.index(stat),
                     f"{v:.3f}", va="center", fontsize=8)
    ax2.set_xlabel(r"$||g^{(i)}||_2$")
    ax2.set_title(
        "Summary statistics\n"
        + (f"Clipped: {frac*100:.1f}%  (C = {C})" if frac is not None else "")
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4, axis="x")

    plt.tight_layout()
    name = f"clip_fraction_{label or 'ep' + str(epoch)}_{_ts()}.png"
    _save(fig, out_dir / name)


def plot_stage_norms(data: dict, out_dir: Path, label: str = "",
                     use_rms: bool = True) -> None:
    """Two-panel bar chart: L2 norms (left) and RMS norms (right) per stage.

    The L2 panel shows the raw norm ||g_s^(i)||_2 — useful to see which
    stage contributes most to the global clipping norm.
    The RMS panel shows ||g_s^(i)||_2 / sqrt(d_s) — scale-free, comparable
    across stages regardless of how many parameters each stage has.
    d_s (number of conv parameters in the stage) is annotated on the RMS bars.
    """
    epoch = data.get("epoch", "?")
    C = data.get("C")
    param_dims = data.get("param_dims", {})

    # Compute d_s per stage from the stored param_dims
    stage_dims: dict[str, int] = {}
    for s in STAGE_ORDER:
        stage_dims[s] = sum(
            param_dims.get(k, 0) for k in CONV_LAYER_NAMES
            if STAGE_ASSIGNMENT.get(k) == s
        )

    l2_summaries = data.get("stage_summaries", {})
    rms_summaries = data.get("stage_rms_summaries", {})

    stages = [s for s in STAGE_ORDER if s in l2_summaries]

    def _vals(summaries: dict, key: str) -> list[float]:
        return [summaries.get(s, {}).get(key, float("nan")) for s in stages]

    colors = [STAGE_COLORS.get(s, "#888") for s in stages]
    x = np.arange(len(stages))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ckpt_name = label or Path(data["checkpoint"]).stem[:50]

    # ── Panel 1: L2 norms ──
    medians_l2 = _vals(l2_summaries, "median")
    p10_l2 = _vals(l2_summaries, "p10")
    p90_l2 = _vals(l2_summaries, "p90")
    ax1.bar(x, medians_l2,
            yerr=[np.array(medians_l2) - np.array(p10_l2),
                  np.array(p90_l2) - np.array(medians_l2)],
            capsize=5, color=colors, alpha=0.85, edgecolor="white")
    if C:
        ax1.axhline(C, color="red", linestyle="--", linewidth=1.5, label=f"C = {C}")
        ax1.legend(fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(stages)
    ax1.set_ylabel(r"$||g_s^{(i)}||_2$  (median, P10–P90)")
    ax1.set_title(f"Stage norms — L2 (raw)\n{ckpt_name}  (epoch {epoch})")
    for xi, v in zip(x, medians_l2):
        if not np.isnan(v):
            ax1.text(xi, v * 1.02, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # ── Panel 2: RMS norms with d_s annotated ──
    medians_rms = _vals(rms_summaries, "median")
    p10_rms = _vals(rms_summaries, "p10")
    p90_rms = _vals(rms_summaries, "p90")
    bars = ax2.bar(x, medians_rms,
                   yerr=[np.array(medians_rms) - np.array(p10_rms),
                         np.array(p90_rms) - np.array(medians_rms)],
                   capsize=5, color=colors, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x); ax2.set_xticklabels(stages)
    ax2.set_ylabel(r"$||g_s^{(i)}||_2 / \sqrt{d_s}$  (median, P10–P90)")
    ax2.set_title(f"Stage norms — RMS (scale-free)\n{ckpt_name}  (epoch {epoch})")
    for xi, s, v, bar in zip(x, stages, medians_rms, bars):
        if not np.isnan(v):
            ds = stage_dims.get(s, 0)
            # Norm value above bar
            ax2.text(xi, v * 1.02, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
            # d_s below x-axis label
            ax2.text(xi, -max(medians_rms) * 0.12,
                     f"$d_s$={ds:,}", ha="center", va="top", fontsize=7, color="#555")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    # Make room for d_s annotations below x-axis
    ax2.set_ylim(bottom=-max(v for v in medians_rms if not np.isnan(v)) * 0.18)

    plt.tight_layout()
    name = f"stage_norms_{label or 'ep' + str(epoch)}_{_ts()}.png"
    _save(fig, out_dir / name)


def plot_layer_norms(datasets: list[dict], labels: list[str], out_dir: Path,
                     use_rms: bool = True) -> None:
    """Compare layer-level (conv only) median norms across checkpoints."""
    key = "layer_rms_summaries" if use_rms else "layer_summaries"
    layers_short = [k.replace(".weight", "") for k in sorted(CONV_LAYER_NAMES)]
    layers_full = sorted(CONV_LAYER_NAMES)

    fig, ax = plt.subplots(figsize=(max(10, len(layers_full) * 0.5), 4))
    x = np.arange(len(layers_full))
    width = 0.8 / max(len(datasets), 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(datasets)))

    for idx, (data, lbl) in enumerate(zip(datasets, labels)):
        summaries = data.get(key, {})
        medians = [summaries.get(k, {}).get("median", float("nan")) for k in layers_full]
        offset = (idx - len(datasets) / 2 + 0.5) * width
        ax.bar(x + offset, medians, width=width * 0.9,
               color=colors[idx], alpha=0.8, label=lbl)

    unit = r"$||g_l^{(i)}||_2 / \sqrt{d_l}$" if use_rms else r"$||g_l^{(i)}||_2$"
    ax.set_ylabel(unit + "  (median)")
    ax.set_xticks(x); ax.set_xticklabels(layers_short, rotation=90, fontsize=8)
    ax.set_title(f"Per-layer gradient norms  {'(RMS)' if use_rms else '(L2)'}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    plt.tight_layout()
    rms_tag = "rms" if use_rms else "l2"
    name = f"layer_norms_{rms_tag}_{_ts()}.png"
    _save(fig, out_dir / name)


def plot_stage_norms_over_epochs(datasets: list[dict], labels: list[str],
                                  out_dir: Path, use_rms: bool = True) -> None:
    """Evolution of stage norms across epochs — one line per stage.

    Each point is the median norm over examples for that stage at that epoch.
    The shaded band is the P10-P90 range across examples (within-epoch variability).
    d_s = number of conv parameters in the stage, shown in the legend.
    """
    paired = sorted(zip(datasets, labels), key=lambda x: x[0].get("epoch", 0))
    datasets, labels = zip(*paired) if paired else ([], [])

    epochs = [d.get("epoch", i + 1) for i, d in enumerate(datasets)]
    key_sum = "stage_rms_summaries" if use_rms else "stage_summaries"

    # Retrieve d_s from the first dataset that has param_dims
    param_dims = next((d.get("param_dims", {}) for d in datasets if d.get("param_dims")), {})
    stage_dims: dict[str, int] = {}
    for s in STAGE_ORDER:
        stage_dims[s] = sum(
            param_dims.get(k, 0) for k in CONV_LAYER_NAMES
            if STAGE_ASSIGNMENT.get(k) == s
        )

    fig, ax = plt.subplots(figsize=(11, 5))

    for stage in STAGE_ORDER:
        medians = [d.get(key_sum, {}).get(stage, {}).get("median", float("nan"))
                   for d in datasets]
        p10 = [d.get(key_sum, {}).get(stage, {}).get("p10", float("nan"))
               for d in datasets]
        p90 = [d.get(key_sum, {}).get(stage, {}).get("p90", float("nan"))
               for d in datasets]
        color = STAGE_COLORS.get(stage, "#888")
        ds = stage_dims.get(stage, 0)
        ds_str = f"{ds:,}" if ds else "?"
        label = f"{stage}  ($d_s$ = {ds_str})"
        ax.plot(epochs, medians, "o-", color=color, linewidth=2, label=label)
        ax.fill_between(epochs, p10, p90, color=color, alpha=0.1)

    unit = r"$||g_s^{(i)}||_2 / \sqrt{d_s}$" if use_rms else r"$||g_s^{(i)}||_2$"
    ax.set_xlabel("Epoch")
    ax.set_ylabel(unit)
    ax.set_title(
        f"Stage gradient norms over epochs  {'(RMS, scale-free)' if use_rms else '(L2)'}\n"
        r"Solid line = median over examples  |  Band = P10–P90 over examples"
    )
    ax.legend(title="Stage  (shaded band = P10–P90)", fontsize=9,
              loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    rms_tag = "rms" if use_rms else "l2"
    _save(fig, out_dir / f"stage_norms_over_epochs_{rms_tag}_{_ts()}.png")


def plot_clip_fraction_over_epochs(datasets: list[dict], labels: list[str],
                                    out_dir: Path) -> None:
    """Evolution of clip fraction and norm distribution across epochs.

    Datasets are sorted by epoch number automatically, so the order of
    --json arguments on the command line doesn't matter.
    """
    # Sort by epoch to be robust to argument order and sort -t'p' issues
    paired = sorted(zip(datasets, labels), key=lambda x: x[0].get("epoch", 0))
    datasets, labels = zip(*paired) if paired else ([], [])

    epochs = [d.get("epoch", i + 1) for i, d in enumerate(datasets)]
    C = next((d.get("C") for d in datasets if d.get("C")), None)

    frac = [d["global_summary"].get("frac_clipped", float("nan")) for d in datasets]
    medians = [d["global_summary"]["median"] for d in datasets]
    p10 = [d["global_summary"]["p10"] for d in datasets]
    p90 = [d["global_summary"]["p90"] for d in datasets]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(epochs, [v * 100 for v in frac], "o-", color="#c44e52", linewidth=2,
             label=r"$\hat{p}_{clip}$")
    if C:
        ax1.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% ref (optimal C)")
    ax1.set_ylabel("Examples clipped (%)")
    ax1.set_title(f"Clipping fraction over epochs  (C = {C})")
    ax1.legend(fontsize=9); ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(epochs, medians, "o-", color="#4c72b0", linewidth=2,
             label=r"Median $||g^{(i)}||_2$")
    ax2.fill_between(epochs, p10, p90, color="#4c72b0", alpha=0.15, label="P10-P90 range")
    if C:
        ax2.axhline(C, color="red", linestyle="--", linewidth=1.5, label=f"C = {C}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"$||g^{(i)}||_2$")
    ax2.set_title(r"Per-sample global norm $||g^{(i)}||_2$ over epochs")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    _save(fig, out_dir / f"clip_fraction_over_epochs_{_ts()}.png")


# ─────────────────────────────────────────────
# PLOT dispatcher
# ─────────────────────────────────────────────

def cmd_plot(args: argparse.Namespace) -> None:
    out_dir = Path(args.results_dir) / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [_load_json(p) for p in args.json]
    labels = args.labels if args.labels else [
        Path(p).stem.replace("grad_norms_", "").rsplit("_", 2)[0]
        for p in args.json
    ]

    use_rms = not args.no_rms

    for plot in args.plots:
        if plot == "clip-fraction":
            for data, lbl in zip(datasets, labels):
                plot_clip_fraction(data, out_dir, label=lbl)

        elif plot == "stage-norms":
            for data, lbl in zip(datasets, labels):
                plot_stage_norms(data, out_dir, label=lbl)

        elif plot == "layer-norms":
            plot_layer_norms(datasets, labels, out_dir, use_rms=use_rms)

        elif plot == "clip-fraction-over-epochs":
            plot_clip_fraction_over_epochs(list(datasets), list(labels), out_dir)

        elif plot == "stage-norms-over-epochs":
            plot_stage_norms_over_epochs(list(datasets), list(labels), out_dir,
                                          use_rms=not args.no_rms)

        else:
            logger.warning("Unknown plot type: '%s'. Valid: clip-fraction, stage-norms, "
                           "layer-norms, clip-fraction-over-epochs", plot)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute and plot per-sample gradient norm statistics."
    )
    parser.add_argument("--experiment", default="default",
                        help="Experiment name — results go to results/<experiment>/")
    parser.add_argument("--results-dir", default="../results",
                        help="Base directory for results (default: ../results)")
    subs = parser.add_subparsers(dest="command", required=True)

    # ── compute ──
    pc = subs.add_parser("compute",
                          help="Run backward pass on train set and save norm JSON files.")
    pc.add_argument("--ckpt", nargs="+", required=True,
                    help="One or more checkpoint .pth files.")
    pc.add_argument("--data-root", default="../cifar10")
    pc.add_argument("--max-examples", type=int, default=None,
                    help="Stop after N examples (default: full train set). "
                         "Use ~5000 for quick C-tuning estimates.")
    pc.add_argument("--C", type=float, default=None,
                    help="Override the clipping constant from the checkpoint "
                         "(e.g. to simulate a different C without retraining).")
    pc.add_argument("--seed", type=int, default=42)
    pc.add_argument("--num-classes", type=int, default=10)
    pc.set_defaults(func=cmd_compute)

    # ── plot ──
    pp = subs.add_parser("plot",
                          help="Load previously computed JSON files and produce figures.")
    pp.add_argument("--json", nargs="+", required=True,
                    help="One or more grad_norms_*.json files produced by 'compute'.")
    pp.add_argument(
        "--plots", nargs="+", required=True,
        choices=["clip-fraction", "stage-norms", "layer-norms",
                 "clip-fraction-over-epochs", "stage-norms-over-epochs"],
        help=(
            "clip-fraction              : histogram + stats for one JSON\n"
            "stage-norms                : L2 + RMS bar chart per stage for one JSON\n"
            "layer-norms                : grouped bar chart comparing all JSONs\n"
            "clip-fraction-over-epochs  : clip %% + median norm evolution (all JSONs)\n"
            "stage-norms-over-epochs    : one line per stage across epochs (all JSONs)"
        ),
    )
    pp.add_argument("--labels", nargs="+", default=None,
                    help="Legend labels for each JSON file (default: auto from filename).")
    pp.add_argument("--no-rms", action="store_true",
                    help="Use raw L2 norms instead of RMS (stage-norms and layer-norms).")
    pp.set_defaults(func=cmd_plot)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()