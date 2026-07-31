#!/usr/bin/env python
"""
grad_stats.py
=============
Standalone script for analyzing gradient statistics saved during DP-SGD
training (the *_grad_stats.json files produced by train_dp.py).

Three subcommands:

    stage-norms    : Plot grad_mean_norm per stage (conv1/layer1-4/fc) over epochs.
                     Answers: "which stage dominates the global clipping norm?"

    clip-fraction  : Plot fraction of examples clipped per epoch.
                     Answers: "is C too small (>80% clipped) or too large (<20%)?"

    layer-stats    : Full statistics for a single conv layer over epochs:
                     mean, std, min/max, quantiles, 95% CI of a chosen metric.
                     Answers: "how does this specific layer's gradient evolve?"

Examples:
    # 1. Stage norms (conv1, layer1..4, fc) — default metric: grad_mean_norm
    python grad_stats.py stage-norms \\
        --grad-json ../networks/grad_stats/dp_resnet18_..._grad_stats.json \\
        --out-dir ../results/grad_stats

    # Same with SNR
    python grad_stats.py stage-norms \\
        --grad-json ../networks/grad_stats/dp_resnet18_..._grad_stats.json \\
        --metric grad_snr --out-dir ../results/grad_stats

    # 2. Clipping fraction over epochs
    python grad_stats.py clip-fraction \\
        --grad-json ../networks/grad_stats/dp_resnet18_..._grad_stats.json \\
        --out-dir ../results/grad_stats

    # 3. Single layer statistics (metric choices: grad_mean_norm, grad_snr,
    #    grad_std_norm, grad_mean_absmax, grad_std_mean)
    python grad_stats.py layer-stats \\
        --grad-json ../networks/grad_stats/dp_resnet18_..._grad_stats.json \\
        --layer layer2.0.conv1 \\
        --metric grad_mean_norm \\
        --out-dir ../results/grad_stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.stats as _stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STAGE_ORDER = ["conv1", "layer1", "layer2", "layer3", "layer4", "fc"]
STAGE_COLORS = {
    "conv1": "#4c72b0",
    "layer1": "#dd8452",
    "layer2": "#55a868",
    "layer3": "#c44e52",
    "layer4": "#8172b3",
    "fc": "#937860",
}

METRIC_LABEL = {
    "grad_mean_norm": r"$||E[g^{(i)}]||_2$  (signal magnitude)",
    "grad_snr": r"SNR $= ||E[g^{(i)}]||_2 \;/\; ||Std[g^{(i)}]||_2$",
    "grad_std_norm": r"$||Std[g^{(i)}]||_2$  (variability across examples)",
    "grad_mean_absmax": r"$\max_k |E[g^{(i)}_k]|$  (largest mean coordinate)",
    "grad_std_mean": r"$mean_k\, Std[g^{(i)}_k]$  (avg per-coord variability)",
}

ALL_CONV_LAYERS = [
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
]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load(path: str) -> tuple[list[int], list[dict], dict[str, dict]]:
    """Load grad_stats JSON.

    Returns:
        (epochs, pre_clip_history, clipping_raw_norms)

    clipping_raw_norms: dict keyed by epoch string (e.g. "1", "2", ...).
    Each value contains raw per-sample norm arrays:
      {
        "n_samples": int,
        "n_real_steps": int,
        "max_grad_norm_C": float,
        "global": [float, ...],           # n_samples values
        "by_stage": {"conv1": [...], ...},
        "by_conv_layer": {"conv1.weight": [...], ...}
      }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    d = json.load(open(p))
    epochs = d.get("epochs", list(range(1, len(d.get("pre_clip_stats_history", [])) + 1)))
    pre_clip = d.get("pre_clip_stats_history", [])
    raw_norms = d.get("clipping_raw_norms", {})
    if not raw_norms and not pre_clip:
        raise ValueError(
            "JSON produced by an older version of train_dp.py — no raw norms found.\n"
            "Re-run training with the current version to get reconstructable statistics."
        )
    return epochs, pre_clip, raw_norms


def _compute_clip_stats(raw: dict, C: float) -> dict[str, float]:
    """Compute clipping statistics from raw norm arrays (any quantile, a posteriori)."""
    norms = np.array(raw["global"])
    return {
        "frac_clipped": float((norms > C).mean()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_median": float(np.median(norms)),
        "norm_p10": float(np.percentile(norms, 10)),
        "norm_p25": float(np.percentile(norms, 25)),
        "norm_p75": float(np.percentile(norms, 75)),
        "norm_p90": float(np.percentile(norms, 90)),
        "n_samples": int(len(norms)),
    }


def _resolve_layer_key(layer_arg: str, sample_epoch: dict) -> str | None:
    """Resolve 'layer2.0.conv1' or 'layer2.0.conv1.weight' to the actual JSON key."""
    by_param = sample_epoch.get("by_parameter", {})
    candidates = [layer_arg, layer_arg + ".weight"]
    for c in candidates:
        if c in by_param:
            return c
    # Try prefix match (in case of partial name)
    matches = [k for k in by_param if k.startswith(layer_arg)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        logger.warning("Ambiguous layer '%s', matches: %s. Using first.", layer_arg, matches)
        return matches[0]
    return None


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{_timestamp()}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)
    return path


# ─────────────────────────────────────────────
# Subcommand 1: stage-norms
# ─────────────────────────────────────────────

def cmd_stage_norms(args: argparse.Namespace) -> None:
    """Plot grad_mean_norm (or another metric) per stage over epochs."""
    epochs, pre_clip, _ = _load(args.grad_json)
    metric = args.metric
    out_dir = Path(args.out_dir)

    if not pre_clip:
        raise ValueError("pre_clip_stats_history is empty.")

    # Extract per-stage, per-epoch from Welford stats
    series: dict[str, list[float]] = {s: [] for s in STAGE_ORDER}
    for ep in pre_clip:
        by_stage = ep.get("by_stage", {})
        for stage in STAGE_ORDER:
            val = by_stage.get(stage, {}).get(metric)
            series[stage].append(float(val) if val is not None else float("nan"))

    fig, ax = plt.subplots(figsize=(10, 5))
    for stage in STAGE_ORDER:
        vals = series[stage]
        if all(np.isnan(vals)):
            continue
        ax.plot(epochs[:len(vals)], vals,
                label=stage, color=STAGE_COLORS.get(stage), linewidth=2, marker="o", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_title(fr"Pre-clip gradient {metric} by stage (before $\|g^{{(i)}}\|_2 \leq C$ clipping)")
    ax.legend(title="Stage", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    _save(fig, out_dir, f"stage_norms_{metric}")

    # Also save a JSON summary
    summary = {
        stage: {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "min": float(np.nanmin(vals)),
            "max": float(np.nanmax(vals)),
        }
        for stage, vals in series.items() if not all(np.isnan(vals))
    }
    json_path = out_dir / f"stage_norms_{metric}_{_timestamp()}.json"
    json.dump({"metric": metric, "source": args.grad_json, "summary": summary}, open(json_path, "w"), indent=2)
    logger.info("Summary JSON: %s", json_path)


# ─────────────────────────────────────────────
# Subcommand 2: clip-fraction
# ─────────────────────────────────────────────

def cmd_clip_fraction(args: argparse.Namespace) -> None:
    """Plot the fraction of examples clipped per epoch + norm distribution + stage decomposition.

    All statistics are computed from raw per-sample norm arrays stored in
    clipping_raw_norms — so you can change quantile thresholds, C values,
    or any other parameter without re-running training.
    """
    epochs, _, raw_norms = _load(args.grad_json)
    out_dir = Path(args.out_dir)

    if not raw_norms:
        raise ValueError(
            "clipping_raw_norms is empty. "
            "This may happen if grad_sample was not available (ablation --no-clip), "
            "or if the JSON was produced before the current version of train_dp.py."
        )

    C = next(iter(raw_norms.values())).get("max_grad_norm_C", None)

    # Compute per-epoch stats from raw arrays (a posteriori — change quantile here freely)
    frac, norm_med, norm_p10, norm_p90, n_samp = [], [], [], [], []
    epoch_keys = [str(ep) for ep in epochs]
    for ek in epoch_keys:
        raw = raw_norms.get(ek, {})
        if not raw:
            frac.append(float("nan")); norm_med.append(float("nan"))
            norm_p10.append(float("nan")); norm_p90.append(float("nan"))
            n_samp.append(0); continue
        s = _compute_clip_stats(raw, C)
        frac.append(s["frac_clipped"])
        norm_med.append(s["norm_median"])
        norm_p10.append(s["norm_p10"])
        norm_p90.append(s["norm_p90"])
        n_samp.append(s["n_samples"])

    # Stage medians at last epoch
    last_raw = raw_norms.get(epoch_keys[-1], {})
    last_stage_med = {
        s: float(np.median(last_raw["by_stage"][s]))
        for s in STAGE_ORDER if s in last_raw.get("by_stage", {})
    }

    n_panels = 3 if last_stage_med else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(11, 4 * n_panels))
    ax1, ax2 = axes[0], axes[1]

    # ── Panel 1: clipping fraction ──
    ax1.plot(epochs, [v * 100 for v in frac], color="#c44e52", linewidth=2,
             label=r"$\hat{p}_{clip}$  (fraction clipped)")
    if C is not None:
        ax1.axhline(50, color="gray", linestyle=":", linewidth=1, label="50% reference (optimal C)")
    ax1.set_ylabel(r"Examples clipped (\%)")
    ax1.set_xlabel("Epoch")
    ax1.set_title(f"Clipping fraction over epochs  (C = {C})")
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    if n_samp[0] > 0:
        ax1.text(0.98, 0.05, f"~{n_samp[0]} samples/epoch",
                 transform=ax1.transAxes, ha="right", fontsize=8, color="gray")

    # ── Panel 2: global norm distribution ──
    ax2.plot(epochs, norm_med, color="#4c72b0", linewidth=2,
             label=r"Median $||g^{(i)}||_2$")
    ax2.fill_between(epochs, norm_p10, norm_p90, color="#4c72b0", alpha=0.15,
                     label="P10-P90 range")
    if C is not None:
        ax2.axhline(C, color="red", linestyle=":", linewidth=1.5, label=f"C = {C}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"$||g^{(i)}||_2$")
    ax2.set_title(
        r"Pre-clip per-sample global norm  $||g^{(i)}||_2 = \sqrt{\sum_l ||g^{(i)}_l||_2^2}$"
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # ── Panel 3: per-stage median at last epoch ──
    if last_stage_med and n_panels == 3:
        ax3 = axes[2]
        stages = [s for s in STAGE_ORDER if s in last_stage_med]
        vals = [last_stage_med[s] for s in stages]
        bars = ax3.bar(stages, vals,
                       color=[STAGE_COLORS.get(s, "#888") for s in stages], alpha=0.85)
        if C is not None:
            ax3.axhline(C, color="red", linestyle=":", linewidth=1.5, label=f"C = {C}")
        ax3.set_ylabel(r"$||g^{(i)}_{stage}||_2$  (median)")
        ax3.set_xlabel("Stage")
        ax3.set_title(
            f"Stage contribution to global norm at epoch {epochs[-1]}\n"
            r"$||g^{(i)}_{stage}||_2 = \sqrt{\sum_{l \in stage} ||g^{(i)}_l||_2^2}$"
            f"  (median, {n_samp[-1]} samples)"
        )
        ax3.legend(fontsize=9)
        ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "clip_fraction")


# ─────────────────────────────────────────────
# Subcommand 3: layer-stats
# ─────────────────────────────────────────────

def cmd_layer_stats(args: argparse.Namespace) -> None:
    """Full statistics for a single conv layer over epochs.

    Two sources of data:
    - Welford signal stats (pre_clip_stats_history): one scalar per epoch,
      metrics: grad_mean_norm, grad_snr, grad_std_norm, grad_mean_absmax, grad_std_mean.
    - Raw per-sample norms (clipping_raw_norms): a distribution per epoch,
      metric: grad_norm_raw. This allows computing any quantile a posteriori.
    """
    epochs, pre_clip, raw_norms = _load(args.grad_json)
    metric = args.metric
    out_dir = Path(args.out_dir)
    is_raw = (metric == "grad_norm_raw")

    if is_raw:
        # Source: raw per-sample norms from clipping_raw_norms
        first_raw = next(iter(raw_norms.values()), {})
        by_conv = first_raw.get("by_conv_layer", {})
        key = _resolve_layer_key(args.layer, {"by_parameter": by_conv})
        if key is None:
            raise ValueError(
                f"Layer '{args.layer}' not found in clipping_raw_norms.by_conv_layer.\n"
                f"Available: {sorted(by_conv.keys())}"
            )
        vals_per_epoch = []
        for ek in [str(ep) for ep in epochs]:
            raw = raw_norms.get(ek, {})
            arr_list = raw.get("by_conv_layer", {}).get(key)
            vals_per_epoch.append(np.array(arr_list, dtype=float) if arr_list else np.array([float("nan")]))
        # Scalar summary: median per epoch
        epoch_meds = [float(np.nanmedian(v)) for v in vals_per_epoch]
        p10_ep = [float(np.nanpercentile(v[~np.isnan(v)], 10)) if np.sum(~np.isnan(v)) > 0 else float("nan") for v in vals_per_epoch]
        p90_ep = [float(np.nanpercentile(v[~np.isnan(v)], 90)) if np.sum(~np.isnan(v)) > 0 else float("nan") for v in vals_per_epoch]
        arr = np.array(epoch_meds)
        y_label = fr"$||g^{{(i)}}_{{{args.layer}}}||_2$"
        plot_title = fr"Per-sample layer norm: {args.layer}  (median per epoch)"
        # Pooled distribution across all epochs
        all_vals = np.concatenate([v[~np.isnan(v)] for v in vals_per_epoch])
    else:
        # Source: Welford mean/std from pre_clip_stats_history
        if not pre_clip:
            raise ValueError("pre_clip_stats_history is empty.")
        key = _resolve_layer_key(args.layer, pre_clip[0])
        if key is None:
            raise ValueError(
                f"Layer '{args.layer}' not found in by_parameter.\n"
                f"Available: {sorted(pre_clip[0].get('by_parameter', {}).keys())}"
            )
        epoch_meds = [
            float(ep.get("by_parameter", {}).get(key, {}).get(metric, float("nan")))
            for ep in pre_clip
        ]
        p10_ep = None; p90_ep = None; all_vals = None
        arr = np.array(epoch_meds)
        y_label = METRIC_LABEL.get(metric, metric)
        plot_title = f"{key.replace('.weight','')} — {metric}"

    n = int(np.sum(~np.isnan(arr)))
    if n == 0:
        raise ValueError("No valid data found for the requested layer/metric.")

    mean = float(np.nanmean(arr))
    std  = float(np.nanstd(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float((std / np.sqrt(n)) * _stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0

    summary: dict = {
        "layer": key, "metric": metric,
        "mean_over_epochs": mean, "std_over_epochs": std, "ci95_over_epochs": ci95,
        "min": float(np.nanmin(arr)), "max": float(np.nanmax(arr)),
        "median_over_epochs": float(np.nanmedian(arr)),
        "p10_over_epochs": float(np.nanpercentile(arr, 10)),
        "p90_over_epochs": float(np.nanpercentile(arr, 90)),
        "n_epochs": n,
    }
    if is_raw and all_vals is not None:
        summary["pooled_n"] = int(len(all_vals))
        summary["pooled_median"] = float(np.median(all_vals))
        summary["pooled_p10"] = float(np.percentile(all_vals, 10))
        summary["pooled_p70"] = float(np.percentile(all_vals, 70))
        summary["pooled_p90"] = float(np.percentile(all_vals, 90))
        logger.info("Pooled distribution (all epochs): n=%d | p10=%.4f | median=%.4f | p70=%.4f | p90=%.4f",
                    summary["pooled_n"], summary["pooled_p10"], summary["pooled_median"],
                    summary["pooled_p70"], summary["pooled_p90"])
    logger.info("  mean=%.6f ± %.6f (95%% CI, over epochs)", mean, ci95)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(epochs[:len(epoch_meds)], epoch_meds, color="#4c72b0", linewidth=2, label="Median per epoch")
    if p10_ep and p90_ep:
        ax1.fill_between(epochs[:len(epoch_meds)], p10_ep, p90_ep,
                         color="#4c72b0", alpha=0.15, label="P10-P90 (within epoch)")
    ax1.axhline(mean, color="#4c72b0", linestyle="--", alpha=0.7, label=f"Mean = {mean:.4f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(y_label)
    ax1.set_title(plot_title)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.barh([0], [mean],
             xerr=[[mean - float(np.nanpercentile(arr, 10))],
                   [float(np.nanpercentile(arr, 90)) - mean]],
             color="#4c72b0", alpha=0.8, capsize=6, label="P10-P90 (over epochs)")
    ax2.barh([0], [mean], xerr=[[ci95], [ci95]],
             color="#dd8452", alpha=0.9, capsize=6, label="95% CI (over epochs)")
    ax2.set_yticks([0]); ax2.set_yticklabels([args.layer])
    ax2.set_xlabel(y_label)
    ax2.set_title(f"Summary over {n} epochs")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")

    plt.suptitle(f"{args.layer} | {metric}", y=1.01)
    plt.tight_layout()
    layer_tag = args.layer.replace(".", "_")
    _save(fig, out_dir, f"layer_stats_{layer_tag}_{metric}")

    json_path = out_dir / f"layer_stats_{layer_tag}_{metric}_{_timestamp()}.json"
    json.dump({"source": args.grad_json, **summary, "time_series_median": epoch_meds},
              open(json_path, "w"), indent=2)
    logger.info("Summary JSON: %s", json_path)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze gradient statistics from DP-SGD training (_grad_stats.json)."
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # shared options
    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--grad-json", required=True,
                       help="Path to the *_grad_stats.json produced by train_dp.py.")
        p.add_argument("--out-dir", default="../results/grad_stats",
                       help="Output directory for figures and JSON summaries.")

    def _add_metric(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--metric", default="grad_mean_norm",
            choices=list(METRIC_LABEL.keys()),
            help="Metric from Welford stats. Choices:\n"
                 + "\n".join(f"  {k}: {v}" for k, v in METRIC_LABEL.items()),
        )

    def _add_metric_layer(p: argparse.ArgumentParser) -> None:
        """Metric choices for layer-stats: Welford stats + raw per-sample norms."""
        p.add_argument(
            "--metric", default="grad_mean_norm",
            choices=list(METRIC_LABEL.keys()) + ["grad_norm_raw"],
            help=(
                "Metric to analyze.\n"
                "  Welford stats (pre_clip_stats_history): "
                + ", ".join(METRIC_LABEL.keys()) + "\n"
                "  Raw per-sample norm (clipping_raw_norms):\n"
                "    grad_norm_raw : ||g_l^(i)||_2 per example — compute any quantile a posteriori"
            ),
        )

    # stage-norms
    p1 = subs.add_parser("stage-norms",
                          help="Per-stage grad norm (conv1/layer1-4/fc) over epochs.")
    _add_common(p1); _add_metric(p1)
    p1.set_defaults(func=cmd_stage_norms)

    # clip-fraction
    p2 = subs.add_parser("clip-fraction",
                          help="Fraction of examples clipped per epoch + norm distribution.")
    _add_common(p2)
    p2.set_defaults(func=cmd_clip_fraction)

    # layer-stats
    p3 = subs.add_parser("layer-stats",
                          help="Full stats for a single conv layer over epochs.")
    _add_common(p3); _add_metric_layer(p3)
    p3.add_argument("--layer", required=True,
                    help="Layer name, e.g. 'layer2.0.conv1' or 'layer2.0.conv1.weight'.")
    p3.set_defaults(func=cmd_layer_stats)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()