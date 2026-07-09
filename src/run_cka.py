#!/usr/bin/env python
"""
run_cka.py
==========
CKA analysis between trained checkpoints. Several subcommands covering the
comparisons from the original notebook, plus multi-seed comparisons with
confidence intervals:

    compare      : CKA between any two checkpoints (high-level layers)
    multi-eps    : CKA between a baseline and several DP checkpoints (epsilon sweep)
    fine-grained : fine-grained CKA (all learnable layers, or a filtered subset)
    multi-seed   : CKA across one or two groups of checkpoints, with full
                   layer x layer heatmaps (mean + std across pairs) and a
                   diagonal mean/95% CI/min-max plot. Two modes:
                   - one group (--ckpts): ALL pairs are compared (e.g. 5
                     baseline seeds -> 10 pairs). Use this to establish how
                     low same-config CKA naturally goes from seed variance
                     alone, before attributing any drop to DP.
                   - two groups (--ckpts-a / --ckpts-b): pairs are formed by
                     matching POSITION (ckpts_a[0] vs ckpts_b[0], etc.) — use
                     this for DP vs baseline or DP vs DP across seeds, where
                     ckpts_a[i] and ckpts_b[i] should share the same seed.
                   With a single pair, only the heatmap is produced (no
                   meaningful variance to report); with 2+ pairs, all three
                   plots (heatmap mean, heatmap std, diagonal stats) are produced.

Examples:
    # Simple baseline vs DP comparison (the most common one day-to-day)
    python run_cka.py compare --ckpt-a networks/baseline_..._seed42_*.pth \\
                               --ckpt-b networks/dp_..._seed42_*.pth \\
                               --label-a Baseline --label-b DP

    # Epsilon sweep against a reference baseline
    python run_cka.py multi-eps --baseline-ckpt networks/baseline_...pth \\
                                 --dp-ckpts networks/dp_eps2...pth networks/dp_eps8...pth \\
                                 --epsilons 2 8

    # Fine-grained (checkerboard) heatmap between two checkpoints
    python run_cka.py fine-grained --ckpt-a networks/baseline_seed42.pth \\
                                    --ckpt-b networks/baseline_seed43.pth

    # Baseline-vs-baseline variance across 5 seeds, all 10 pairs compared
    python run_cka.py multi-seed --ckpts networks/baseline_seed1.pth networks/baseline_seed2.pth \\
                                          networks/baseline_seed3.pth networks/baseline_seed4.pth \\
                                          networks/baseline_seed5.pth \\
                                  --label baseline_seed_variance

    # DP vs baseline across 3 matching seeds (ckpts_a[i] vs ckpts_b[i])
    python run_cka.py multi-seed --ckpts-a networks/dp_seed1.pth networks/dp_seed2.pth networks/dp_seed3.pth \\
                                  --ckpts-b networks/baseline_seed1.pth networks/baseline_seed2.pth networks/baseline_seed3.pth \\
                                  --label dp_vs_baseline
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cka import compute_cka_matrix, summarize_diagonal
from config import add_config_overrides_args, apply_overrides, load_config
from checkpoint import load_checkpoint
from data import load_cifar10
from logging_utils import setup_logging
from model import build_resnet18_dp_compatible, list_learnable_layers
from visualization import (
    plot_accuracy_curves,
    plot_cka_results,
    plot_diagonal_stats,
    plot_epsilon_layer_heatmap,
    plot_fine_grained_matrix,
    plot_mean_heatmap,
    plot_std_heatmap,
    plot_zscore_heatmap,
)

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(ckpt_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Rebuild the DP-compatible architecture and load the checkpoint's weights.

    Reads `cifar_stem` from the checkpoint JSON (if present) to ensure the
    architecture matches exactly what was used during training. This is
    critical: loading a checkpoint trained with cifar_stem=True into a
    standard stem model would silently succeed (same parameter names) but
    produce wrong results because conv1 would have wrong kernel size.
    """
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))

    # Read cifar_stem from the JSON sidecar if available, else from the pth payload
    json_path = Path(ckpt_path).with_suffix(".json")
    cifar_stem = False
    if json_path.exists():
        import json as _json
        with open(json_path) as f:
            meta = _json.load(f)
        cifar_stem = meta.get("cifar_stem", False)
    else:
        cifar_stem = ckpt.get("cifar_stem", False)

    model = build_resnet18_dp_compatible(num_classes=num_classes, cifar_stem=cifar_stem).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def cmd_compare(args: argparse.Namespace, cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, _ = load_cifar10(cfg.data_root, batch_size=cfg.cka_batch_size, shuffle_train=False)

    model_a = load_model_from_checkpoint(args.ckpt_a, cfg.num_classes, device)
    model_b = load_model_from_checkpoint(args.ckpt_b, cfg.num_classes, device)

    logger.info("Computing CKA: %s vs %s", args.label_a, args.label_b)
    accum = compute_cka_matrix(
        model_a, model_b, cfg.cka_layers, test_loader, device,
        num_batches=cfg.cka_num_batches, kernel_type=cfg.cka_kernel,
    )

    ts = _timestamp()
    out_prefix = Path(cfg.results_path()) / f"cka_compare_{args.label_a}_vs_{args.label_b}_{ts}"
    plot_cka_results(accum, cfg.cka_layers, title_suffix=f"({args.label_a} vs {args.label_b})",
                      save_path=out_prefix.with_suffix(".png"))

    summary = summarize_diagonal(accum, cfg.cka_layers)
    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump({"label_a": args.label_a, "label_b": args.label_b, "summary": summary}, f, indent=2)

    logger.info("Results saved: %s.{png,json}", out_prefix)


def cmd_multi_eps(args: argparse.Namespace, cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, _ = load_cifar10(cfg.data_root, batch_size=cfg.cka_batch_size, shuffle_train=False)

    if len(args.dp_ckpts) != len(args.epsilons):
        raise ValueError("--dp-ckpts and --epsilons must have the same number of elements")

    baseline_model = load_model_from_checkpoint(args.baseline_ckpt, cfg.num_classes, device)

    epsilon_layer_matrix = np.zeros((len(args.epsilons), len(cfg.cka_layers)))

    for eps_idx, (eps, ckpt_path) in enumerate(zip(args.epsilons, args.dp_ckpts)):
        logger.info("Analyzing privacy budget epsilon=%s ...", eps)
        dp_model = load_model_from_checkpoint(ckpt_path, cfg.num_classes, device)

        accum = compute_cka_matrix(
            baseline_model, dp_model, cfg.cka_layers, test_loader, device,
            num_batches=cfg.cka_num_batches, kernel_type=cfg.cka_kernel,
        )
        for l_idx in range(len(cfg.cka_layers)):
            epsilon_layer_matrix[eps_idx, l_idx] = np.mean(accum[(l_idx, l_idx)])

        del dp_model

    ts = _timestamp()
    out_prefix = Path(cfg.results_path()) / f"cka_multi_epsilon_{ts}"
    plot_epsilon_layer_heatmap(epsilon_layer_matrix, args.epsilons, cfg.cka_layers,
                                save_path=out_prefix.with_suffix(".png"))

    np.save(out_prefix.with_suffix(".npy"), epsilon_layer_matrix)
    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump(
            {"epsilons": args.epsilons, "layers": cfg.cka_layers, "matrix": epsilon_layer_matrix.tolist()},
            f, indent=2,
        )

    logger.info("Results saved: %s.{png,npy,json}", out_prefix)


def cmd_multi_seed(args: argparse.Namespace, cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, _ = load_cifar10(cfg.data_root, batch_size=cfg.cka_batch_size, shuffle_train=False)

    # --layers-override (sous-parser multi-seed) prend la priorité sur cfg.cka_layers (YAML/parser principal)
    # On utilise un attribut interne distinct pour éviter le conflit avec apply_overrides
    layers_override = getattr(args, "layers_override", None)
    layers = [s.strip() for s in layers_override.split(",") if s.strip()] if layers_override else cfg.cka_layers
    logger.info("Layers used (%d): %s", len(layers), ", ".join(layers))

    # Déduire les labels d'axes depuis --label-a / --label-b ou depuis le mode
    if args.ckpts_a and args.ckpts_b:
        if len(args.ckpts_a) != len(args.ckpts_b):
            raise ValueError("--ckpts-a and --ckpts-b must have the same length (matched by position/seed)")
        ckpt_pairs = list(zip(args.ckpts_a, args.ckpts_b))
        axis_y = args.label_a or "Group A"   # Y = lignes = groupe A (ckpts-a)
        axis_x = args.label_b or "Group B"   # X = colonnes = groupe B (ckpts-b)
        logger.info("Two-group mode: %d matched pairs | Y=%s  X=%s", len(ckpt_pairs), axis_y, axis_x)
    elif args.ckpts:
        if len(args.ckpts) < 2:
            raise ValueError("--ckpts needs at least 2 checkpoints to form pairs")
        ckpt_pairs = list(itertools.combinations(args.ckpts, 2))
        axis_y = args.label_a or args.label
        axis_x = args.label_b or args.label
        logger.info("One-group mode: %d checkpoints -> %d pairs (all combinations)", len(args.ckpts), len(ckpt_pairs))
    else:
        raise ValueError("Provide either --ckpts (one group, all pairs) or both --ckpts-a and --ckpts-b (matched pairs)")

    num_layers = len(layers)
    all_pair_matrices: list[np.ndarray] = []

    for pair_idx, (ckpt_i, ckpt_j) in enumerate(ckpt_pairs):
        logger.info("Pair (%d/%d): %s vs %s", pair_idx + 1, len(ckpt_pairs), Path(ckpt_i).name, Path(ckpt_j).name)
        model_i = load_model_from_checkpoint(ckpt_i, cfg.num_classes, device)
        model_j = load_model_from_checkpoint(ckpt_j, cfg.num_classes, device)

        accum = compute_cka_matrix(
            model_i, model_j, layers, test_loader, device,
            num_batches=cfg.cka_num_batches, kernel_type=cfg.cka_kernel,
        )
        matrix = np.zeros((num_layers, num_layers))
        for a in range(num_layers):
            for b in range(num_layers):
                matrix[a, b] = np.mean(accum[(a, b)])
        all_pair_matrices.append(matrix)
        del model_i, model_j

    mean_matrix = np.mean(all_pair_matrices, axis=0)
    n_pairs = len(all_pair_matrices)

    ts = _timestamp()
    out_prefix = Path(cfg.results_path()) / f"cka_multiseed_{args.label}_{ts}"

    plot_mean_heatmap(
        mean_matrix, layers,
        title=f"Mean CKA — {args.label} (n={n_pairs} pair{'s' if n_pairs != 1 else ''})",
        ylabel=axis_y, xlabel=axis_x,
        annotate=(num_layers <= 15),
        save_path=out_prefix.with_suffix(".png"),
    )

    result_payload: dict = {
        "label": args.label,
        "label_a": axis_y,
        "label_b": axis_x,
        "mode": "two-group" if (args.ckpts_a and args.ckpts_b) else "one-group",
        "checkpoint_pairs": [[i, j] for i, j in ckpt_pairs],
        "layers": layers,
        "n_pairs": n_pairs,
        "mean_matrix": mean_matrix.tolist(),
    }

    if n_pairs > 1:
        std_matrix = np.std(all_pair_matrices, axis=0)
        plot_std_heatmap(
            std_matrix, layers,
            title=f"CKA std. dev. across pairs — {args.label}",
            ylabel=axis_y, xlabel=axis_x,
            save_path=out_prefix.parent / f"{out_prefix.name}_std.png",
        )
        diag_summary = plot_diagonal_stats(
            all_pair_matrices, layers,
            title=f"Diagonal CKA across pairs — {args.label}",
            save_path=out_prefix.parent / f"{out_prefix.name}_diagonal_stats.png",
        )
        result_payload["std_matrix"] = std_matrix.tolist()
        result_payload["diagonal_summary"] = diag_summary
    else:
        logger.info("Only 1 pair: heatmap only (std/diagonal-stats plots skipped).")

    np.save(out_prefix.with_suffix(".npy"), np.array(all_pair_matrices))
    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump(result_payload, f, indent=2)

    logger.info("Results saved: %s.*", out_prefix)


def cmd_fine_grained(args: argparse.Namespace, cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, _ = load_cifar10(cfg.data_root, batch_size=cfg.cka_batch_size, shuffle_train=False)

    model_a = load_model_from_checkpoint(args.ckpt_a, cfg.num_classes, device)
    model_b = load_model_from_checkpoint(args.ckpt_b, cfg.num_classes, device)

    layers = resolve_fine_grained_layers(model_a, args.layer_types)
    logger.info(
        "Computing fine-grained (checkerboard) CKA: %s vs %s (%d layers, types=%s)",
        args.label_a, args.label_b, len(layers), args.layer_types,
    )
    accum = compute_cka_matrix(
        model_a, model_b, layers, test_loader, device,
        num_batches=cfg.cka_num_batches, kernel_type=cfg.cka_kernel,
    )

    num_layers = len(layers)
    matrix_scores = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            matrix_scores[i, j] = np.mean(accum[(i, j)])

    ts = _timestamp()
    out_prefix = Path(cfg.results_path()) / f"cka_fine_grained_{args.label_a}_vs_{args.label_b}_{ts}"
    plot_fine_grained_matrix(matrix_scores, layers, args.label_a, args.label_b,
                              kernel_type=cfg.cka_kernel, save_path=out_prefix.with_suffix(".png"))
    np.save(out_prefix.with_suffix(".npy"), matrix_scores)
    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump({"layers": layers, "layer_types": args.layer_types}, f, indent=2)

    logger.info("Results saved: %s.{png,npy,json}", out_prefix)


LAYER_TYPE_MAP = {
    "conv": torch.nn.Conv2d,
    "linear": torch.nn.Linear,
    "groupnorm": torch.nn.GroupNorm,
}


def resolve_fine_grained_layers(model: torch.nn.Module, layer_types: str) -> list[str]:
    """Resolve the --layer-types CLI string (e.g. "conv,linear") into an
    actual list of layer names, by introspecting the model."""
    if layer_types == "all":
        return list_learnable_layers(model)
    types = tuple(LAYER_TYPE_MAP[t.strip()] for t in layer_types.split(",") if t.strip())
    return list_learnable_layers(model, include_types=types)


def cmd_replot(args: argparse.Namespace, cfg) -> None:
    """Reload saved .npy + .json and regenerate all plots without recomputing CKA.

    The .npy file must contain the full (n_pairs, n_layers, n_layers) array
    saved by multi-seed (not just the mean matrix). If you ran multi-seed
    before this format was introduced, re-run multi-seed once to regenerate.
    """
    json_path = Path(args.json)
    npy_path = Path(args.npy) if args.npy else json_path.with_suffix(".npy")

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"npy file not found: {npy_path}")

    with open(json_path) as f:
        meta = json.load(f)

    all_pair_matrices_raw = np.load(npy_path)
    layers = meta["layers"]
    label = args.label or meta.get("label", "replot")
    n_pairs = meta.get("n_pairs", 1)

    # Handle both old format (mean_matrix only, shape n_layers x n_layers)
    # and new format (all pairs, shape n_pairs x n_layers x n_layers)
    if all_pair_matrices_raw.ndim == 3:
        all_pair_matrices = list(all_pair_matrices_raw)
    else:
        logger.warning(
            "npy file contains only the mean matrix (old format). "
            "Diagonal stats and std heatmap cannot be regenerated exactly. "
            "Re-run multi-seed to save all pair matrices and enable full replotting."
        )
        all_pair_matrices = [all_pair_matrices_raw]
        n_pairs = 1  # treat as single-pair run

    mean_matrix = np.mean(all_pair_matrices, axis=0)

    out_dir = Path(args.out_dir) if args.out_dir else json_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    out_prefix = out_dir / f"cka_replot_{label}_{ts}"

    logger.info("Replotting '%s': %d pairs, %d layers -> %s", label, n_pairs, len(layers), out_prefix.parent)

    plot_mean_heatmap(
        mean_matrix, layers,
        title=f"Mean CKA — {label} (n={n_pairs} pair{'s' if n_pairs != 1 else ''})",
        annotate=(len(layers) <= 15),
        save_path=out_prefix.with_suffix(".png"),
    )

    if n_pairs > 1:
        std_matrix = np.std(all_pair_matrices, axis=0)
        plot_std_heatmap(
            std_matrix, layers,
            title=f"CKA std. dev. across pairs — {label}",
            save_path=out_prefix.parent / f"{out_prefix.name}_std.png",
        )
        plot_diagonal_stats(
            all_pair_matrices, layers,
            title=f"Diagonal CKA across pairs — {label}",
            save_path=out_prefix.parent / f"{out_prefix.name}_diagonal_stats.png",
        )

    logger.info("Replot done: %s.*", out_prefix)


def _make_series_label(meta: dict) -> str:
    """Build a readable legend label from checkpoint metadata."""
    run_type = meta.get("run_type", "unknown")
    seed = meta.get("seed", "?")
    epoch = meta.get("epoch", "?")
    eps = meta.get("epsilon", "")
    C = meta.get("max_grad_norm", "")
    lr = meta.get("lr", "")

    parts = [run_type, f"seed{seed}", f"ep{epoch}"]
    if eps not in ("", "NA", None):
        parts.append(f"ε={eps:.4g}" if isinstance(eps, float) else f"ε={eps}")
    if C not in ("", None):
        parts.append(f"C={C}")
    if lr not in ("", None):
        parts.append(f"lr={lr}")
    return " | ".join(parts)


def _make_output_name(json_paths: list[str]) -> str:
    """Build a filename encoding the key parameters of all checkpoints."""
    tags = []
    for p in json_paths:
        with open(p) as f:
            meta = json.load(f)
        run_type = meta.get("run_type", "unk")
        seed = meta.get("seed", "?")
        epoch = meta.get("epoch", "?")
        eps = meta.get("epsilon", "")
        C = meta.get("max_grad_norm", "")
        tag = f"{run_type}_s{seed}_ep{epoch}"
        if eps not in ("", "NA", None):
            tag += f"_eps{eps:.4g}" if isinstance(eps, float) else f"_eps{eps}"
        if C not in ("", None):
            tag += f"_C{C}"
        tags.append(tag)
    return "__vs__".join(tags)


def cmd_plot_accuracy(args: argparse.Namespace, cfg) -> None:
    """Load checkpoint JSONs and plot convergence curves for all of them."""
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:olive"]

    series = []
    for idx, json_path in enumerate(args.jsons):
        with open(json_path) as f:
            meta = json.load(f)

        label = _make_series_label(meta)
        series.append({
            "train_acc_history": meta["train_acc_history"],
            "test_acc_history": meta["test_acc_history"],
            "label": label,
            "color": colors[idx % len(colors)],
        })
        logger.info(
            "Loaded: %s | final test acc=%.1f%%",
            Path(json_path).name,
            meta["test_acc_history"][-1] * 100,
        )

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.networks_path())
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    param_tag = _make_output_name(args.jsons)
    # Truncate if too long for a filename
    if len(param_tag) > 120:
        param_tag = param_tag[:120] + "_etc"
    filename = f"accuracy_{param_tag}_{ts}.png"
    save_path = out_dir / filename

    title = args.title or f"Accuracy convergence ({len(series)} checkpoint{'s' if len(series) != 1 else ''})"
    plot_accuracy_curves(series, title=title, save_path=save_path)

    # Save a JSON summary alongside the figure
    summary_path = save_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump({
            "title": title,
            "source_jsons": args.jsons,
            "final_test_accs": {
                s["label"]: s["test_acc_history"][-1] * 100
                for s in series
            },
        }, f, indent=2)

    logger.info("Accuracy plot saved: %s", save_path)
    logger.info("Summary JSON saved: %s", summary_path)


def cmd_zscore(args: argparse.Namespace, cfg) -> None:
    """Compare a target CKA matrix against a reference (baseline vs baseline)
    using a z-score, cell by cell — including off-diagonal blocks."""

    # Load target .npy (e.g. DP vs baseline, shape n_pairs x n_layers x n_layers or n_layers x n_layers)
    target_raw = np.load(args.target_npy)
    ref_raw = np.load(args.ref_npy)

    # Accept both old format (2D mean matrix) and new format (3D all-pairs)
    mean_target = target_raw.mean(axis=0) if target_raw.ndim == 3 else target_raw
    mean_ref = ref_raw.mean(axis=0) if ref_raw.ndim == 3 else ref_raw
    std_ref = ref_raw.std(axis=0) if ref_raw.ndim == 3 else np.zeros_like(ref_raw)

    if ref_raw.ndim != 3:
        logger.warning(
            "Reference .npy is 2D (old format — only mean matrix, no std). "
            "Z-scores will use std_floor=%.3f everywhere. "
            "Re-run multi-seed on baseline-vs-baseline with the current code to get proper per-cell std.",
            args.std_floor,
        )

    # Load layers list from JSON sidecar (preferred) or from the target JSON
    if args.target_json:
        with open(args.target_json) as f:
            layers = json.load(f)["layers"]
    elif args.ref_json:
        with open(args.ref_json) as f:
            layers = json.load(f)["layers"]
    else:
        raise ValueError("Provide --target-json or --ref-json to read the layer list.")

    # Verify dimensions match
    if mean_target.shape != mean_ref.shape:
        raise ValueError(
            f"Shape mismatch: target {mean_target.shape} vs ref {mean_ref.shape}. "
            "Both .npy files must have been computed with the same layer list."
        )

    label = args.label or "zscore"
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.target_npy).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()
    out_prefix = out_dir / f"cka_zscore_{label}_{ts}"

    logger.info(
        "Z-score: target=%s | ref=%s | %d layers | std_floor=%.3f",
        Path(args.target_npy).name, Path(args.ref_npy).name, len(layers), args.std_floor,
    )

    zscore = plot_zscore_heatmap(
        mean_matrix_target=mean_target,
        mean_matrix_ref=mean_ref,
        std_matrix_ref=std_ref,
        layers=layers,
        title=f"CKA z-score — {label}",
        ylabel=args.label_a or "Target (rows)",
        xlabel=args.label_b or "Reference (columns)",
        std_floor=args.std_floor,
        save_path=out_prefix.with_suffix(".png"),
    )

    np.save(out_prefix.with_suffix(".npy"), zscore)
    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump({
            "label": label,
            "label_a": args.label_a,
            "label_b": args.label_b,
            "target_npy": str(args.target_npy),
            "ref_npy": str(args.ref_npy),
            "layers": layers,
            "std_floor": args.std_floor,
            "zscore_matrix": zscore.tolist(),
        }, f, indent=2)

    logger.info("Results saved: %s.*", out_prefix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CKA analysis between ResNet18 checkpoints (baseline vs DP)")
    add_config_overrides_args(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_compare = subparsers.add_parser("compare", help="CKA between two checkpoints (layer1-4)")
    p_compare.add_argument("--ckpt-a", required=True)
    p_compare.add_argument("--ckpt-b", required=True)
    p_compare.add_argument("--label-a", default="ModelA")
    p_compare.add_argument("--label-b", default="ModelB")
    p_compare.set_defaults(func=cmd_compare)

    p_multi = subparsers.add_parser("multi-eps", help="CKA between baseline and several DP checkpoints (epsilon sweep)")
    p_multi.add_argument("--baseline-ckpt", required=True)
    p_multi.add_argument("--dp-ckpts", nargs="+", required=True)
    p_multi.add_argument("--epsilons", nargs="+", type=float, required=True)
    p_multi.set_defaults(func=cmd_multi_eps)

    p_fine = subparsers.add_parser("fine-grained", help="Fine-grained CKA (all learnable layers, or a filtered subset)")
    p_fine.add_argument("--ckpt-a", required=True)
    p_fine.add_argument("--ckpt-b", required=True)
    p_fine.add_argument("--label-a", default="ModelA")
    p_fine.add_argument("--label-b", default="ModelB")
    p_fine.add_argument(
        "--layer-types", default="conv,linear",
        help="Comma-separated subset of {conv,linear,groupnorm}, or 'all' for every learnable layer "
             "(conv1/conv2/downsample, GroupNorm affine params, fc). Default: conv,linear (no normalization layers).",
    )
    p_fine.set_defaults(func=cmd_fine_grained)

    p_seed = subparsers.add_parser(
        "multi-seed",
        help="CKA heatmap (+ stats if 2+ pairs) across checkpoints: one group (all pairs) or two matched groups",
    )
    p_seed.add_argument(
        "--ckpts", nargs="+", default=None,
        help="One group of 2+ checkpoints (e.g. baselines with different seeds) — ALL pairs are compared.",
    )
    p_seed.add_argument(
        "--ckpts-a", nargs="+", default=None,
        help="First group for two-group mode (e.g. DP checkpoints, one per seed). Use with --ckpts-b.",
    )
    p_seed.add_argument(
        "--ckpts-b", nargs="+", default=None,
        help="Second group for two-group mode (e.g. baseline checkpoints, one per seed, same order/seeds as --ckpts-a).",
    )
    p_seed.add_argument("--label", default="multiseed", help="Label used in plot titles and output filenames")
    p_seed.add_argument("--label-a", default=None, help="Label for group A (--ckpts-a) — shown on Y axis (rows)")
    p_seed.add_argument("--label-b", default=None, help="Label for group B (--ckpts-b) — shown on X axis (columns)")
    p_seed.add_argument(
        "--layers-override", dest="layers_override", default=None,
        help="Comma-separated list of layer names for this specific run, e.g. "
             "'conv1,layer1.0.conv1,fc'. Takes priority over --cka-layers / the YAML config. "
             "Tip: get the full list with: python -c \"from model import build_resnet18_dp_compatible, "
             "list_learnable_layers as L; print(','.join(L(build_resnet18_dp_compatible(10))))\"",
    )
    p_seed.set_defaults(func=cmd_multi_seed)

    p_replot = subparsers.add_parser(
        "replot",
        help="Regenerate plots from a previously saved .json + .npy, without recomputing CKA",
    )
    p_replot.add_argument(
        "--json", required=True,
        help="Path to the .json file from a previous multi-seed run (e.g. results/.../cka_multiseed_...json)",
    )
    p_replot.add_argument(
        "--npy", default=None,
        help="Path to the .npy file (mean_matrix). Defaults to the .json path with .npy extension.",
    )
    p_replot.add_argument(
        "--label", default=None,
        help="Override the label shown in plot titles (defaults to the label stored in the JSON).",
    )
    p_replot.add_argument(
        "--out-dir", default=None,
        help="Directory to write the new plots into (defaults to the same folder as the .json).",
    )
    p_replot.set_defaults(func=cmd_replot)

    p_z = subparsers.add_parser(
        "zscore",
        help="3-panel plot: target CKA | reference mean | z-score (target vs reference, cell by cell)",
    )
    p_z.add_argument(
        "--target-npy", required=True,
        help="Path to the .npy from a multi-seed run for the comparison of interest "
             "(e.g. DP vs baseline). Shape: (n_pairs, n_layers, n_layers) or (n_layers, n_layers).",
    )
    p_z.add_argument(
        "--ref-npy", required=True,
        help="Path to the .npy from the baseline-vs-baseline multi-seed run (the reference / noise floor). "
             "Must have the same layer list as --target-npy.",
    )
    p_z.add_argument("--target-json", default=None, help="JSON sidecar of --target-npy (to read the layer list).")
    p_z.add_argument("--ref-json", default=None, help="JSON sidecar of --ref-npy (fallback for layer list).")
    p_z.add_argument("--label", default=None, help="Label used in the plot title and output filename.")
    p_z.add_argument("--label-a", default=None, help="Y-axis label (rows = target group A, e.g. 'DP').")
    p_z.add_argument("--label-b", default=None, help="X-axis label (columns = reference group B, e.g. 'Baseline').")
    p_z.add_argument(
        "--std-floor", type=float, default=0.01,
        help="Minimum std per cell before dividing (avoids inflating z-scores in cells where the "
             "reference is nearly constant). Default: 0.01.",
    )
    p_z.add_argument("--out-dir", default=None, help="Output directory (default: same folder as --target-npy).")
    p_z.set_defaults(func=cmd_zscore)

    p_acc = subparsers.add_parser(
        "plot-accuracy",
        help="Plot accuracy convergence curves from one or several checkpoint JSON files",
    )
    p_acc.add_argument(
        "--jsons", nargs="+", required=True,
        help="One or more checkpoint .json files (the metadata files saved alongside .pth). "
             "Each becomes one line in the plot. Order determines color assignment.",
    )
    p_acc.add_argument(
        "--title", default=None,
        help="Custom figure title (default: auto-generated from number of checkpoints).",
    )
    p_acc.add_argument(
        "--out-dir", default=None,
        help="Output directory (default: results/<experiment>/ from config).",
    )
    p_acc.set_defaults(func=cmd_plot_accuracy)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    setup_logging(cfg.logs_dir, f"cka_{args.command}_{_timestamp()}")
    logger.info("=== CKA: subcommand '%s' (experiment: %s) ===", args.command, cfg.experiment)

    Path(cfg.results_path()).mkdir(parents=True, exist_ok=True)
    args.func(args, cfg)


if __name__ == "__main__":
    main()