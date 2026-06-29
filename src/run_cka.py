#!/usr/bin/env python
"""
run_cka.py
==========
CKA analysis between trained checkpoints. Several subcommands covering the
comparisons from the original notebook:

    compare      : CKA between any two checkpoints (high-level layers)
    multi-eps    : CKA between a baseline and several DP checkpoints (epsilon sweep)
    fine-grained : fine-grained CKA (per-block conv1/conv2 sub-layers)

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
"""

from __future__ import annotations

import argparse
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
from model import build_resnet18_dp_compatible
from visualization import plot_cka_results, plot_epsilon_layer_heatmap, plot_fine_grained_matrix

logger = logging.getLogger(__name__)

DAMIER_LAYERS = [
    "conv1",
    "layer1.0.conv1", "layer1.0.conv2", "layer1.1.conv1", "layer1.1.conv2",
    "layer2.0.conv1", "layer2.0.conv2", "layer2.1.conv1", "layer2.1.conv2",
    "layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2",
    "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2",
    "fc",
]


def load_model_from_checkpoint(ckpt_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """Rebuild the DP-compatible architecture and load the checkpoint's weights.

    Uses the same `build_resnet18_dp_compatible` function as for training
    (baseline or DP): the architecture is identical in both cases
    (GroupNorm, non-inplace ReLU), only the optimization procedure differed.
    This is what allows any checkpoint, baseline or DP, to be reloaded
    without a state_dict error.
    """
    model = build_resnet18_dp_compatible(num_classes=num_classes).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
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
    out_prefix = Path(cfg.results_dir) / f"cka_compare_{args.label_a}_vs_{args.label_b}_{ts}"
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
    out_prefix = Path(cfg.results_dir) / f"cka_multi_epsilon_{ts}"
    plot_epsilon_layer_heatmap(epsilon_layer_matrix, args.epsilons, cfg.cka_layers,
                                save_path=out_prefix.with_suffix(".png"))

    np.save(out_prefix.with_suffix(".npy"), epsilon_layer_matrix)
    with open(out_prefix.with_suffix(".json"), "w") as f:
        json.dump(
            {"epsilons": args.epsilons, "layers": cfg.cka_layers, "matrix": epsilon_layer_matrix.tolist()},
            f, indent=2,
        )

    logger.info("Results saved: %s.{png,npy,json}", out_prefix)


def cmd_fine_grained(args: argparse.Namespace, cfg) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader, _, _ = load_cifar10(cfg.data_root, batch_size=cfg.cka_batch_size, shuffle_train=False)

    model_a = load_model_from_checkpoint(args.ckpt_a, cfg.num_classes, device)
    model_b = load_model_from_checkpoint(args.ckpt_b, cfg.num_classes, device)

    logger.info("Computing fine-grained (checkerboard) CKA: %s vs %s", args.label_a, args.label_b)
    accum = compute_cka_matrix(
        model_a, model_b, DAMIER_LAYERS, test_loader, device,
        num_batches=cfg.cka_num_batches, kernel_type=cfg.cka_kernel,
    )

    num_layers = len(DAMIER_LAYERS)
    matrix_scores = np.zeros((num_layers, num_layers))
    for i in range(num_layers):
        for j in range(num_layers):
            matrix_scores[i, j] = np.mean(accum[(i, j)])

    ts = _timestamp()
    out_prefix = Path(cfg.results_dir) / f"cka_fine_grained_{args.label_a}_vs_{args.label_b}_{ts}"
    plot_fine_grained_matrix(matrix_scores, DAMIER_LAYERS, args.label_a, args.label_b,
                              kernel_type=cfg.cka_kernel, save_path=out_prefix.with_suffix(".png"))
    np.save(out_prefix.with_suffix(".npy"), matrix_scores)

    logger.info("Results saved: %s.{png,npy}", out_prefix)


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

    p_fine = subparsers.add_parser("fine-grained", help="Fine-grained CKA (conv1/conv2 sub-layers)")
    p_fine.add_argument("--ckpt-a", required=True)
    p_fine.add_argument("--ckpt-b", required=True)
    p_fine.add_argument("--label-a", default="ModelA")
    p_fine.add_argument("--label-b", default="ModelB")
    p_fine.set_defaults(func=cmd_fine_grained)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    setup_logging(cfg.logs_dir, f"cka_{args.command}_{_timestamp()}")
    logger.info("=== CKA: subcommand '%s' ===", args.command)

    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)
    args.func(args, cfg)


if __name__ == "__main__":
    main()