#!/usr/bin/env python
"""
train_baseline.py
==================
Trains the "DP-compatible" ResNet18 model WITHOUT differential privacy
(standard optimization), and saves a uniquely named checkpoint (with a
timestamp).

Examples:
    python train_baseline.py --config configs/default.yaml
    python train_baseline.py --epochs 5 --seed 43 --lr 5e-4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import add_config_overrides_args, apply_overrides, load_config
from checkpoint import get_checkpoint_path, make_run_id, save_checkpoint
from data import download_cifar10, load_cifar10, set_seed
from logging_utils import setup_logging
from model import build_resnet18_dp_compatible
from training import evaluate, train_one_epoch_baseline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline training (no DP)")
    add_config_overrides_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    run_id = make_run_id("baseline_resnet18", "NA", cfg.delta, cfg.epochs, cfg.max_grad_norm, cfg.seed)
    log_path = setup_logging(cfg.logs_dir, run_id)
    logger.info("=== Baseline run: %s ===", run_id)
    logger.info("Experiment: %s | Full log at: %s", cfg.experiment, log_path)

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    download_cifar10(cfg.data_root, cfg.cifar10_url)
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10(
        cfg.data_root, batch_size=cfg.batch_size
    )

    model = build_resnet18_dp_compatible(num_classes=cfg.num_classes).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=cfg.lr)

    train_acc_history: list[float] = []
    test_acc_history: list[float] = []

    training_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()

        train_acc = train_one_epoch_baseline(model, train_loader, optimizer, epoch + 1, device)
        train_acc_history.append(train_acc)

        test_acc = evaluate(model, test_loader, device, prefix="Baseline test")
        test_acc_history.append(test_acc)

        epoch_duration = time.perf_counter() - epoch_start
        logger.info("Epoch %d completed in %.1fs", epoch + 1, epoch_duration)

    total_duration = time.perf_counter() - training_start
    logger.info(
        "Total training time: %.1fs (%.2f min) for %d epoch(s)",
        total_duration, total_duration / 60, cfg.epochs,
    )

    save_path = get_checkpoint_path(
        prefix="baseline_resnet18",
        epsilon="NA",
        delta=cfg.delta,
        epochs=cfg.epochs,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
        save_dir=cfg.networks_path(),
    )

    save_checkpoint(
        save_path,
        payload={
            "epoch": cfg.epochs,
            "model_state_dict": model.state_dict(),
            "train_acc_history": train_acc_history,
            "test_acc_history": test_acc_history,
            "seed": cfg.seed,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "training_duration_seconds": total_duration,
        },
        extra_metadata={"run_type": "baseline", "run_id": run_id, "experiment": cfg.experiment},
    )

    logger.info("Baseline training finished. Checkpoint: %s", save_path)


if __name__ == "__main__":
    main()