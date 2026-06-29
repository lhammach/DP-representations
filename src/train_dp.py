#!/usr/bin/env python
"""
train_dp.py
===========
Trains the ResNet18 model with DP-SGD (Opacus), for a given privacy budget
(epsilon, delta), and saves a uniquely named checkpoint (with a timestamp).

Examples:
    python train_dp.py --config configs/default.yaml
    python train_dp.py --epsilon 2 --seed 43
    python train_dp.py --epsilon 8 --max-grad-norm 1.0
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
from training import apply_fsdp_compat_patch, evaluate, make_private, train_one_epoch_dp, unwrap_state_dict

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DP-SGD training (Opacus)")
    add_config_overrides_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    run_id = make_run_id("dp_resnet18", cfg.epsilon, cfg.delta, cfg.epochs, cfg.max_grad_norm, cfg.seed)
    log_path = setup_logging(cfg.logs_dir, run_id)
    logger.info("=== DP-SGD run: %s ===", run_id)
    logger.info("Full log at: %s", log_path)
    logger.info("Target privacy budget: epsilon=%s, delta=%s", cfg.epsilon, cfg.delta)

    set_seed(cfg.seed)
    apply_fsdp_compat_patch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    download_cifar10(cfg.data_root, cfg.cifar10_url)
    train_loader, test_loader, train_dataset, test_dataset = load_cifar10(
        cfg.data_root, batch_size=cfg.batch_size
    )

    model = build_resnet18_dp_compatible(num_classes=cfg.num_classes).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=cfg.lr)

    dp_model, dp_optimizer, dp_train_loader, privacy_engine = make_private(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        epochs=cfg.epochs,
        target_epsilon=cfg.epsilon,
        target_delta=cfg.delta,
        max_grad_norm=cfg.max_grad_norm,
        accountant=cfg.accountant,
    )

    train_acc_history: list[float] = []
    test_acc_history: list[float] = []
    current_epsilon = 0.0

    training_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()

        train_acc, current_epsilon = train_one_epoch_dp(
            dp_model, dp_train_loader, dp_optimizer, epoch + 1, device,
            privacy_engine, cfg.max_physical_batch_size, cfg.delta,
        )
        train_acc_history.append(train_acc)

        test_acc = evaluate(dp_model, test_loader, device, prefix="DP test")
        test_acc_history.append(test_acc)

        epoch_duration = time.perf_counter() - epoch_start
        logger.info("Epoch %d completed in %.1fs", epoch + 1, epoch_duration)

    total_duration = time.perf_counter() - training_start
    logger.info(
        "Total training time: %.1fs (%.2f min) for %d epoch(s)",
        total_duration, total_duration / 60, cfg.epochs,
    )

    save_path = get_checkpoint_path(
        prefix="dp_resnet18",
        epsilon=cfg.epsilon,
        delta=cfg.delta,
        epochs=cfg.epochs,
        max_grad_norm=cfg.max_grad_norm,
        seed=cfg.seed,
        save_dir=cfg.networks_dir,
    )

    save_checkpoint(
        save_path,
        payload={
            "epoch": cfg.epochs,
            "model_state_dict": unwrap_state_dict(dp_model),
            "train_acc_history": train_acc_history,
            "test_acc_history": test_acc_history,
            "noise_multiplier": dp_optimizer.noise_multiplier,
            "max_grad_norm": cfg.max_grad_norm,
            "epsilon": current_epsilon,
            "delta": cfg.delta,
            "seed": cfg.seed,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "training_duration_seconds": total_duration,
        },
        extra_metadata={"run_type": "dp", "run_id": run_id, "target_epsilon": cfg.epsilon},
    )

    logger.info("DP training finished (final epsilon = %.2f). Checkpoint: %s", current_epsilon, save_path)


if __name__ == "__main__":
    main()
