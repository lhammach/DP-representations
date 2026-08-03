#!/usr/bin/env python
"""
train_baseline.py
==================
Trains the "DP-compatible" ResNet18 WITHOUT differential privacy
and saves checkpoints at regular intervals.

Examples:
    python train_baseline.py --config configs/default.yaml
    python train_baseline.py --epochs 100 --seed 1 --optimizer sgd --lr 0.1 \
        --cifar-stem True --lr-scheduler cosine --checkpoint-every 10
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import add_config_overrides_args, apply_overrides, load_config
from checkpoint import (
    get_checkpoint_path, make_run_id, save_checkpoint, save_intermediate_checkpoint,
)
from data import download_cifar10, load_cifar10, set_seed
from logging_utils import setup_logging
from model import build_model
from training import build_optimizer, build_scheduler, evaluate, train_one_epoch_baseline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline training (no DP)")
    add_config_overrides_args(parser)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    model_prefix = f"baseline_{cfg.model.replace('-', '').replace('_', '')}"

    run_id = make_run_id(
        model_prefix, "NA", cfg.delta, cfg.epochs, cfg.max_grad_norm, cfg.seed,
        lr=cfg.lr, optimizer=cfg.optimizer, cifar_stem=cfg.cifar_stem,
        lr_scheduler=cfg.lr_scheduler, batch_size=cfg.batch_size, model_name=cfg.model,
    )
    log_path = setup_logging(cfg.logs_dir, run_id)
    logger.info("=== Baseline run: %s ===", run_id)
    logger.info("Experiment: %s | Full log at: %s", cfg.experiment, log_path)
    if cfg.cifar_stem:
        logger.info("Architecture: CIFAR-10 stem (3x3 conv, no MaxPool)")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    download_cifar10(cfg.data_root, cfg.cifar10_url)
    train_loader, test_loader, _, _ = load_cifar10(cfg.data_root, batch_size=cfg.batch_size)

    model = build_model(cfg.model, num_classes=cfg.num_classes,
                        cifar_stem=cfg.cifar_stem).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.momentum,
                                weight_decay=cfg.weight_decay)
    scheduler = build_scheduler(optimizer, cfg.lr_scheduler, cfg.epochs, cfg.lr_min)

    checkpoint_every = cfg.checkpoint_every  # 0 = disabled, N = every N epochs

    save_path = get_checkpoint_path(
        prefix=model_prefix, epsilon="NA", delta=cfg.delta,
        epochs=cfg.epochs, max_grad_norm=cfg.max_grad_norm, seed=cfg.seed,
        lr=cfg.lr, optimizer=cfg.optimizer, cifar_stem=cfg.cifar_stem,
        lr_scheduler=cfg.lr_scheduler, batch_size=cfg.batch_size,
        model_name=cfg.model, save_dir=cfg.networks_path(),
    )

    train_acc_history: list[float] = []
    train_loss_history: list[float] = []
    test_acc_history: list[float] = []
    lr_history: list[float] = []
    training_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        train_acc, train_loss = train_one_epoch_baseline(
            model, train_loader, optimizer, epoch + 1, device)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        test_acc = evaluate(model, test_loader, device, prefix="Baseline test")
        test_acc_history.append(test_acc)

        if scheduler is not None:
            scheduler.step()

        logger.info("Epoch %d completed in %.1fs | lr=%.2e",
                    epoch + 1, time.perf_counter() - epoch_start, current_lr)

        # Intermediate checkpoints
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0 and (epoch + 1) < cfg.epochs:
            inter_path = save_intermediate_checkpoint(
                base_path=save_path, epoch=epoch + 1,
                model_state_dict=model.state_dict(),
                payload={
                    "train_acc_history": train_acc_history,
                    "train_loss_history": train_loss_history,
                    "test_acc_history": test_acc_history,
                    "lr_history": lr_history,
                    "seed": cfg.seed, "lr": cfg.lr,
                    "optimizer": cfg.optimizer, "cifar_stem": cfg.cifar_stem,
                    "batch_size": cfg.batch_size,
                },
                milestones=tuple(range(checkpoint_every, cfg.epochs, checkpoint_every)),
            )
            if inter_path:
                logger.info("Intermediate checkpoint saved at epoch %d: %s", epoch + 1, inter_path)

    total_duration = time.perf_counter() - training_start
    logger.info("Total training time: %.1fs (%.2f min) for %d epoch(s)",
                total_duration, total_duration / 60, cfg.epochs)

    save_checkpoint(
        save_path,
        payload={
            "epoch": cfg.epochs,
            "model_state_dict": model.state_dict(),
            "train_acc_history": train_acc_history,
            "train_loss_history": train_loss_history,
            "test_acc_history": test_acc_history,
            "lr_history": lr_history,
            "lr_scheduler": cfg.lr_scheduler,
            "lr_min": cfg.lr_min,
            "seed": cfg.seed,
            "lr": cfg.lr,
            "optimizer": cfg.optimizer,
            "momentum": cfg.momentum,
            "cifar_stem": cfg.cifar_stem,
            "model": cfg.model,
            "batch_size": cfg.batch_size,
            "training_duration_seconds": total_duration,
        },
        extra_metadata={"run_type": "baseline", "run_id": run_id,
                        "model": cfg.model, "experiment": cfg.experiment},
    )
    logger.info("Baseline training finished. Checkpoint: %s", save_path)


if __name__ == "__main__":
    main()