#!/usr/bin/env python
"""
train_dp.py
===========
Trains the ResNet18 model with DP-SGD (Opacus), for a given privacy budget
(epsilon, delta), and saves a uniquely named checkpoint (with a timestamp).

Ablation flags:
  --no-clip   : disables gradient clipping (sets max_grad_norm=1e6). Keeps noise.
  --no-noise  : disables Gaussian noise (sets noise_multiplier=0). Keeps clipping.
                With --no-noise, epsilon=inf (no privacy). The noise multiplier
                is computed from target epsilon only in the standard case (neither
                flag set). With either ablation flag, make_private_ablation is
                used and the epsilon column in the checkpoint is set to 'inf'.

Examples:
    python train_dp.py --config configs/default.yaml
    python train_dp.py --epsilon 2 --seed 43
    python train_dp.py --optimizer sgd --lr 0.1 --cifar-stem True --seed 1
    python train_dp.py --no-clip --seed 1 --experiment ablation
    python train_dp.py --no-noise --seed 1 --experiment ablation
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
from model import build_model
from training import (
    apply_fsdp_compat_patch,
    build_optimizer,
    build_scheduler,
    evaluate,
    make_private,
    make_private_ablation,
    train_one_epoch_dp,
    unwrap_state_dict,
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="DP-SGD training (Opacus)")
    add_config_overrides_args(parser)

    # Ablation flags — added directly to this parser, not through Config,
    # because they are experiment-specific overrides, not stable hyperparameters.
    ablation = parser.add_argument_group("ablation (mutually exclusive with standard DP)")
    ablation.add_argument(
        "--no-clip", action="store_true",
        help="Disable gradient clipping (max_grad_norm=1e6). Noise is preserved. "
             "Isolates the effect of noise alone on representations.",
    )
    ablation.add_argument(
        "--no-noise", action="store_true",
        help="Disable Gaussian noise (noise_multiplier=0). Clipping is preserved. "
             "Isolates the effect of clipping alone on representations. epsilon=inf.",
    )

    args = parser.parse_args()

    if args.no_clip and args.no_noise:
        parser.error("--no-clip and --no-noise cannot be used together (that would be plain SGD, not a DP ablation).")

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    ablation_tag = "_noclip" if args.no_clip else "_nonoise" if args.no_noise else ""
    model_prefix = f"dp_{cfg.model.replace('-', '').replace('_', '')}"
    prefix = f"{model_prefix}{ablation_tag}"
    epsilon_label = "inf" if args.no_noise else cfg.epsilon

    run_id = make_run_id(
        prefix, epsilon_label, cfg.delta, cfg.epochs, cfg.max_grad_norm, cfg.seed,
        lr=cfg.lr, optimizer=cfg.optimizer, cifar_stem=cfg.cifar_stem,
        lr_scheduler=cfg.lr_scheduler, batch_size=cfg.batch_size, model_name=cfg.model,
    )
    log_path = setup_logging(cfg.logs_dir, run_id)
    logger.info("=== DP-SGD run: %s ===", run_id)
    logger.info("Experiment: %s | Full log at: %s", cfg.experiment, log_path)

    if args.no_clip:
        logger.info("ABLATION: gradient clipping DISABLED (max_grad_norm=1e6). Noise is active.")
    elif args.no_noise:
        logger.info("ABLATION: Gaussian noise DISABLED (noise_multiplier=0). Clipping is active. epsilon=inf.")
    else:
        logger.info("Target privacy budget: epsilon=%s, delta=%s", cfg.epsilon, cfg.delta)

    if cfg.cifar_stem:
        logger.info("Architecture: CIFAR-10 stem (3x3 conv, no MaxPool)")

    set_seed(cfg.seed)
    apply_fsdp_compat_patch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    download_cifar10(cfg.data_root, cfg.cifar10_url)
    train_loader, test_loader, _, _ = load_cifar10(cfg.data_root, batch_size=cfg.batch_size)

    model = build_model(cfg.model, num_classes=cfg.num_classes,
                        cifar_stem=cfg.cifar_stem).to(device)
    optimizer = build_optimizer(model, cfg.optimizer, cfg.lr, cfg.momentum)

    if args.no_clip or args.no_noise:
        if args.no_clip:
            # Strategy: use make_private_ablation with:
            #   - noise_multiplier = sigma calibrated for (epsilon, C) via a temp model
            #   - max_grad_norm = C (the real C, so noise = sigma * C, identical to DP)
            # Then disable clipping by monkey-patching clip_and_accumulate to skip
            # the clipping step (multiply each grad_sample by 1.0 instead of
            # min(1, C/||g||)).
            # This gives: g_update = mean(g^(i)) + N(0, sigma^2 * C^2 * I)
            # with no clipping applied, and noise identical to full DP.
            from opacus import PrivacyEngine as _PE
            _tmp = build_model(cfg.model, num_classes=cfg.num_classes,
                               cifar_stem=cfg.cifar_stem).to(device)
            _tmp_opt = build_optimizer(_tmp, cfg.optimizer, cfg.lr, cfg.momentum)
            _pe = _PE(accountant=cfg.accountant)
            _, _wrapped_opt, _ = _pe.make_private_with_epsilon(
                module=_tmp,
                optimizer=_tmp_opt,
                data_loader=train_loader,
                epochs=cfg.epochs,
                target_epsilon=cfg.epsilon,
                target_delta=cfg.delta,
                max_grad_norm=cfg.max_grad_norm,
            )
            sigma = _wrapped_opt.noise_multiplier
            del _tmp, _tmp_opt, _pe, _wrapped_opt

            dp_model, dp_optimizer, dp_train_loader, privacy_engine = make_private_ablation(
                model=model, optimizer=optimizer, train_loader=train_loader,
                noise_multiplier=sigma, max_grad_norm=cfg.max_grad_norm,
                accountant=cfg.accountant,
            )
            # Disable clipping: override clip_and_accumulate to skip the clipping step.
            # Opacus calls this method to clip each grad_sample before accumulation.
            # By making it a no-op (scale factor = 1 for all examples), gradients
            # pass through unclipped. The noise added afterwards is still sigma*C.
            _orig_clip = dp_optimizer.clip_and_accumulate

            def _no_clip_and_accumulate(*a, **kw):
                # Accumulate without clipping: just sum grad_sample directly
                for p in (pp for pp in dp_model.parameters()
                          if hasattr(pp, 'grad_sample') and pp.grad_sample is not None):
                    if p.summed_grad is None:
                        p.summed_grad = p.grad_sample.sum(dim=0)
                    else:
                        p.summed_grad += p.grad_sample.sum(dim=0)

            dp_optimizer.clip_and_accumulate = _no_clip_and_accumulate
            logger.info(
                "ABLATION no-clip: sigma=%.4f, C=%.2f → effective noise std=%.4f. "
                "clip_and_accumulate patched to skip clipping.",
                sigma, cfg.max_grad_norm, sigma * cfg.max_grad_norm,
            )
        else:  # no_noise
            dp_model, dp_optimizer, dp_train_loader, privacy_engine = make_private_ablation(
                model=model, optimizer=optimizer, train_loader=train_loader,
                noise_multiplier=0.0, max_grad_norm=cfg.max_grad_norm,
                accountant=cfg.accountant,
            )
    else:
        dp_model, dp_optimizer, dp_train_loader, privacy_engine = make_private(
            model=model, optimizer=optimizer, train_loader=train_loader,
            epochs=cfg.epochs, target_epsilon=cfg.epsilon, target_delta=cfg.delta,
            max_grad_norm=cfg.max_grad_norm, accountant=cfg.accountant,
        )

    train_acc_history: list[float] = []
    test_acc_history: list[float] = []
    lr_history: list[float] = []
    current_epsilon: float | str = 0.0

    scheduler = build_scheduler(dp_optimizer, cfg.lr_scheduler, cfg.epochs, cfg.lr_min)
    # effective_C is the C used for noise calibration (always cfg.max_grad_norm now).
    # For no-clip, clipping is disabled post-hoc but C still determines the noise level.
    effective_C = cfg.max_grad_norm

    # Build save_path BEFORE the loop so intermediate checkpoints can reference it
    save_path = get_checkpoint_path(
        prefix=prefix, epsilon=epsilon_label, delta=cfg.delta,
        epochs=cfg.epochs, max_grad_norm=effective_C, seed=cfg.seed,
        lr=cfg.lr, optimizer=cfg.optimizer, cifar_stem=cfg.cifar_stem,
        lr_scheduler=cfg.lr_scheduler, batch_size=cfg.batch_size,
        model_name=cfg.model, save_dir=cfg.networks_path(),
    )

    training_start = time.perf_counter()

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        current_lr = dp_optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

        train_acc, current_epsilon = train_one_epoch_dp(
            dp_model, dp_train_loader, dp_optimizer, epoch + 1, device,
            privacy_engine, cfg.max_physical_batch_size, cfg.delta,
        )
        train_acc_history.append(train_acc)

        test_acc = evaluate(dp_model, test_loader, device, prefix=f"DP{ablation_tag} test")
        test_acc_history.append(test_acc)

        if scheduler is not None:
            scheduler.step()

        epoch_duration = time.perf_counter() - epoch_start
        logger.info("Epoch %d completed in %.1fs | lr=%.2e", epoch + 1, epoch_duration, current_lr)

        # Intermediate checkpoint every N epochs (if checkpoint_every > 0)
        if cfg.checkpoint_every > 0 and (epoch + 1) % cfg.checkpoint_every == 0 and (epoch + 1) < cfg.epochs:
            inter_path = save_path.parent / f"{save_path.stem}_ep{epoch + 1}{save_path.suffix}"
            save_checkpoint(
                inter_path,
                payload={
                    "epoch": epoch + 1,
                    "model_state_dict": unwrap_state_dict(dp_model),
                    "train_acc_history": train_acc_history,
                    "test_acc_history": test_acc_history,
                    "lr_history": lr_history,
                    "epsilon": current_epsilon,
                    "delta": cfg.delta,
                    "seed": cfg.seed,
                    "lr": cfg.lr,
                    "optimizer": cfg.optimizer,
                    "cifar_stem": cfg.cifar_stem,
                    "batch_size": cfg.batch_size,
                    "noise_multiplier": dp_optimizer.noise_multiplier,
                    "max_grad_norm": effective_C,
                },
                extra_metadata={"intermediate_checkpoint": True, "saved_at_epoch": epoch + 1},
            )
            logger.info("Intermediate checkpoint saved at epoch %d: %s", epoch + 1, inter_path)

    total_duration = time.perf_counter() - training_start
    logger.info(
        "Total training time: %.1fs (%.2f min) for %d epoch(s)",
        total_duration, total_duration / 60, cfg.epochs,
    )

    if args.no_noise:
        current_epsilon = "inf"

    save_checkpoint(
        save_path,
        payload={
            "epoch": cfg.epochs,
            "model_state_dict": unwrap_state_dict(dp_model),
            "train_acc_history": train_acc_history,
            "test_acc_history": test_acc_history,
            "lr_history": lr_history,
            "lr_scheduler": cfg.lr_scheduler,
            "lr_min": cfg.lr_min,
            "noise_multiplier": dp_optimizer.noise_multiplier,
            "max_grad_norm": effective_C,
            "epsilon": current_epsilon,
            "delta": cfg.delta,
            "seed": cfg.seed,
            "lr": cfg.lr,
            "optimizer": cfg.optimizer,
            "momentum": cfg.momentum,
            "cifar_stem": cfg.cifar_stem,
            "model": cfg.model,
            "batch_size": cfg.batch_size,
            "training_duration_seconds": total_duration,
            "ablation_no_clip": args.no_clip,
            "ablation_no_noise": args.no_noise,
        },
        extra_metadata={
            "run_type": f"dp{ablation_tag}",
            "run_id": run_id,
            "target_epsilon": str(epsilon_label),
            "experiment": cfg.experiment,
        },
    )

    logger.info(
        "DP%s training finished (epsilon=%s). Checkpoint: %s",
        ablation_tag, current_epsilon, save_path,
    )


if __name__ == "__main__":
    main()