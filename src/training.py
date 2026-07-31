"""
training.py
===========
Training loops (baseline and DP-SGD via Opacus) and evaluation.

Gradient statistics are NOT computed here anymore. They are computed
a posteriori by compute_grad_stats.py, which loads a checkpoint and
runs a clean backward pass on the full training set — independently
of Opacus. This avoids all the complexity and memory overhead of
reading grad_sample during training (virtual steps, grad_sample
lifetime, OnlineGradStats RAM cost, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def build_optimizer(
    model: nn.Module,
    optimizer: str,
    lr: float,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """Build the optimizer from a name string.

    Supported values: "sgd", "adam", "rmsprop".

    Recommended starting points for DP-SGD on CIFAR-10:
    - SGD   : lr=0.1, momentum=0.9  (classic, theoretically well-understood with DP)
    - Adam  : lr=1e-3
    - RMSprop: lr=1e-3 (generally worse for DP due to adaptive stat corruption by noise)
    """
    name = optimizer.lower().strip()
    if name == "sgd":
        opt = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        logger.info("Optimizer: SGD (lr=%g, momentum=%g)", lr, momentum)
    elif name == "adam":
        opt = optim.Adam(model.parameters(), lr=lr)
        logger.info("Optimizer: Adam (lr=%g)", lr)
    elif name == "rmsprop":
        opt = optim.RMSprop(model.parameters(), lr=lr)
        logger.info("Optimizer: RMSprop (lr=%g)", lr)
    else:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: sgd, adam, rmsprop.")
    return opt


def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler: str,
    epochs: int,
    lr_min: float = 1e-4,
) -> optim.lr_scheduler.LRScheduler | None:
    """Build a learning rate scheduler, or return None for a constant LR.

    Args:
        scheduler: "none" (constant LR) or "cosine" (CosineAnnealingLR).
        epochs: total number of training epochs — used as T_max for cosine.
        lr_min: minimum LR at the end of cosine annealing (eta_min).
    """
    name = scheduler.lower().strip()
    if name == "none":
        return None
    elif name == "cosine":
        sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
        logger.info("Scheduler: CosineAnnealingLR (T_max=%d, eta_min=%g)", epochs, lr_min)
        return sched
    else:
        raise ValueError(f"Unknown scheduler '{scheduler}'. Choose from: none, cosine.")


def train_one_epoch_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
) -> float:
    """One epoch of standard (non-DP) training. Returns mean train accuracy."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []

    progress = tqdm(train_loader, desc=f"Baseline epoch {epoch}", unit="batch", leave=False)
    for images, target in progress:
        images, target = images.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, target)
        acc = accuracy_from_logits(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append(acc)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.1f}%")

    mean_acc = float(np.mean(accs))
    logger.info("[Baseline] Epoch %d | Loss: %.4f | Train Acc: %.2f%%",
                epoch, np.mean(losses), mean_acc * 100)
    return mean_acc


def train_one_epoch_dp(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    device: torch.device,
    privacy_engine: PrivacyEngine,
    max_physical_batch_size: int,
    delta: float,
) -> tuple[float, float]:
    """One epoch of DP-SGD training. Returns (mean train accuracy, current epsilon)."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=max_physical_batch_size,
        optimizer=optimizer,
    ) as memory_safe_data_loader:
        progress = tqdm(memory_safe_data_loader, desc=f"DP epoch {epoch}", unit="batch", leave=False)
        for images, target in progress:
            images, target = images.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, target)
            acc = accuracy_from_logits(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            accs.append(acc)
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.1f}%")

    epsilon = privacy_engine.get_epsilon(delta)
    mean_acc = float(np.mean(accs))
    logger.info(
        "[DP] Epoch %d | Loss: %.4f | Train Acc: %.2f%% | (ε = %.2f, δ = %.2e)",
        epoch, np.mean(losses), mean_acc * 100, epsilon, delta,
    )
    return mean_acc, epsilon


@torch.no_grad()
def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device, prefix: str = "Test") -> float:
    """Evaluation on the test set. Returns the mean accuracy."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []

    progress = tqdm(test_loader, desc=prefix, unit="batch", leave=False)
    for images, target in progress:
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)
        acc = accuracy_from_logits(output, target)
        losses.append(loss.item())
        accs.append(acc)
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.1f}%")

    mean_acc = float(np.mean(accs))
    logger.info("[%s] Loss: %.4f | Test Acc: %.2f%%", prefix, np.mean(losses), mean_acc * 100)
    return mean_acc


def make_private(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    epochs: int,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    accountant: str = "rdp",
) -> tuple[nn.Module, optim.Optimizer, DataLoader, PrivacyEngine]:
    """Wrap model/optimizer/dataloader with Opacus for DP-SGD."""
    privacy_engine = PrivacyEngine(accountant=accountant)
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=epochs,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
    )
    logger.info("Sigma computed by Opacus: %.4f (C=%.2f)", dp_optimizer.noise_multiplier, max_grad_norm)
    return dp_model, dp_optimizer, dp_loader, privacy_engine


def make_private_ablation(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    noise_multiplier: float,
    max_grad_norm: float,
    accountant: str = "rdp",
) -> tuple[nn.Module, optim.Optimizer, DataLoader, PrivacyEngine]:
    """Wrap model/optimizer/dataloader with Opacus for ablation studies.

    Takes noise_multiplier and max_grad_norm directly instead of deriving
    them from a target epsilon. Use noise_multiplier=0.0 for clipping-only
    ablation, or max_grad_norm=1e6 for noise-only ablation.
    """
    privacy_engine = PrivacyEngine(accountant=accountant)
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    logger.info("Ablation mode: noise_multiplier=%.4f, max_grad_norm=%.2f",
                noise_multiplier, max_grad_norm)
    return dp_model, dp_optimizer, dp_loader, privacy_engine


def unwrap_state_dict(model: nn.Module) -> dict[str, Any]:
    """Get a 'clean' state_dict (unwrapped from Opacus if necessary)."""
    inner = getattr(model, "_module", None)
    return inner.state_dict() if inner is not None else model.state_dict()


def apply_fsdp_compat_patch() -> None:
    """Temporary compatibility patch for some torch/Opacus version mismatches."""
    if hasattr(torch, "distributed") and hasattr(torch.distributed, "fsdp"):
        if not hasattr(torch.distributed.fsdp, "FSDPModule"):
            class _DummyFSDPModule:
                pass
            torch.distributed.fsdp.FSDPModule = _DummyFSDPModule
            logger.debug("FSDP compatibility patch applied.")