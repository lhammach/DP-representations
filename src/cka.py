"""
cka.py
======
Computation of Centered Kernel Alignment (CKA) between representations of
two models, plus activation extraction utilities via forward hooks.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Activation extraction
# --------------------------------------------------------------------------- #

def get_activations(model: nn.Module, layer_names: list[str]) -> tuple[dict[str, torch.Tensor], list]:
    """Register forward hooks on the named layers and return
    (activations dict [empty until the forward pass happens], list of hooks).

    Remember to remove the hooks after use (`for h in hooks: h.remove()`)
    to avoid memory leaks / hooks piling up.
    """
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str) -> Callable:
        def hook(module, inputs, output):
            activations[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(make_hook(name)))

    return activations, hooks


# --------------------------------------------------------------------------- #
# CKA formulas
# --------------------------------------------------------------------------- #

def gram_linear(x: torch.Tensor) -> torch.Tensor:
    return x @ x.T


def gram_rbf(x: torch.Tensor, sigma: float | None = None) -> torch.Tensor:
    dists = torch.cdist(x, x, p=2) ** 2
    if sigma is None:
        median_dist = torch.median(dists[dists > 0])
        sigma = torch.sqrt(median_dist)
    return torch.exp(-dists / (2 * (sigma ** 2)))


def center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - torch.ones(n, n, device=K.device) / n
    return H @ K @ H


def hsic(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    return torch.sum(K * L)


def compute_cka(X: torch.Tensor, Y: torch.Tensor, kernel_type: str = "linear") -> float:
    """CKA between two activation tensors (N, C, H, W) or (N, D)."""
    X = X.flatten(1)
    Y = Y.flatten(1)

    if kernel_type == "linear":
        K = center_gram(gram_linear(X))
        L = center_gram(gram_linear(Y))
    elif kernel_type == "gaussian":
        K = center_gram(gram_rbf(X))
        L = center_gram(gram_rbf(Y))
    else:
        raise ValueError("kernel_type must be 'linear' or 'gaussian'")

    hsic_xy = hsic(K, L)
    hsic_xx = hsic(K, K)
    hsic_yy = hsic(L, L)

    denom = torch.sqrt(hsic_xx * hsic_yy + 1e-12)
    return (hsic_xy / denom).item()


# --------------------------------------------------------------------------- #
# Multi-batch / multi-layer CKA matrices
# --------------------------------------------------------------------------- #

def compute_cka_matrix(
    model_A: nn.Module,
    model_B: nn.Module,
    layers: list[str],
    loader: DataLoader,
    device: torch.device,
    num_batches: int = 30,
    kernel_type: str = "linear",
) -> dict[tuple[int, int], list[float]]:
    """Compute the cross CKA matrix (all layer pairs) between two models,
    averaged over several test batches.

    Returns:
        dict {(i, j): [per-batch scores]} where i, j are indices into `layers`.
    """
    model_A.eval()
    model_B.eval()

    cross_cka_accum: dict[tuple[int, int], list[float]] = {
        (i, j): [] for i in range(len(layers)) for j in range(len(layers))
    }

    for batch_idx, (images, _) in enumerate(
        tqdm(loader, desc=f"CKA ({kernel_type})", leave=False)
    ):
        if batch_idx >= num_batches:
            break

        images = images.to(device, non_blocking=True)
        acts_A, hooks_A = get_activations(model_A, layers)
        acts_B, hooks_B = get_activations(model_B, layers)

        with torch.no_grad():
            model_A(images)
            model_B(images)

        for h in hooks_A:
            h.remove()
        for h in hooks_B:
            h.remove()

        for i, layer_A in enumerate(layers):
            for j, layer_B in enumerate(layers):
                score = compute_cka(acts_A[layer_A], acts_B[layer_B], kernel_type=kernel_type)
                cross_cka_accum[(i, j)].append(score)

        del images, acts_A, acts_B

    return cross_cka_accum


def summarize_diagonal(
    cross_cka_accum: dict[tuple[int, int], list[float]], layers: list[str]
) -> dict[str, dict[str, float]]:
    """Summarize the diagonal (layer i vs layer i) with mean and 95% CI.

    Useful for exporting a numeric summary (JSON/CSV) without depending on
    matplotlib.
    """
    import scipy.stats as stats

    summary = {}
    for i, layer in enumerate(layers):
        scores = cross_cka_accum[(i, i)]
        mean = float(np.mean(scores))
        std_err = stats.sem(scores) if len(scores) > 1 else 0.0
        ci = float(std_err * stats.t.ppf(0.975, len(scores) - 1)) if len(scores) > 1 else 0.0
        summary[layer] = {"mean": mean, "ci95": ci, "n_batches": len(scores)}
    return summary