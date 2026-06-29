"""
visualization.py
=================
Plotting functions (matplotlib), kept separate from the computation logic
so that `cka.py` and `training.py` can be reused/tested without a graphical
backend dependency (useful for non-interactive / headless execution).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend: safe in script/headless mode
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


def plot_accuracy_curves(
    baseline_train: list[float],
    baseline_test: list[float],
    dp_train: list[float],
    dp_test: list[float],
    epochs: int,
    save_path: str | Path | None = None,
) -> None:
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, [x * 100 for x in baseline_train], label="Baseline Train", marker="o", color="tab:blue")
    plt.plot(epochs_range, [x * 100 for x in dp_train], label="DP Train", marker="s", color="tab:orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Train Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, [x * 100 for x in baseline_test], label="Baseline Test", marker="o", linestyle="--", color="tab:blue")
    plt.plot(epochs_range, [x * 100 for x in dp_test], label="DP Test", marker="s", linestyle="--", color="tab:orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title("Test Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    _save_or_show(save_path)


def plot_cka_results(
    cross_cka_accum: dict[tuple[int, int], list[float]],
    layers: list[str],
    title_suffix: str = "",
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Plot the diagonal barplot (with 95% CI) plus the cross heatmap. Returns the mean matrix."""
    num_layers = len(layers)
    matrix_means = np.zeros((num_layers, num_layers))
    diag_means, diag_cis = [], []

    for i in range(num_layers):
        for j in range(num_layers):
            scores = cross_cka_accum[(i, j)]
            matrix_means[i, j] = np.mean(scores)
            if i == j:
                diag_means.append(matrix_means[i, j])
                std_err = stats.sem(scores) if len(scores) > 1 else 0
                ci_range = std_err * stats.t.ppf(0.975, len(scores) - 1) if len(scores) > 1 else 0
                diag_cis.append(ci_range)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(layers, diag_means, yerr=diag_cis, capsize=5, color="skyblue", edgecolor="black")
    plt.ylabel("CKA Score (with 95% CI)")
    plt.title(f"Diagonal CKA Similarity\n{title_suffix}")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    cax = plt.imshow(matrix_means, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(cax, label="CKA Similarity")
    plt.xticks(range(num_layers), layers)
    plt.yticks(range(num_layers), layers)
    plt.xlabel("Model B")
    plt.ylabel("Model A")
    plt.title(f"Inter-layer CKA Matrix\n{title_suffix}")

    for i in range(num_layers):
        for j in range(num_layers):
            val = matrix_means[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.6 else "black")

    plt.tight_layout()
    _save_or_show(save_path)

    logger.info("--- Diagonal scores (± 95%% CI) ---")
    for idx, layer in enumerate(layers):
        logger.info("%s : %.4f (± %.4f)", layer, diag_means[idx], diag_cis[idx])

    return matrix_means


def plot_epsilon_layer_heatmap(
    epsilon_layer_matrix: np.ndarray,
    epsilon_list: list[float],
    layers: list[str],
    save_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    cax = plt.imshow(epsilon_layer_matrix, cmap="plasma", vmin=0, vmax=1, origin="lower")
    plt.colorbar(cax, label="CKA Similarity with Baseline")

    plt.xticks(range(len(layers)), layers)
    plt.yticks(range(len(epsilon_list)), [f"ε = {e}" for e in epsilon_list])
    plt.xlabel("Network Layers")
    plt.ylabel("Privacy Constraint Level (DP)")
    plt.title("Evolution of Representation Similarity to Baseline across ε")

    for i in range(len(epsilon_list)):
        for j in range(len(layers)):
            val = epsilon_layer_matrix[i, j]
            plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val < 0.5 else "black")

    plt.tight_layout()
    _save_or_show(save_path)


def plot_fine_grained_matrix(
    fine_matrix_scores: np.ndarray,
    damier_layers: list[str],
    model_A_name: str,
    model_B_name: str,
    kernel_type: str = "linear",
    save_path: str | Path | None = None,
) -> None:
    num_layers = len(damier_layers)
    plt.figure(figsize=(10, 8))
    cax = plt.imshow(fine_matrix_scores, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(cax, label=f"CKA Similarity ({kernel_type})")

    plt.xticks(range(num_layers), damier_layers, rotation=90, fontsize=9)
    plt.yticks(range(num_layers), damier_layers, fontsize=9)
    plt.xlabel(f"Model: {model_B_name} — successive layers", fontsize=11, labelpad=10)
    plt.ylabel(f"Model: {model_A_name} — successive layers", fontsize=11, labelpad=10)
    plt.title(f"Fine-grained CKA analysis: {model_A_name} vs {model_B_name}\n(checkerboard pattern)", fontsize=13, pad=15)

    plt.tight_layout()
    _save_or_show(save_path)


def _save_or_show(save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    plt.close()