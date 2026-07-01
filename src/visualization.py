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


def plot_mean_heatmap(
    mean_matrix: np.ndarray,
    layers: list[str],
    title: str = "",
    annotate: bool = False,
    save_path: str | Path | None = None,
) -> None:
    """Full layer-by-layer CKA heatmap, averaged across seed/checkpoint pairs.

    Args:
        mean_matrix: (n_layers, n_layers) array, mean CKA across pairs.
        annotate: if True, print the numeric score in each cell (gets
            cluttered fast beyond ~15 layers — leave off for full layer lists).
    """
    num_layers = len(layers)
    fig_size = max(8, num_layers * 0.4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    cax = plt.imshow(mean_matrix, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(cax, label="Mean CKA Similarity (across pairs)")

    plt.xticks(range(num_layers), layers, rotation=90, fontsize=8)
    plt.yticks(range(num_layers), layers, fontsize=8)
    plt.title(title)

    if annotate:
        for i in range(num_layers):
            for j in range(num_layers):
                val = mean_matrix[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center",
                          color="white" if val < 0.6 else "black", fontsize=6)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_std_heatmap(
    std_matrix: np.ndarray,
    layers: list[str],
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """Full layer-by-layer heatmap of the standard deviation across pairs.

    Shows where CKA is most sensitive to seed (or checkpoint) variation,
    without needing a 3D plot: same 2D layer x layer grid as the mean
    heatmap, just colored by spread instead of by level.
    """
    num_layers = len(layers)
    fig_size = max(8, num_layers * 0.4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    cax = plt.imshow(std_matrix, cmap="magma", vmin=0, origin="lower")
    plt.colorbar(cax, label="Std. dev. of CKA across pairs")

    plt.xticks(range(num_layers), layers, rotation=90, fontsize=8)
    plt.yticks(range(num_layers), layers, fontsize=8)
    plt.title(title)

    plt.tight_layout()
    _save_or_show(save_path)


def plot_diagonal_stats(
    all_pair_matrices: list[np.ndarray],
    layers: list[str],
    title: str = "",
    save_path: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    """Diagonal (layer i vs layer i) CKA across pairs: mean, 95% CI, min/max.

    This is the plot that directly answers "how low does same-layer CKA
    naturally go across seeds/checkpoints?" — the rest of the matrix
    (off-diagonal, cross-layer) is summarized separately by the heatmaps.

    Args:
        all_pair_matrices: one (n_layers, n_layers) array per pair.

    Returns:
        {layer_name: {"mean", "ci95", "min", "max", "n_pairs"}}
    """
    diag_values = np.array([np.diag(m) for m in all_pair_matrices])  # (n_pairs, n_layers)
    n_pairs = diag_values.shape[0]

    means = diag_values.mean(axis=0)
    mins = diag_values.min(axis=0)
    maxs = diag_values.max(axis=0)
    if n_pairs > 1:
        sem = diag_values.std(axis=0, ddof=1) / np.sqrt(n_pairs)
        cis = sem * stats.t.ppf(0.975, n_pairs - 1)
    else:
        cis = np.zeros_like(means)

    x = np.arange(len(layers))
    plt.figure(figsize=(max(10, len(layers) * 0.4), 5))
    plt.fill_between(x, mins, maxs, color="gray", alpha=0.2, label="Min–max range")
    plt.fill_between(x, means - cis, means + cis, color="blue", alpha=0.3, label="95% CI")
    plt.plot(x, means, "o-", color="blue", linewidth=2, label="Mean")
    plt.xticks(x, layers, rotation=90, fontsize=8)
    plt.ylabel("CKA Score")
    plt.ylim(-0.05, 1.05)
    plt.title(f"{title}\n(n = {n_pairs} pair{'s' if n_pairs != 1 else ''})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left")

    plt.tight_layout()
    _save_or_show(save_path)

    summary = {}
    logger.info("--- Diagonal CKA across pairs (n=%d) ---", n_pairs)
    for idx, layer in enumerate(layers):
        summary[layer] = {
            "mean": float(means[idx]), "ci95": float(cis[idx]),
            "min": float(mins[idx]), "max": float(maxs[idx]), "n_pairs": n_pairs,
        }
        logger.info(
            "%s : mean=%.4f ± %.4f (95%% CI) | range=[%.4f, %.4f]",
            layer, means[idx], cis[idx], mins[idx], maxs[idx],
        )

    return summary


def _save_or_show(save_path: str | Path | None) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Figure saved: %s", save_path)
    plt.close()