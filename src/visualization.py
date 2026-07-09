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
    series: list[dict],
    title: str = "",
    save_path: str | Path | None = None,
) -> None:
    """Plot train and test accuracy curves for one or several checkpoints.

    Args:
        series: list of dicts, one per checkpoint. Each dict must contain:
            - "train_acc_history": list[float]
            - "test_acc_history": list[float]
            - "label": str  (shown in the legend)
            - "color": str  (matplotlib color, optional — auto-assigned if absent)
            - "linestyle": str  (optional, default "-")
        title: overall figure title.
        save_path: path to save the figure.

    Example:
        series = [
            {"train_acc_history": [...], "test_acc_history": [...], "label": "Baseline seed1", "color": "tab:blue"},
            {"train_acc_history": [...], "test_acc_history": [...], "label": "DP eps10 seed1", "color": "tab:orange"},
        ]
    """
    default_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                      "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    # Infer epoch count from the longest history (they should all be the same, but be defensive)
    max_epochs = max(len(s["train_acc_history"]) for s in series)
    epochs_range = range(1, max_epochs + 1)

    fig, (ax_train, ax_test) = plt.subplots(1, 2, figsize=(12, 4))

    for idx, s in enumerate(series):
        color = s.get("color") or default_colors[idx % len(default_colors)]
        linestyle = s.get("linestyle", "-")
        label = s.get("label", f"Series {idx+1}")
        n = len(s["train_acc_history"])

        ax_train.plot(
            range(1, n + 1), [x * 100 for x in s["train_acc_history"]],
            label=label, color=color, linestyle=linestyle, marker="o", markersize=4,
        )
        ax_test.plot(
            range(1, n + 1), [x * 100 for x in s["test_acc_history"]],
            label=label, color=color, linestyle="--", marker="s", markersize=4,
        )

    for ax, panel_title in [(ax_train, "Train Accuracy"), (ax_test, "Test Accuracy")]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(panel_title)
        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)

    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

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
    ylabel: str = "Model A (rows)",
    xlabel: str = "Model B (columns)",
    annotate: bool = False,
    save_path: str | Path | None = None,
) -> None:
    """Full layer-by-layer CKA heatmap, averaged across seed/checkpoint pairs."""
    num_layers = len(layers)
    fig_size = max(8, num_layers * 0.4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    cax = plt.imshow(mean_matrix, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(cax, label="Mean CKA Similarity")
    plt.xticks(range(num_layers), layers, rotation=90, fontsize=8)
    plt.yticks(range(num_layers), layers, fontsize=8)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
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
    ylabel: str = "Model A (rows)",
    xlabel: str = "Model B (columns)",
    save_path: str | Path | None = None,
) -> None:
    """Full layer-by-layer heatmap of the standard deviation across pairs."""
    num_layers = len(layers)
    fig_size = max(8, num_layers * 0.4)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    cax = plt.imshow(std_matrix, cmap="magma", vmin=0, origin="lower")
    plt.colorbar(cax, label="Std. dev. of CKA across pairs")
    plt.xticks(range(num_layers), layers, rotation=90, fontsize=8)
    plt.yticks(range(num_layers), layers, fontsize=8)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_zscore_heatmap(
    mean_matrix_target: np.ndarray,
    mean_matrix_ref: np.ndarray,
    std_matrix_ref: np.ndarray,
    layers: list[str],
    title: str = "",
    ylabel: str = "Model A (rows)",
    xlabel: str = "Model B (columns)",
    std_floor: float = 0.01,
    save_path: str | Path | None = None,
) -> np.ndarray:
    """Three-panel plot: target CKA | reference mean CKA | z-score (target vs reference).

    The z-score answers "where does target CKA deviate significantly from
    the reference baseline variability?" for every (layer_i, layer_j) pair,
    including off-diagonal blocks.

    Args:
        mean_matrix_target: (n_layers, n_layers) mean CKA for the comparison
            of interest (e.g. DP vs baseline).
        mean_matrix_ref: (n_layers, n_layers) mean CKA from the reference
            distribution (e.g. baseline vs baseline across seeds).
        std_matrix_ref: (n_layers, n_layers) std of the reference distribution.
        std_floor: minimum std applied to every cell before dividing, to avoid
            artificially inflating z-scores in cells where the reference is
            nearly constant (e.g. very dissimilar cross-layer pairs that are
            always ~0 with tiny variance).
        ylabel: label for the Y axis (rows) — typically the "A" group.
        xlabel: label for the X axis (columns) — typically the "B" group.

    Returns:
        zscore_matrix: (n_layers, n_layers) array of z-scores.
    """
    std_safe = np.maximum(std_matrix_ref, std_floor)
    zscore = (mean_matrix_target - mean_matrix_ref) / std_safe
    num_layers = len(layers)
    fig_size = max(8, num_layers * 0.4)

    fig, axes = plt.subplots(1, 3, figsize=(fig_size * 3, fig_size * 0.85))

    # Panel 1 : target (e.g. DP vs baseline)
    ax = axes[0]
    im = ax.imshow(mean_matrix_target, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(im, ax=ax, label="CKA")
    ax.set_xticks(range(num_layers)); ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_yticks(range(num_layers)); ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title("Target: " + (title or "DP vs Baseline"))

    # Panel 2 : reference mean (e.g. baseline vs baseline)
    ax = axes[1]
    im = ax.imshow(mean_matrix_ref, cmap="viridis", vmin=0, vmax=1, origin="lower")
    plt.colorbar(im, ax=ax, label="CKA")
    ax.set_xticks(range(num_layers)); ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_yticks(range(num_layers)); ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel("Baseline (columns)", fontsize=9); ax.set_ylabel("Baseline (rows)", fontsize=9)
    ax.set_title("Reference mean: Baseline vs Baseline")

    # Panel 3 : z-score (symmetric colormap centered on 0)
    ax = axes[2]
    vabs = max(abs(zscore.min()), abs(zscore.max()), 1.0)
    im = ax.imshow(zscore, cmap="RdBu_r", vmin=-vabs, vmax=vabs, origin="lower")
    plt.colorbar(im, ax=ax, label="z-score")
    ax.set_xticks(range(num_layers)); ax.set_xticklabels(layers, rotation=90, fontsize=7)
    ax.set_yticks(range(num_layers)); ax.set_yticklabels(layers, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(f"Z-score (std floor={std_floor})\nBlue=below ref, Red=above ref")

    plt.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    _save_or_show(save_path)
    return zscore


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