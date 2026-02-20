from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# Chart color constants (enforced across all visualizations)
COLOR_BASELINE = "#9E9E9E"  # gray
COLOR_STANDARD = "#2196F3"  # blue
COLOR_LORA = "#FF9800"  # orange


def plot_comparison_cosine_distributions(
    baseline_sims: np.ndarray,
    baseline_labels: list[int],
    standard_sims: np.ndarray,
    standard_labels: list[int],
    lora_sims: np.ndarray,
    lora_labels: list[int],
    output_path: Path,
) -> Path:
    """Plot cosine similarity distributions for all three models.

    WHY 2-row subplot: separates compatible/incompatible distributions for clarity.
    Compatible pairs should cluster at high similarity (right side), incompatible
    at low similarity (left side). Good models show clear separation.

    Args:
        baseline_sims: Baseline cosine similarities (N,)
        baseline_labels: Baseline labels (N,)
        standard_sims: Standard model similarities (N,)
        standard_labels: Standard model labels (N,)
        lora_sims: LoRA model similarities (N,)
        lora_labels: LoRA labels (N,)
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Convert labels to numpy arrays for indexing
    baseline_labels_arr = np.array(baseline_labels)
    standard_labels_arr = np.array(standard_labels)
    lora_labels_arr = np.array(lora_labels)

    # --- Top subplot: Compatible pairs (label=1) ---
    ax_compat = axes[0]
    baseline_compat = baseline_sims[baseline_labels_arr == 1]
    standard_compat = standard_sims[standard_labels_arr == 1]
    lora_compat = lora_sims[lora_labels_arr == 1]

    sns.kdeplot(baseline_compat, ax=ax_compat, color=COLOR_BASELINE, label="Baseline", linewidth=2)
    sns.kdeplot(standard_compat, ax=ax_compat, color=COLOR_STANDARD, label="Standard", linewidth=2)
    sns.kdeplot(lora_compat, ax=ax_compat, color=COLOR_LORA, label="LoRA", linewidth=2)

    ax_compat.set_title("Compatible Pairs — Cosine Similarity Distributions", fontsize=14, fontweight="bold")
    ax_compat.set_xlabel("Cosine Similarity", fontsize=12)
    ax_compat.set_ylabel("Density", fontsize=12)
    ax_compat.legend(fontsize=11)
    ax_compat.grid(alpha=0.3)

    # --- Bottom subplot: Incompatible pairs (label=0) ---
    ax_incompat = axes[1]
    baseline_incompat = baseline_sims[baseline_labels_arr == 0]
    standard_incompat = standard_sims[standard_labels_arr == 0]
    lora_incompat = lora_sims[lora_labels_arr == 0]

    sns.kdeplot(baseline_incompat, ax=ax_incompat, color=COLOR_BASELINE, label="Baseline", linewidth=2)
    sns.kdeplot(standard_incompat, ax=ax_incompat, color=COLOR_STANDARD, label="Standard", linewidth=2)
    sns.kdeplot(lora_incompat, ax=ax_incompat, color=COLOR_LORA, label="LoRA", linewidth=2)

    ax_incompat.set_title("Incompatible Pairs — Cosine Similarity Distributions", fontsize=14, fontweight="bold")
    ax_incompat.set_xlabel("Cosine Similarity", fontsize=12)
    ax_incompat.set_ylabel("Density", fontsize=12)
    ax_incompat.legend(fontsize=11)
    ax_incompat.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_comparison_umap(
    baseline_projections: np.ndarray,
    baseline_labels: list[int],
    standard_projections: np.ndarray,
    standard_labels: list[int],
    lora_projections: np.ndarray,
    lora_labels: list[int],
    output_path: Path,
) -> Path:
    """Plot UMAP projections for all three models in 1x3 panel layout.

    WHY side-by-side panels: enables visual comparison of clustering quality.
    Compatible (orange) and incompatible (blue) pairs should form distinct
    clusters. Better models show tighter, more separated clusters.

    Args:
        baseline_projections: Baseline UMAP (2N, 2)
        baseline_labels: Baseline labels (N,)
        standard_projections: Standard UMAP (2N, 2)
        standard_labels: Standard labels (N,)
        lora_projections: LoRA UMAP (2N, 2)
        lora_labels: LoRA labels (N,)
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # WHY average text_1 + text_2 projections: each pair has two UMAP points
    # (one per person). Averaging gives a single 2D location per pair.
    def average_pair_projections(projections: np.ndarray, labels: list[int]) -> tuple[np.ndarray, np.ndarray]:
        n = len(labels)
        pair_projections = (projections[:n] + projections[n:]) / 2
        return pair_projections, np.array(labels)

    # --- Panel 1: Baseline ---
    baseline_avg, baseline_labels_arr = average_pair_projections(baseline_projections, baseline_labels)
    ax_baseline = axes[0]
    _ = ax_baseline.scatter(
        baseline_avg[:, 0],
        baseline_avg[:, 1],
        c=baseline_labels_arr,
        cmap="RdYlGn",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidths=0.5,
    )
    ax_baseline.set_title("Baseline UMAP", fontsize=14, fontweight="bold")
    ax_baseline.set_xlabel("UMAP 1", fontsize=12)
    ax_baseline.set_ylabel("UMAP 2", fontsize=12)
    ax_baseline.grid(alpha=0.3)

    # --- Panel 2: Standard ---
    standard_avg, standard_labels_arr = average_pair_projections(standard_projections, standard_labels)
    ax_standard = axes[1]
    _ = ax_standard.scatter(
        standard_avg[:, 0],
        standard_avg[:, 1],
        c=standard_labels_arr,
        cmap="RdYlGn",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidths=0.5,
    )
    ax_standard.set_title("Standard Fine-Tuned UMAP", fontsize=14, fontweight="bold")
    ax_standard.set_xlabel("UMAP 1", fontsize=12)
    ax_standard.set_ylabel("UMAP 2", fontsize=12)
    ax_standard.grid(alpha=0.3)

    # --- Panel 3: LoRA ---
    lora_avg, lora_labels_arr = average_pair_projections(lora_projections, lora_labels)
    ax_lora = axes[2]
    scatter_lora = ax_lora.scatter(
        lora_avg[:, 0],
        lora_avg[:, 1],
        c=lora_labels_arr,
        cmap="RdYlGn",
        alpha=0.6,
        s=50,
        edgecolors="black",
        linewidths=0.5,
    )
    ax_lora.set_title("LoRA Fine-Tuned UMAP", fontsize=14, fontweight="bold")
    ax_lora.set_xlabel("UMAP 1", fontsize=12)
    ax_lora.set_ylabel("UMAP 2", fontsize=12)
    ax_lora.grid(alpha=0.3)

    # Add shared colorbar
    fig.colorbar(scatter_lora, ax=axes, label="Label (0=Incompatible, 1=Compatible)", shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_comparison_cluster_purity(
    baseline_purity: float,
    standard_purity: float,
    lora_purity: float,
    output_path: Path,
) -> Path:
    """Plot cluster purity comparison as a 3-bar chart.

    WHY cluster purity: measures how well HDBSCAN clusters align with ground
    truth labels. Higher purity = better unsupervised clustering quality.

    Args:
        baseline_purity: Baseline cluster purity (0-1)
        standard_purity: Standard cluster purity (0-1)
        lora_purity: LoRA cluster purity (0-1)
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["Baseline", "Standard", "LoRA"]
    purities = [baseline_purity, standard_purity, lora_purity]
    colors = [COLOR_BASELINE, COLOR_STANDARD, COLOR_LORA]

    bars = ax.bar(models, purities, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels on top of bars
    for bar, purity in zip(bars, purities):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{purity:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Cluster Purity", fontsize=12)
    ax.set_title("HDBSCAN Cluster Purity Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_comparison_roc_curves(
    baseline_sims: np.ndarray,
    baseline_labels: list[int],
    standard_sims: np.ndarray,
    standard_labels: list[int],
    lora_sims: np.ndarray,
    lora_labels: list[int],
    output_path: Path,
) -> Path:
    """Plot ROC curves for all three models on the same axes.

    WHY ROC curves: shows classification performance across all thresholds.
    AUC-ROC close to 1.0 indicates excellent discrimination between compatible
    and incompatible pairs. Curve closer to top-left corner = better performance.

    Args:
        baseline_sims: Baseline cosine similarities (N,)
        baseline_labels: Baseline labels (N,)
        standard_sims: Standard model similarities (N,)
        standard_labels: Standard model labels (N,)
        lora_sims: LoRA model similarities (N,)
        lora_labels: LoRA labels (N,)
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Compute ROC curves for each model
    baseline_fpr, baseline_tpr, _ = roc_curve(baseline_labels, baseline_sims)
    baseline_auc = auc(baseline_fpr, baseline_tpr)

    standard_fpr, standard_tpr, _ = roc_curve(standard_labels, standard_sims)
    standard_auc = auc(standard_fpr, standard_tpr)

    lora_fpr, lora_tpr, _ = roc_curve(lora_labels, lora_sims)
    lora_auc = auc(lora_fpr, lora_tpr)

    # Plot curves
    ax.plot(
        baseline_fpr,
        baseline_tpr,
        color=COLOR_BASELINE,
        linewidth=2,
        label=f"Baseline (AUC = {baseline_auc:.4f})",
    )
    ax.plot(
        standard_fpr,
        standard_tpr,
        color=COLOR_STANDARD,
        linewidth=2,
        label=f"Standard (AUC = {standard_auc:.4f})",
    )
    ax.plot(
        lora_fpr,
        lora_tpr,
        color=COLOR_LORA,
        linewidth=2,
        label=f"LoRA (AUC = {lora_auc:.4f})",
    )

    # Plot diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
