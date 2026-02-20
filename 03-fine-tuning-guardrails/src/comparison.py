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


def plot_comparison_category_heatmap(
    baseline_category_metrics: list,
    standard_category_metrics: list,
    lora_category_metrics: list,
    output_path: Path,
) -> Path:
    """Plot category margin comparison as a heatmap.

    WHY heatmap: enables quick visual comparison of per-category performance
    across all three models. Rows = categories, columns = models, cells = margin.
    Darker colors indicate better separation.

    Args:
        baseline_category_metrics: Baseline CategoryMetrics list
        standard_category_metrics: Standard CategoryMetrics list
        lora_category_metrics: LoRA CategoryMetrics list
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    # WHY dict comprehension: CategoryMetrics have category + margin attributes
    baseline_margins = {cm.category: cm.margin for cm in baseline_category_metrics}
    standard_margins = {cm.category: cm.margin for cm in standard_category_metrics}
    lora_margins = {cm.category: cm.margin for cm in lora_category_metrics}

    # Get all unique categories (union across all three models)
    all_categories = sorted(set(baseline_margins.keys()) | set(standard_margins.keys()) | set(lora_margins.keys()))

    # Build 2D array: rows=categories, cols=[Baseline, Standard, LoRA]
    data = []
    for cat in all_categories:
        row = [
            baseline_margins.get(cat, 0.0),
            standard_margins.get(cat, 0.0),
            lora_margins.get(cat, 0.0),
        ]
        data.append(row)

    # WHY numpy array: seaborn heatmap expects numeric 2D array
    import pandas as pd

    df = pd.DataFrame(data, index=all_categories, columns=["Baseline", "Standard", "LoRA"])

    fig, ax = plt.subplots(figsize=(10, max(6, len(all_categories) * 0.5)))

    # WHY annot=True: show numeric values in cells for exact comparison
    sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn", center=0, ax=ax, cbar_kws={"label": "Margin"})

    ax.set_title("Category Margin Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Category", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_comparison_cohens_d(
    baseline_d: float,
    standard_d: float,
    lora_d: float,
    output_path: Path,
) -> Path:
    """Plot Cohen's d comparison with color-coded magnitude bars.

    WHY color coding: Cohen's d interpretation is standardized
    (small <0.2, medium 0.2-0.8, large >0.8). Color helps communicate effect size.

    Args:
        baseline_d: Baseline Cohen's d
        standard_d: Standard Cohen's d
        lora_d: LoRA Cohen's d
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = ["Baseline", "Standard", "LoRA"]
    d_values = [baseline_d, standard_d, lora_d]

    # WHY color coding: matches Cohen's d interpretation thresholds
    def get_color(d: float) -> str:
        if abs(d) < 0.2:
            return "#EF5350"  # red = small effect
        elif abs(d) < 0.8:
            return "#FFC107"  # yellow = medium effect
        else:
            return "#66BB6A"  # green = large effect

    colors = [get_color(d) for d in d_values]

    bars = ax.bar(models, d_values, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

    # Add value labels on top of bars
    for bar, d in zip(bars, d_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{d:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Cohen's d", fontsize=12)
    ax.set_title("Effect Size Comparison (Cohen's d)", fontsize=14, fontweight="bold")
    ax.axhline(y=0.2, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Small (0.2)")
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Large (0.8)")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_comparison_classification_metrics(
    baseline_metrics,
    standard_metrics,
    lora_metrics,
    output_path: Path,
) -> Path:
    """Plot classification metrics (Accuracy, Precision, Recall, F1) for all models.

    WHY grouped bar: enables side-by-side comparison of 4 related metrics across
    3 models. Each metric group shows relative performance.

    Args:
        baseline_metrics: Baseline BaselineMetrics instance
        standard_metrics: Standard BaselineMetrics instance
        lora_metrics: LoRA BaselineMetrics instance
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # WHY these 4 metrics: standard classification performance indicators
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    baseline_values = [
        baseline_metrics.accuracy_at_best_threshold,
        baseline_metrics.precision_at_best_threshold,
        baseline_metrics.recall_at_best_threshold,
        baseline_metrics.best_f1,
    ]
    standard_values = [
        standard_metrics.accuracy_at_best_threshold,
        standard_metrics.precision_at_best_threshold,
        standard_metrics.recall_at_best_threshold,
        standard_metrics.best_f1,
    ]
    lora_values = [
        lora_metrics.accuracy_at_best_threshold,
        lora_metrics.precision_at_best_threshold,
        lora_metrics.recall_at_best_threshold,
        lora_metrics.best_f1,
    ]

    # WHY x positioning: 3 bars per metric group with spacing
    x = np.arange(len(metric_names))
    width = 0.25  # bar width

    ax.bar(x - width, baseline_values, width, label="Baseline", color=COLOR_BASELINE, alpha=0.8, edgecolor="black")
    ax.bar(x, standard_values, width, label="Standard", color=COLOR_STANDARD, alpha=0.8, edgecolor="black")
    ax.bar(x + width, lora_values, width, label="LoRA", color=COLOR_LORA, alpha=0.8, edgecolor="black")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Metrics Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_comparison_false_positives(
    baseline_fp_counts: dict[str, int],
    standard_fp_counts: dict[str, int],
    lora_fp_counts: dict[str, int],
    output_path: Path,
) -> Path:
    """Plot false positive counts by pair type for all models.

    WHY horizontal grouped bar: easier to read pair_type labels (e.g., "CC", "II", "CI", "IC"),
    and grouped bars enable direct comparison across models for each type.

    Args:
        baseline_fp_counts: Baseline FP counts dict {pair_type: count}
        standard_fp_counts: Standard FP counts dict
        lora_fp_counts: LoRA FP counts dict
        output_path: Path to save PNG

    Returns:
        Path to saved chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get all unique pair types (union across all three models)
    all_pair_types = sorted(set(baseline_fp_counts.keys()) | set(standard_fp_counts.keys()) | set(lora_fp_counts.keys()))

    # Build lists aligned by pair_type
    baseline_values = [baseline_fp_counts.get(pt, 0) for pt in all_pair_types]
    standard_values = [standard_fp_counts.get(pt, 0) for pt in all_pair_types]
    lora_values = [lora_fp_counts.get(pt, 0) for pt in all_pair_types]

    # WHY horizontal: better label readability
    y = np.arange(len(all_pair_types))
    height = 0.25

    ax.barh(y - height, baseline_values, height, label="Baseline", color=COLOR_BASELINE, alpha=0.8, edgecolor="black")
    ax.barh(y, standard_values, height, label="Standard", color=COLOR_STANDARD, alpha=0.8, edgecolor="black")
    ax.barh(y + height, lora_values, height, label="LoRA", color=COLOR_LORA, alpha=0.8, edgecolor="black")

    ax.set_xlabel("False Positive Count", fontsize=12)
    ax.set_title("False Positives by Pair Type", fontsize=14, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(all_pair_types, fontsize=11)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def write_false_positive_analysis(
    baseline_metrics,
    standard_metrics,
    lora_metrics,
    output_path: Path,
) -> Path:
    """Write false positive analysis comparing all three models.

    WHY text report: provides numerical breakdown and reduction percentages
    that complement the visual chart. Enables quick assessment of FP improvement.

    Args:
        baseline_metrics: Baseline BaselineMetrics instance
        standard_metrics: Standard BaselineMetrics instance
        lora_metrics: LoRA BaselineMetrics instance
        output_path: Path to save .txt file

    Returns:
        Path to saved file
    """
    baseline_fp = baseline_metrics.false_positive_counts
    standard_fp = standard_metrics.false_positive_counts
    lora_fp = lora_metrics.false_positive_counts

    # Get all unique pair types
    all_pair_types = sorted(set(baseline_fp.keys()) | set(standard_fp.keys()) | set(lora_fp.keys()))

    # Build markdown report
    lines = [
        "# False Positive Analysis",
        "",
        "Comparison of false positive counts across baseline, standard fine-tuned, and LoRA fine-tuned models.",
        "",
        "## Summary by Pair Type",
        "",
        "| Pair Type | Baseline | Standard | LoRA | Std Reduction | LoRA Reduction |",
        "|-----------|----------|----------|------|---------------|----------------|",
    ]

    for pt in all_pair_types:
        b_count = baseline_fp.get(pt, 0)
        s_count = standard_fp.get(pt, 0)
        l_count = lora_fp.get(pt, 0)

        # WHY avoid division by zero: baseline might have 0 FPs for some types
        if b_count > 0:
            std_reduction = ((b_count - s_count) / b_count) * 100
            lora_reduction = ((b_count - l_count) / b_count) * 100
        else:
            std_reduction = 0.0
            lora_reduction = 0.0

        lines.append(
            f"| {pt:<9} | {b_count:>8} | {s_count:>8} | {l_count:>4} | "
            f"{std_reduction:>13.1f}% | {lora_reduction:>14.1f}% |"
        )

    # Total FP counts
    baseline_total = sum(baseline_fp.values())
    standard_total = sum(standard_fp.values())
    lora_total = sum(lora_fp.values())

    if baseline_total > 0:
        std_total_reduction = ((baseline_total - standard_total) / baseline_total) * 100
        lora_total_reduction = ((baseline_total - lora_total) / baseline_total) * 100
    else:
        std_total_reduction = 0.0
        lora_total_reduction = 0.0

    lines.extend(
        [
            "",
            "## Overall Totals",
            "",
            f"- **Baseline**: {baseline_total} false positives",
            f"- **Standard**: {standard_total} false positives ({std_total_reduction:.1f}% reduction)",
            f"- **LoRA**: {lora_total} false positives ({lora_total_reduction:.1f}% reduction)",
            "",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))

    return output_path
