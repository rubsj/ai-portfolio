from __future__ import annotations

import base64
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, roc_curve

from src.data_loader import load_pairs
from src.models import BaselineMetrics, ComparisonResult, EvaluationBundle

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


def generate_all_comparison_charts(
    baseline_bundle: EvaluationBundle,
    standard_bundle: EvaluationBundle,
    lora_bundle: EvaluationBundle,
) -> dict[str, Path]:
    """Generate all 8 comparison charts and return path dict.

    WHY orchestrator: centralizes chart generation logic, enables HTML report
    to embed all charts without duplicating calls.

    Args:
        baseline_bundle: Baseline EvaluationBundle
        standard_bundle: Standard EvaluationBundle
        lora_bundle: LoRA EvaluationBundle

    Returns:
        Dict mapping chart keys to saved PNG paths
    """
    output_dir = Path("eval/visualizations/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = {}

    # Chart 1: Cosine distributions
    chart_paths["distributions"] = plot_comparison_cosine_distributions(
        baseline_sims=baseline_bundle.similarities,
        baseline_labels=baseline_bundle.labels,
        standard_sims=standard_bundle.similarities,
        standard_labels=standard_bundle.labels,
        lora_sims=lora_bundle.similarities,
        lora_labels=lora_bundle.labels,
        output_path=output_dir / "cosine_distributions.png",
    )

    # Chart 2: UMAP projections
    chart_paths["umap"] = plot_comparison_umap(
        baseline_projections=baseline_bundle.projections,
        baseline_labels=baseline_bundle.labels,
        standard_projections=standard_bundle.projections,
        standard_labels=standard_bundle.labels,
        lora_projections=lora_bundle.projections,
        lora_labels=lora_bundle.labels,
        output_path=output_dir / "umap.png",
    )

    # Chart 3: Cluster purity
    chart_paths["cluster_purity"] = plot_comparison_cluster_purity(
        baseline_purity=baseline_bundle.metrics.cluster_purity,
        standard_purity=standard_bundle.metrics.cluster_purity,
        lora_purity=lora_bundle.metrics.cluster_purity,
        output_path=output_dir / "cluster_purity.png",
    )

    # Chart 4: ROC curves
    chart_paths["roc"] = plot_comparison_roc_curves(
        baseline_sims=baseline_bundle.similarities,
        baseline_labels=baseline_bundle.labels,
        standard_sims=standard_bundle.similarities,
        standard_labels=standard_bundle.labels,
        lora_sims=lora_bundle.similarities,
        lora_labels=lora_bundle.labels,
        output_path=output_dir / "roc_curves.png",
    )

    # Chart 5: Category heatmap
    chart_paths["category_heatmap"] = plot_comparison_category_heatmap(
        baseline_category_metrics=baseline_bundle.metrics.category_metrics,
        standard_category_metrics=standard_bundle.metrics.category_metrics,
        lora_category_metrics=lora_bundle.metrics.category_metrics,
        output_path=output_dir / "category_heatmap.png",
    )

    # Chart 6: Cohen's d comparison
    chart_paths["cohens_d"] = plot_comparison_cohens_d(
        baseline_d=baseline_bundle.metrics.cohens_d,
        standard_d=standard_bundle.metrics.cohens_d,
        lora_d=lora_bundle.metrics.cohens_d,
        output_path=output_dir / "cohens_d.png",
    )

    # Chart 7: Classification metrics
    chart_paths["classification_metrics"] = plot_comparison_classification_metrics(
        baseline_metrics=baseline_bundle.metrics,
        standard_metrics=standard_bundle.metrics,
        lora_metrics=lora_bundle.metrics,
        output_path=output_dir / "classification_metrics.png",
    )

    # Chart 8: False positives
    chart_paths["false_positives"] = plot_comparison_false_positives(
        baseline_fp_counts=baseline_bundle.metrics.false_positive_counts,
        standard_fp_counts=standard_bundle.metrics.false_positive_counts,
        lora_fp_counts=lora_bundle.metrics.false_positive_counts,
        output_path=output_dir / "false_positives.png",
    )

    return chart_paths


def build_comparison_result(
    baseline_metrics: BaselineMetrics,
    standard_metrics: BaselineMetrics,
    lora_metrics: BaselineMetrics,
) -> ComparisonResult:
    """Build ComparisonResult with improvement deltas.

    WHY deltas: portfolio documentation needs clear improvement metrics
    (e.g., "73% margin improvement from fine-tuning").

    Args:
        baseline_metrics: Baseline BaselineMetrics
        standard_metrics: Standard BaselineMetrics
        lora_metrics: LoRA BaselineMetrics

    Returns:
        ComparisonResult with computed improvement deltas
    """
    # WHY absolute improvement: baseline could be negative (inverted embeddings)
    margin_improvement = standard_metrics.compatibility_margin - baseline_metrics.compatibility_margin

    # WHY avoid division by zero: baseline margin might be zero or negative
    if baseline_metrics.compatibility_margin != 0:
        margin_improvement_pct = (margin_improvement / abs(baseline_metrics.compatibility_margin)) * 100
    else:
        margin_improvement_pct = 0.0

    # WHY Cohen's d improvement: measures effect size change
    cohens_d_improvement = standard_metrics.cohens_d - baseline_metrics.cohens_d

    # WHY Spearman improvement: rank correlation change
    spearman_improvement = standard_metrics.spearman_correlation - baseline_metrics.spearman_correlation

    return ComparisonResult(
        baseline=baseline_metrics,
        standard_finetuned=standard_metrics,
        lora_finetuned=lora_metrics,
        margin_improvement=margin_improvement,
        margin_improvement_pct=margin_improvement_pct,
        cohens_d_improvement=cohens_d_improvement,
        spearman_improvement=spearman_improvement,
    )


def generate_comparison_report_html(
    baseline_metrics: BaselineMetrics,
    standard_metrics: BaselineMetrics,
    lora_metrics: BaselineMetrics,
    chart_paths: dict[str, Path],
) -> Path:
    """Generate self-contained HTML report with all comparison charts.

    WHY self-contained: base64-embeds PNGs so report is a single file
    (easy to share, no broken links if files move).

    Args:
        baseline_metrics: Baseline BaselineMetrics
        standard_metrics: Standard BaselineMetrics
        lora_metrics: LoRA BaselineMetrics
        chart_paths: Dict mapping chart keys to PNG paths

    Returns:
        Path to saved HTML report
    """
    output_path = Path("eval/comparison_report.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _embed_png(path: Path) -> str:
        """Embed PNG as base64 data URI."""
        if not path.exists():
            return "<p><em>Chart not found.</em></p>"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;border-radius:6px;">'

    # WHY exclude complex fields: focus on core numeric metrics for table
    baseline_dict = baseline_metrics.model_dump(
        exclude={"category_metrics", "pair_type_metrics", "false_positive_counts"}
    )
    standard_dict = standard_metrics.model_dump(
        exclude={"category_metrics", "pair_type_metrics", "false_positive_counts"}
    )
    lora_dict = lora_metrics.model_dump(
        exclude={"category_metrics", "pair_type_metrics", "false_positive_counts"}
    )

    # Build comparison table rows
    rows = ""
    for key in baseline_dict.keys():
        label = key.replace("_", " ").title()
        b_val = baseline_dict[key]
        s_val = standard_dict[key]
        l_val = lora_dict[key]

        # WHY formatting: floats get 4 decimals, ints/strings as-is
        if isinstance(b_val, float):
            b_fmt = f"{b_val:.4f}"
            s_fmt = f"{s_val:.4f}"
            l_fmt = f"{l_val:.4f}"
        else:
            b_fmt = str(b_val)
            s_fmt = str(s_val)
            l_fmt = str(l_val)

        rows += f"<tr><td><strong>{label}</strong></td><td>{b_fmt}</td><td>{s_fmt}</td><td>{l_fmt}</td></tr>\n"

    # Build chart sections
    sections = ""
    for key, label in [
        ("distributions", "Cosine Similarity Distributions"),
        ("umap", "UMAP Projections"),
        ("cluster_purity", "Cluster Purity Comparison"),
        ("roc", "ROC Curves"),
        ("category_heatmap", "Category Margin Heatmap"),
        ("cohens_d", "Cohen's d Effect Size"),
        ("classification_metrics", "Classification Metrics"),
        ("false_positives", "False Positives by Pair Type"),
    ]:
        path = chart_paths.get(key)
        if path is None:
            continue
        chart_html = _embed_png(path)
        sections += f"""
        <section>
            <h2>{label}</h2>
            {chart_html}
        </section>
        """

    # WHY inline CSS: self-contained file, no external dependencies
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>P3 — Comparison Report (Baseline vs Standard vs LoRA)</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 2rem; background: #f9f9f9; color: #222; }}
  h1 {{ border-bottom: 3px solid #2196F3; padding-bottom: 0.5rem; }}
  h2 {{ color: #1565C0; margin-top: 2rem; }}
  section {{ background: white; border-radius: 8px; padding: 1.5rem; margin: 1.5rem 0;
             box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #E3F2FD; color: #1565C0; }}
  tr:hover {{ background: #f5f5f5; }}
</style>
</head>
<body>
<h1>P3 — Fine-Tuning Comparison Report</h1>
<p>Comparison of baseline (pre-trained <code>all-MiniLM-L6-v2</code>), standard fine-tuned, and LoRA fine-tuned models on the dating compatibility eval split (n=295 pairs).</p>

<section>
  <h2>Core Metrics Comparison</h2>
  <table>
    <tr><th>Metric</th><th>Baseline</th><th>Standard</th><th>LoRA</th></tr>
    {rows}
  </table>
</section>

{sections}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_comparison() -> None:
    """Main orchestrator for comparison analysis.

    WHY separate from evaluation: enables fast re-run from cached data
    (no model loading, just metrics + embeddings). Useful for iterating
    on visualization tweaks.

    Steps:
    1. Load baseline, standard, LoRA metrics JSONs
    2. Load baseline, standard, LoRA embeddings NPZs
    3. Rebuild EvaluationBundles from saved data
    4. Generate all 8 comparison charts
    5. Generate HTML report
    6. Generate false positive analysis
    7. Save ComparisonResult JSON
    """
    from src.post_training_eval import evaluate_from_embeddings, resolve_baseline_path

    print("=== COMPARISON ANALYSIS ===\n")

    # --- Step 1: Load metrics JSONs ---
    print("Loading metrics JSONs...")
    baseline_metrics_path = resolve_baseline_path("baseline_metrics.json")
    baseline_metrics = BaselineMetrics.model_validate_json(baseline_metrics_path.read_text())

    standard_metrics_path = Path("eval/finetuned_metrics.json")
    if not standard_metrics_path.exists():
        msg = f"Standard metrics not found at {standard_metrics_path}. Run 'evaluate --mode all' first."
        raise FileNotFoundError(msg)
    standard_metrics = BaselineMetrics.model_validate_json(standard_metrics_path.read_text())

    lora_metrics_path = Path("eval/lora_metrics.json")
    if not lora_metrics_path.exists():
        msg = f"LoRA metrics not found at {lora_metrics_path}. Run 'evaluate --mode all' first."
        raise FileNotFoundError(msg)
    lora_metrics = BaselineMetrics.model_validate_json(lora_metrics_path.read_text())

    print(f"  ✓ Baseline: {baseline_metrics_path}")
    print(f"  ✓ Standard: {standard_metrics_path}")
    print(f"  ✓ LoRA: {lora_metrics_path}\n")

    # --- Step 2: Load eval pairs ---
    print("Loading eval pairs...")
    eval_pairs = load_pairs(Path("data/raw/eval_pairs.jsonl"))
    print(f"  ✓ Loaded {len(eval_pairs)} evaluation pairs\n")

    # --- Step 3: Rebuild EvaluationBundles from embeddings ---
    # WHY rebuild: we need intermediate arrays (similarities, projections, cluster_labels) for charts
    print("Rebuilding EvaluationBundles from cached embeddings...")

    baseline_emb_path = Path("data/embeddings/baseline_eval.npz")
    if not baseline_emb_path.exists():
        msg = f"Baseline embeddings not found at {baseline_emb_path}"
        raise FileNotFoundError(msg)
    baseline_bundle = evaluate_from_embeddings(baseline_emb_path, eval_pairs)
    print("  ✓ Baseline bundle ready")

    standard_emb_path = Path("data/embeddings/finetuned_eval.npz")
    if not standard_emb_path.exists():
        msg = f"Standard embeddings not found at {standard_emb_path}"
        raise FileNotFoundError(msg)
    standard_bundle = evaluate_from_embeddings(standard_emb_path, eval_pairs)
    print("  ✓ Standard bundle ready")

    lora_emb_path = Path("data/embeddings/lora_eval.npz")
    if not lora_emb_path.exists():
        msg = f"LoRA embeddings not found at {lora_emb_path}"
        raise FileNotFoundError(msg)
    lora_bundle = evaluate_from_embeddings(lora_emb_path, eval_pairs)
    print("  ✓ LoRA bundle ready\n")

    # --- Step 4: Generate all comparison charts ---
    print("Generating comparison charts...")
    chart_paths = generate_all_comparison_charts(baseline_bundle, standard_bundle, lora_bundle)
    print(f"  ✓ Generated {len(chart_paths)} charts in eval/visualizations/comparison/\n")

    # --- Step 5: Generate HTML report ---
    print("Generating HTML report...")
    html_path = generate_comparison_report_html(
        baseline_metrics, standard_metrics, lora_metrics, chart_paths
    )
    html_size_kb = html_path.stat().st_size / 1024
    print(f"  ✓ {html_path} ({html_size_kb:.1f} KB)\n")

    # --- Step 6: Generate false positive analysis ---
    print("Generating false positive analysis...")
    fp_analysis_path = write_false_positive_analysis(
        baseline_metrics, standard_metrics, lora_metrics, Path("eval/false_positive_analysis.txt")
    )
    print(f"  ✓ {fp_analysis_path}\n")

    # --- Step 7: Save ComparisonResult JSON ---
    print("Computing improvement deltas...")
    comparison_result = build_comparison_result(baseline_metrics, standard_metrics, lora_metrics)
    comparison_result_path = Path("eval/comparison_result.json")
    comparison_result_path.write_text(comparison_result.model_dump_json(indent=2))
    print(f"  ✓ {comparison_result_path}\n")

    # --- Summary ---
    print("=== SUMMARY ===")
    print(f"Margin improvement: {comparison_result.margin_improvement:+.4f} "
          f"({comparison_result.margin_improvement_pct:+.1f}%)")
    print(f"Cohen's d improvement: {comparison_result.cohens_d_improvement:+.4f}")
    print(f"Spearman improvement: {comparison_result.spearman_improvement:+.4f}")
    print(f"\nHTML report: {html_path}")
    print(f"Charts: eval/visualizations/comparison/ ({len(chart_paths)} files)")


if __name__ == "__main__":  # pragma: no cover
    run_comparison()
