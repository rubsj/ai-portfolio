from __future__ import annotations

import base64
from pathlib import Path

import matplotlib

# WHY Agg: non-interactive backend — no display required, works in scripts/CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from src.models import BaselineMetrics, CategoryMetrics

sns.set_theme(style="whitegrid", font_scale=1.1)

BASELINE_VIZ_DIR = Path("eval/visualizations/baseline")


def plot_cosine_distributions(
    similarities: np.ndarray,
    labels: list[int],
    save_path: Path | None = None,
) -> Path:
    """Overlapping KDE histograms for compatible vs incompatible cosine similarities.

    WHY KDE overlay: the raw distribution shape matters — if they overlap heavily,
    fine-tuning has a clear job to do. Annotated means + margin text makes the
    'before' story concrete for the portfolio write-up.
    """
    save_path = save_path or BASELINE_VIZ_DIR / "cosine_distributions.png"
    labels_arr = np.array(labels)
    compat_sims = similarities[labels_arr == 1]
    incompat_sims = similarities[labels_arr == 0]

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        compat_sims, kde=True, alpha=0.5, color="#2196F3",
        label=f"Compatible (n={len(compat_sims)})", ax=ax, bins=30,
    )
    sns.histplot(
        incompat_sims, kde=True, alpha=0.5, color="#F44336",
        label=f"Incompatible (n={len(incompat_sims)})", ax=ax, bins=30,
    )

    compat_mean = float(np.mean(compat_sims))
    incompat_mean = float(np.mean(incompat_sims))
    margin = compat_mean - incompat_mean

    ax.axvline(compat_mean, color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Compatible mean = {compat_mean:.3f}")
    ax.axvline(incompat_mean, color="#B71C1C", linestyle="--", linewidth=1.5,
               label=f"Incompatible mean = {incompat_mean:.3f}")

    ax.set_title(f"Baseline Cosine Similarity Distributions  (margin = {margin:.3f})", fontsize=13)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_umap_scatter(
    projections: np.ndarray,
    labels: list[int],
    categories: list[str],
    save_path: Path | None = None,
) -> Path:
    """Interactive Plotly scatter plot of UMAP projections.

    WHY Plotly HTML: hover tooltips show category per point — static PNG can't do this.
    The portfolio reader can explore the embedding space interactively.

    projections shape: (2N, 2) — first N = text_1, last N = text_2.
    We plot both points per pair, colored by label.
    """
    save_path = save_path or BASELINE_VIZ_DIR / "umap_scatter.html"
    n = len(labels)
    labels_arr = np.array(labels)
    categories_arr = np.array(categories)

    # Build combined dataframe entries
    x = projections[:, 0].tolist()
    y = projections[:, 1].tolist()
    label_strs = (
        [("compatible" if lbl == 1 else "incompatible") for lbl in labels_arr] * 2
    )
    cat_strs = categories_arr.tolist() + categories_arr.tolist()
    side_strs = ["text_1"] * n + ["text_2"] * n

    import pandas as pd
    df = pd.DataFrame({"x": x, "y": y, "label": label_strs,
                       "category": cat_strs, "side": side_strs})

    color_map = {"compatible": "#2196F3", "incompatible": "#F44336"}
    fig = px.scatter(
        df, x="x", y="y", color="label",
        color_discrete_map=color_map,
        hover_data={"category": True, "side": True, "label": True},
        title="UMAP Projection — Baseline Embeddings (eval split)",
        labels={"x": "UMAP-1", "y": "UMAP-2"},
        opacity=0.6,
    )
    fig.update_traces(marker={"size": 5})
    fig.update_layout(legend_title_text="Compatibility")
    fig.write_html(str(save_path))
    return save_path


def plot_hdbscan_clusters(
    projections: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path | None = None,
) -> Path:
    """Scatter plot of UMAP pair projections colored by HDBSCAN cluster.

    WHY averaged projections: cluster_labels come from averaged pair points,
    so we recompute the averaged coordinates to plot them correctly.
    Noise points (label=-1) plotted in gray at 30% opacity.
    """
    save_path = save_path or BASELINE_VIZ_DIR / "hdbscan_clusters.png"
    n = len(cluster_labels)
    # Recover averaged pair projections (same operation as in run_hdbscan_clustering)
    pair_proj = (projections[:n] + projections[n:]) / 2  # shape (N, 2)

    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len([c for c in unique_clusters if c != -1])
    noise_ratio = float(np.sum(cluster_labels == -1) / n)

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab20")

    for c in unique_clusters:
        mask = cluster_labels == c
        if c == -1:
            ax.scatter(pair_proj[mask, 0], pair_proj[mask, 1],
                       c="gray", alpha=0.3, s=20, label="Noise")
        else:
            ax.scatter(pair_proj[mask, 0], pair_proj[mask, 1],
                       c=[cmap(c % 20)], alpha=0.7, s=25, label=f"Cluster {c}")

    ax.set_title(
        f"HDBSCAN Clusters — {n_clusters} clusters  (noise ratio = {noise_ratio:.1%})",
        fontsize=12,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if n_clusters <= 10:
        ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_roc_curve(
    similarities: np.ndarray,
    labels: list[int],
    save_path: Path | None = None,
) -> Path:
    """ROC curve with AUC annotation and diagonal reference line."""
    save_path = save_path or BASELINE_VIZ_DIR / "roc_curve.png"
    fpr, tpr, _ = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"Baseline (AUC = {auc_score:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1, label="Random (AUC = 0.500)")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")
    ax.set_title("ROC Curve — Baseline (eval split)", fontsize=13)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_category_margins(
    category_metrics: list[CategoryMetrics],
    title: str,
    save_path: Path | None = None,
) -> Path:
    """Horizontal bar chart of compatibility margin per category or pair_type.

    WHY sorted descending: immediately shows which categories are easiest /
    hardest for the pre-trained model — highlights where fine-tuning will help most.
    """
    save_path = save_path or BASELINE_VIZ_DIR / "category_margins.png"
    sorted_metrics = sorted(category_metrics, key=lambda m: m.margin)
    categories = [m.category for m in sorted_metrics]
    margins = [m.margin for m in sorted_metrics]

    # Color gradient: low margin = red, high margin = blue
    norm = plt.Normalize(vmin=min(margins), vmax=max(margins))
    colors = [plt.cm.RdYlBu(norm(v)) for v in margins]  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(10, max(5, len(categories) * 0.45)))
    bars = ax.barh(categories, margins, color=colors, edgecolor="none")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    for bar, m in zip(bars, margins):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{m:.3f}", va="center", ha="left", fontsize=8,
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Margin (compatible_mean − incompatible_mean)")
    ax.set_xlim(min(margins) - 0.05, max(margins) + 0.08)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_false_positive_breakdown(
    false_positive_counts: dict[str, int],
    save_path: Path | None = None,
) -> Path:
    """PRD Metric #4: False positive count per pair_type (horizontal bar chart).

    This is the 'before' reference for Day 3's false positive reduction story.
    If no FPs, saves a placeholder chart with explanatory text.
    """
    save_path = save_path or BASELINE_VIZ_DIR / "false_positive_breakdown.png"
    fig, ax = plt.subplots(figsize=(9, 5))

    if not false_positive_counts:
        ax.text(0.5, 0.5, "No false positives at the selected threshold",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.set_title("False Positive Breakdown — Baseline", fontsize=13)
        ax.axis("off")
    else:
        pair_types = list(false_positive_counts.keys())
        counts = list(false_positive_counts.values())
        # Already sorted descending by compute_false_positive_analysis
        colors = ["#EF5350"] * len(pair_types)

        bars = ax.barh(pair_types, counts, color=colors, edgecolor="none")
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", ha="left", fontsize=10, fontweight="bold",
            )
        ax.set_title("False Positives by Pair Type — Baseline (eval split)", fontsize=13)
        ax.set_xlabel("False Positive Count")
        ax.set_xlim(0, max(counts) * 1.2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_baseline_report_html(
    metrics: BaselineMetrics,
    chart_paths: dict[str, Path],
    save_path: Path | None = None,
) -> Path:
    """HTML report embedding all charts + metrics summary table.

    PNG charts embedded as base64 data URIs (self-contained file).
    The UMAP HTML chart embedded via <iframe> pointing to a relative path.
    """
    save_path = save_path or Path("eval") / "baseline_report.html"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    def _embed_png(path: Path) -> str:
        if not path.exists():
            return "<p><em>Chart not found.</em></p>"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;border-radius:6px;">'

    def _embed_html(path: Path, report_path: Path) -> str:
        if not path.exists():
            return "<p><em>Interactive chart not found.</em></p>"
        # Use relative path from report location
        rel = path.relative_to(report_path.parent)
        return f'<iframe src="{rel}" width="100%" height="550" frameborder="0"></iframe>'

    metrics_dict = metrics.model_dump(exclude={"category_metrics", "pair_type_metrics",
                                                "false_positive_counts"})

    rows = ""
    for k, v in metrics_dict.items():
        label = k.replace("_", " ").title()
        if isinstance(v, float):
            formatted = f"{v:.6f}"
        else:
            formatted = str(v)
        rows += f"<tr><td><strong>{label}</strong></td><td>{formatted}</td></tr>\n"

    # FP breakdown table
    fp_rows = ""
    for pt, count in metrics.false_positive_counts.items():
        fp_rows += f"<tr><td>{pt}</td><td>{count}</td></tr>\n"

    sections = ""
    for key, label in [
        ("distributions", "Cosine Similarity Distributions"),
        ("roc", "ROC Curve"),
        ("category_margins", "Category Margins"),
        ("pairtype_margins", "Pair Type Margins"),
        ("false_positives", "False Positive Breakdown"),
        ("hdbscan", "HDBSCAN Cluster Map"),
    ]:
        path = chart_paths.get(key)
        if path is None:
            continue
        suffix = path.suffix.lower()
        if suffix == ".png":
            chart_html = _embed_png(path)
        elif suffix == ".html":
            chart_html = _embed_html(path, save_path)
        else:
            chart_html = "<p><em>Unknown chart type.</em></p>"
        sections += f"""
        <section>
            <h2>{label}</h2>
            {chart_html}
        </section>
        """

    umap_path = chart_paths.get("umap")
    if umap_path is not None:
        sections += f"""
        <section>
            <h2>UMAP Embedding Projection (Interactive)</h2>
            {_embed_html(umap_path, save_path)}
        </section>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>P3 — Baseline Analysis Report</title>
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
<h1>P3 — Baseline Analysis Report</h1>
<p>Pre-training metrics for <code>all-MiniLM-L6-v2</code> on the dating compatibility eval split (n=295 pairs).</p>

<section>
  <h2>Core Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {rows}
  </table>
</section>

<section>
  <h2>False Positive Counts by Pair Type</h2>
  <table>
    <tr><th>Pair Type</th><th>Count</th></tr>
    {fp_rows if fp_rows else "<tr><td colspan='2'>No false positives at best threshold.</td></tr>"}
  </table>
</section>

{sections}
</body>
</html>
"""
    save_path.write_text(html, encoding="utf-8")
    return save_path
