"""Visualization module — generates charts from grid search results.

Produces 4 charts for Day 3 checkpoint:
1. Config x Metric Heatmap — 16 rows x 7 metrics, seaborn heatmap
2. Metric Bar Chart — grouped bars for R@5, P@5, MRR@5 across all configs
3. BM25 vs Vector — side-by-side bars: BM25 vs best vector config
4. Semantic vs Fixed-Size — Config E vs Config B per embedding model

WHY matplotlib + seaborn (not Plotly): static charts for reports and PRs.
Plotly is great for interactive dashboards but overkill for PNG exports.
Seaborn wraps matplotlib with cleaner defaults and heatmap support.

Java/TS parallel: like generating JFreeChart PNGs in a Java batch job —
read data, configure chart, write to disk. No interactivity needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import CHARTS_DIR, METRICS_DIR
from src.models import ConfigEvaluation

logger = logging.getLogger(__name__)

# WHY Agg backend: prevents "no display" errors on headless servers and CI.
# Must be set before any plt calls. Safe to call multiple times.
import matplotlib
matplotlib.use("Agg")

# WHY these style settings: consistent look across all 4 charts.
# seaborn "whitegrid" is clean for academic/portfolio presentations.
sns.set_theme(style="whitegrid", font_scale=1.1)

# Metric column order for heatmap and tables
_METRIC_COLS = [
    "avg_recall_at_1",
    "avg_recall_at_3",
    "avg_recall_at_5",
    "avg_precision_at_1",
    "avg_precision_at_3",
    "avg_precision_at_5",
    "avg_mrr_at_5",
]

# WHY short labels: long field names clutter chart axes
_METRIC_LABELS = {
    "avg_recall_at_1": "R@1",
    "avg_recall_at_3": "R@3",
    "avg_recall_at_5": "R@5",
    "avg_precision_at_1": "P@1",
    "avg_precision_at_3": "P@3",
    "avg_precision_at_5": "P@5",
    "avg_mrr_at_5": "MRR@5",
}


# ===========================================================================
# Data Loading
# ===========================================================================

def load_evaluations(path: Path | None = None) -> list[ConfigEvaluation]:
    """Load grid search results from JSON.

    WHY separate from grid_search.py: visualization shouldn't depend on
    the grid search orchestrator. Load from the JSON file directly.
    """
    if path is None:
        path = METRICS_DIR / "grid_search_results.json"

    data = json.loads(path.read_text())
    return [ConfigEvaluation.model_validate(item) for item in data]


def _evals_to_dataframe(evaluations: list[ConfigEvaluation]) -> pd.DataFrame:
    """Convert evaluations to a DataFrame with config_id as index.

    WHY DataFrame: seaborn heatmap and pandas pivot tables require tabular
    data. DataFrame is the standard format for matplotlib/seaborn plotting.

    Java/TS parallel: like mapping List<ConfigEvaluation> to a 2D array
    for a charting library. DataFrame is Python's equivalent of a ResultSet.
    """
    rows = []
    for ev in evaluations:
        rows.append({
            "config_id": ev.config_id,
            "chunk_config": ev.chunk_config,
            "embedding_model": ev.embedding_model,
            "retrieval_method": ev.retrieval_method.value,
            "num_chunks": ev.num_chunks,
            **{col: getattr(ev, col) for col in _METRIC_COLS},
        })

    df = pd.DataFrame(rows)
    # WHY sort by R@5 descending: best configs at top of heatmap
    df = df.sort_values("avg_recall_at_5", ascending=False)
    return df


# ===========================================================================
# Chart 1: Config x Metric Heatmap
# ===========================================================================

def plot_config_heatmap(
    evaluations: list[ConfigEvaluation],
    save_path: Path | None = None,
) -> Path:
    """Generate a heatmap of all 16 configs x 7 metrics.

    WHY heatmap: shows patterns at a glance — which configs dominate,
    which metrics correlate. A table of 112 numbers is unreadable;
    color-coding makes outliers and trends instantly visible.
    """
    if save_path is None:
        save_path = CHARTS_DIR / "config_heatmap.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = _evals_to_dataframe(evaluations)

    # Extract just the metric columns, use config_id as row labels
    heatmap_data = df.set_index("config_id")[_METRIC_COLS]
    heatmap_data.columns = [_METRIC_LABELS[c] for c in _METRIC_COLS]

    # WHY figsize (10, 8): 16 rows need vertical space; 7 columns need width.
    fig, ax = plt.subplots(figsize=(10, 8))

    # WHY annot=True, fmt=".3f": show exact values in each cell.
    # WHY cmap="YlOrRd": yellow-orange-red diverging — higher is redder (better).
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        vmin=0.0,
        vmax=1.0,
    )

    ax.set_title("Retrieval Config x Metric Heatmap (sorted by R@5)", fontsize=14)
    ax.set_ylabel("Configuration")
    ax.set_xlabel("Metric")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved heatmap to %s", save_path)
    return save_path


# ===========================================================================
# Chart 2: Metric Bar Chart (R@5, P@5, MRR@5)
# ===========================================================================

def plot_metric_bars(
    evaluations: list[ConfigEvaluation],
    save_path: Path | None = None,
) -> Path:
    """Grouped bar chart — R@5, P@5, MRR@5 for all 16 configs.

    WHY grouped bars (not stacked): R@5, P@5, MRR@5 are independent metrics
    on different scales. Stacking would imply they sum to something meaningful.
    Grouped bars allow direct comparison of each metric across configs.
    """
    if save_path is None:
        save_path = CHARTS_DIR / "metric_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = _evals_to_dataframe(evaluations)

    # WHY these 3 metrics: R@5 (coverage), P@5 (precision), MRR@5 (ranking quality).
    # @5 is the most representative K value — enough results to be meaningful.
    metrics = ["avg_recall_at_5", "avg_precision_at_5", "avg_mrr_at_5"]
    labels = [_METRIC_LABELS[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(df))
    width = 0.25  # WHY 0.25: 3 bars per group, 0.25 * 3 = 0.75, leaves 0.25 gap

    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(x + i * width, df[metric].values, width, label=label)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("R@5, P@5, MRR@5 by Configuration")
    ax.set_xticks(x + width)
    ax.set_xticklabels(df["config_id"].values, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved metric bars to %s", save_path)
    return save_path


# ===========================================================================
# Chart 3: BM25 vs Best Vector Config
# ===========================================================================

def plot_bm25_comparison(
    evaluations: list[ConfigEvaluation],
    save_path: Path | None = None,
) -> Path:
    """Side-by-side bars — BM25 baseline vs best vector config.

    WHY this chart: the key research question is "do embeddings beat keywords?"
    This chart gives a clear visual answer. If BM25 wins, the embeddings
    aren't adding value for this corpus.
    """
    if save_path is None:
        save_path = CHARTS_DIR / "bm25_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = _evals_to_dataframe(evaluations)

    # Find BM25 row and best vector row (by R@5)
    bm25_row = df[df["retrieval_method"] == "bm25"]
    vector_rows = df[df["retrieval_method"] == "vector"]
    best_vector = vector_rows.iloc[0]  # already sorted by R@5 desc

    metrics = _METRIC_COLS
    labels = [_METRIC_LABELS[m] for m in metrics]

    bm25_vals = [bm25_row[m].values[0] for m in metrics]
    vector_vals = [best_vector[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics))
    width = 0.35

    ax.bar(x - width / 2, bm25_vals, width, label=f"BM25 ({bm25_row['config_id'].values[0]})", color="#4c72b0")
    ax.bar(x + width / 2, vector_vals, width, label=f"Best Vector ({best_vector['config_id']})", color="#dd8452")

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("BM25 Baseline vs Best Vector Configuration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved BM25 comparison to %s", save_path)
    return save_path


# ===========================================================================
# Chart 4: Semantic (Config E) vs Fixed-Size (Config B)
# ===========================================================================

def plot_semantic_vs_fixed(
    evaluations: list[ConfigEvaluation],
    save_path: Path | None = None,
) -> Path:
    """Config E vs Config B per embedding model — tests semantic chunking.

    WHY this chart: Config E uses LLM-based semantic chunking (expensive).
    Config B is the standard fixed-size baseline. This chart shows whether
    the semantic approach justifies its cost for each embedding model.
    """
    if save_path is None:
        save_path = CHARTS_DIR / "semantic_vs_fixed.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = _evals_to_dataframe(evaluations)

    # Filter to Config B and E vector results only
    b_rows = df[(df["chunk_config"] == "B") & (df["retrieval_method"] == "vector")]
    e_rows = df[(df["chunk_config"] == "E") & (df["retrieval_method"] == "vector")]

    # WHY merge on embedding_model: ensures B and E are paired per model
    merged = pd.merge(
        b_rows[["embedding_model", "avg_recall_at_5"]],
        e_rows[["embedding_model", "avg_recall_at_5"]],
        on="embedding_model",
        suffixes=("_B", "_E"),
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(merged))
    width = 0.35

    ax.bar(x - width / 2, merged["avg_recall_at_5_B"], width,
           label="Config B (256/64 fixed)", color="#4c72b0")
    ax.bar(x + width / 2, merged["avg_recall_at_5_E"], width,
           label="Config E (semantic)", color="#dd8452")

    ax.set_xlabel("Embedding Model")
    ax.set_ylabel("Recall@5")
    ax.set_title("Semantic Chunking (E) vs Fixed-Size (B) — Recall@5")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["embedding_model"].values)
    ax.legend()
    ax.set_ylim(0, 1.0)

    # WHY annotate delta: shows the exact improvement/regression per model
    for i, row in merged.iterrows():
        delta = row["avg_recall_at_5_E"] - row["avg_recall_at_5_B"]
        sign = "+" if delta >= 0 else ""
        ax.annotate(
            f"{sign}{delta:.3f}",
            xy=(i, max(row["avg_recall_at_5_B"], row["avg_recall_at_5_E"]) + 0.02),
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="green" if delta > 0 else "red",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved semantic vs fixed chart to %s", save_path)
    return save_path


# ===========================================================================
# Generate All Charts
# ===========================================================================

def generate_all_charts(
    evaluations: list[ConfigEvaluation] | None = None,
) -> list[Path]:
    """Generate all 4 Day 3 charts from grid search results.

    WHY a wrapper: single entry point for the __main__ block and
    for grid_search.py to call after evaluation completes.
    """
    if evaluations is None:
        evaluations = load_evaluations()

    paths = [
        plot_config_heatmap(evaluations),
        plot_metric_bars(evaluations),
        plot_bm25_comparison(evaluations),
        plot_semantic_vs_fixed(evaluations),
    ]

    logger.info("Generated %d charts in %s", len(paths), CHARTS_DIR)
    return paths


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    evaluations = load_evaluations()
    logger.info("Loaded %d config evaluations", len(evaluations))

    paths = generate_all_charts(evaluations)

    print(f"\n{'='*60}")
    print("Charts Generated")
    print(f"{'='*60}")
    for p in paths:
        print(f"  {p}")
