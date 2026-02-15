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

from src.config import CHARTS_DIR, METRICS_DIR, REPORTS_DIR
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


def load_reranking_results(path: Path | None = None) -> list[dict]:
    """Load reranking comparison results from JSON.

    WHY separate loader: reranking data has different structure from evaluations.
    Returns raw dicts because RerankingComparison model is already validated.
    """
    if path is None:
        path = METRICS_DIR / "reranking_results.json"

    data = json.loads(path.read_text())
    return data  # List of 3 dicts: E-openai, B-openai, D-openai


def load_ragas_results(path: Path | None = None) -> dict:
    """Load RAGAS evaluation results from JSON.

    WHY single dict: only one config (E-openai) was evaluated with RAGAS.
    """
    if path is None:
        path = METRICS_DIR / "ragas_results.json"

    data = json.loads(path.read_text())
    return data  # Single dict with faithfulness, answer_relevancy, etc.


def load_judge_results(path: Path | None = None) -> list[dict]:
    """Load LLM-as-Judge results from JSON.

    WHY list: 56 JudgeResult objects (one per question).
    """
    if path is None:
        path = METRICS_DIR / "judge_results.json"

    data = json.loads(path.read_text())
    return data  # List of 56 dicts


def load_qa_report(path: Path | None = None) -> dict:
    """Load QA dataset quality report from JSON.

    WHY separate: QADatasetReport is a summary, not per-question data.
    """
    if path is None:
        path = REPORTS_DIR / "qa_dataset_report.json"

    data = json.loads(path.read_text())
    return data  # Single dict with coverage, type distribution, etc.


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
# Additional Charts (Tasks 27.1–27.8)
# ===========================================================================

def generate_chunk_size_effect_chart(
    evaluations: list[ConfigEvaluation],
    output_path: Path | None = None,
) -> Path:
    """Generate line chart showing chunk size impact on Recall@5.

    WHY line chart: shows trend as chunk size increases (128 → 256 → 512).
    Compares Configs A/B/C with constant 25% overlap and openai embeddings.

    Args:
        evaluations: All 16 configs
        output_path: Where to save PNG (defaults to CHARTS_DIR/chunk_size_effect.png)
    """
    if output_path is None:
        output_path = CHARTS_DIR / "chunk_size_effect.png"

    # Filter to Configs A, B, C with openai embeddings (constant overlap 25%)
    configs_of_interest = ["A-openai", "B-openai", "C-openai"]
    filtered = [ev for ev in evaluations if ev.config_id in configs_of_interest]

    # Map config to chunk size
    chunk_sizes = {"A-openai": 128, "B-openai": 256, "C-openai": 512}

    # Extract data
    data = []
    for ev in filtered:
        data.append({
            "chunk_size": chunk_sizes[ev.config_id],
            "recall_at_5": ev.avg_recall_at_5,
        })

    # Sort by chunk size
    data.sort(key=lambda x: x["chunk_size"])

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        df["chunk_size"],
        df["recall_at_5"],
        marker="o",
        linewidth=2.5,
        markersize=10,
        color="steelblue",
        label="Recall@5",
    )

    ax.set_xlabel("Chunk Size (tokens)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@5", fontsize=12, fontweight="bold")
    ax.set_title("Chunk Size Impact on Retrieval Performance", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Annotate each point with value
    for _, row in df.iterrows():
        ax.annotate(
            f"{row['recall_at_5']:.3f}",
            xy=(row["chunk_size"], row["recall_at_5"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ Chunk size effect chart saved to %s", output_path)
    return output_path


def generate_overlap_effect_chart(
    evaluations: list[ConfigEvaluation],
    output_path: Path | None = None,
) -> Path:
    """Generate multi-panel comparison of Config B (25%) vs Config D (50%) overlap.

    WHY multi-panel: shows overlap impact across 3 key metrics (R@5, P@5, MRR@5).
    Uses openai embeddings to isolate the overlap variable.

    Args:
        evaluations: All 16 configs
        output_path: Where to save PNG
    """
    if output_path is None:
        output_path = CHARTS_DIR / "overlap_effect.png"

    # Filter to B-openai (25%) and D-openai (50%)
    b_config = next(ev for ev in evaluations if ev.config_id == "B-openai")
    d_config = next(ev for ev in evaluations if ev.config_id == "D-openai")

    metrics = ["avg_recall_at_5", "avg_precision_at_5", "avg_mrr_at_5"]
    metric_labels = ["Recall@5", "Precision@5", "MRR@5"]

    # Create 1x3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        b_value = getattr(b_config, metric)
        d_value = getattr(d_config, metric)

        x = ["Config B\n(25% overlap)", "Config D\n(50% overlap)"]
        y = [b_value, d_value]
        colors = ["steelblue", "coral"]

        bars = ax.bar(x, y, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5)

        # Annotate bars with values
        for bar, value in zip(bars, y):
            height = bar.get_height()
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )

        ax.set_ylabel(label, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(y) * 1.15)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Overlap Impact: 25% vs 50% (Configs B vs D, OpenAI Embeddings)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ Overlap effect chart saved to %s", output_path)
    return output_path


def generate_embedding_comparison_chart(
    evaluations: list[ConfigEvaluation],
    output_path: Path | None = None,
) -> Path:
    """Generate grouped bar chart comparing embedding models (minilm, mpnet, openai).

    WHY grouped bars: shows 3 metrics side-by-side for each embedding model.
    Uses Config B (256/64) to isolate embedding quality from chunk size.

    Args:
        evaluations: All 16 configs
        output_path: Where to save PNG
    """
    if output_path is None:
        output_path = CHARTS_DIR / "embedding_comparison.png"

    # Filter to B-minilm, B-mpnet, B-openai
    configs_of_interest = ["B-minilm", "B-mpnet", "B-openai"]
    filtered = [ev for ev in evaluations if ev.config_id in configs_of_interest]

    # Sort by a logical order: minilm, mpnet, openai (quality ascending)
    order = {"B-minilm": 0, "B-mpnet": 1, "B-openai": 2}
    filtered.sort(key=lambda ev: order[ev.config_id])

    # Extract data
    model_labels = ["MiniLM\n(384d)", "MPnet\n(768d)", "OpenAI\n(1536d)"]
    recall_5 = [ev.avg_recall_at_5 for ev in filtered]
    precision_5 = [ev.avg_precision_at_5 for ev in filtered]
    mrr_5 = [ev.avg_mrr_at_5 for ev in filtered]

    x = np.arange(len(model_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width, recall_5, width, label="Recall@5", color="steelblue", alpha=0.8)
    ax.bar(x, precision_5, width, label="Precision@5", color="coral", alpha=0.8)
    ax.bar(x + width, mrr_5, width, label="MRR@5", color="seagreen", alpha=0.8)

    ax.set_xlabel("Embedding Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Embedding Model Comparison (Config B: 256 tokens, 25% overlap)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ Embedding comparison chart saved to %s", output_path)
    return output_path


def generate_question_type_breakdown_chart(
    evaluations: list[ConfigEvaluation],
    output_path: Path | None = None,
) -> Path:
    """Generate stacked bar chart showing Recall@5 by question type.

    WHY stacked bars: shows how each config performs across different question types.
    Focuses on top-5 configs for readability.

    Args:
        evaluations: All 16 configs
        output_path: Where to save PNG
    """
    if output_path is None:
        output_path = CHARTS_DIR / "question_type_breakdown.png"

    # Sort by avg_recall_at_5 and take top-5
    sorted_evals = sorted(evaluations, key=lambda ev: ev.avg_recall_at_5, reverse=True)
    top_5 = sorted_evals[:5]

    # Guard: check if metrics_by_question_type exists for at least one config
    # WHY: This field may be missing if evaluation was run without per-type breakdowns
    has_question_type_data = any(
        hasattr(ev, 'metrics_by_question_type') and ev.metrics_by_question_type
        for ev in top_5
    )

    if not has_question_type_data:
        logger.warning("⚠️  Skipping question type breakdown chart — metrics_by_question_type field not found in data")
        return output_path

    # Extract question type metrics
    # Structure: metrics_by_question_type = {"factual": {"avg_recall_at_5": 0.5, ...}, ...}
    question_types = ["factual", "analytical", "multi_hop"]  # Most common types

    data = {qt: [] for qt in question_types}
    config_labels = []

    for ev in top_5:
        config_labels.append(ev.config_id)
        # Safe access with empty dict fallback
        type_metrics = getattr(ev, 'metrics_by_question_type', {}) or {}
        for qt in question_types:
            if qt in type_metrics:
                data[qt].append(type_metrics[qt]["avg_recall_at_5"])
            else:
                data[qt].append(0.0)  # No questions of this type

    x = np.arange(len(config_labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))

    # Stacked bars
    bottom = np.zeros(len(config_labels))
    colors = ["steelblue", "coral", "seagreen"]

    for qt, color in zip(question_types, colors):
        ax.bar(x, data[qt], width, label=qt.replace("_", " ").title(), bottom=bottom, color=color, alpha=0.8)
        bottom += np.array(data[qt])

    ax.set_xlabel("Configuration", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@5 (Stacked by Question Type)", fontsize=12, fontweight="bold")
    ax.set_title("Recall@5 Breakdown by Question Type (Top-5 Configs)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, rotation=15, ha="right")
    ax.legend(title="Question Type")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ Question type breakdown chart saved to %s", output_path)
    return output_path


def generate_reranking_impact_chart(
    output_path: Path | None = None,
) -> Path:
    """Generate paired horizontal bar chart showing reranking before/after.

    WHY horizontal bars: easier to read config names and improvement percentages.
    Shows 3 configs that were reranked: E-openai, B-openai, D-openai.

    Args:
        output_path: Where to save PNG
    """
    if output_path is None:
        output_path = CHARTS_DIR / "reranking_impact.png"

    # Load reranking results
    rerank_data = load_reranking_results()

    # Extract data for plotting
    # WHY flat field names: reranking JSON uses flat structure, not nested original_metrics/reranked_metrics
    config_labels = []
    before_scores = []
    after_scores = []
    improvements = []

    for item in rerank_data:
        config_labels.append(item["config_id"])
        before_scores.append(item["recall_at_5_before"])  # FIXED: was "before_recall_at_5"
        after_scores.append(item["recall_at_5_after"])    # FIXED: was "after_recall_at_5"
        improvements.append(item["recall_improvement_pct"])  # FIXED: was "improvement_percent_recall_at_5"

    y = np.arange(len(config_labels))
    height = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bars
    bars1 = ax.barh(y - height/2, before_scores, height, label="Before Reranking", color="lightcoral", alpha=0.8)
    bars2 = ax.barh(y + height/2, after_scores, height, label="After Reranking", color="seagreen", alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(config_labels)
    ax.set_xlabel("Recall@5", fontsize=12, fontweight="bold")
    ax.set_title("Reranking Impact on Top-3 Configurations", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    # Annotate improvement percentages
    for i, (before, after, improvement) in enumerate(zip(before_scores, after_scores, improvements)):
        ax.annotate(
            f"+{improvement:.1f}%",
            xy=(max(before, after) + 0.02, y[i]),
            fontsize=11,
            fontweight="bold",
            color="darkgreen",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ Reranking impact chart saved to %s", output_path)
    return output_path


def generate_ragas_radar_chart(
    output_path: Path | None = None,
) -> Path:
    """Generate radar/polar chart showing 4 RAGAS metrics for best config.

    WHY radar chart: shows performance across multiple dimensions on same scale.
    All RAGAS metrics are 0-1, making them directly comparable.

    Args:
        output_path: Where to save PNG
    """
    if output_path is None:
        output_path = CHARTS_DIR / "ragas_radar.png"

    # Load RAGAS results
    ragas_data = load_ragas_results()

    # Extract metrics (all 0-1 scale)
    metrics = [
        ("Faithfulness", ragas_data["faithfulness"]),
        ("Answer Relevancy", ragas_data["answer_relevancy"]),
        ("Context Recall", ragas_data["context_recall"]),
        ("Context Precision", ragas_data["context_precision"]),
    ]

    labels = [m[0] for m in metrics]
    values = [m[1] for m in metrics]

    # Number of variables
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # Complete the circle
    values += values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

    ax.plot(angles, values, "o-", linewidth=2, color="steelblue", label="E-openai")
    ax.fill(angles, values, alpha=0.25, color="steelblue")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_title("RAGAS Metrics — Best Configuration (E-openai)", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ RAGAS radar chart saved to %s", output_path)
    return output_path


def generate_bloom_distribution_chart(
    output_path: Path | None = None,
) -> Path:
    """Generate horizontal bar chart showing Bloom taxonomy distribution.

    WHY horizontal bars: easier to read Bloom level names (Remember, Understand, etc.).
    Shows cognitive complexity distribution of the 56 questions.

    Args:
        output_path: Where to save PNG
    """
    if output_path is None:
        output_path = CHARTS_DIR / "bloom_distribution.png"

    # Load judge results
    judge_data = load_judge_results()

    # Count questions per Bloom level
    bloom_counts = {}
    for item in judge_data:
        bloom_level = item.get("bloom_level")
        if bloom_level and bloom_level != "None":  # Filter out None values
            bloom_counts[bloom_level] = bloom_counts.get(bloom_level, 0) + 1

    # Sort by Bloom taxonomy order
    bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
    sorted_items = [(level, bloom_counts.get(level, 0)) for level in bloom_order if level in bloom_counts]

    labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(labels, counts, color="steelblue", alpha=0.8, edgecolor="black", linewidth=1.5)

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        width = bar.get_width()
        ax.annotate(
            f"{count}",
            xy=(width, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Number of Questions", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bloom Taxonomy Level", fontsize=12, fontweight="bold")
    ax.set_title("Question Distribution by Cognitive Complexity (Bloom Taxonomy)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ Bloom distribution chart saved to %s", output_path)
    return output_path


def generate_qa_quality_dashboard(
    output_path: Path | None = None,
) -> Path:
    """Generate 2x2 subplot grid showing QA dataset quality metrics.

    WHY 2x2 grid: shows 4 different quality dimensions in one comprehensive view.
    Includes type distribution, hierarchy, strategy, and coverage.

    Args:
        output_path: Where to save PNG
    """
    if output_path is None:
        output_path = CHARTS_DIR / "qa_quality.png"

    # Load QA quality report
    qa_data = load_qa_report()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Question Type Distribution (top-left)
    ax1 = axes[0, 0]
    type_dist = qa_data["questions_per_type"]
    ax1.bar(type_dist.keys(), type_dist.values(), color="steelblue", alpha=0.8)
    ax1.set_title("Question Type Distribution", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(axis="y", alpha=0.3)

    # Subplot 2: Hierarchy Distribution (top-right)
    ax2 = axes[0, 1]
    hierarchy_dist = qa_data["questions_per_hierarchy"]
    ax2.bar(hierarchy_dist.keys(), hierarchy_dist.values(), color="coral", alpha=0.8)
    ax2.set_title("Hierarchy Distribution", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Count")
    ax2.grid(axis="y", alpha=0.3)

    # Subplot 3: Strategy Distribution (bottom-left)
    ax3 = axes[1, 0]
    strategy_dist = qa_data["questions_per_strategy"]
    ax3.barh(list(strategy_dist.keys()), list(strategy_dist.values()), color="seagreen", alpha=0.8)
    ax3.set_title("Generation Strategy Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Count")
    ax3.grid(axis="x", alpha=0.3)

    # Subplot 4: Coverage Gauge (bottom-right)
    ax4 = axes[1, 1]
    coverage = qa_data["chunk_coverage_percent"]

    # Simple bar showing coverage percentage
    ax4.barh(["Chunk Coverage"], [coverage], color="purple", alpha=0.8, height=0.4)
    ax4.set_xlim(0, 100)
    ax4.set_xlabel("Percentage (%)")
    ax4.set_title("Chunk Coverage", fontsize=12, fontweight="bold")
    ax4.annotate(
        f"{coverage:.1f}%",
        xy=(coverage, 0),
        xytext=(5, 0),
        textcoords="offset points",
        fontsize=14,
        fontweight="bold",
    )
    ax4.grid(axis="x", alpha=0.3)

    fig.suptitle("QA Dataset Quality Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("✓ QA quality dashboard saved to %s", output_path)
    return output_path


# ===========================================================================
# Generate All Charts
# ===========================================================================

def generate_all_charts(
    evaluations: list[ConfigEvaluation] | None = None,
) -> list[Path]:
    """Generate all 12 charts for the grid search report.

    WHY wrapper function: single command to regenerate all visualizations.
    Useful for re-running after data updates or fixing chart bugs.
    """
    logger.info("Generating all 12 charts...")

    if evaluations is None:
        evaluations = load_evaluations()

    # Ensure output directory exists
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = []

    # Charts 1-4: Original Day 3 charts
    logger.info("Chart 1/12: Config heatmap...")
    paths.append(plot_config_heatmap(evaluations))

    logger.info("Chart 2/12: Metric comparison...")
    paths.append(plot_metric_bars(evaluations))

    logger.info("Chart 3/12: BM25 comparison...")
    paths.append(plot_bm25_comparison(evaluations))

    logger.info("Chart 4/12: Semantic vs fixed-size...")
    paths.append(plot_semantic_vs_fixed(evaluations))

    # Charts 5-12: Additional Day 5 charts
    logger.info("Chart 5/12: Chunk size effect...")
    paths.append(generate_chunk_size_effect_chart(evaluations))

    logger.info("Chart 6/12: Overlap effect...")
    paths.append(generate_overlap_effect_chart(evaluations))

    logger.info("Chart 7/12: Embedding comparison...")
    paths.append(generate_embedding_comparison_chart(evaluations))

    logger.info("Chart 8/12: Question type breakdown...")
    paths.append(generate_question_type_breakdown_chart(evaluations))

    logger.info("Chart 9/12: Reranking impact...")
    paths.append(generate_reranking_impact_chart())

    logger.info("Chart 10/12: RAGAS radar...")
    paths.append(generate_ragas_radar_chart())

    logger.info("Chart 11/12: Bloom distribution...")
    paths.append(generate_bloom_distribution_chart())

    logger.info("Chart 12/12: QA quality dashboard...")
    paths.append(generate_qa_quality_dashboard())

    logger.info("✅ All 12 charts generated successfully!")
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
