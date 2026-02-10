"""Analysis module: DataFrame construction, charts, and summary metrics.

Loads generated records + LLM labels, builds a combined Pandas DataFrame,
produces 6 visualizations (PRD Section 7b), and saves summary metrics.

Java/TS parallel: like a reporting service that joins data from two sources
(generation + evaluation) and produces dashboards. Pandas is the Python
equivalent of Java Streams + collectors, but optimized for tabular data.

WHY Pandas over raw dicts:
- GroupBy/pivot/corr operations are one-liners vs nested loops
- seaborn/matplotlib integrate natively with DataFrames
- .describe() gives instant summary stats — no manual aggregation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
# Use non-interactive backend so charts render to file without a display
# (Java/TS parallel: like setting a headless browser for server-side rendering)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"
_LABELS_DIR = _PROJECT_ROOT / "data" / "labels"
_CHARTS_DIR = _PROJECT_ROOT / "results" / "charts"
_RESULTS_DIR = _PROJECT_ROOT / "results"

# The 6 failure modes tracked by the LLM judge (PRD Section 6a)
FAILURE_MODES = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataFrame construction (PRD Section 7a)
# ---------------------------------------------------------------------------

def build_analysis_dataframe(
    batch_file: str = "batch_v1.json",
    labels_file: str = "llm_labels.csv",
) -> pd.DataFrame:
    """Build a combined DataFrame of records + LLM labels.

    Joins generated records (for category/difficulty metadata) with LLM
    judge labels (for failure mode flags). Adds computed columns:
    total_failures and quality_score.

    Args:
        batch_file: JSON filename in data/generated/.
        labels_file: CSV filename in data/labels/.

    Returns:
        DataFrame with columns: trace_id, category, difficulty,
        6 failure mode columns, total_failures, quality_score.
    """
    # Load generated records for metadata
    raw_records = json.loads((_GENERATED_DIR / batch_file).read_text())
    records_df = pd.DataFrame([
        {
            "trace_id": r["trace_id"],
            "category": r["category"],
            "difficulty": r["difficulty"],
        }
        for r in raw_records
    ])

    # Load LLM labels
    labels_df = pd.read_csv(_LABELS_DIR / labels_file)

    # Join on trace_id — inner join ensures we only keep records with labels
    df = records_df.merge(labels_df, on="trace_id", how="inner")

    # Ensure failure mode columns are int (CSV may load as float if any NaN)
    for mode in FAILURE_MODES:
        df[mode] = df[mode].astype(int)

    # Computed columns
    df["total_failures"] = df[FAILURE_MODES].sum(axis=1)
    # quality_score from the LLM judge (already in labels CSV)
    if "overall_quality_score" in df.columns:
        df["quality_score"] = df["overall_quality_score"]
    else:
        # Fallback: derive from failures (6 - failures, clamped to 1-5)
        df["quality_score"] = (6 - df["total_failures"]).clip(1, 5)

    logger.info(
        "Built analysis DataFrame: %d records, %d with failures",
        len(df),
        (df["total_failures"] > 0).sum(),
    )
    return df


# ---------------------------------------------------------------------------
# Chart generation (PRD Section 7b)
# ---------------------------------------------------------------------------

def _ensure_charts_dir() -> None:
    """Create the charts output directory if needed."""
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_failure_heatmap(df: pd.DataFrame) -> Path:
    """Records x failure modes heatmap.

    Each cell shows 0 (pass) or 1 (fail). Rows are trace_ids (truncated),
    columns are failure modes. Uses a red colormap so failures stand out.
    """
    _ensure_charts_dir()

    # Use short trace_id labels for readability
    plot_df = df.set_index(df["trace_id"].str[:8])[FAILURE_MODES]

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(
        plot_df,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        cbar_kws={"label": "0 = pass, 1 = fail"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Failure Mode Heatmap (Records × Failure Modes)", fontsize=14)
    ax.set_ylabel("Record (trace_id prefix)")
    ax.set_xlabel("Failure Mode")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = _CHARTS_DIR / "failure_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_failure_frequency(df: pd.DataFrame) -> Path:
    """Bar chart showing count of failures per failure mode."""
    _ensure_charts_dir()

    counts = df[FAILURE_MODES].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("YlOrRd_r", n_colors=len(counts))
    counts.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
    ax.set_title("Failure Frequency by Mode", fontsize=14)
    ax.set_ylabel("Number of Records Failing")
    ax.set_xlabel("Failure Mode")

    # Add count labels on top of each bar
    for i, v in enumerate(counts):
        ax.text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = _CHARTS_DIR / "failure_frequency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_failure_correlation(df: pd.DataFrame) -> Path:
    """Correlation matrix of failure mode co-occurrence.

    Shows which failure modes tend to appear together. High positive
    correlation = they co-occur; near-zero = independent.
    """
    _ensure_charts_dir()

    corr = df[FAILURE_MODES].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Failure Mode Correlation Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = _CHARTS_DIR / "failure_correlation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_category_failures(df: pd.DataFrame) -> Path:
    """Failures broken down by repair category (grouped bar chart)."""
    _ensure_charts_dir()

    # Group by category, sum each failure mode
    cat_failures = df.groupby("category")[FAILURE_MODES].sum()

    fig, ax = plt.subplots(figsize=(12, 6))
    cat_failures.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Failure Modes by Repair Category", fontsize=14)
    ax.set_ylabel("Number of Failures")
    ax.set_xlabel("Category")
    ax.legend(title="Failure Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = _CHARTS_DIR / "category_failures.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_difficulty_failures(df: pd.DataFrame) -> Path:
    """Failures broken down by difficulty level (grouped bar chart)."""
    _ensure_charts_dir()

    # Order difficulties logically
    diff_order = ["beginner", "intermediate", "advanced"]
    diff_failures = (
        df.groupby("difficulty")[FAILURE_MODES]
        .sum()
        .reindex(diff_order)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    diff_failures.plot(kind="bar", ax=ax, edgecolor="black")
    ax.set_title("Failure Modes by Difficulty Level", fontsize=14)
    ax.set_ylabel("Number of Failures")
    ax.set_xlabel("Difficulty")
    ax.legend(title="Failure Mode", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()

    path = _CHARTS_DIR / "difficulty_failures.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_agreement_matrix(
    manual_file: str = "manual_labels.csv",
    llm_file: str = "llm_labels.csv",
) -> Path:
    """Manual vs LLM labels comparison heatmap for overlapping records.

    Shows a side-by-side comparison: for each of the 10 overlapping records,
    displays manual label and LLM label for each failure mode. Disagreements
    are highlighted.
    """
    _ensure_charts_dir()

    manual_df = pd.read_csv(_LABELS_DIR / manual_file)
    llm_df = pd.read_csv(_LABELS_DIR / llm_file)

    # Only keep records present in both
    common_ids = set(manual_df["trace_id"]) & set(llm_df["trace_id"])
    manual_df = manual_df[manual_df["trace_id"].isin(common_ids)].set_index("trace_id")
    llm_df = llm_df[llm_df["trace_id"].isin(common_ids)].set_index("trace_id")

    # Reindex to same order
    common_list = sorted(common_ids)
    manual_sub = manual_df.loc[common_list, FAILURE_MODES].astype(int)
    llm_sub = llm_df.loc[common_list, FAILURE_MODES].astype(int)

    # Build agreement matrix: 0 = both pass, 1 = both fail, -1 = disagree
    # We'll show the difference (manual - LLM) so 0 = agree
    diff = manual_sub - llm_sub

    # Truncate trace_ids for labels
    short_ids = [tid[:8] for tid in common_list]

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Panel 1: Manual labels
    sns.heatmap(
        manual_sub.values,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=FAILURE_MODES,
        yticklabels=short_ids,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Manual Labels", fontsize=12)

    # Panel 2: LLM labels
    sns.heatmap(
        llm_sub.values,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=FAILURE_MODES,
        yticklabels=short_ids,
        ax=axes[1],
        cbar=False,
    )
    axes[1].set_title("LLM Judge Labels", fontsize=12)

    # Panel 3: Disagreements (0 = agree, non-zero = disagree)
    sns.heatmap(
        diff.values,
        annot=True,
        fmt="d",
        cmap="RdYlGn",
        center=0,
        xticklabels=FAILURE_MODES,
        yticklabels=short_ids,
        ax=axes[2],
        cbar_kws={"label": "-1: LLM flagged, Human didn't | +1: Human flagged, LLM didn't"},
    )
    axes[2].set_title("Disagreements (Manual − LLM)", fontsize=12)

    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.suptitle("Inter-Rater Agreement: Manual vs LLM Judge (10 records)", fontsize=14, y=1.02)
    plt.tight_layout()

    path = _CHARTS_DIR / "agreement_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


# ---------------------------------------------------------------------------
# Summary metrics (PRD Section 7c)
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute summary metrics answering the PRD's key analysis questions.

    Returns a dict suitable for saving as results/metrics.json.
    """
    total_records = len(df)
    total_possible = total_records * len(FAILURE_MODES)
    total_failures = int(df[FAILURE_MODES].sum().sum())
    records_with_failures = int((df["total_failures"] > 0).sum())

    # Per-mode failure counts and rates
    per_mode = {}
    for mode in FAILURE_MODES:
        count = int(df[mode].sum())
        per_mode[mode] = {
            "count": count,
            "rate": f"{count / total_records * 100:.1f}%",
        }

    # Per-category failure rates
    per_category = {}
    for cat, group in df.groupby("category"):
        cat_failures = int(group[FAILURE_MODES].sum().sum())
        cat_possible = len(group) * len(FAILURE_MODES)
        per_category[cat] = {
            "records": len(group),
            "total_failures": cat_failures,
            "failure_rate": f"{cat_failures / cat_possible * 100:.1f}%",
        }

    # Per-difficulty failure rates
    per_difficulty = {}
    for diff, group in df.groupby("difficulty"):
        diff_failures = int(group[FAILURE_MODES].sum().sum())
        diff_possible = len(group) * len(FAILURE_MODES)
        per_difficulty[diff] = {
            "records": len(group),
            "total_failures": diff_failures,
            "failure_rate": f"{diff_failures / diff_possible * 100:.1f}%",
        }

    # Load agreement report if it exists
    agreement_path = _LABELS_DIR / "agreement_report.json"
    agreement = {}
    if agreement_path.exists():
        agreement = json.loads(agreement_path.read_text())

    return {
        "dataset_summary": {
            "total_records": total_records,
            "total_possible_failures": total_possible,
            "total_failures": total_failures,
            "overall_failure_rate": f"{total_failures / total_possible * 100:.1f}%",
            "records_with_at_least_one_failure": records_with_failures,
            "records_clean": total_records - records_with_failures,
            "avg_quality_score": round(float(df["quality_score"].mean()), 2),
        },
        "per_mode_failures": per_mode,
        "per_category": per_category,
        "per_difficulty": per_difficulty,
        "inter_rater_agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Orchestrator — runs all analysis and saves outputs
# ---------------------------------------------------------------------------

def run_full_analysis(
    batch_file: str = "batch_v1.json",
    labels_file: str = "llm_labels.csv",
) -> dict:
    """Run the complete analysis pipeline: DataFrame + 6 charts + metrics.

    Args:
        batch_file: Which batch JSON to analyze.
        labels_file: Which labels CSV to use.

    Returns:
        The metrics dict (also saved to results/metrics.json).
    """
    print("Building analysis DataFrame...")
    df = build_analysis_dataframe(batch_file, labels_file)
    print(f"  {len(df)} records, {(df['total_failures'] > 0).sum()} with failures")

    print("\nGenerating charts...")
    charts = [
        ("failure_heatmap", plot_failure_heatmap),
        ("failure_frequency", plot_failure_frequency),
        ("failure_correlation", plot_failure_correlation),
        ("category_failures", plot_category_failures),
        ("difficulty_failures", plot_difficulty_failures),
        ("agreement_matrix", plot_agreement_matrix),
    ]
    for name, fn in charts:
        try:
            if name == "agreement_matrix":
                path = fn()
            else:
                path = fn(df)
            print(f"  Saved {name} -> {path}")
        except Exception as exc:
            print(f"  WARN: {name} failed: {exc}")

    print("\nComputing metrics...")
    metrics = compute_metrics(df)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = _RESULTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics -> {metrics_path}")

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    metrics = run_full_analysis()
    print("\n=== Summary Metrics ===")
    print(json.dumps(metrics, indent=2))
