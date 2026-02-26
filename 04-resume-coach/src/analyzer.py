"""
analyzer.py — DataFrame construction, 9 charts, A/B chi-squared, and pipeline summary.

No API calls. Reads 5 data sources, joins them on pair_id, and produces:
  - results/charts/*.png  (9 chart files)
  - results/pipeline_summary.json

Run: uv run python -m src.analyzer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

# WHY Agg backend: Non-interactive server-safe renderer. Must be set before pyplot import
# to prevent "no display" crashes in headless environments.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy import stats  # noqa: E402

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"
_LABELED_DIR = _PROJECT_ROOT / "data" / "labeled"
_CORRECTED_DIR = _PROJECT_ROOT / "data" / "corrected"
_CHARTS_DIR = _PROJECT_ROOT / "results" / "charts"
_RESULTS_DIR = _PROJECT_ROOT / "results"

# Exact filenames from Day 1 generation run
_JOBS_FILE = _GENERATED_DIR / "jobs_20260225_141615.jsonl"
_RESUMES_FILE = _GENERATED_DIR / "resumes_20260225_142052.jsonl"
_PAIRS_FILE = _GENERATED_DIR / "pairs_20260225_142052.jsonl"
_FAILURE_LABELS_FILE = _LABELED_DIR / "failure_labels.jsonl"
_JUDGE_RESULTS_FILE = _LABELED_DIR / "judge_results.jsonl"
_CORRECTION_SUMMARY_FILE = _CORRECTED_DIR / "correction_summary.json"

# Canonical failure mode columns (rule-based)
_FAILURE_COLS = [
    "experience_mismatch",
    "seniority_mismatch",
    "missing_core_skills",
    "has_hallucinations",
    "has_awkward_language",
]

# Canonical fit level ordering for charts (worst to best reversed for display)
_FIT_ORDER = ["excellent", "good", "partial", "poor", "mismatch"]

# Seaborn theme applied once at module level — all charts inherit this
sns.set_theme(style="whitegrid", font_scale=1.1)


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------


def build_analysis_dataframe() -> pd.DataFrame:
    """
    Join 5 data sources on pair_id into one analysis DataFrame.

    WHY join rather than query each file per chart: Joining once amortizes file I/O
    and keeps chart functions pure (they receive a df, not file paths).
    DataFrame columns:
      - pair_id, fit_level (from pairs)
      - template (writing_style stripped of 'v1-' prefix from resumes)
      - is_niche, industry (from jobs)
      - all failure_labels fields (skills_overlap, 5 failure flags, seniority levels, etc.)
      - overall_quality_score, judge_has_hallucinations, judge_has_awkward_language (from judge)
      - total_flags (sum of 5 boolean failure flags)
    """
    # --- Load pairs ---
    pairs_rows = []
    with _PAIRS_FILE.open() as f:
        for line in f:
            d = json.loads(line.strip())
            pairs_rows.append({
                "pair_id": d["pair_id"],
                "fit_level": d["fit_level"],
                "resume_trace_id": d["resume_trace_id"],
                "job_trace_id": d["job_trace_id"],
            })
    pairs_df = pd.DataFrame(pairs_rows)

    # --- Load resumes (template column from writing_style) ---
    resume_rows = []
    with _RESUMES_FILE.open() as f:
        for line in f:
            d = json.loads(line.strip())
            resume_rows.append({
                "resume_trace_id": d["trace_id"],
                # WHY writing_style not prompt_template: writing_style is the clean label
                # ('technical', 'formal', etc.) while prompt_template is 'v1-technical'.
                # Use writing_style directly as the template column.
                "template": d["writing_style"],
            })
    resumes_df = pd.DataFrame(resume_rows)

    # --- Load jobs (is_niche + industry) ---
    job_rows = []
    with _JOBS_FILE.open() as f:
        for line in f:
            d = json.loads(line.strip())
            job_rows.append({
                "job_trace_id": d["trace_id"],
                "is_niche": d["is_niche_role"],
                "industry": d["job"].get("company", {}).get("industry", "Unknown"),
            })
    jobs_df = pd.DataFrame(job_rows)

    # --- Load failure labels ---
    label_rows = []
    with _FAILURE_LABELS_FILE.open() as f:
        for line in f:
            d = json.loads(line.strip())
            label_rows.append(d)
    labels_df = pd.DataFrame(label_rows)

    # --- Load judge results ---
    judge_rows = []
    with _JUDGE_RESULTS_FILE.open() as f:
        for line in f:
            d = json.loads(line.strip())
            judge_rows.append({
                "pair_id": d["pair_id"],
                "overall_quality_score": d["overall_quality_score"],
                # WHY rename: avoid column name collision with rule-based flags
                "judge_has_hallucinations": d["has_hallucinations"],
                "judge_has_awkward_language": d["has_awkward_language"],
            })
    judge_df = pd.DataFrame(judge_rows)

    # --- Join all sources on pair_id ---
    df = (
        pairs_df
        .merge(resumes_df, on="resume_trace_id", how="left")
        .merge(jobs_df, on="job_trace_id", how="left")
        .merge(labels_df, on="pair_id", how="left")
        .merge(judge_df, on="pair_id", how="left")
    )

    # Computed column: sum of all 5 boolean failure flags (0-5 range)
    df["total_flags"] = df[_FAILURE_COLS].sum(axis=1)

    logger.info("DataFrame built: %d rows, %d columns", len(df), len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def _save_fig(fig: plt.Figure, filename: str) -> Path:
    """Save figure to results/charts/, return path."""
    _CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _CHARTS_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Chart saved: %s", output_path.name)
    return output_path


# ---------------------------------------------------------------------------
# 9 Chart functions
# ---------------------------------------------------------------------------


def plot_failure_correlation(df: pd.DataFrame) -> Path:
    """
    Chart #1: Heatmap of pairwise Pearson correlations between the 5 failure flags.

    WHY coolwarm: Diverging colormap highlights positive (red) and negative (blue)
    correlations symmetrically around zero (white).
    """
    corr = df[_FAILURE_COLS].astype(float).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        xticklabels=[c.replace("_", "\n") for c in _FAILURE_COLS],
        yticklabels=[c.replace("_", "\n") for c in _FAILURE_COLS],
    )
    ax.set_title("Failure Mode Correlations", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    return _save_fig(fig, "failure_correlation.png")


def plot_failure_rates_by_fit(df: pd.DataFrame) -> Path:
    """
    Chart #2: Grouped bar chart — failure rate for each mode, grouped by fit level.

    WHY melt: seaborn barplot expects long-form data (one row per data point).
    Wide-form (one column per failure mode) must be melted first.
    """
    # Convert boolean flags to float (0/1) for rate calculation
    rate_df = df[["fit_level"] + _FAILURE_COLS].copy()
    for col in _FAILURE_COLS:
        rate_df[col] = rate_df[col].astype(float)

    # Melt to long form: fit_level | failure_mode | value
    melted = rate_df.melt(id_vars="fit_level", var_name="failure_mode", value_name="rate")

    # Aggregate: mean = failure rate per group
    grouped = melted.groupby(["fit_level", "failure_mode"], as_index=False)["rate"].mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=grouped,
        x="failure_mode",
        y="rate",
        hue="fit_level",
        hue_order=_FIT_ORDER,
        ax=ax,
        palette="tab10",
    )
    ax.set_title("Failure Rates by Fit Level", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Failure Mode")
    ax.set_ylabel("Rate (0–1)")
    ax.set_xticklabels([c.replace("_", "\n") for c in _FAILURE_COLS])
    ax.legend(title="Fit Level", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return _save_fig(fig, "failure_by_fit_level.png")


def plot_failure_rates_by_template(df: pd.DataFrame) -> Path:
    """
    Chart #3: Grouped bar chart — failure rate per mode, grouped by resume template.

    5 templates × 5 failure modes = 25 bars total.
    """
    rate_df = df[["template"] + _FAILURE_COLS].copy()
    for col in _FAILURE_COLS:
        rate_df[col] = rate_df[col].astype(float)

    melted = rate_df.melt(id_vars="template", var_name="failure_mode", value_name="rate")
    grouped = melted.groupby(["template", "failure_mode"], as_index=False)["rate"].mean()

    templates = sorted(df["template"].unique())
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=grouped,
        x="failure_mode",
        y="rate",
        hue="template",
        hue_order=templates,
        ax=ax,
        palette="Set2",
    )
    ax.set_title("Failure Rates by Resume Template", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Failure Mode")
    ax.set_ylabel("Rate (0–1)")
    ax.set_xticklabels([c.replace("_", "\n") for c in _FAILURE_COLS])
    ax.legend(title="Template", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return _save_fig(fig, "failure_by_template.png")


def plot_niche_vs_standard(df: pd.DataFrame) -> Path:
    """
    Chart #4: Overall failure rate (any flag) for niche vs standard roles.

    WHY 'any_failure' derived column: a single binary target is cleaner than
    5 separate bars when the question is just "does niche-ness cause more failures".
    """
    df2 = df.copy()
    df2["any_failure"] = (df2[_FAILURE_COLS].any(axis=1)).astype(float)
    df2["role_type"] = df2["is_niche"].map({True: "Niche", False: "Standard"})

    # Per-failure-mode rates by niche status
    rate_df = df2[["role_type"] + _FAILURE_COLS].copy()
    for col in _FAILURE_COLS:
        rate_df[col] = rate_df[col].astype(float)

    melted = rate_df.melt(id_vars="role_type", var_name="failure_mode", value_name="rate")
    grouped = melted.groupby(["role_type", "failure_mode"], as_index=False)["rate"].mean()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=grouped,
        x="failure_mode",
        y="rate",
        hue="role_type",
        hue_order=["Niche", "Standard"],
        ax=ax,
        palette={"Niche": "#e74c3c", "Standard": "#3498db"},
    )
    ax.set_title("Failure Rates: Niche vs Standard Roles", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Failure Mode")
    ax.set_ylabel("Rate (0–1)")
    ax.set_xticklabels([c.replace("_", "\n") for c in _FAILURE_COLS])
    ax.legend(title="Role Type")
    fig.tight_layout()
    return _save_fig(fig, "niche_vs_standard.png")


def plot_validation_summary() -> Path:
    """
    Chart #5: Annotated bar chart showing 300/300 records validated with 0 errors.

    WHY single annotated bar (not heatmap): Day 1 validation produced 0 failures.
    An empty heatmap conveys nothing. A 100% bar with a bold annotation communicates
    "the pipeline is clean" which IS the result worth showing.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ["Jobs (50)", "Resumes (250)"]
    rates = [100, 100]
    bars = ax.bar(categories, rates, color=["#2ecc71", "#27ae60"], width=0.4, edgecolor="white")

    # Bold annotation on each bar
    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{rate}% Valid",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="white",
        )

    ax.set_ylim(0, 120)
    ax.set_ylabel("Validation Rate (%)")
    ax.set_title(
        "Day 1 Validation Summary\n300/300 records valid — 0 field errors",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.axhline(100, color="black", linestyle="--", linewidth=1, alpha=0.4)
    ax.set_yticks([0, 25, 50, 75, 100])
    fig.tight_layout()
    return _save_fig(fig, "validation_summary.png")


def plot_skills_overlap_distribution(df: pd.DataFrame) -> Path:
    """
    Chart #6: Box plot of Jaccard skills overlap by fit level.

    WHY boxplot: shows median, IQR, and outliers — more informative than a bar.
    This is the most important chart: it proves the Jaccard gradient is real and not
    just an artifact of the average (boxes should not overlap for adjacent fit levels).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # WHY hue=fit_level + legend=False: seaborn >=0.13 deprecates palette without hue.
    # Setting hue to the same column as x gives per-bar colors with no redundant legend.
    sns.boxplot(
        data=df,
        x="fit_level",
        y="skills_overlap",
        order=_FIT_ORDER,
        hue="fit_level",
        hue_order=_FIT_ORDER,
        legend=False,
        ax=ax,
        palette="Blues_r",
    )
    ax.set_title(
        "Skills Overlap (Jaccard) by Fit Level",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Fit Level")
    ax.set_ylabel("Jaccard Similarity (0–1)")
    # Reference lines
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6, label="0.5 threshold")
    ax.legend()
    fig.tight_layout()
    return _save_fig(fig, "skills_overlap_distribution.png")


def plot_hallucination_by_seniority(df: pd.DataFrame) -> Path:
    """
    Chart #7: Rule-based hallucination rate by candidate seniority level (0-4).

    WHY rule-based not judge: shows whether the labeler's hallucination detector
    correlates with seniority level — junior candidates faking senior skills.
    Seniority 0=intern/entry, 4=executive.
    """
    seniority_labels = {0: "Entry\n(0)", 1: "Mid\n(1)", 2: "Senior\n(2)", 3: "Lead\n(3)", 4: "Exec\n(4)"}

    df2 = df.copy()
    df2["has_hallucinations"] = df2["has_hallucinations"].astype(float)
    df2["seniority_label"] = df2["seniority_level_resume"].map(seniority_labels)

    grouped = df2.groupby("seniority_level_resume")["has_hallucinations"].mean().reset_index()
    grouped["seniority_label"] = grouped["seniority_level_resume"].map(seniority_labels)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        grouped["seniority_label"],
        grouped["has_hallucinations"],
        color="#e67e22",
        edgecolor="white",
        linewidth=0.5,
    )
    # Annotate bar heights
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title(
        "Hallucination Rate by Candidate Seniority Level",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Seniority Level")
    ax.set_ylabel("Hallucination Rate (rule-based)")
    ax.set_ylim(0, min(1.1, grouped["has_hallucinations"].max() + 0.15))
    fig.tight_layout()
    return _save_fig(fig, "hallucination_by_seniority.png")


def plot_correction_success() -> Path:
    """
    Chart #8: Horizontal bar chart of correction pipeline results from correction_summary.json.

    WHY barh not stacked: two bars (corrected, failed) side-by-side are immediately
    readable without needing a legend for a simple 2-category result.
    """
    with _CORRECTION_SUMMARY_FILE.open() as f:
        summary = json.load(f)

    total = summary["total_invalid"]
    corrected = summary["total_corrected"]
    failed = total - corrected
    rate = summary["correction_rate"]
    avg_attempts = summary["avg_attempts_per_success"]

    fig, ax = plt.subplots(figsize=(8, 4))
    categories = ["Corrected", "Failed"]
    values = [corrected, failed]
    colors = ["#2ecc71", "#e74c3c"]

    bars = ax.barh(categories, values, color=colors, edgecolor="white", height=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlim(0, total + 1)
    ax.set_xlabel("Number of Records")
    ax.set_title(
        f"LLM Correction Pipeline Results\n"
        f"{corrected}/{total} corrected ({rate:.0%}) — avg {avg_attempts:.1f} attempt(s) each",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    fig.tight_layout()
    return _save_fig(fig, "correction_success.png")


def plot_judge_vs_rules_agreement(df: pd.DataFrame) -> Path:
    """
    Chart #9: 2×2 confusion matrix heatmaps — judge vs rule-based signals.

    WHY two side-by-side heatmaps: one for hallucinations, one for awkward language.
    The agreement between independent signals (rule-based vs LLM judge) validates
    both detectors. High off-diagonal counts → disagreement → investigate.

    Rows = rule-based label, Cols = judge label.
    Cell values are counts, annotated as strings.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, rule_col, judge_col, title in [
        (axes[0], "has_hallucinations", "judge_has_hallucinations", "Hallucinations"),
        (axes[1], "has_awkward_language", "judge_has_awkward_language", "Awkward Language"),
    ]:
        # Build 2×2 confusion matrix: rule_col (rows) × judge_col (cols)
        ct = pd.crosstab(
            df[rule_col].astype(bool),
            df[judge_col].astype(bool),
            rownames=["Rule-based"],
            colnames=["Judge (GPT-4o)"],
        )
        # WHY reindex: if all values are True or all False, crosstab omits the missing
        # category. reindex ensures the 2×2 matrix always has both rows and columns.
        ct = ct.reindex(index=[False, True], columns=[False, True], fill_value=0)

        # WHY iloc not loc: pd.crosstab with bool columns stores index as Python bool,
        # but loc(False, False) can trigger ambiguous length-2 tuple lookups. iloc[0,0]
        # is unambiguous: row 0 = False, row 1 = True (matching the reindex order above).
        agree = (ct.iloc[0, 0] + ct.iloc[1, 1]) / ct.values.sum()

        sns.heatmap(
            ct,
            ax=ax,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            xticklabels=["No (Judge)", "Yes (Judge)"],
            yticklabels=["No (Rule)", "Yes (Rule)"],
        )
        ax.set_title(f"{title}\nAgreement: {agree:.1%}", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Judge vs Rule-Based Agreement (250 pairs)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save_fig(fig, "judge_vs_rules_agreement.png")


# ---------------------------------------------------------------------------
# A/B chi-squared test
# ---------------------------------------------------------------------------


def compute_template_ab_test(df: pd.DataFrame) -> dict:
    """
    Chi-squared test: are failure rates significantly different across resume templates?

    WHY chi-squared: Tests independence between template (categorical) and failure
    outcome (binary). If p < 0.05, template choice is a statistically significant
    predictor of failure — actionable for resume advice.

    Contingency table: rows = templates (5), cols = [any_failure=True, any_failure=False].
    """
    df2 = df.copy()
    # "Failed" = any of the 5 rule-based failure flags triggered
    df2["any_failure"] = df2[_FAILURE_COLS].any(axis=1)

    ct = pd.crosstab(df2["template"], df2["any_failure"])
    # WHY reindex not ct[[False, True]]: passing a 2-element list of booleans to []
    # triggers pandas boolean-mask logic (expects len == len(df)), not column selection.
    ct = ct.reindex(columns=[False, True], fill_value=0)  # [not_failed, failed]

    chi2, p_value, dof, _expected = stats.chi2_contingency(ct)

    # Best/worst template by raw failure rate
    failure_rates = (ct[True] / ct.sum(axis=1)).to_dict()
    best_template = min(failure_rates, key=failure_rates.get)
    worst_template = max(failure_rates, key=failure_rates.get)

    return {
        "chi_squared_statistic": round(chi2, 4),
        "chi_squared_p_value": round(p_value, 4),
        "degrees_of_freedom": dof,
        "significant": bool(p_value < 0.05),
        "best_template": best_template,
        "worst_template": worst_template,
        "failure_rates_by_template": {k: round(v, 4) for k, v in failure_rates.items()},
    }


# ---------------------------------------------------------------------------
# Pipeline summary
# ---------------------------------------------------------------------------


def generate_pipeline_summary(df: pd.DataFrame, ab_results: dict) -> dict:
    """
    Aggregate statistics from all pipeline stages into one summary JSON.

    WHY save to results/: Separate from data/ (raw/processed) to signal it's
    the final human-readable output. pipeline_summary.json is the portfolio artifact.
    """
    # --- Generation stats ---
    generation = {
        "jobs_generated": int(df["job_trace_id"].nunique()),
        "resumes_generated": len(df),
        "pairs_generated": len(df),
        "validation_rate": 1.0,
        "field_errors": 0,
    }

    # --- Labeling stats ---
    avg_jaccard = df.groupby("fit_level")["skills_overlap"].mean().round(3).to_dict()
    failure_rates = {col: round(float(df[col].astype(float).mean()), 4) for col in _FAILURE_COLS}
    labeling = {
        "pairs_labeled": len(df),
        "failure_mode_rates": failure_rates,
        "avg_jaccard_by_fit_level": avg_jaccard,
    }

    # --- Judge stats ---
    judge = {
        "pairs_evaluated": int(df["overall_quality_score"].notna().sum()),
        "avg_quality_score": round(float(df["overall_quality_score"].mean()), 4),
        "hallucination_rate": round(float(df["judge_has_hallucinations"].astype(float).mean()), 4),
        "awkward_language_rate": round(float(df["judge_has_awkward_language"].astype(float).mean()), 4),
    }

    # --- Correction stats ---
    with _CORRECTION_SUMMARY_FILE.open() as f:
        correction_raw = json.load(f)
    correction = {
        "records_processed": correction_raw["total_invalid"],
        "records_corrected": correction_raw["total_corrected"],
        "correction_rate": correction_raw["correction_rate"],
        "avg_attempts_per_success": correction_raw["avg_attempts_per_success"],
    }

    # --- A/B testing ---
    ab_testing = {
        "test": "chi_squared_independence",
        "null_hypothesis": "Resume template has no effect on failure rate",
        **ab_results,
    }

    summary = {
        "project": "P4 — Resume Coach",
        "day": "Day 2",
        "generation": generation,
        "labeling": labeling,
        "judge": judge,
        "correction": correction,
        "ab_testing": ab_testing,
        "charts_generated": 9,
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _RESULTS_DIR / "pipeline_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    logger.info("Pipeline summary saved: %s", output_path)
    return summary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_all_charts(df: pd.DataFrame) -> list[Path]:
    """Run all 9 chart functions and return their output paths."""
    return [
        plot_failure_correlation(df),        # Chart 1
        plot_failure_rates_by_fit(df),       # Chart 2
        plot_failure_rates_by_template(df),  # Chart 3
        plot_niche_vs_standard(df),          # Chart 4
        plot_validation_summary(),           # Chart 5 (no df needed)
        plot_skills_overlap_distribution(df), # Chart 6
        plot_hallucination_by_seniority(df), # Chart 7
        plot_correction_success(),           # Chart 8 (reads correction_summary.json)
        plot_judge_vs_rules_agreement(df),   # Chart 9
    ]


# ---------------------------------------------------------------------------
# CLI entry point — T2.8+T2.9
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console
    from rich.table import Table

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    console = Console()
    console.print("[bold cyan]P4 Day 2 — Analysis Pipeline[/bold cyan]\n")

    # Build DataFrame
    console.print("[yellow]Building analysis DataFrame...[/yellow]")
    df = build_analysis_dataframe()
    console.print(f"  {len(df)} rows × {len(df.columns)} columns\n")

    # Generate charts
    console.print("[yellow]Generating 9 charts...[/yellow]")
    chart_paths = generate_all_charts(df)
    for i, path in enumerate(chart_paths, 1):
        console.print(f"  [green]Chart {i:2d}[/green]: {path.name}")

    # A/B test
    console.print("\n[yellow]Running chi-squared A/B test...[/yellow]")
    ab_results = compute_template_ab_test(df)
    console.print(f"  χ² = {ab_results['chi_squared_statistic']:.4f}, p = {ab_results['chi_squared_p_value']:.4f}")
    sig = "[green]YES[/green]" if ab_results["significant"] else "[yellow]NO[/yellow]"
    console.print(f"  Significant (p<0.05): {sig}")
    console.print(f"  Best template:  {ab_results['best_template']}")
    console.print(f"  Worst template: {ab_results['worst_template']}")

    # Pipeline summary
    console.print("\n[yellow]Generating pipeline summary...[/yellow]")
    summary = generate_pipeline_summary(df, ab_results)

    # Print labeling stats table
    table = Table(title="Avg Jaccard by Fit Level", show_lines=True)
    table.add_column("Fit Level", style="bold")
    table.add_column("Avg Jaccard", justify="right")
    table.add_column("N Pairs", justify="right")
    for fl in ["excellent", "good", "partial", "poor", "mismatch"]:
        n = int((df["fit_level"] == fl).sum())
        j = df.loc[df["fit_level"] == fl, "skills_overlap"].mean()
        table.add_row(fl, f"{j:.3f}", str(n))
    console.print(table)

    # Print judge stats
    console.print("\n[bold]Judge Stats (GPT-4o):[/bold]")
    console.print(f"  Avg quality score:    {summary['judge']['avg_quality_score']:.3f}")
    console.print(f"  Hallucination rate:   {summary['judge']['hallucination_rate']:.1%}")
    console.print(f"  Awkward language:     {summary['judge']['awkward_language_rate']:.1%}")

    console.print(
        f"\n[bold green]Done! {len(chart_paths)} charts → {_CHARTS_DIR}[/bold green]"
    )
    console.print(f"[bold green]Pipeline summary → {_RESULTS_DIR / 'pipeline_summary.json'}[/bold green]")
