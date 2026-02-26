"""
run_labeling.py — Batch rule-based failure labeling for all 250 resume-job pairs.

No API calls. Reads three generated data files, runs label_pair() on each pair,
writes failure_labels.jsonl, and prints a Rich summary table.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table

from .labeler import label_pair
from .normalizer import SkillNormalizer
from .schemas import (
    FitLevel,
    GeneratedJob,
    GeneratedResume,
    ResumeJobPair,
)

# ---------------------------------------------------------------------------
# File paths (exact filenames from Day 1 generation run)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"
GENERATED_DIR = DATA_DIR / "generated"
LABELED_DIR = DATA_DIR / "labeled"

JOBS_FILE = GENERATED_DIR / "jobs_20260225_141615.jsonl"
RESUMES_FILE = GENERATED_DIR / "resumes_20260225_142052.jsonl"
PAIRS_FILE = GENERATED_DIR / "pairs_20260225_142052.jsonl"

OUTPUT_FILE = LABELED_DIR / "failure_labels.jsonl"

# Failure mode column names for the summary table
FAILURE_COLS = [
    "experience_mismatch",
    "seniority_mismatch",
    "missing_core_skills",
    "has_hallucinations",
    "has_awkward_language",
]

console = Console()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_jobs(path: Path) -> dict[str, GeneratedJob]:
    """Load all jobs into a dict keyed by trace_id."""
    jobs: dict[str, GeneratedJob] = {}
    with path.open() as f:
        for line in f:
            gj = GeneratedJob.model_validate_json(line.strip())
            jobs[gj.trace_id] = gj
    return jobs


def load_resumes(path: Path) -> dict[str, GeneratedResume]:
    """Load all resumes into a dict keyed by trace_id."""
    resumes: dict[str, GeneratedResume] = {}
    with path.open() as f:
        for line in f:
            gr = GeneratedResume.model_validate_json(line.strip())
            resumes[gr.trace_id] = gr
    return resumes


def load_pairs(path: Path) -> list[ResumeJobPair]:
    """Load all 250 resume-job pairs."""
    pairs: list[ResumeJobPair] = []
    with path.open() as f:
        for line in f:
            pairs.append(ResumeJobPair.model_validate_json(line.strip()))
    return pairs


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run() -> None:
    console.print("[bold cyan]P4 Day 2 — Failure Labeling Pipeline[/bold cyan]")
    console.print(f"Jobs file:    {JOBS_FILE.name}")
    console.print(f"Resumes file: {RESUMES_FILE.name}")
    console.print(f"Pairs file:   {PAIRS_FILE.name}\n")

    # Load all data
    console.print("[yellow]Loading data...[/yellow]")
    jobs = load_jobs(JOBS_FILE)
    resumes = load_resumes(RESUMES_FILE)
    pairs = load_pairs(PAIRS_FILE)
    console.print(f"  Loaded {len(jobs)} jobs, {len(resumes)} resumes, {len(pairs)} pairs\n")

    # WHY shared normalizer: avoids re-instantiating the alias dict 250 times
    normalizer = SkillNormalizer()

    # Ensure output directory exists
    LABELED_DIR.mkdir(parents=True, exist_ok=True)

    # Label all pairs
    console.print("[yellow]Labeling pairs...[/yellow]")
    all_labels = []
    errors: list[str] = []

    for pair in pairs:
        try:
            gen_resume = resumes[pair.resume_trace_id]
            gen_job = jobs[pair.job_trace_id]
            labels = label_pair(gen_resume.resume, gen_job.job, pair.pair_id, normalizer)
            all_labels.append((pair.fit_level, labels))
        except KeyError as e:
            errors.append(f"Missing trace_id {e} for pair {pair.pair_id}")
        except Exception as e:
            errors.append(f"Error labeling {pair.pair_id}: {e}")

    if errors:
        console.print(f"[red]Errors: {len(errors)}[/red]")
        for err in errors:
            console.print(f"  [red]{err}[/red]")

    console.print(f"  Labeled {len(all_labels)} pairs ({len(errors)} errors)\n")

    # Write output
    console.print("[yellow]Writing failure_labels.jsonl...[/yellow]")
    with OUTPUT_FILE.open("w") as f:
        for _fit_level, labels in all_labels:
            f.write(labels.model_dump_json() + "\n")
    console.print(f"  Written to {OUTPUT_FILE}\n")

    # ---------------------------------------------------------------------------
    # Summary table: failure rates per fit level
    # ---------------------------------------------------------------------------

    # Accumulate stats by fit level
    # WHY dict of dict: allows indexing by (fit_level, column_name) without a DataFrame
    FitLevel_ORDER = [
        FitLevel.EXCELLENT,
        FitLevel.GOOD,
        FitLevel.PARTIAL,
        FitLevel.POOR,
        FitLevel.MISMATCH,
    ]

    stats: dict[str, dict] = {fl.value: {"count": 0, "jaccard_sum": 0.0, **{c: 0 for c in FAILURE_COLS}} for fl in FitLevel_ORDER}

    for fit_level, labels in all_labels:
        s = stats[fit_level.value]
        s["count"] += 1
        s["jaccard_sum"] += labels.skills_overlap
        for col in FAILURE_COLS:
            if getattr(labels, col):
                s[col] += 1

    # Failure mode rates table
    table = Table(title="Failure Mode Rates by Fit Level", show_lines=True)
    table.add_column("Fit Level", style="bold")
    table.add_column("N")
    table.add_column("Avg Jaccard", justify="right")
    for col in FAILURE_COLS:
        table.add_column(col.replace("_", " ").title(), justify="right")

    for fl in FitLevel_ORDER:
        s = stats[fl.value]
        n = s["count"]
        if n == 0:
            continue
        avg_j = s["jaccard_sum"] / n
        row = [fl.value, str(n), f"{avg_j:.3f}"]
        for col in FAILURE_COLS:
            pct = s[col] / n * 100
            row.append(f"{s[col]} ({pct:.0f}%)")
        table.add_row(*row)

    console.print(table)

    # Overall failure mode counts
    overall_table = Table(title="Overall Failure Mode Counts (250 pairs)", show_lines=True)
    overall_table.add_column("Failure Mode", style="bold")
    overall_table.add_column("Count", justify="right")
    overall_table.add_column("Rate", justify="right")

    total = len(all_labels)
    for col in FAILURE_COLS:
        count = sum(s[col] for s in stats.values())
        overall_table.add_row(col.replace("_", " ").title(), str(count), f"{count/total*100:.1f}%")

    console.print(overall_table)
    console.print(f"\n[bold green]Done! {total} failure labels written to {OUTPUT_FILE}[/bold green]")


if __name__ == "__main__":
    run()
