from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

from src.generator import _create_client, generate_all_jobs, generate_all_resumes
from src.schemas import GeneratedJob, GeneratedResume
from src.validator import ValidationTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()

_PROJECT_ROOT = Path(__file__).parent.parent

# WHY: 2 industries × 1 job = 2 jobs × 5 resumes = 10 pairs — validates full pipeline fast
_DRY_RUN_INDUSTRIES = 2
_DRY_RUN_JOBS_PER_INDUSTRY = 1


def write_stats_from_generated_files(stats_path: Path | None = None) -> Path:
    """
    Derive validation stats from the existing generated JSONL files — no API calls.

    WHY this function: saves_stats() is called inside main() which needs a live client.
    --stats-only lets us regenerate the artifact after schema changes without re-running
    the expensive (even if cached) generation loop.
    """
    generated_dir = _PROJECT_ROOT / "data" / "generated"
    stats_path = stats_path or (_PROJECT_ROOT / "data" / "validated" / "validation_stats.json")

    tracker = ValidationTracker()

    # WHY max(key=stat().st_size): partial dry-run files share the same glob pattern but
    # have fewer records. Selecting by file size picks the canonical full-run artifact.
    jobs_candidates = list(generated_dir.glob("jobs_*.jsonl"))
    resumes_candidates = list(generated_dir.glob("resumes_*.jsonl"))

    if jobs_candidates:
        jobs_file = max(jobs_candidates, key=lambda p: p.stat().st_size)
        for line in jobs_file.read_text().splitlines():
            line = line.strip()
            if line:
                gj = GeneratedJob.model_validate_json(line)
                tracker.record_success("JobDescription", gj.trace_id)

    if resumes_candidates:
        resumes_file = max(resumes_candidates, key=lambda p: p.stat().st_size)
        for line in resumes_file.read_text().splitlines():
            line = line.strip()
            if line:
                gr = GeneratedResume.model_validate_json(line)
                tracker.record_success("Resume", gr.trace_id)

    tracker.save_stats(stats_path)

    stats = tracker.get_stats()
    console.print(f"  total={stats['total']}, valid={stats['success_count']}, "
                  f"invalid={stats['failure_count']}, "
                  f"success_rate={stats['success_rate']:.1%}")
    console.print(f"  Written to [cyan]{stats_path}[/cyan]")
    return stats_path


def main() -> None:
    """
    End-to-end generation pipeline:
    1. Generate 50 jobs (5 per industry × 10 industries)
    2. Generate 250 resumes (5 per job: 1 each of excellent/good/partial/poor/mismatch)
    3. Track validation outcomes
    4. Save validation stats

    Use --dry-run for quick validation: 2 jobs × 5 resumes = 10 pairs.
    """
    dry_run = "--dry-run" in sys.argv
    stats_only = "--stats-only" in sys.argv

    # WHY early exit: --stats-only reads existing JSONL files and rewrites the stats
    # artifact without touching the API — useful after schema changes.
    if stats_only:
        console.print("[bold green]P4 Resume Coach — Stats-Only Mode[/bold green]")
        console.print("[yellow]Reading existing generated files (no API calls)...[/yellow]")
        write_stats_from_generated_files()
        return

    console.print("[bold green]P4 Resume Coach — Generation Pipeline[/bold green]")
    if dry_run:
        console.print(
            "[yellow]DRY RUN: 2 industries × 1 job = 2 jobs × 5 resumes = 10 pairs[/yellow]"
        )
    console.print(f"Started at {datetime.now().isoformat()}")

    client = _create_client()
    tracker = ValidationTracker()

    # Phase 1: Generate jobs
    count_per_industry = _DRY_RUN_JOBS_PER_INDUSTRY if dry_run else 5
    max_industries = _DRY_RUN_INDUSTRIES if dry_run else None
    expected_jobs = (_DRY_RUN_INDUSTRIES if dry_run else 10) * count_per_industry
    console.print(f"\n[bold]Phase 1: Generating {expected_jobs} job descriptions...[/bold]")

    jobs, job_api_calls = generate_all_jobs(
        client,
        count_per_industry=count_per_industry,
        max_industries=max_industries,
    )
    console.print(f"  Jobs generated: {len(jobs)}")

    for job in jobs:
        tracker.record_success("JobDescription", job.trace_id)

    # Phase 2: Generate resumes
    expected_resumes = len(jobs) * 5
    console.print(
        f"\n[bold]Phase 2: Generating {expected_resumes} resumes "
        f"({len(jobs)} jobs × 5 fit levels)...[/bold]"
    )
    resumes, pairs, resume_api_calls = generate_all_resumes(client, jobs)
    console.print(f"  Resumes generated: {len(resumes)}")
    console.print(f"  Pairs created: {len(pairs)}")

    for resume in resumes:
        tracker.record_success("Resume", resume.trace_id)

    # Phase 3: Save validation stats
    stats_path = _PROJECT_ROOT / "data" / "validated" / "validation_stats.json"
    tracker.save_stats(stats_path)

    # Summary
    stats = tracker.get_stats()
    total_api_calls = job_api_calls + resume_api_calls
    total_records = len(jobs) + len(resumes)
    # WHY: Cache hits = records that didn't need an API call (prompt already cached)
    cache_hits = total_records - total_api_calls

    console.print("\n[bold]Pipeline Summary[/bold]")
    console.print(f"  Total records: {stats['total']}")
    console.print(f"  Success rate: {stats['success_rate']:.1%}")
    console.print(f"  Jobs: {len(jobs)}")
    console.print(f"  Resumes: {len(resumes)}")
    console.print(f"  Pairs: {len(pairs)}")
    if total_records > 0:
        console.print(
            f"  Cache hits: {cache_hits}/{total_records} ({cache_hits / total_records:.0%})"
        )
    console.print(f"  API calls: {total_api_calls}")
    console.print(f"\nCompleted at {datetime.now().isoformat()}")


if __name__ == "__main__":  # pragma: no cover
    main()
