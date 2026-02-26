from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.normalizer import SkillNormalizer
from src.schemas import GeneratedJob, GeneratedResume

_PROJECT_ROOT = Path(__file__).parent.parent
console = Console()


def _load_latest_jsonl(directory: Path, prefix: str) -> list[dict]:
    """Load the most recent JSONL file matching prefix."""
    files = sorted(directory.glob(f"{prefix}*.jsonl"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No {prefix}*.jsonl files in {directory}")
    records = []
    with open(files[0]) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main() -> None:
    """Spot-check 5 resume-job pairs for quality."""
    gen_dir = _PROJECT_ROOT / "data" / "generated"
    normalizer = SkillNormalizer()

    # Load data
    jobs_raw = _load_latest_jsonl(gen_dir, "jobs_")
    resumes_raw = _load_latest_jsonl(gen_dir, "resumes_")
    pairs_raw = _load_latest_jsonl(gen_dir, "pairs_")

    # Build lookup maps
    jobs = {j["trace_id"]: GeneratedJob.model_validate(j) for j in jobs_raw}
    resumes = {r["trace_id"]: GeneratedResume.model_validate(r) for r in resumes_raw}

    console.print(f"Loaded {len(jobs)} jobs, {len(resumes)} resumes, {len(pairs_raw)} pairs")

    # Pick 1 pair per fit level (5 total)
    fit_levels_seen: dict[str, dict] = {}
    for pair_data in pairs_raw:
        fl = pair_data["fit_level"]
        if fl not in fit_levels_seen:
            fit_levels_seen[fl] = pair_data
        if len(fit_levels_seen) >= 5:
            break

    # Display table
    table = Table(title="Sanity Check: 5 Sample Pairs (1 per Fit Level)")
    table.add_column("Fit Level", style="bold")
    table.add_column("Job Title")
    table.add_column("Candidate Name")
    table.add_column("Resume Skills")
    table.add_column("Job Req Skills")
    table.add_column("Overlap", style="cyan")

    for fit_level, pair_data in fit_levels_seen.items():
        job = jobs.get(pair_data["job_trace_id"])
        resume = resumes.get(pair_data["resume_trace_id"])

        if not job or not resume:
            console.print(f"[red]Missing data for pair {pair_data['pair_id'][:8]}[/red]")
            continue

        resume_skills = normalizer.normalize_set([s.name for s in resume.resume.skills])
        job_skills = normalizer.normalize_set(job.job.requirements.required_skills)
        overlap = resume_skills & job_skills
        union = resume_skills | job_skills
        jaccard = len(overlap) / len(union) if union else 0.0

        table.add_row(
            fit_level,
            job.job.title[:40],
            resume.resume.contact_info.name[:25],
            str(len(resume_skills)),
            str(len(job_skills)),
            f"{len(overlap)}/{len(union)} ({jaccard:.0%})",
        )

    console.print(table)
    console.print(
        "\n[bold]Expected:[/bold] Excellent > Good > Partial > Poor > Mismatch in overlap %"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
