"""
multi_hop.py — Rule-based multi-hop evaluation questions for resume-job pairs.

Multi-hop questions require reasoning across 2+ resume/job sections simultaneously.
Unlike single-field lookups, each question chains evidence from multiple sources:
  Q1: education + job requirements → experience alignment
  Q2: skills + experience titles + years → consistency check
  Q3: career progression (oldest→newest title) → seniority realism
  Q4: Jaccard skills overlap → fit sufficiency for target industry

No API calls — all answers derived deterministically from FailureLabels.
These are designed as evaluation inputs for Day 3 vector store retrieval.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from .schemas import (
    FailureLabels,
    GeneratedJob,
    GeneratedResume,
    JobDescription,
    MultiHopQuestion,
    MultiHopResponse,
    Resume,
    ResumeJobPair,
)

_PROJECT_ROOT = Path(__file__).parent.parent
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"
_LABELED_DIR = _PROJECT_ROOT / "data" / "labeled"
_ANALYSIS_DIR = _PROJECT_ROOT / "data" / "analysis"

# WHY int keys not str: seniority_level_resume/job are stored as 0-4 ints in FailureLabels
_SENIORITY_LABELS: dict[int, str] = {
    0: "entry-level",
    1: "mid-level",
    2: "senior",
    3: "lead/principal",
    4: "executive",
}

# File paths (exact filenames from Day 1 generation run)
_JOBS_FILE = _GENERATED_DIR / "jobs_20260225_141615.jsonl"
_RESUMES_FILE = _GENERATED_DIR / "resumes_20260225_142052.jsonl"
_PAIRS_FILE = _GENERATED_DIR / "pairs_20260225_142052.jsonl"
_LABELS_FILE = _LABELED_DIR / "failure_labels.jsonl"
OUTPUT_FILE = _ANALYSIS_DIR / "multi_hop_questions.jsonl"

# Pairs sampled per fit level (excellent/good/partial/poor/mismatch × 2 = 10 total)
_PAIRS_PER_FIT_LEVEL = 2


# ---------------------------------------------------------------------------
# Individual question generators — 4 distinct reasoning chains
# ---------------------------------------------------------------------------


def _q_education_vs_experience(
    resume: Resume, job: JobDescription, labels: FailureLabels
) -> MultiHopQuestion:
    """
    Q1: Education level vs. job experience requirements.

    WHY multi-hop: requires cross-referencing resume.education (degree level)
    with job.requirements.experience_years AND experience_level. Neither field
    alone is sufficient — both inform the alignment judgment.
    """
    degree = resume.education[0].degree if resume.education else "No formal degree"
    exp_level = job.requirements.experience_level.value
    exp_years_required = job.requirements.experience_years

    question = (
        f"Does this candidate's education ({degree}) align with the job's "
        f"required experience level ({exp_level}) and {exp_years_required}+ years?"
    )

    if labels.experience_mismatch:
        gap = max(0.0, exp_years_required - labels.experience_years_resume)
        answer = (
            f"Partial misalignment. The candidate has {labels.experience_years_resume:.1f} years "
            f"but the role requires {exp_years_required}+. "
            f"Education ({degree}) does not compensate for the {gap:.1f}-year gap."
        )
        assessment = "mismatch"
    else:
        answer = (
            f"Aligned. The candidate's {labels.experience_years_resume:.1f} years meets the "
            f"{exp_years_required}-year requirement, and {degree} is appropriate for {exp_level} roles."
        )
        assessment = "aligned"

    return MultiHopQuestion(
        question=question,
        requires_sections=["education", "requirements"],
        answer=answer,
        assessment=assessment,
    )


def _q_skills_consistency(
    resume: Resume, job: JobDescription, labels: FailureLabels
) -> MultiHopQuestion:
    """
    Q2: Claimed skills vs. job titles vs. experience years — credibility check.

    WHY multi-hop: each hop narrows credibility: skills listed → titles held →
    years worked. All three must align for the profile to be credible.
    """
    top_skills = ", ".join(labels.resume_skills_normalized[:3]) or "none listed"
    job_titles = [exp.title for exp in resume.experience[:3]]
    titles_str = ", ".join(job_titles) if job_titles else "no prior roles"
    years = labels.experience_years_resume

    question = (
        f"Are the claimed skills ({top_skills}) consistent with the "
        f"job titles held ({titles_str}) and {years:.1f} years of experience?"
    )

    if labels.has_hallucinations:
        reasons = (
            "; ".join(labels.hallucination_reasons[:2])
            if labels.hallucination_reasons
            else "skill claims appear inflated for the experience level"
        )
        answer = (
            f"Inconsistency detected. {reasons}. "
            f"The claimed skills do not credibly follow from {years:.1f} years in roles like {titles_str}."
        )
        assessment = "inconsistent"
    elif labels.missing_core_skills:
        missing = ", ".join(labels.missing_skills[:3]) if labels.missing_skills else "core job skills"
        answer = (
            f"Partially consistent. The background ({titles_str}) supports {years:.1f} years of experience, "
            f"but key requirements are unmet: {missing}."
        )
        assessment = "partial"
    else:
        answer = (
            f"Consistent. The claimed skills ({top_skills}) are plausible given "
            f"{years:.1f} years in {titles_str}. No credibility concerns detected."
        )
        assessment = "consistent"

    return MultiHopQuestion(
        question=question,
        requires_sections=["skills", "experience", "requirements"],
        answer=answer,
        assessment=assessment,
    )


def _q_career_progression(
    resume: Resume, job: JobDescription, labels: FailureLabels
) -> MultiHopQuestion:
    """
    Q3: Career progression — first title → most recent title over total years.

    WHY multi-hop: requires reading the full experience timeline (multiple entries)
    to infer trajectory, then comparing inferred seniority to the job requirement.
    """
    experiences = resume.experience
    if len(experiences) >= 2:
        # WHY sort by start_date: experience list order is not guaranteed chronological
        sorted_exp = sorted(experiences, key=lambda e: e.start_date)
        first_title = sorted_exp[0].title
        last_title = sorted_exp[-1].title
    elif experiences:
        first_title = last_title = experiences[0].title
    else:
        first_title = last_title = "no experience listed"

    total_years = labels.experience_years_resume
    resume_seniority = _SENIORITY_LABELS.get(labels.seniority_level_resume, "unknown")
    job_seniority = _SENIORITY_LABELS.get(labels.seniority_level_job, "unknown")

    question = (
        f"Given the career progression from '{first_title}' to '{last_title}' "
        f"over {total_years:.1f} years, is the {resume_seniority} seniority realistic "
        f"for a {job_seniority} role at {job.company.name}?"
    )

    if labels.seniority_mismatch:
        gap = abs(labels.seniority_level_resume - labels.seniority_level_job)
        answer = (
            f"Seniority mismatch ({gap} level(s)). Progression from {first_title} to "
            f"{last_title} over {total_years:.1f} years suggests {resume_seniority}, "
            f"but the role requires {job_seniority}."
        )
        assessment = "mismatch"
    else:
        answer = (
            f"Realistic. The {first_title} → {last_title} progression over "
            f"{total_years:.1f} years is consistent with {resume_seniority} level, "
            f"matching the {job_seniority} requirement."
        )
        assessment = "aligned"

    return MultiHopQuestion(
        question=question,
        requires_sections=["experience"],
        answer=answer,
        assessment=assessment,
    )


def _q_skills_overlap_fit(
    resume: Resume, job: JobDescription, labels: FailureLabels
) -> MultiHopQuestion:
    """
    Q4: Jaccard skills overlap as fit proxy for the target industry.

    WHY multi-hop: combines resume.skills (normalized) + job.requirements.required_skills
    (normalized) + job.company.industry context. The industry matters because a 30%
    overlap might suffice in a generalist role but be inadequate in a specialised one.
    """
    target_industry = job.company.industry
    top_job_skills = ", ".join(labels.job_skills_normalized[:3]) or "unspecified"
    overlap_pct = labels.skills_overlap * 100
    missing = ", ".join(labels.missing_skills[:3]) if labels.missing_skills else "none"

    question = (
        f"Does the candidate's skill set provide sufficient coverage "
        f"({overlap_pct:.0f}% Jaccard overlap) for the core {target_industry} "
        f"requirements ({top_job_skills})?"
    )

    if labels.skills_overlap >= 0.5:
        suffix = f" Minor gaps: {missing}." if labels.missing_skills else " No critical gaps."
        answer = (
            f"Sufficient. {overlap_pct:.0f}% Jaccard overlap indicates strong skill alignment "
            f"for {target_industry}.{suffix}"
        )
        assessment = "sufficient"
    elif labels.skills_overlap >= 0.2:
        answer = (
            f"Partial. {overlap_pct:.0f}% Jaccard overlap shows relevant skills but notable "
            f"gaps in {target_industry} requirements: {missing}."
        )
        assessment = "partial"
    else:
        answer = (
            f"Insufficient. {overlap_pct:.0f}% Jaccard overlap is too low for {target_industry}. "
            f"Key missing skills: {missing}."
        )
        assessment = "insufficient"

    return MultiHopQuestion(
        question=question,
        requires_sections=["skills", "requirements", "company"],
        answer=answer,
        assessment=assessment,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_multi_hop_questions(
    resume: Resume,
    job: JobDescription,
    labels: FailureLabels,
    pair_id: str,
) -> MultiHopResponse:
    """
    Generate 4 multi-hop evaluation questions for one resume-job pair.

    WHY 4 questions: each covers a distinct 2+-hop reasoning chain, exercising
    diverse retrieval patterns for Day 3 vector store evaluation. All answers
    are deterministic (no LLM calls) — reproducible across runs.

    Returns: MultiHopResponse with pair_id, questions list, and processing time.
    """
    start = time.monotonic()
    questions = [
        _q_education_vs_experience(resume, job, labels),
        _q_skills_consistency(resume, job, labels),
        _q_career_progression(resume, job, labels),
        _q_skills_overlap_fit(resume, job, labels),
    ]
    elapsed = time.monotonic() - start

    return MultiHopResponse(
        pair_id=pair_id,
        questions=questions,
        processing_time_seconds=round(elapsed, 4),
    )


# ---------------------------------------------------------------------------
# Loaders (mirrors run_labeling.py pattern)
# ---------------------------------------------------------------------------


def _load_pairs(path: Path) -> list[ResumeJobPair]:
    pairs = []
    with path.open() as f:
        for line in f:
            pairs.append(ResumeJobPair.model_validate_json(line.strip()))
    return pairs


def _load_jobs(path: Path) -> dict[str, GeneratedJob]:
    jobs: dict[str, GeneratedJob] = {}
    with path.open() as f:
        for line in f:
            gj = GeneratedJob.model_validate_json(line.strip())
            jobs[gj.trace_id] = gj
    return jobs


def _load_resumes(path: Path) -> dict[str, GeneratedResume]:
    resumes: dict[str, GeneratedResume] = {}
    with path.open() as f:
        for line in f:
            gr = GeneratedResume.model_validate_json(line.strip())
            resumes[gr.trace_id] = gr
    return resumes


def _load_labels(path: Path) -> dict[str, FailureLabels]:
    labels: dict[str, FailureLabels] = {}
    with path.open() as f:
        for line in f:
            fl = FailureLabels.model_validate_json(line.strip())
            labels[fl.pair_id] = fl
    return labels


# ---------------------------------------------------------------------------
# CLI entry point — T2.10: generate multi-hop questions for 10 pairs (2 per fit level)
# ---------------------------------------------------------------------------


def run() -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("[bold cyan]P4 Day 2 — Multi-Hop Evaluation Questions[/bold cyan]\n")

    console.print("[yellow]Loading data...[/yellow]")
    pairs = _load_pairs(_PAIRS_FILE)
    jobs = _load_jobs(_JOBS_FILE)
    resumes = _load_resumes(_RESUMES_FILE)
    labels = _load_labels(_LABELS_FILE)
    console.print(
        f"  Loaded {len(pairs)} pairs, {len(jobs)} jobs, "
        f"{len(resumes)} resumes, {len(labels)} labels\n"
    )

    # Group pairs by fit level; sort by pair_id for deterministic selection
    pairs_by_fit: dict[str, list[ResumeJobPair]] = defaultdict(list)
    for pair in sorted(pairs, key=lambda p: p.pair_id):
        pairs_by_fit[pair.fit_level.value].append(pair)

    selected: list[ResumeJobPair] = []
    for fit_val, fit_pairs in pairs_by_fit.items():
        n = min(_PAIRS_PER_FIT_LEVEL, len(fit_pairs))
        selected.extend(fit_pairs[:n])
        console.print(f"  {fit_val}: selected {n}/{len(fit_pairs)} pairs")

    console.print(f"\n  Total selected: {len(selected)} pairs\n")

    # Generate questions
    console.print("[yellow]Generating questions...[/yellow]")
    _ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    responses: list[MultiHopResponse] = []
    errors: list[str] = []

    for pair in selected:
        try:
            gen_resume = resumes[pair.resume_trace_id]
            gen_job = jobs[pair.job_trace_id]
            label = labels[pair.pair_id]
            response = generate_multi_hop_questions(
                gen_resume.resume, gen_job.job, label, pair.pair_id
            )
            responses.append(response)
        except KeyError as exc:
            errors.append(f"Missing key {exc} for pair {pair.pair_id}")
        except Exception as exc:
            errors.append(f"Error on {pair.pair_id}: {exc}")

    if errors:
        for err in errors:
            console.print(f"  [red]{err}[/red]")

    with OUTPUT_FILE.open("w") as f:
        for resp in responses:
            f.write(resp.model_dump_json() + "\n")

    console.print(f"  Generated {len(responses)} records, {len(responses) * 4} questions total\n")

    # Summary table: one row per selected pair
    table = Table(
        title=f"Multi-Hop Assessment Summary ({len(responses)} pairs × 4 questions)",
        show_lines=True,
    )
    table.add_column("Pair ID", style="dim", max_width=18)
    table.add_column("Fit Level")
    table.add_column("Q1: Edu/Exp", justify="center")
    table.add_column("Q2: Skills", justify="center")
    table.add_column("Q3: Career", justify="center")
    table.add_column("Q4: Overlap", justify="center")

    pair_lookup = {p.pair_id: p for p in selected}
    for resp in responses:
        fit_val = pair_lookup[resp.pair_id].fit_level.value
        assessments = [q.assessment for q in resp.questions]
        table.add_row(resp.pair_id[:16] + "…", fit_val, *assessments)

    console.print(table)
    console.print(f"\n[bold green]Done! {len(responses) * 4} questions → {OUTPUT_FILE}[/bold green]")


if __name__ == "__main__":
    run()
