from __future__ import annotations

from datetime import datetime

from .normalizer import SkillNormalizer
from .schemas import (
    Experience,
    FailureLabels,
    JobDescription,
    ProficiencyLevel,
    Resume,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENIORITY_MAP: dict[str, int] = {
    "entry": 0,
    "junior": 0,
    "intern": 0,
    "associate": 0,
    "mid": 1,
    "intermediate": 1,
    "regular": 1,
    "senior": 2,
    "sr": 2,
    "lead": 3,
    "principal": 3,
    "staff": 3,
    "architect": 3,
    "executive": 4,
    "director": 4,
    "vp": 4,
    "chief": 4,
    "head": 4,
    "c-level": 4,
}

# WHY sorted descending by level: "Senior Associate" must match "senior" (2) before
# "associate" (0). Python dicts are insertion-ordered, so naive iteration would
# return the wrong level.  Sorting highest→lowest ensures the strongest signal wins.
_SENIORITY_SORTED: list[tuple[str, int]] = sorted(
    SENIORITY_MAP.items(), key=lambda kv: kv[1], reverse=True
)

BUZZWORDS: list[str] = [
    "synergy",
    "synergize",
    "leverage",
    "paradigm",
    "disrupt",
    "innovative",
    "thought leader",
    "move the needle",
    "circle back",
    "deep dive",
    "bleeding edge",
    "cutting edge",
    "game changer",
    "rockstar",
    "ninja",
    "guru",
    "wizard",
    "unicorn",
    "best-of-breed",
    "world-class",
    "results-driven",
    "self-starter",
    "detail-oriented",
    "team player",
    "proactive",
    "dynamic",
    "passionate",
    "motivated",
    "thinking outside the box",
    "low-hanging fruit",
    "bandwidth",
    "pivot",
    "ecosystem",
    "holistic",
    "scalable",
    "robust",
]

AI_PATTERNS: list[str] = [
    "as a seasoned",
    "in today's fast-paced",
    "proven track record of",
    "demonstrated ability to",
    "spearheaded initiatives",
    "orchestrated cross-functional",
    "championed the adoption",
    "leveraged cutting-edge",
    "passionate about driving",
]


# ---------------------------------------------------------------------------
# Core metric calculations
# ---------------------------------------------------------------------------


def calculate_jaccard(
    resume_skills: list[str],
    job_skills: list[str],
    normalizer: SkillNormalizer,
) -> tuple[float, int, int, set[str], set[str]]:
    """
    Compute Jaccard similarity between resume and job skill sets.

    WHY return 5-tuple instead of just the score: FailureLabels has 4 separate
    fields for the raw counts and normalized skill sets. Computing everything
    once here avoids a second normalization pass per pair.

    Returns:
        (jaccard_score, intersection_size, union_size, resume_norm_set, job_norm_set)
    """
    resume_norm = normalizer.normalize_set(resume_skills)
    job_norm = normalizer.normalize_set(job_skills)

    intersection = resume_norm & job_norm
    union = resume_norm | job_norm

    # WHY guard: empty union means both skill lists are empty — score = 0.0
    score = len(intersection) / len(union) if union else 0.0

    return score, len(intersection), len(union), resume_norm, job_norm


def calculate_total_experience(experiences: list[Experience]) -> float:
    """
    Sum total years across all experience entries.

    WHY slice [:7]: Experience dates can be "YYYY-MM" or "YYYY-MM-DD".
    Taking the first 7 chars normalizes both formats for strptime("%Y-%m").

    WHY max(0, months) guard: The date_order validator allows start == end (same month),
    producing 0 months — which is fine. This prevents any float rounding issues.
    """
    total_months = 0.0

    for exp in experiences:
        try:
            start = datetime.strptime(exp.start_date[:7], "%Y-%m")
            # WHY: None end_date means current job — use today as end
            end = datetime.now() if exp.end_date is None else datetime.strptime(exp.end_date[:7], "%Y-%m")
            months = (end.year - start.year) * 12 + (end.month - start.month)
            total_months += max(0.0, float(months))
        except ValueError:
            # WHY: Skip unparseable dates rather than crash — validator already
            # caught format errors; this is a belt-and-suspenders guard
            continue

    return total_months / 12.0


def infer_seniority(title: str, years_experience: float) -> int:
    """
    Infer seniority level (0–4) from job/resume title with experience fallback.

    WHY title takes precedence: A "Senior Engineer" with 3 years is still
    functioning at senior level for this role — title is the explicit signal.

    WHY descending sort: See _SENIORITY_SORTED constant above.

    Levels: 0=entry, 1=mid, 2=senior, 3=lead/principal, 4=executive
    """
    title_lower = title.lower()

    for keyword, level in _SENIORITY_SORTED:
        if keyword in title_lower:
            return level

    # Fallback: map experience years to seniority bands
    if years_experience < 2:
        return 0
    elif years_experience < 5:
        return 1
    elif years_experience < 10:
        return 2
    elif years_experience < 15:
        return 3
    else:
        return 4


# ---------------------------------------------------------------------------
# Failure mode detectors
# ---------------------------------------------------------------------------


def check_experience_mismatch(resume_years: float, job_years: int) -> bool:
    """
    True if resume years fall significantly short of job requirements.

    WHY two conditions with OR: The 50% threshold catches severe gaps
    (2yr for a 5yr role). The -3 threshold catches absolute gaps for
    senior roles (7yr for a 10yr role). Either alone is a red flag.

    WHY job_years == 0 guard: Entry-level roles require 0 years — no
    mismatch is possible, and 0.5 * 0 = 0 would incorrectly flag everyone.
    """
    if job_years == 0:
        return False
    return resume_years < job_years * 0.5 or resume_years < job_years - 3


def check_seniority_mismatch(resume_level: int, job_level: int) -> bool:
    """
    True if resume and job seniority differ by more than 1 level.

    WHY gap > 1 not gap > 0: A 1-level gap is normal (applying slightly
    above/below current level is common). A 2+ level gap signals misalignment.
    """
    return abs(resume_level - job_level) > 1


def check_missing_core_skills(
    resume_skills_norm: set[str],
    job_required: list[str],
    normalizer: SkillNormalizer,
) -> tuple[bool, list[str]]:
    """
    Check if any of the top-3 required job skills are absent from the resume.

    WHY top-3: The first 3 required skills in the list are assumed to be
    the most critical (LLM generator was prompted to order by importance).
    Missing any of them is a significant red flag.

    Returns:
        (has_missing_skills, list_of_missing_original_skill_names)
    """
    top_3 = job_required[:3]
    missing = [skill for skill in top_3 if normalizer.normalize(skill) not in resume_skills_norm]
    return len(missing) > 0, missing


def detect_hallucinations(
    resume: Resume,
    experience_years: float,
) -> tuple[bool, list[str]]:
    """
    Rule-based hallucination detection — 4 rules from PRD Section 6d.

    WHY rule-based first, LLM-judge second:
    - Rules run in microseconds and are deterministic → fast CI
    - LLM judge is ~5s per call, non-deterministic, costs $0.75 for 250 pairs
    - Having both independent signals enables Chart #9 (agreement analysis)
    """
    reasons: list[str] = []

    expert_count = sum(1 for s in resume.skills if s.proficiency_level == ProficiencyLevel.EXPERT)

    # Rule 1: Entry-level (<2yr) claiming Expert in more than 3 skills
    if experience_years < 2 and expert_count > 3:
        reasons.append(
            f"Entry-level ({experience_years:.1f} years) claims Expert in {expert_count} skills"
        )

    # Rule 2: Unrealistic total skill count combined with too many Expert ratings
    if len(resume.skills) > 20 and expert_count > 10:
        reasons.append(
            f"{len(resume.skills)} total skills with {expert_count} at Expert level"
        )

    # Rule 3: Individual skill years exceed total career experience
    for skill in resume.skills:
        if skill.years is not None and skill.years > experience_years + 1:
            # WHY +1 tolerance: half a year of rounding shouldn't flag a mismatch
            reasons.append(
                f"Claims {skill.years}yr in {skill.name} but total experience is {experience_years:.1f}yr"
            )

    # Rule 4: Senior executive title with very little total experience
    # WHY only when len > 1: A single-job resume where the person started as VP is
    # plausible (founder, etc.). Multiple entries confirm the timeline is real.
    senior_exec_titles = ["director", "vp", "vice president", "chief", "head of"]
    if len(resume.experience) > 1:
        for exp in resume.experience:
            if any(st in exp.title.lower() for st in senior_exec_titles) and experience_years < 5:
                reasons.append(
                    f"Senior title '{exp.title}' with only {experience_years:.1f} years experience"
                )

    return len(reasons) > 0, reasons


def detect_awkward_language(resume: Resume) -> tuple[bool, list[str]]:
    """
    Detect AI-generated or buzzword-heavy language — 3 rules from PRD Section 6e.

    WHY pattern matching over LLM classification: Same performance/cost reasoning
    as hallucination detection. The judge catches subtle phrasing; rules catch obvious.
    """
    reasons: list[str] = []

    # Combine all resume text for a single-pass analysis
    all_text = " ".join([
        resume.summary or "",
        *[r for exp in resume.experience for r in exp.responsibilities],
        *[a for exp in resume.experience for a in (exp.achievements or [])],
    ]).lower()

    # Rule 1: Buzzword density — more than 5 buzzwords is a red flag
    buzzword_count = sum(1 for bw in BUZZWORDS if bw in all_text)
    if buzzword_count > 5:
        reasons.append(f"High buzzword density: {buzzword_count} buzzwords detected")

    # Rule 2: AI-pattern phrases — more than 2 matches signals LLM boilerplate
    ai_pattern_count = sum(1 for pattern in AI_PATTERNS if pattern in all_text)
    if ai_pattern_count > 2:
        reasons.append(f"AI-generated patterns detected: {ai_pattern_count} matches")

    # Rule 3: Repeated words in close proximity (3+ times in a 50-word window)
    words = all_text.split()
    for i in range(max(0, len(words) - 49)):
        window = words[i : i + 50]
        word_counts: dict[str, int] = {}
        for w in window:
            if len(w) > 4:  # WHY: Skip short stop words (the, and, with…)
                word_counts[w] = word_counts.get(w, 0) + 1
        repetitions = {w: c for w, c in word_counts.items() if c >= 3}
        if repetitions:
            reasons.append(f"Repeated words in close proximity: {repetitions}")
            break  # One proximity finding is sufficient

    return len(reasons) > 0, reasons


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def label_pair(
    resume: Resume,
    job: JobDescription,
    pair_id: str,
    normalizer: SkillNormalizer | None = None,
) -> FailureLabels:
    """
    Compute all 6 failure labels for a single resume-job pair.

    WHY accept shared normalizer: Passing one SkillNormalizer instance from the
    caller avoids re-instantiating the alias dict 250 times in the batch pipeline.
    Passing None creates a fresh instance (convenient for isolated unit tests).

    Returns a fully-populated FailureLabels validated by Pydantic (18 fields).
    """
    if normalizer is None:
        normalizer = SkillNormalizer()

    # --- Jaccard similarity ---
    resume_skill_names = [s.name for s in resume.skills]
    job_skill_names = job.requirements.required_skills

    jaccard_score, overlap_raw, union_raw, resume_norm, job_norm = calculate_jaccard(
        resume_skill_names, job_skill_names, normalizer
    )

    # --- Experience years ---
    total_years = calculate_total_experience(resume.experience)
    job_years_required = job.requirements.experience_years
    exp_mismatch = check_experience_mismatch(total_years, job_years_required)

    # --- Seniority levels ---
    # WHY experience[0]: Most-recent job (index 0) is most relevant for current level
    latest_title = resume.experience[0].title if resume.experience else ""
    resume_level = infer_seniority(latest_title, total_years)
    # WHY cast to float: infer_seniority expects float; experience_years is int on job
    job_level = infer_seniority(job.title, float(job_years_required))
    seniority_mismatch = check_seniority_mismatch(resume_level, job_level)

    # --- Missing core skills ---
    has_missing, missing_skills = check_missing_core_skills(
        resume_norm, job_skill_names, normalizer
    )

    # --- Hallucination detection ---
    has_hallucinations, hallucination_reasons = detect_hallucinations(resume, total_years)

    # --- Awkward language detection ---
    has_awkward, awkward_reasons = detect_awkward_language(resume)

    return FailureLabels(
        pair_id=pair_id,
        skills_overlap=jaccard_score,
        skills_overlap_raw=overlap_raw,
        skills_union_raw=union_raw,
        experience_mismatch=exp_mismatch,
        seniority_mismatch=seniority_mismatch,
        missing_core_skills=has_missing,
        has_hallucinations=has_hallucinations,
        has_awkward_language=has_awkward,
        experience_years_resume=total_years,
        experience_years_required=job_years_required,
        seniority_level_resume=resume_level,
        seniority_level_job=job_level,
        missing_skills=missing_skills,
        hallucination_reasons=hallucination_reasons,
        awkward_language_reasons=awkward_reasons,
        resume_skills_normalized=sorted(resume_norm),
        job_skills_normalized=sorted(job_norm),
    )
