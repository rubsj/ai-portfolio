from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from rich.progress import Progress

from src.schemas import (
    ExperienceLevel,
    FitLevel,
    GeneratedJob,
    GeneratedResume,
    JobDescription,
    Resume,
    ResumeJobPair,
    WritingStyle,
)
from src.templates import INDUSTRIES, PromptTemplateLibrary, TEMPLATE_VERSION

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"

_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.8
_MAX_RETRIES = 5


def _create_client() -> instructor.Instructor:
    """Create Instructor-wrapped OpenAI client."""
    import os

    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
    # WHY: Fail fast with clear message instead of cryptic OpenAI auth error deep in the call stack
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Check .env file in 04-resume-coach/")
    return instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON)


def _prompt_hash(system_prompt: str, user_prompt: str, model_name: str) -> str:
    """MD5 hash of prompt combination for cache key."""
    # WHY: Include model name so switching models (mini → 4o) doesn't serve stale mini responses
    combined = f"{model_name}\n{system_prompt}\n---\n{user_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


def _load_cache(cache_key: str, response_model: type[BaseModel]) -> BaseModel | None:
    """Load cached response; return None on miss or corruption."""
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        return response_model.model_validate(data["response"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        logger.warning("Cache corruption for %s: %s", cache_key[:8], exc)
        return None


def _save_cache(
    cache_key: str,
    prompt_key: str,
    result: BaseModel,
) -> None:
    """Save validated result to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "prompt_key": prompt_key,
        "model": _MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response": result.model_dump(),
    }
    (_CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(payload, indent=2))


def generate_with_cache(
    client: instructor.Instructor,
    prompt_key: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
) -> tuple[BaseModel, bool]:
    """
    Generate structured output with cache-first pattern.
    Returns: (result, from_cache)
    """
    cache_key = _prompt_hash(system_prompt, user_prompt, _MODEL)
    cached = _load_cache(cache_key, response_model)
    if cached is not None:
        return cached, True

    result = client.chat.completions.create(
        model=_MODEL,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=_TEMPERATURE,
        max_retries=_MAX_RETRIES,
    )

    _save_cache(cache_key, prompt_key, result)
    return result, False


def generate_job(
    client: instructor.Instructor,
    industry: str,
    is_niche: bool,
    experience_level: ExperienceLevel,
    templates: PromptTemplateLibrary,
) -> tuple[GeneratedJob, bool]:
    """Generate a single job description with metadata wrapper. Returns (job, from_cache)."""
    system_prompt, user_prompt = templates.get_job_prompt(industry, is_niche, experience_level)
    prompt_key = f"job_{industry}_{is_niche}_{experience_level.value}"

    job, from_cache = generate_with_cache(
        client, prompt_key, system_prompt, user_prompt, JobDescription
    )

    return GeneratedJob(
        trace_id=str(uuid.uuid4()),
        job=job,  # type: ignore[arg-type]
        is_niche_role=is_niche,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="job-standard",
        model_used=_MODEL,
    ), from_cache


def generate_resume(
    client: instructor.Instructor,
    job: GeneratedJob,
    fit_level: FitLevel,
    writing_style: WritingStyle,
    templates: PromptTemplateLibrary,
) -> tuple[GeneratedResume, bool]:
    """Generate a single resume matched to a job at specified fit level. Returns (resume, from_cache)."""
    system_prompt, user_prompt = templates.get_resume_prompt(job.job, fit_level, writing_style)
    template_id = templates.get_template_id(writing_style)
    prompt_key = f"resume_{job.trace_id}_{fit_level.value}_{writing_style.value}"

    resume, from_cache = generate_with_cache(
        client, prompt_key, system_prompt, user_prompt, Resume
    )

    return GeneratedResume(
        trace_id=str(uuid.uuid4()),
        resume=resume,  # type: ignore[arg-type]
        fit_level=fit_level,
        writing_style=writing_style,
        template_version=TEMPLATE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template=template_id,
        model_used=_MODEL,
    ), from_cache


def _append_jsonl(record: BaseModel, filepath: Path) -> None:
    """Append a single record to JSONL file (crash-resilient writes)."""
    # WHY: Write each record immediately so a mid-run crash doesn't lose prior work
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record.model_dump()) + "\n")


def generate_all_jobs(
    client: instructor.Instructor,
    count_per_industry: int = 5,
    max_industries: int | None = None,
) -> tuple[list[GeneratedJob], int]:
    """
    Generate jobs across industries.

    Strategy: count_per_industry per industry (default 5 × 10 = 50 total).
    ~20% are niche roles (1 per industry).
    Experience levels rotate: Entry, Mid, Senior, Lead, Executive.
    max_industries: limit number of industries (for --dry-run).

    Returns: (jobs, api_calls)
    """
    templates = PromptTemplateLibrary()
    industries = INDUSTRIES[:max_industries] if max_industries else INDUSTRIES

    experience_levels = list(ExperienceLevel)
    jobs: list[GeneratedJob] = []
    total = len(industries) * count_per_industry
    failed = 0
    api_calls = 0  # WHY: Track non-cached calls; rate limit only fires on real API hits

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = _GENERATED_DIR / f"jobs_{timestamp}.jsonl"

    with Progress() as progress:
        task = progress.add_task("Generating jobs...", total=total)

        for industry in industries:
            for i in range(count_per_industry):
                is_niche = i == 0  # First job per industry is niche (~20%)
                exp_level = experience_levels[i % len(experience_levels)]

                try:
                    job, from_cache = generate_job(client, industry, is_niche, exp_level, templates)
                    jobs.append(job)
                    _append_jsonl(job, output_file)
                    if not from_cache:
                        api_calls += 1
                except (ValidationError, Exception) as exc:
                    failed += 1
                    api_calls += 1  # Assume API was called on failure
                    logger.error("Failed job %s/%d: %s", industry, i, exc)

                progress.advance(task)

                # WHY: Only rate-limit on real API calls, not cache hits — re-runs stay fast
                if api_calls > 0 and api_calls % 10 == 0:
                    time.sleep(2)

    logger.info(
        "Jobs generated: %d succeeded, %d failed, %d API calls", len(jobs), failed, api_calls
    )
    return jobs, api_calls


def generate_all_resumes(
    client: instructor.Instructor,
    jobs: list[GeneratedJob],
) -> tuple[list[GeneratedResume], list[ResumeJobPair], int]:
    """
    Generate 5 resumes per job (one per fit level).
    Writing styles rotate evenly across all resumes.

    Returns: (resumes, pairs, api_calls)
    """
    templates = PromptTemplateLibrary()
    writing_styles = list(WritingStyle)
    resumes: list[GeneratedResume] = []
    pairs: list[ResumeJobPair] = []
    total = len(jobs) * len(FitLevel)
    failed = 0
    api_calls = 0
    style_index = 0  # WHY: Global index gives even style distribution (50/style for 250 resumes)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_file = _GENERATED_DIR / f"resumes_{timestamp}.jsonl"
    pair_file = _GENERATED_DIR / f"pairs_{timestamp}.jsonl"

    with Progress() as progress:
        task = progress.add_task("Generating resumes...", total=total)

        for job in jobs:
            for fit_level in FitLevel:
                style = writing_styles[style_index % len(writing_styles)]
                style_index += 1

                try:
                    resume, from_cache = generate_resume(
                        client, job, fit_level, style, templates
                    )
                    resumes.append(resume)
                    _append_jsonl(resume, resume_file)
                    if not from_cache:
                        api_calls += 1

                    pair = ResumeJobPair(
                        pair_id=str(uuid.uuid4()),
                        resume_trace_id=resume.trace_id,
                        job_trace_id=job.trace_id,
                        fit_level=fit_level,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    pairs.append(pair)
                    _append_jsonl(pair, pair_file)

                except (ValidationError, Exception) as exc:
                    failed += 1
                    api_calls += 1  # Assume API was called on failure
                    logger.error(
                        "Failed resume for job %s, fit=%s: %s",
                        job.trace_id[:8],
                        fit_level.value,
                        exc,
                    )

                progress.advance(task)

                # WHY: Only rate-limit on real API calls, not cache hits — re-runs stay fast
                if api_calls > 0 and api_calls % 10 == 0:
                    time.sleep(2)

    logger.info(
        "Resumes generated: %d succeeded, %d failed (%d pairs, %d API calls)",
        len(resumes),
        failed,
        len(pairs),
        api_calls,
    )
    return resumes, pairs, api_calls
