"""
judge.py — LLM-as-Judge evaluation of 250 resume-job pairs using GPT-4o.

Uses Instructor + ThreadPoolExecutor for concurrent evaluation with MD5 caching.
First run makes ~250 GPT-4o API calls (~$0.75). Subsequent runs serve from cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError, field_validator

from .schemas import (
    GeneratedJob,
    GeneratedResume,
    JobDescription,
    JudgeResult,
    Resume,
    ResumeJobPair,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_LABELED_DIR = _PROJECT_ROOT / "data" / "labeled"
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"

# WHY gpt-4o not gpt-4o-mini: Judge quality matters — we want nuanced hallucination
# detection and fit assessments. Mini produces shallower evaluations.
_MODEL = "gpt-4o"
_TEMPERATURE = 0.3  # WHY 0.3: Low temperature for consistent, reproducible judgments
_MAX_INSTRUCTOR_RETRIES = 3
_MAX_WORKERS = 1
# WHY every=1 sleep=7s: At 30K TPM with ~3K tokens/call → max 10 calls/min.
# Sleeping 7s between every real API call keeps us under the limit (≈8.5 calls/min).
# The original 4-worker config hit all 250 pairs simultaneously → instant 429s.
_RATE_LIMIT_EVERY = 1
_RATE_LIMIT_SLEEP = 7.0  # Seconds to sleep between API calls

# WHY thread-safe counter: Multiple worker threads may call the API simultaneously.
# The lock ensures the counter is incremented atomically, preventing race conditions
# that could skip or double-apply the sleep.
_api_call_lock = threading.Lock()
_api_call_count = 0


# ---------------------------------------------------------------------------
# Private LLM output schema (pair_id injected post-hoc)
# ---------------------------------------------------------------------------


class _JudgeLLMOutput(BaseModel):
    """
    LLM response schema — mirrors JudgeResult but omits pair_id.

    WHY separate from JudgeResult: The LLM receives job+resume data, not pair_id.
    Including pair_id in the prompt/schema would waste tokens and pollute the
    cache key (same pair evaluated twice with different pair_ids → cache miss).
    We inject pair_id after the LLM call.
    """

    has_hallucinations: bool
    hallucination_details: str
    has_awkward_language: bool
    awkward_language_details: str
    overall_quality_score: float
    fit_assessment: str
    recommendations: list[str]
    red_flags: list[str]

    @field_validator("overall_quality_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Quality score must be 0.0-1.0, got {v}")
        return v


# ---------------------------------------------------------------------------
# Client + cache helpers (mirrors generator.py pattern)
# ---------------------------------------------------------------------------


def _create_judge_client() -> instructor.Instructor:
    """Create Instructor-wrapped OpenAI client for GPT-4o evaluation."""
    import os

    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Check .env file in 04-resume-coach/")
    return instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON)


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """MD5 hash for cache key — includes model name to prevent cross-model cache hits."""
    combined = f"{_MODEL}\n{system_prompt}\n---\n{user_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


def _load_cache(cache_key: str) -> _JudgeLLMOutput | None:
    """Load cached judge output; return None on miss or corruption."""
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        return _JudgeLLMOutput.model_validate(data["response"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        logger.warning("Cache corruption for %s: %s", cache_key[:8], exc)
        return None


def _save_cache(cache_key: str, result: _JudgeLLMOutput) -> None:
    """Persist judge output to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "model": _MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response": result.model_dump(),
    }
    (_CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_judge_prompt(job: JobDescription, resume: Resume) -> tuple[str, str]:
    """
    Build system + user prompts for the LLM-as-Judge evaluation.

    WHY model_dump_json(indent=2): Gives the LLM the full structured data.
    Indented JSON is more readable than minified — helps the model parse
    nested fields like requirements.required_skills or skills[].years.

    Returns: (system_prompt, user_prompt)
    """
    system_prompt = (
        "You are an expert resume reviewer and career coach with 20+ years of experience. "
        "Evaluate whether a resume is well-suited for a specific job description.\n\n"
        "Assess three dimensions:\n"
        "1. HALLUCINATIONS: Identify implausible claims given the candidate's experience "
        "level, timeline contradictions, or unrealistic skill proficiency claims.\n"
        "2. LANGUAGE QUALITY: Flag AI-generated boilerplate, excessive buzzwords, "
        "repetitive phrasing, or awkward language that would make a recruiter skeptical.\n"
        "3. OVERALL FIT: Score the resume-job match on a 0.0-1.0 scale "
        "(0.0=completely unsuitable, 0.5=partial fit, 1.0=perfect match).\n\n"
        "Be specific — reference actual content from the resume and job description. "
        "Provide actionable recommendations and concrete red flags if present."
    )

    user_prompt = (
        "Evaluate this resume against the job description:\n\n"
        f"JOB DESCRIPTION:\n{job.model_dump_json(indent=2)}\n\n"
        f"RESUME:\n{resume.model_dump_json(indent=2)}"
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Single pair evaluation
# ---------------------------------------------------------------------------


def judge_pair(
    client: instructor.Instructor,
    job: JobDescription,
    resume: Resume,
    pair_id: str,
    *,
    use_cache: bool = True,
) -> JudgeResult:
    """
    Evaluate one resume-job pair using LLM-as-Judge.

    WHY pair_id injected after LLM call: The LLM receives job+resume data only.
    Injecting pair_id into the prompt would pollute the cache key — the same
    pair evaluated under different IDs would miss the cache unnecessarily.

    Returns: JudgeResult with pair_id set.
    """
    global _api_call_count

    system_prompt, user_prompt = _build_judge_prompt(job, resume)
    cache_key = _prompt_hash(system_prompt, user_prompt)

    if use_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            logger.debug("Cache hit for pair %s", pair_id[:8])
            # Construct JudgeResult with the correct pair_id injected
            return JudgeResult(pair_id=pair_id, **cached.model_dump())

    # Rate limiting: increment shared counter; sleep outside lock to unblock other threads
    with _api_call_lock:
        _api_call_count += 1
        should_sleep = _api_call_count % _RATE_LIMIT_EVERY == 0

    if should_sleep:
        logger.info(
            "Rate limit: sleeping %.1fs after %d API calls", _RATE_LIMIT_SLEEP, _api_call_count
        )
        time.sleep(_RATE_LIMIT_SLEEP)

    llm_output: _JudgeLLMOutput = client.chat.completions.create(
        model=_MODEL,
        response_model=_JudgeLLMOutput,
        temperature=_TEMPERATURE,
        # WHY max_retries=0: Rate limit (429) errors propagate to the batch handler which
        # logs the failure. Instructor retrying on 429 consumes MORE tokens per attempt,
        # worsening the rate limit problem. We handle retries at the batch level via re-runs.
        max_retries=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    if use_cache:
        _save_cache(cache_key, llm_output)

    # WHY construct here not before cache: We need llm_output validated first.
    return JudgeResult(pair_id=pair_id, **llm_output.model_dump())


# ---------------------------------------------------------------------------
# Batch evaluation (threaded)
# ---------------------------------------------------------------------------


def judge_batch(
    pairs: list[ResumeJobPair],
    jobs: dict[str, GeneratedJob],
    resumes: dict[str, GeneratedResume],
    *,
    max_workers: int = _MAX_WORKERS,
    use_cache: bool = True,
) -> list[JudgeResult]:
    """
    Evaluate all pairs concurrently using ThreadPoolExecutor.

    WHY ThreadPoolExecutor: 250 pairs × ~2s per LLM call = ~500s sequential.
    With 4 workers and caching, first run ~125s, re-runs ~5s (all cache hits).

    WHY dict lookups not list scan: O(1) vs O(n) per pair lookup.
    Pre-building dicts in the caller avoids O(n²) total complexity.
    """
    results: list[JudgeResult] = []
    failed_ids: list[str] = []
    results_lock = threading.Lock()

    client = _create_judge_client()

    def _eval_one(pair: ResumeJobPair) -> JudgeResult | None:
        try:
            gen_job = jobs[pair.job_trace_id]
            gen_resume = resumes[pair.resume_trace_id]
            return judge_pair(
                client, gen_job.job, gen_resume.resume, pair.pair_id, use_cache=use_cache
            )
        except KeyError as exc:
            logger.error("Missing trace_id %s for pair %s", exc, pair.pair_id)
            return None
        except Exception as exc:
            logger.error("Judge failed for pair %s: %s", pair.pair_id, exc)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_eval_one, pair): pair.pair_id for pair in pairs}
        completed = 0

        for future in as_completed(futures):
            completed += 1
            pair_id = futures[future]
            result = future.result()
            if result is not None:
                with results_lock:
                    results.append(result)
            else:
                with results_lock:
                    failed_ids.append(pair_id)

            if completed % 25 == 0 or completed == len(pairs):
                logger.info("Judge progress: %d/%d pairs evaluated", completed, len(pairs))

    logger.info(
        "Judge batch complete: %d succeeded, %d failed", len(results), len(failed_ids)
    )
    if failed_ids:
        logger.warning("Failed pair IDs: %s", failed_ids)

    return results


# ---------------------------------------------------------------------------
# Loaders
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


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------


def save_judge_results(results: list[JudgeResult]) -> Path:
    """Write all JudgeResult records to data/labeled/judge_results.jsonl."""
    _LABELED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _LABELED_DIR / "judge_results.jsonl"
    with output_path.open("w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point — T2.4: Run judge on 250 pairs (~$0.75)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    console.print("[bold cyan]P4 Day 2 — LLM-as-Judge Pipeline[/bold cyan]\n")

    JOBS_FILE = _GENERATED_DIR / "jobs_20260225_141615.jsonl"
    RESUMES_FILE = _GENERATED_DIR / "resumes_20260225_142052.jsonl"
    PAIRS_FILE = _GENERATED_DIR / "pairs_20260225_142052.jsonl"

    console.print("[yellow]Loading data...[/yellow]")
    pairs = _load_pairs(PAIRS_FILE)
    jobs = _load_jobs(JOBS_FILE)
    resumes = _load_resumes(RESUMES_FILE)
    console.print(f"  Loaded {len(pairs)} pairs, {len(jobs)} jobs, {len(resumes)} resumes\n")

    console.print(
        f"[yellow]Running LLM-as-Judge on {len(pairs)} pairs "
        f"(model: {_MODEL}, workers: {_MAX_WORKERS})...[/yellow]"
    )
    console.print("[dim]First run: ~250 API calls (~$0.75). Re-runs: all cache hits.[/dim]\n")

    results = judge_batch(pairs, jobs, resumes)
    output_path = save_judge_results(results)

    # Print summary
    console.print("\n[bold]Judge Summary:[/bold]")
    console.print(f"  Total evaluated:      {len(results)}/{len(pairs)}")

    if results:
        avg_score = sum(r.overall_quality_score for r in results) / len(results)
        hallucination_rate = sum(1 for r in results if r.has_hallucinations) / len(results)
        awkward_rate = sum(1 for r in results if r.has_awkward_language) / len(results)
        console.print(f"  Avg quality score:    {avg_score:.3f}")
        console.print(f"  Hallucination rate:   {hallucination_rate:.1%}")
        console.print(f"  Awkward language:     {awkward_rate:.1%}")

    console.print(f"\n[bold green]Done! Results → {output_path}[/bold green]")
