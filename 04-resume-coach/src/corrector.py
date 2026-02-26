"""
corrector.py — LLM-based correction loop for invalid Resume records.

Uses GPT-4o-mini via Instructor to fix Pydantic validation errors.
Demonstrates the correction loop on 8 seeded broken records (Day 1 generation
had 100% validity rate — seeding provides Chart #8 data).
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import instructor
from openai import OpenAI
from pydantic import ValidationError

from .schemas import (
    CorrectionResult,
    CorrectionSummary,
    Resume,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_CORRECTED_DIR = _PROJECT_ROOT / "data" / "corrected"

_MODEL = "gpt-4o-mini"
# WHY 0.3: Low temperature → deterministic correction. We want targeted fixes,
# not creative rewriting of the candidate's resume.
_TEMPERATURE = 0.3
# WHY 3: Instructor retries the structured extraction up to 3 times per outer attempt
_MAX_INSTRUCTOR_RETRIES = 3


# ---------------------------------------------------------------------------
# Client + cache helpers (mirrors generator.py pattern)
# ---------------------------------------------------------------------------


def _create_client() -> instructor.Instructor:
    """Create Instructor-wrapped OpenAI client — same pattern as generator.py."""
    import os

    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Check .env file in 04-resume-coach/")
    return instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON)


def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """MD5 hash for cache key — includes model to prevent cross-model cache hits."""
    combined = f"{_MODEL}\n{system_prompt}\n---\n{user_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


def _load_cache(cache_key: str) -> dict | None:
    """Load cached correction response dict; return None on miss or corruption."""
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        return json.loads(cache_file.read_text())["response"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Cache corruption for %s: %s", cache_key[:8], exc)
        return None


def _save_cache(cache_key: str, response_dict: dict) -> None:
    """Persist corrected resume dict to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"cache_key": cache_key, "model": _MODEL, "response": response_dict}
    (_CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Core correction logic
# ---------------------------------------------------------------------------


def extract_validation_errors(
    raw_data: dict[str, Any],
    model_class: type,
) -> list[dict]:
    """
    Attempt to validate raw_data against model_class; return structured error list.

    WHY return list[dict] instead of raising: Callers need the error details to
    build a correction prompt. Raw ValidationError.errors() has awkward nested
    loc tuples — we flatten to a human-readable format here.

    Returns:
        [] if valid.
        [{field_path, error_message, invalid_value}] for each error if invalid.
    """
    try:
        model_class.model_validate(raw_data)
        return []
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            # WHY join with ".": loc is a tuple like ('contact_info', 'email').
            # Joining gives "contact_info.email" which is human-readable and
            # also matches JSON pointer notation (minus the leading /).
            field_path = ".".join(str(loc) for loc in err["loc"])
            errors.append({
                "field_path": field_path,
                "error_message": err["msg"],
                "invalid_value": str(err.get("input", "N/A")),
            })
        return errors


def build_correction_prompt(
    raw_data: dict[str, Any],
    errors: list[dict],
    record_type: str = "Resume",
) -> tuple[str, str]:
    """
    Build system + user prompts for the correction LLM call.

    WHY show original data: LLM needs full context to preserve valid fields
    while fixing only the broken ones. Without it, the LLM might invent values.

    WHY list errors explicitly: Vague "fix the errors" prompts produce poor results.
    Specific field paths + messages let the model target exactly what's wrong.

    Returns: (system_prompt, user_prompt)
    """
    error_lines = "\n".join(
        f"  - Field: {e['field_path']}\n"
        f"    Error: {e['error_message']}\n"
        f"    Invalid value: {e['invalid_value']}"
        for e in errors
    )

    system_prompt = (
        f"You are a data correction specialist. Fix validation errors in a {record_type} "
        "JSON object while preserving all valid fields unchanged.\n\n"
        "RULES:\n"
        "1. Fix ONLY the fields listed in the errors — do NOT change any other fields.\n"
        "2. Output a complete, valid JSON object matching the exact schema structure.\n"
        "3. For email: ensure it has a valid TLD (e.g., user@company.com).\n"
        "4. For phone: ensure it has at least 10 digits.\n"
        "5. For dates: use ISO format YYYY-MM or YYYY-MM-DD.\n"
        "6. For GPA: use 0.0–4.0 scale (e.g., 3.5 not 85.0).\n"
        "7. For date order: end_date must be after or equal to start_date.\n"
        "8. For responsibilities: include at least 1 item.\n"
        "9. For skill years: must be 0–30."
    )

    user_prompt = (
        f"Fix the following validation errors in this {record_type}:\n\n"
        f"ERRORS TO FIX:\n{error_lines}\n\n"
        f"ORIGINAL DATA:\n{json.dumps(raw_data, indent=2)}"
    )

    return system_prompt, user_prompt


def correct_record(
    client: instructor.Instructor,
    raw_data: dict[str, Any],
    record_type: str = "Resume",
    max_attempts: int = 3,
    *,
    use_cache: bool = True,
) -> CorrectionResult:
    """
    Attempt to correct a single invalid record, with up to max_attempts outer retries.

    WHY outer retry loop: Instructor retries internally (MAX_INSTRUCTOR_RETRIES).
    The outer loop lets us update the prompt with remaining errors after each attempt,
    giving the LLM more targeted guidance on subsequent tries.

    WHY use_cache kwarg: Allows unit tests to skip file I/O by passing use_cache=False.

    Returns a CorrectionResult regardless of success/failure.
    """
    record_id = str(raw_data.get("record_id", "unknown"))
    errors = extract_validation_errors(raw_data, Resume)
    original_errors = [e["error_message"] for e in errors]

    # WHY early return: Record passed validation — nothing to correct.
    if not errors:
        return CorrectionResult(
            pair_id=record_id,
            attempt_number=0,
            original_errors=[],
            corrected_successfully=True,
            remaining_errors=[],
        )

    current_data = raw_data.copy()

    for attempt in range(1, max_attempts + 1):
        system_prompt, user_prompt = build_correction_prompt(current_data, errors, record_type)

        if use_cache:
            cache_key = _prompt_hash(system_prompt, user_prompt)
            cached = _load_cache(cache_key)
            if cached is not None:
                logger.info("Cache hit for correction %s attempt %d", record_id, attempt)
                remaining_errors = extract_validation_errors(cached, Resume)
                if not remaining_errors:
                    return CorrectionResult(
                        pair_id=record_id,
                        attempt_number=attempt,
                        original_errors=original_errors,
                        corrected_successfully=True,
                        remaining_errors=[],
                    )
                # Cached result still has errors — use it as updated context for next attempt
                errors = remaining_errors
                current_data = cached
                continue
        else:
            cache_key = _prompt_hash(system_prompt, user_prompt)

        try:
            corrected: Resume = client.chat.completions.create(
                model=_MODEL,
                response_model=Resume,
                temperature=_TEMPERATURE,
                max_retries=_MAX_INSTRUCTOR_RETRIES,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            # WHY: If instructor succeeded, the result IS a valid Resume — Pydantic validated.
            corrected_dict = corrected.model_dump()
            if use_cache:
                _save_cache(cache_key, corrected_dict)
            return CorrectionResult(
                pair_id=record_id,
                attempt_number=attempt,
                original_errors=original_errors,
                corrected_successfully=True,
                remaining_errors=[],
            )

        except Exception as exc:
            logger.warning("Correction attempt %d/%d failed for %s: %s", attempt, max_attempts, record_id, exc)
            # Continue to next attempt with same errors and current_data

    # All attempts exhausted
    return CorrectionResult(
        pair_id=record_id,
        attempt_number=max_attempts,
        original_errors=original_errors,
        corrected_successfully=False,
        remaining_errors=[e["error_message"] for e in errors],
    )


# ---------------------------------------------------------------------------
# Seeded broken records
# ---------------------------------------------------------------------------


def generate_seeded_broken_records() -> list[dict[str, Any]]:
    """
    Return 8 deliberately broken resume dicts that fail Resume.model_validate().

    WHY 8 specific error types: Covers all 8 validators in schemas.py, giving
    the correction loop diverse failure modes to fix. Each dict includes a
    'record_id' key used as pair_id in CorrectionResult.

    Error types:
        broken-001: Invalid email (no TLD)
        broken-002: Malformed education date ("March 2020")
        broken-003: GPA out of range (85.0 instead of 0-4 scale)
        broken-004: Phone too short (6 digits instead of >=10)
        broken-005: End date before start date
        broken-006: Empty responsibilities list
        broken-007: Skill years out of range (35 > max 30)
        broken-008: Compound — invalid email + bad date + GPA out of range
    """

    def _base(record_id: str) -> dict:
        """Minimal valid resume dict — mutate specific fields to introduce errors."""
        return {
            "record_id": record_id,
            "contact_info": {
                "name": "Test Candidate",
                "email": "test@example.com",
                "phone": "555-123-4567",
                "location": "San Francisco, CA",
            },
            "education": [
                {
                    "degree": "B.S. Computer Science",
                    "institution": "State University",
                    "graduation_date": "2019-05",
                    "gpa": 3.7,
                }
            ],
            "experience": [
                {
                    "company": "Acme Corp",
                    "title": "Software Engineer",
                    "start_date": "2019-06",
                    "end_date": "2023-01",
                    "responsibilities": ["Built REST APIs", "Mentored junior developers"],
                }
            ],
            "skills": [
                {"name": "Python", "proficiency_level": "Advanced", "years": 4},
                {"name": "SQL", "proficiency_level": "Intermediate", "years": 3},
            ],
            "summary": "Experienced software engineer focused on backend systems.",
        }

    records: list[dict] = []

    # broken-001: Invalid email — missing TLD
    r1 = copy.deepcopy(_base("broken-001"))
    r1["contact_info"]["email"] = "john@company"
    records.append(r1)

    # broken-002: Malformed education date — not ISO format
    r2 = copy.deepcopy(_base("broken-002"))
    r2["education"][0]["graduation_date"] = "March 2020"
    records.append(r2)

    # broken-003: GPA percentage instead of 0-4 scale
    r3 = copy.deepcopy(_base("broken-003"))
    r3["education"][0]["gpa"] = 85.0
    records.append(r3)

    # broken-004: Phone with only 6 digits
    r4 = copy.deepcopy(_base("broken-004"))
    r4["contact_info"]["phone"] = "123456"
    records.append(r4)

    # broken-005: End date before start date
    r5 = copy.deepcopy(_base("broken-005"))
    r5["experience"][0]["start_date"] = "2023-01"
    r5["experience"][0]["end_date"] = "2020-01"
    records.append(r5)

    # broken-006: Empty responsibilities list
    r6 = copy.deepcopy(_base("broken-006"))
    r6["experience"][0]["responsibilities"] = []
    records.append(r6)

    # broken-007: Skill years exceeds maximum (0-30)
    r7 = copy.deepcopy(_base("broken-007"))
    r7["skills"][0]["years"] = 35
    records.append(r7)

    # broken-008: Compound — email + date + GPA all wrong
    r8 = copy.deepcopy(_base("broken-008"))
    r8["contact_info"]["email"] = "sarah@nodomain"
    r8["education"][0]["graduation_date"] = "June 2018"
    r8["education"][0]["gpa"] = 92.0
    records.append(r8)

    return records


# ---------------------------------------------------------------------------
# Batch + persistence
# ---------------------------------------------------------------------------


def correct_batch(
    invalid_records: list[dict[str, Any]],
    record_type: str,
    client: instructor.Instructor,
    *,
    use_cache: bool = True,
) -> tuple[list[CorrectionResult], CorrectionSummary]:
    """
    Correct all records sequentially; return results + aggregate summary.

    WHY sequential: Records are few (8) and correction is stateful per-record.
    No benefit from threading here (contrast with judge.py which does 250 records).
    """
    from rich.console import Console
    from rich.progress import track

    console = Console()
    results: list[CorrectionResult] = []

    for record in track(invalid_records, description="Correcting records..."):
        result = correct_record(client, record, record_type, use_cache=use_cache)
        results.append(result)
        status = "[green]✓[/green]" if result.corrected_successfully else "[red]✗[/red]"
        console.print(
            f"  {status} {result.pair_id} (attempt {result.attempt_number}): "
            f"{'corrected' if result.corrected_successfully else 'failed'}"
        )

    # --- Build summary ---
    total = len(results)
    corrected_count = sum(1 for r in results if r.corrected_successfully)

    # WHY filter attempt > 0: Records that were already valid (attempt=0) had nothing to correct
    # and shouldn't inflate the avg_attempts_per_success metric.
    successful_fixed = [r for r in results if r.corrected_successfully and r.attempt_number > 0]
    avg_attempts = (
        sum(r.attempt_number for r in successful_fixed) / len(successful_fixed)
        if successful_fixed
        else 0.0
    )

    failure_reasons: dict[str, int] = {}
    for r in results:
        if not r.corrected_successfully and r.remaining_errors:
            for reason in r.remaining_errors:
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    summary = CorrectionSummary(
        total_invalid=total,
        total_corrected=corrected_count,
        correction_rate=corrected_count / total if total > 0 else 0.0,
        avg_attempts_per_success=avg_attempts,
        common_failure_reasons=failure_reasons,
    )

    return results, summary


def save_correction_results(
    results: list[CorrectionResult],
    summary: CorrectionSummary,
) -> tuple[Path, Path]:
    """Write results to JSONL and summary to JSON in data/corrected/."""
    _CORRECTED_DIR.mkdir(parents=True, exist_ok=True)

    results_path = _CORRECTED_DIR / "correction_results.jsonl"
    with results_path.open("w") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")

    summary_path = _CORRECTED_DIR / "correction_summary.json"
    summary_path.write_text(summary.model_dump_json(indent=2))

    return results_path, summary_path


# ---------------------------------------------------------------------------
# CLI entry point — T2.7: Run correction pipeline on seeded records
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console

    console = Console()
    console.print("[bold cyan]P4 Day 2 — Correction Pipeline[/bold cyan]\n")

    broken_records = generate_seeded_broken_records()
    console.print(f"[yellow]Generated {len(broken_records)} seeded broken records[/yellow]")

    # Sanity-check: all records must fail validation before we start
    for record in broken_records:
        errors = extract_validation_errors(record, Resume)
        assert errors, f"Expected {record['record_id']} to be invalid but it passed!"
    console.print("[green]All 8 records confirmed invalid ✓[/green]\n")

    client = _create_client()
    results, summary = correct_batch(broken_records, "Resume", client)
    results_path, summary_path = save_correction_results(results, summary)

    console.print("\n[bold]Correction Summary:[/bold]")
    console.print(f"  Total invalid:    {summary.total_invalid}")
    console.print(f"  Total corrected:  {summary.total_corrected}")
    console.print(f"  Correction rate:  {summary.correction_rate:.1%}")
    console.print(f"  Avg attempts:     {summary.avg_attempts_per_success:.2f}")
    console.print(f"\n  Results: {results_path}")
    console.print(f"  Summary: {summary_path}")
