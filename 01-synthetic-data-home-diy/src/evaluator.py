"""LLM-as-Judge evaluation using GPT-4o via Instructor.

Sends each generated record to GPT-4o for quality evaluation across 6 failure
modes. Also loads manual labels (CSV) and computes inter-rater agreement
between human and LLM labels.

Java/TS parallel: like a separate evaluation service that calls a different
model — analogous to having a QA microservice that validates output from a
generation microservice.

WHY GPT-4o for judging (not GPT-4o-mini):
PRD Section 2: "cheap model generates, expensive model evaluates." The judge
needs higher reasoning capability to catch subtle quality issues. This pattern
(weak-generates, strong-evaluates) is reused in P2–P9.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.schemas import GeneratedRecord, JudgeResult

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LABELS_DIR = _PROJECT_ROOT / "data" / "labels"
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"

# Judge uses GPT-4o — higher quality evaluation (PRD Section 2)
_JUDGE_MODEL = "gpt-4o"
_TEMPERATURE = 0.3  # Lower temperature for more consistent judgments
_MAX_RETRIES = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge prompt (PRD Section 6c)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """\
You are a STRICT quality evaluator for DIY home repair guides.
Your job is to find deficiencies. Most guides have at least 1-2 issues.
Evaluate the following repair guide for these 6 failure modes.
For each mode, respond with 0 (pass) or 1 (fail) and a brief reason.

Failure mode definitions (apply these STRICTLY):

1. incomplete_answer: Fail if the answer has fewer than 3 substantive sentences, \
omits key information a homeowner needs (e.g., how to identify the part, where to \
buy replacements, what to do if the fix doesn't work), or lacks troubleshooting advice.

2. safety_violations: Fail if ANY of these are missing when relevant: \
power-off/water-off warning BEFORE starting work, PPE requirements (gloves, goggles), \
warning about when to call a licensed professional instead, specific hazard callouts \
(e.g., electrical shock risk, gas leak risk, hot surfaces). For electrical/HVAC repairs, \
ALWAYS check that circuit breaker/power disconnect is mentioned as the FIRST step.

3. unrealistic_tools: Fail if any tool is specialized enough that a typical homeowner \
would need to buy or rent it specifically for this repair (e.g., multimeter, cartridge \
puller, torque wrench). Also fail if the tool list is suspiciously short for the complexity \
of the repair.

4. overcomplicated_solution: For beginner tasks, fail if more than 8 steps or requires \
any specialty tools. For intermediate, fail if more than 12 steps. For advanced tasks, \
fail if the repair should ONLY be done by a licensed professional (e.g., working inside \
an electrical panel, replacing gas lines, modifying structural elements). Also fail if \
the guide doesn't explicitly state when to stop and call a professional.

5. missing_context: Fail if the answer gives generic advice not specifically tied to \
the equipment_problem field. The answer should reference the specific equipment/problem \
described, not just give a general category overview.

6. poor_quality_tips: Fail if tips are generic platitudes ("be careful", "take your time"), \
repeat information already in the steps or safety_info, or miss an opportunity to provide \
genuinely useful advice specific to this repair.

Be thorough and critical. A guide that is merely "okay" should still get failures flagged.
Only give 0 (pass) when the guide is genuinely strong in that dimension.
Quality scores: 1=very poor, 2=poor, 3=adequate, 4=good, 5=excellent. Most guides are 3-4."""


def _build_judge_user_prompt(record: GeneratedRecord) -> str:
    """Build the user prompt for the LLM judge.

    Includes category, difficulty, and the full record as JSON so the judge
    has complete context for evaluation.
    """
    record_json = record.record.model_dump_json(indent=2)
    return (
        f"Category: {record.category}\n"
        f"Difficulty: {record.difficulty}\n\n"
        f"Record:\n{record_json}\n\n"
        "Evaluate for all 6 failure modes. Provide your overall quality score (1-5)."
    )


# ---------------------------------------------------------------------------
# Cache for judge calls (same pattern as generator, different prefix)
# ---------------------------------------------------------------------------

def _judge_cache_key(record: GeneratedRecord) -> str:
    """Cache key for a judge evaluation.

    Combines the judge prompt + record trace_id to produce a unique key.
    Using trace_id ensures each record gets its own cached evaluation.
    """
    prompt = f"{_JUDGE_SYSTEM_PROMPT}\n---\n{_build_judge_user_prompt(record)}"
    return f"judge_{hashlib.md5(prompt.encode()).hexdigest()}"


def _load_judge_cache(cache_key: str) -> JudgeResult | None:
    """Load a cached judge result if it exists."""
    path = _CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        return JudgeResult.model_validate(data["response"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        logger.warning("Judge cache hit but failed to load %s: %s", cache_key, exc)
        return None


def _save_judge_cache(cache_key: str, trace_id: str, result: JudgeResult) -> None:
    """Save a judge result to the cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "trace_id": trace_id,
        "model": _JUDGE_MODEL,
        "response": result.model_dump(),
    }
    (_CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Single-record evaluation
# ---------------------------------------------------------------------------

def _create_client() -> instructor.Instructor:
    """Create an Instructor-wrapped OpenAI client for the judge."""
    load_dotenv(_PROJECT_ROOT / ".env")
    return instructor.from_openai(OpenAI())


def evaluate_record(
    client: instructor.Instructor,
    record: GeneratedRecord,
    *,
    use_cache: bool = True,
) -> JudgeResult:
    """Evaluate a single record using GPT-4o as judge.

    Args:
        client: Instructor-wrapped OpenAI client.
        record: The GeneratedRecord to evaluate.
        use_cache: If True, check cache before calling the API.

    Returns:
        JudgeResult with 6 failure mode labels + overall quality score.
    """
    cache_key = _judge_cache_key(record)

    if use_cache:
        cached = _load_judge_cache(cache_key)
        if cached is not None:
            logger.info("Judge cache hit for %s", record.trace_id)
            return cached

    logger.info("Evaluating %s via %s...", record.trace_id, _JUDGE_MODEL)

    result = client.chat.completions.create(
        model=_JUDGE_MODEL,
        response_model=JudgeResult,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_judge_user_prompt(record)},
        ],
        temperature=_TEMPERATURE,
        max_retries=_MAX_RETRIES,
        # Pass the trace_id via context so Instructor can populate it
        context={"trace_id": record.trace_id},
    )

    # Instructor might not populate trace_id from context — set it explicitly
    if result.trace_id != record.trace_id:
        result = result.model_copy(update={"trace_id": record.trace_id})

    _save_judge_cache(cache_key, record.trace_id, result)
    return result


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    records: list[GeneratedRecord],
    client: instructor.Instructor | None = None,
    *,
    use_cache: bool = True,
) -> list[JudgeResult]:
    """Evaluate all records in a batch using the LLM judge.

    Args:
        records: List of GeneratedRecords to evaluate.
        client: Instructor-wrapped OpenAI client. Created if not provided.
        use_cache: Whether to use the cache.

    Returns:
        List of JudgeResult objects, one per record.
    """
    if client is None:
        client = _create_client()

    results: list[JudgeResult] = []
    total = len(records)

    for i, record in enumerate(records, 1):
        logger.info("[%d/%d] Evaluating %s...", i, total, record.trace_id)
        try:
            result = evaluate_record(client, record, use_cache=use_cache)
            results.append(result)
        except (ValidationError, Exception) as exc:
            logger.error("Failed to evaluate %s: %s", record.trace_id, exc)

    logger.info("Evaluation complete: %d/%d records evaluated", len(results), total)
    return results


# ---------------------------------------------------------------------------
# Persistence — save/load LLM labels
# ---------------------------------------------------------------------------

def save_llm_labels(results: list[JudgeResult], filename: str = "llm_labels.csv") -> Path:
    """Save LLM judge results as a CSV file.

    CSV format: trace_id, then one column per failure mode (0/1),
    plus overall_quality_score. This matches the Pandas DataFrame
    structure in PRD Section 7a.

    Args:
        results: List of JudgeResult objects.
        filename: Output CSV filename.

    Returns:
        Path to the saved CSV.
    """
    _LABELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _LABELS_DIR / filename

    fieldnames = [
        "trace_id",
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
        "overall_quality_score",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {"trace_id": result.trace_id, "overall_quality_score": result.overall_quality_score}
            for label in result.labels:
                row[label.mode] = label.label
            writer.writerow(row)

    logger.info("Saved %d LLM labels to %s", len(results), output_path)
    return output_path


def save_llm_labels_json(results: list[JudgeResult], filename: str = "llm_labels.json") -> Path:
    """Save full JudgeResult objects as JSON (preserves reasoning).

    The CSV loses the per-mode 'reason' strings. This JSON file keeps them
    for the correction loop (corrector.py needs the reasons).
    """
    _LABELS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _LABELS_DIR / filename
    output_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2)
    )
    logger.info("Saved %d LLM labels (JSON) to %s", len(results), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Manual labels — loading + agreement analysis
# ---------------------------------------------------------------------------

def load_manual_labels(filename: str = "manual_labels.csv") -> list[dict]:
    """Load manual labels from CSV.

    Expected CSV format (same columns as LLM labels):
    trace_id, incomplete_answer, safety_violations, unrealistic_tools,
    overcomplicated_solution, missing_context, poor_quality_tips

    Args:
        filename: CSV filename in data/labels/.

    Returns:
        List of dicts, one per labeled record.
    """
    path = _LABELS_DIR / filename
    if not path.exists():
        logger.warning("Manual labels file not found: %s", path)
        return []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_agreement(
    manual_labels: list[dict],
    llm_labels: list[dict],
) -> dict:
    """Compute per-mode agreement rate between manual and LLM labels.

    For the records that have BOTH manual and LLM labels, compare each
    failure mode's binary label and report the agreement percentage.

    PRD Section 6b, Step 3: per-mode agreement rate.

    Args:
        manual_labels: List of dicts from manual_labels.csv.
        llm_labels: List of dicts from llm_labels.csv.

    Returns:
        Dict with per-mode agreement rates and overall agreement.
    """
    # Index LLM labels by trace_id for fast lookup
    llm_by_id = {row["trace_id"]: row for row in llm_labels}

    failure_modes = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
    ]

    # Track agreements per mode
    mode_agree: dict[str, int] = {m: 0 for m in failure_modes}
    mode_total: dict[str, int] = {m: 0 for m in failure_modes}

    matched_count = 0

    for manual_row in manual_labels:
        trace_id = manual_row["trace_id"]
        llm_row = llm_by_id.get(trace_id)
        if llm_row is None:
            logger.warning("No LLM label found for trace_id %s", trace_id)
            continue

        matched_count += 1
        for mode in failure_modes:
            raw_manual = manual_row.get(mode, "")
            raw_llm = llm_row.get(mode, "")
            # Skip modes where either label is missing (empty string = unlabeled)
            if raw_manual == "" or raw_llm == "":
                continue
            manual_val = int(raw_manual)
            llm_val = int(raw_llm)
            mode_total[mode] += 1
            if manual_val == llm_val:
                mode_agree[mode] += 1

    # Compute per-mode agreement rates
    per_mode = {}
    for mode in failure_modes:
        if mode_total[mode] > 0:
            rate = mode_agree[mode] / mode_total[mode]
            per_mode[mode] = f"{rate * 100:.1f}%"
        else:
            per_mode[mode] = "N/A"

    # Overall agreement (across all modes and records)
    total_comparisons = sum(mode_total.values())
    total_agreements = sum(mode_agree.values())
    overall = total_agreements / total_comparisons if total_comparisons > 0 else 0.0

    return {
        "matched_records": matched_count,
        "per_mode_agreement": per_mode,
        "overall_agreement": f"{overall * 100:.1f}%",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_llm_labels_csv(filename: str = "llm_labels.csv") -> list[dict]:
    """Load LLM labels from CSV for agreement analysis."""
    path = _LABELS_DIR / filename
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.generator import load_generated_records

    print("Loading generated records...")
    records = load_generated_records()
    print(f"Loaded {len(records)} records")

    print("\nRunning LLM-as-Judge evaluation...")
    results = evaluate_batch(records)
    csv_path = save_llm_labels(results)
    json_path = save_llm_labels_json(results)
    print(f"\nDone! Labels saved to:\n  CSV:  {csv_path}\n  JSON: {json_path}")

    # Check for manual labels and compute agreement if available
    manual = load_manual_labels()
    if manual:
        print(f"\nFound {len(manual)} manual labels. Computing agreement...")
        llm_csv = _load_llm_labels_csv()
        agreement = compute_agreement(manual, llm_csv)
        print(json.dumps(agreement, indent=2))
    else:
        print("\nNo manual labels found yet. Create data/labels/manual_labels.csv to enable agreement analysis.")
