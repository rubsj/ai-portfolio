"""Correction loop: individual record fixes + template v2 improvement.

Two correction strategies (PRD Section 8):

Strategy A — Individual record correction:
  Takes each record with >=1 failure, sends original + judge feedback to
  GPT-4o-mini for targeted correction, validates via same Pydantic model.

Strategy B — Template v2 improvement:
  Analyzes which failure modes are most common per category, creates v2
  templates with explicit instructions to prevent those failures, then
  re-generates a fresh batch.

Java/TS parallel: Strategy A is like a retry-with-context pattern (similar
to Polly retry with onRetry callback). Strategy B is like refactoring
the request builder based on QA feedback — upstream fix vs downstream patch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import instructor
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from src.schemas import Category, DIYRepairRecord, Difficulty, GeneratedRecord

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_CORRECTED_DIR = _PROJECT_ROOT / "data" / "corrected"
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"
_LABELS_DIR = _PROJECT_ROOT / "data" / "labels"

_GENERATION_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.7
_MAX_RETRIES = 3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instructor client
# ---------------------------------------------------------------------------

def _create_client() -> instructor.Instructor:
    """Create an Instructor-wrapped OpenAI client."""
    load_dotenv(_PROJECT_ROOT / ".env")
    return instructor.from_openai(OpenAI())


# ---------------------------------------------------------------------------
# Cache helpers (reused from generator — same pattern, different prefix)
# ---------------------------------------------------------------------------

def _correction_cache_key(prompt: str) -> str:
    """MD5 hash for caching correction responses."""
    return f"correct_{hashlib.md5(prompt.encode()).hexdigest()}"


def _load_correction_cache(cache_key: str) -> DIYRepairRecord | None:
    """Load a cached correction if it exists."""
    path = _CACHE_DIR / f"{cache_key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return DIYRepairRecord.model_validate(data["response"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        logger.warning("Correction cache load failed for %s: %s", cache_key, exc)
        return None


def _save_correction_cache(cache_key: str, trace_id: str, record: DIYRepairRecord) -> None:
    """Save a corrected record to the cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "trace_id": trace_id,
        "model": _GENERATION_MODEL,
        "type": "correction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response": record.model_dump(),
    }
    (_CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Strategy A: Individual record correction (PRD Section 8a)
# ---------------------------------------------------------------------------

_CORRECTION_SYSTEM_PROMPT = """\
You are a quality improvement specialist for DIY home repair guides.
Your job is to fix specific quality issues while preserving the overall
structure and useful content of the original guide.

Rules:
- Fix ONLY the flagged issues — do not rewrite content that is already good
- Keep the same repair topic, tools, and general approach
- Improve specificity: replace generic advice with concrete, actionable details
- For incomplete_answer: add troubleshooting, parts sourcing, and what-if scenarios
- For poor_quality_tips: replace generic tips with specific, expert-level advice
- For safety_violations: add explicit safety warnings BEFORE the relevant steps
- For unrealistic_tools: replace specialty tools with common household alternatives
- For overcomplicated_solution: simplify steps and reduce count
- For missing_context: tie advice specifically to the equipment_problem"""


def _build_correction_prompt(
    record: GeneratedRecord,
    failures: list[dict],
) -> str:
    """Build the user prompt for correcting a single record.

    Args:
        record: The original GeneratedRecord.
        failures: List of dicts with 'mode' and 'reason' from the judge.
    """
    record_json = record.record.model_dump_json(indent=2)
    flagged = "\n".join(
        f"- {f['mode']}: {f['reason']}" for f in failures
    )
    return (
        f"The following {record.difficulty}-level {record.category.replace('_', ' ')} "
        f"repair guide was flagged with quality issues.\n\n"
        f"Original record:\n{record_json}\n\n"
        f"Flagged issues:\n{flagged}\n\n"
        "Generate a corrected version that addresses these specific issues "
        "while keeping the same repair topic and overall structure."
    )


def correct_record(
    client: instructor.Instructor,
    record: GeneratedRecord,
    failures: list[dict],
    *,
    use_cache: bool = True,
) -> DIYRepairRecord:
    """Correct a single record based on judge feedback.

    Args:
        client: Instructor-wrapped OpenAI client.
        record: The original GeneratedRecord with failures.
        failures: List of {'mode': ..., 'reason': ...} dicts from judge.
        use_cache: Whether to check cache first.

    Returns:
        Corrected DIYRepairRecord.
    """
    user_prompt = _build_correction_prompt(record, failures)
    full_prompt = f"{_CORRECTION_SYSTEM_PROMPT}\n---\n{user_prompt}"
    cache_key = _correction_cache_key(full_prompt)

    if use_cache:
        cached = _load_correction_cache(cache_key)
        if cached is not None:
            logger.info("Correction cache hit for %s", record.trace_id)
            return cached

    logger.info("Correcting %s (%d failures)...", record.trace_id, len(failures))

    corrected = client.chat.completions.create(
        model=_GENERATION_MODEL,
        response_model=DIYRepairRecord,
        messages=[
            {"role": "system", "content": _CORRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=_TEMPERATURE,
        max_retries=_MAX_RETRIES,
    )

    _save_correction_cache(cache_key, record.trace_id, corrected)
    return corrected


def correct_batch(
    records: list[GeneratedRecord],
    judge_results: list[dict],
    client: instructor.Instructor | None = None,
    *,
    use_cache: bool = True,
) -> list[GeneratedRecord]:
    """Correct all records that have >=1 failure.

    Args:
        records: All generated records (including clean ones).
        judge_results: Full JudgeResult dicts from llm_labels.json.
        client: Instructor client (created if None).
        use_cache: Whether to use cache.

    Returns:
        List of GeneratedRecord objects — corrected records replace originals,
        clean records pass through unchanged.
    """
    if client is None:
        client = _create_client()

    # Index judge results by trace_id
    judge_by_id = {jr["trace_id"]: jr for jr in judge_results}

    corrected_records: list[GeneratedRecord] = []
    corrected_count = 0
    unchanged_count = 0

    for record in records:
        judge = judge_by_id.get(record.trace_id)
        if judge is None:
            logger.warning("No judge result for %s — skipping", record.trace_id)
            corrected_records.append(record)
            unchanged_count += 1
            continue

        # Extract failures (label == 1)
        failures = [
            {"mode": label["mode"], "reason": label["reason"]}
            for label in judge["labels"]
            if label["label"] == 1
        ]

        if not failures:
            # Record is clean — pass through
            corrected_records.append(record)
            unchanged_count += 1
            continue

        try:
            corrected_diy = correct_record(client, record, failures, use_cache=use_cache)
            # Wrap corrected record with original metadata (same trace_id)
            corrected_gen = record.model_copy(
                update={"record": corrected_diy, "template_version": "v1_corrected"}
            )
            corrected_records.append(corrected_gen)
            corrected_count += 1
        except (ValidationError, Exception) as exc:
            logger.error("Failed to correct %s: %s", record.trace_id, exc)
            corrected_records.append(record)
            unchanged_count += 1

    logger.info(
        "Correction complete: %d corrected, %d unchanged",
        corrected_count, unchanged_count,
    )
    return corrected_records


# ---------------------------------------------------------------------------
# Strategy B: Template v2 improvement (PRD Section 8b)
# ---------------------------------------------------------------------------

def analyze_failure_patterns(
    records: list[GeneratedRecord],
    judge_results: list[dict],
) -> dict[str, list[str]]:
    """Identify top failure modes per category to inform v2 templates.

    Returns:
        Dict mapping category -> list of failure mode names (sorted by frequency).
    """
    judge_by_id = {jr["trace_id"]: jr for jr in judge_results}

    # Count failures per (category, mode) pair
    from collections import Counter
    category_mode_counts: dict[str, Counter] = {}

    for record in records:
        judge = judge_by_id.get(record.trace_id)
        if judge is None:
            continue
        if record.category not in category_mode_counts:
            category_mode_counts[record.category] = Counter()
        for label in judge["labels"]:
            if label["label"] == 1:
                category_mode_counts[record.category][label["mode"]] += 1

    # For each category, return failure modes sorted by frequency (most common first)
    patterns: dict[str, list[str]] = {}
    for cat, counter in category_mode_counts.items():
        if counter:
            patterns[cat] = [mode for mode, _ in counter.most_common()]

    return patterns


# V2 template additions — explicit instructions per failure mode to prevent them
_V2_MODE_INSTRUCTIONS: dict[str, str] = {
    "incomplete_answer": (
        "IMPORTANT: Your answer MUST include: (1) how to identify the specific part/problem, "
        "(2) where to buy replacement parts if needed, (3) what to do if the fix doesn't work, "
        "and (4) troubleshooting advice for common complications. The answer should be at least "
        "4-5 detailed sentences."
    ),
    "poor_quality_tips": (
        "IMPORTANT: Tips MUST be specific and expert-level. Do NOT use generic advice like "
        "'be careful' or 'take your time'. Instead, provide concrete tips that a professional "
        "would give — specific product recommendations, pro techniques, common mistakes to avoid, "
        "and maintenance advice to prevent the problem from recurring."
    ),
    "safety_violations": (
        "CRITICAL SAFETY: You MUST include: (1) explicit instruction to disconnect power/water "
        "BEFORE starting work as the FIRST step, (2) required PPE (gloves, goggles, etc.), "
        "(3) specific hazard callouts, and (4) clear guidance on when to call a licensed "
        "professional instead of attempting the repair."
    ),
    "unrealistic_tools": (
        "TOOL REQUIREMENT: Only list tools that a typical homeowner would already own or can "
        "buy at any hardware store for under $20. Do NOT include specialty tools like multimeters, "
        "cartridge pullers, torque wrenches, or pipe threading tools. If a specialty tool is "
        "truly needed, suggest a common household alternative."
    ),
    "overcomplicated_solution": (
        "SIMPLICITY: Keep the solution appropriate for the stated difficulty level. Beginner "
        "tasks should have no more than 8 steps. Intermediate no more than 12. Always indicate "
        "when a homeowner should stop and call a professional."
    ),
    "missing_context": (
        "SPECIFICITY: Your answer must directly reference and address the specific equipment "
        "and problem described. Do not give generic category advice — tie every step to the "
        "actual problem scenario."
    ),
}


def build_v2_templates() -> dict[str, dict]:
    """Build v2 templates with failure-mode-specific additions.

    Uses the failure pattern analysis to add targeted instructions
    to each category's template. Returns a structure that
    generate_v2_batch() uses to build prompts.

    Returns:
        Dict mapping category -> {persona, emphasis, v2_additions: str}.
    """
    from src.templates import CATEGORY_TEMPLATES

    # Load failure patterns from the metrics
    metrics_path = _PROJECT_ROOT / "results" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        per_mode = metrics.get("per_mode_failures", {})
    else:
        per_mode = {}

    # Global top failure modes (modes with >10% failure rate)
    global_failures = [
        mode for mode, data in per_mode.items()
        if isinstance(data, dict) and int(data.get("count", 0)) > 0
    ]

    v2_templates: dict[str, dict] = {}
    for category, template in CATEGORY_TEMPLATES.items():
        # Build v2 additions for modes that affect this category globally
        additions = []
        for mode in global_failures:
            if mode in _V2_MODE_INSTRUCTIONS:
                additions.append(_V2_MODE_INSTRUCTIONS[mode])

        v2_templates[category] = {
            "persona": template.persona,
            "emphasis": template.emphasis,
            "v2_additions": "\n\n".join(additions),
        }

    return v2_templates


def build_v2_system_prompt(category: str, v2_template: dict) -> str:
    """Build a v2 system prompt with failure-prevention instructions."""
    display_category = category.replace("_", " ")
    base = (
        f"You are a {v2_template['persona']}. "
        f"You specialize in {display_category}.\n\n"
        f"Emphasis: {v2_template['emphasis']}."
    )
    if v2_template["v2_additions"]:
        base += f"\n\n--- QUALITY REQUIREMENTS ---\n{v2_template['v2_additions']}"
    return base


def build_v2_user_prompt(category: str, difficulty: str, variant: int = 0) -> str:
    """Build a v2 user prompt (same structure as v1 but with v2 system context)."""
    from src.templates import DIFFICULTY_MODIFIERS
    modifier = DIFFICULTY_MODIFIERS[difficulty]
    display_category = category.replace("_", " ")
    article = "an" if difficulty[0] in "aeiou" else "a"

    prompt = (
        f"Generate {article} {difficulty}-level DIY repair Q&A for "
        f"{display_category}. The homeowner should be able to follow "
        f"your instructions safely.\n\n"
        f"{modifier}\n\n"
        "Include realistic tools that a homeowner would actually own, "
        "clear step-by-step instructions, and appropriate safety warnings."
    )
    if variant > 0:
        prompt += (
            f"\n\nIMPORTANT: This is variation #{variant + 1}. Generate a "
            f"COMPLETELY DIFFERENT repair scenario than what you might have "
            f"generated before. Pick a different specific problem, different "
            f"equipment, and different steps."
        )
    return prompt


def _v2_prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Cache key for v2 generation."""
    combined = f"v2:{system_prompt}\n---\n{user_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


def generate_v2_batch(
    client: instructor.Instructor | None = None,
    *,
    use_cache: bool = True,
    records_per_combo: int = 2,
) -> list[GeneratedRecord]:
    """Generate a fresh batch of 30 records using v2 templates.

    Same generation matrix as v1 (5 categories × 3 difficulties × 2),
    but using v2 system prompts with failure-prevention instructions.
    """
    if client is None:
        client = _create_client()

    from src.generator import CATEGORIES, DIFFICULTIES, load_from_cache, save_to_cache

    v2_templates = build_v2_templates()
    results: list[GeneratedRecord] = []
    total = len(CATEGORIES) * len(DIFFICULTIES) * records_per_combo

    for category in CATEGORIES:
        v2_tmpl = v2_templates[category]
        for difficulty in DIFFICULTIES:
            for i in range(records_per_combo):
                system_prompt = build_v2_system_prompt(category, v2_tmpl)
                user_prompt = build_v2_user_prompt(category, difficulty, variant=i)
                cache_key = _v2_prompt_hash(system_prompt, user_prompt)

                from_cache = False
                if use_cache:
                    cached = load_from_cache(cache_key)
                    if cached is not None:
                        logger.info("V2 cache hit: %s/%s v%d", category, difficulty, i)
                        record = cached
                        from_cache = True

                if not from_cache:
                    logger.info(
                        "[%d/%d] V2 generating %s/%s v%d...",
                        len(results) + 1, total, category, difficulty, i,
                    )
                    record = client.chat.completions.create(
                        model=_GENERATION_MODEL,
                        response_model=DIYRepairRecord,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=_TEMPERATURE,
                        max_retries=_MAX_RETRIES,
                    )
                    save_to_cache(cache_key, category, difficulty, _GENERATION_MODEL, record)

                wrapped = GeneratedRecord(
                    trace_id=str(uuid.uuid4()),
                    category=category,
                    difficulty=difficulty,
                    template_version="v2",
                    generation_timestamp=datetime.now(timezone.utc).isoformat(),
                    model_used=_GENERATION_MODEL,
                    prompt_hash=cache_key,
                    record=record,
                )
                results.append(wrapped)

    logger.info("V2 batch complete: %d records generated", len(results))
    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_corrected_records(
    records: list[GeneratedRecord],
    filename: str = "corrected_records.json",
) -> Path:
    """Save corrected records to data/corrected/."""
    _CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _CORRECTED_DIR / filename
    output_path.write_text(
        json.dumps([r.model_dump() for r in records], indent=2)
    )
    logger.info("Saved %d corrected records to %s", len(records), output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from src.generator import load_generated_records, save_generated_records

    # Load v1 records + judge results
    print("Loading v1 records and judge results...")
    v1_records = load_generated_records("batch_v1.json")
    judge_results = json.loads(
        (_LABELS_DIR / "llm_labels.json").read_text()
    )

    # Strategy A: Individual correction
    print("\n=== Strategy A: Individual Record Correction ===")
    corrected = correct_batch(v1_records, judge_results)
    save_corrected_records(corrected)

    # Strategy B: Template v2 generation
    print("\n=== Strategy B: Template V2 Generation ===")
    print("Analyzing failure patterns...")
    patterns = analyze_failure_patterns(v1_records, judge_results)
    for cat, modes in patterns.items():
        print(f"  {cat}: {modes}")

    print("\nGenerating v2 batch...")
    v2_records = generate_v2_batch()
    save_generated_records(v2_records, filename="batch_v2.json")

    print(f"\nDone! Corrected: {len(corrected)}, V2: {len(v2_records)}")
