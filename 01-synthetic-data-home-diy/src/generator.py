"""Instructor-based generation pipeline with JSON file caching.

Generates synthetic DIY repair Q&A records using GPT-4o-mini via Instructor.
Each record is validated by Pydantic on the way out (Instructor auto-retries
on validation failures). Results are cached to avoid redundant API calls.

Java/TS parallel: like a service class with an HTTP client (Instructor/OpenAI)
and a file-based cache layer — similar to Spring's @Cacheable but manual.
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
from src.templates import (
    TEMPLATE_VERSION,
    build_messages,
    build_system_prompt,
    build_user_prompt,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Paths relative to the project root (01-synthetic-data-home-diy/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"

# LLM parameters (PRD Section 2 decisions)
_GENERATION_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.7
_MAX_RETRIES = 3

# Generation matrix: 2 records per (category, difficulty) combo = 30 total
_RECORDS_PER_COMBO = 2

# All valid categories and difficulties for the generation matrix
CATEGORIES: list[Category] = [
    "appliance_repair",
    "plumbing_repair",
    "electrical_repair",
    "hvac_maintenance",
    "general_home_repair",
]

DIFFICULTIES: list[Difficulty] = ["beginner", "intermediate", "advanced"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache layer (PRD Section 4d)
# ---------------------------------------------------------------------------

def _prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """MD5 hash of the full prompt — used as cache key.

    Combines system + user prompts into a single string before hashing.
    MD5 is fine here: we only need collision-resistant keying, not security.
    """
    combined = f"{system_prompt}\n---\n{user_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


def _cache_path(cache_key: str) -> Path:
    """Return the path to a cache file for a given key."""
    return _CACHE_DIR / f"{cache_key}.json"


def load_from_cache(cache_key: str) -> DIYRepairRecord | None:
    """Load a cached LLM response if it exists.

    Returns None on cache miss or if the cached data fails validation
    (e.g., after a schema change).
    """
    path = _cache_path(cache_key)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        # The "response" key holds the DIYRepairRecord as a dict
        return DIYRepairRecord.model_validate(data["response"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        logger.warning("Cache hit but failed to load %s: %s", cache_key, exc)
        return None


def save_to_cache(
    cache_key: str,
    category: Category,
    difficulty: Difficulty,
    model: str,
    record: DIYRepairRecord,
) -> None:
    """Save an LLM response to the JSON file cache.

    Cache format matches PRD Section 4d spec.
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "prompt_hash": cache_key,
        "category": category,
        "difficulty": difficulty,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response": record.model_dump(),
    }
    _cache_path(cache_key).write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Single-record generation
# ---------------------------------------------------------------------------

def _create_client() -> instructor.Instructor:
    """Create an Instructor-wrapped OpenAI client.

    Loads .env for the OPENAI_API_KEY. Instructor wraps the client to handle
    JSON schema injection, response parsing, and automatic retries.

    Java/TS parallel: like creating a RestTemplate or Axios instance with
    interceptors — Instructor is the middleware layer.
    """
    load_dotenv(_PROJECT_ROOT / ".env")
    return instructor.from_openai(OpenAI())


def generate_record(
    client: instructor.Instructor,
    category: Category,
    difficulty: Difficulty,
    *,
    use_cache: bool = True,
) -> tuple[DIYRepairRecord, str, bool]:
    """Generate a single DIY repair record via Instructor.

    Args:
        client: Instructor-wrapped OpenAI client.
        category: One of 5 repair categories.
        difficulty: beginner, intermediate, or advanced.
        use_cache: If True (default), check cache before calling the API.

    Returns:
        Tuple of (record, prompt_hash, from_cache) where from_cache indicates
        whether the result was loaded from cache (True) or freshly generated (False).
    """
    system_prompt = build_system_prompt(category)
    user_prompt = build_user_prompt(category, difficulty)
    cache_key = _prompt_hash(system_prompt, user_prompt)

    # Check cache first (PRD: "Before each LLM call, check cache")
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached is not None:
            logger.info("Cache hit for %s/%s (%s)", category, difficulty, cache_key[:8])
            return cached, cache_key, True

    # Cache miss — call the LLM via Instructor
    messages = build_messages(category, difficulty)
    logger.info("Generating %s/%s via %s...", category, difficulty, _GENERATION_MODEL)

    record = client.chat.completions.create(
        model=_GENERATION_MODEL,
        response_model=DIYRepairRecord,
        messages=messages,
        temperature=_TEMPERATURE,
        max_retries=_MAX_RETRIES,
    )

    # Cache the result for next time
    save_to_cache(cache_key, category, difficulty, _GENERATION_MODEL, record)
    return record, cache_key, False


# ---------------------------------------------------------------------------
# Batch generation (PRD Section 4b — 5 categories × 3 difficulties × 2)
# ---------------------------------------------------------------------------

def generate_batch(
    client: instructor.Instructor | None = None,
    *,
    use_cache: bool = True,
    records_per_combo: int = _RECORDS_PER_COMBO,
) -> list[GeneratedRecord]:
    """Generate the full matrix of 30 DIY repair records.

    For each (category, difficulty) combo, generates `records_per_combo` records.
    Each call uses a slightly different user prompt (appending the iteration index)
    to ensure unique cache keys and varied outputs.

    Args:
        client: Instructor-wrapped OpenAI client. Created if not provided.
        use_cache: Whether to use the JSON file cache.
        records_per_combo: How many records per (category, difficulty) pair.

    Returns:
        List of GeneratedRecord wrappers with metadata.
    """
    if client is None:
        client = _create_client()

    results: list[GeneratedRecord] = []
    total = len(CATEGORIES) * len(DIFFICULTIES) * records_per_combo
    generated_count = 0
    cache_hit_count = 0
    failed_count = 0

    for category in CATEGORIES:
        for difficulty in DIFFICULTIES:
            for i in range(records_per_combo):
                generated_count += 1
                logger.info(
                    "[%d/%d] %s / %s (variant %d)",
                    generated_count, total, category, difficulty, i + 1,
                )

                try:
                    # For variants beyond the first, append a variation hint
                    # to get a different prompt hash and different LLM output
                    record, prompt_hash, from_cache = _generate_variant(
                        client, category, difficulty, variant=i, use_cache=use_cache
                    )

                    if from_cache:
                        cache_hit_count += 1

                    # Wrap in GeneratedRecord with pipeline metadata
                    wrapped = GeneratedRecord(
                        trace_id=str(uuid.uuid4()),
                        category=category,
                        difficulty=difficulty,
                        template_version=TEMPLATE_VERSION,
                        generation_timestamp=datetime.now(timezone.utc).isoformat(),
                        model_used=_GENERATION_MODEL,
                        prompt_hash=prompt_hash,
                        record=record,
                    )
                    results.append(wrapped)

                except (ValidationError, Exception) as exc:
                    # Instructor retries internally, but if it exhausts max_retries
                    # it raises the last ValidationError. Log and continue.
                    failed_count += 1
                    logger.error(
                        "Failed to generate %s/%s variant %d: %s",
                        category, difficulty, i + 1, exc,
                    )

    logger.info(
        "Batch complete: %d generated, %d from cache, %d failed",
        len(results), cache_hit_count, failed_count,
    )
    return results


def _generate_variant(
    client: instructor.Instructor,
    category: Category,
    difficulty: Difficulty,
    *,
    variant: int,
    use_cache: bool = True,
) -> tuple[DIYRepairRecord, str, bool]:
    """Generate a single variant, appending a variation hint for variants > 0.

    Variant 0 uses the base prompt. Variants 1+ append a short instruction to
    produce a different scenario, ensuring unique cache keys and diverse outputs.
    """
    system_prompt = build_system_prompt(category)
    user_prompt = build_user_prompt(category, difficulty)

    if variant > 0:
        # Append variation hint to produce a different scenario
        user_prompt += (
            f"\n\nIMPORTANT: This is variation #{variant + 1}. Generate a "
            f"COMPLETELY DIFFERENT repair scenario than what you might have "
            f"generated before. Pick a different specific problem, different "
            f"equipment, and different steps."
        )

    cache_key = _prompt_hash(system_prompt, user_prompt)

    # Check cache
    if use_cache:
        cached = load_from_cache(cache_key)
        if cached is not None:
            logger.info("Cache hit for %s/%s v%d (%s)", category, difficulty, variant, cache_key[:8])
            return cached, cache_key, True

    # Call LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    record = client.chat.completions.create(
        model=_GENERATION_MODEL,
        response_model=DIYRepairRecord,
        messages=messages,
        temperature=_TEMPERATURE,
        max_retries=_MAX_RETRIES,
    )

    save_to_cache(cache_key, category, difficulty, _GENERATION_MODEL, record)
    return record, cache_key, False


# ---------------------------------------------------------------------------
# Persistence — save/load generated records to data/generated/
# ---------------------------------------------------------------------------

def save_generated_records(records: list[GeneratedRecord], filename: str = "batch_v1.json") -> Path:
    """Save generated records to a JSON file in data/generated/.

    Args:
        records: List of GeneratedRecord objects to save.
        filename: Output filename (default: batch_v1.json).

    Returns:
        Path to the saved file.
    """
    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _GENERATED_DIR / filename

    # model_dump() converts Pydantic models to dicts for JSON serialization
    payload = [r.model_dump() for r in records]
    output_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved %d records to %s", len(records), output_path)
    return output_path


def load_generated_records(filename: str = "batch_v1.json") -> list[GeneratedRecord]:
    """Load generated records from a JSON file.

    Args:
        filename: Input filename to load from data/generated/.

    Returns:
        List of validated GeneratedRecord objects.
    """
    path = _GENERATED_DIR / filename
    raw = json.loads(path.read_text())
    return [GeneratedRecord.model_validate(item) for item in raw]


# ---------------------------------------------------------------------------
# CLI entry point — run as `python -m src.generator` from project root
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("Starting batch generation (30 records)...")
    records = generate_batch()
    output = save_generated_records(records)
    print(f"\nDone! {len(records)} records saved to {output}")
    print(f"Success rate: {len(records)}/30 ({len(records)/30*100:.0f}%)")
