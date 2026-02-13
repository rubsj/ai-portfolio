"""JSON file cache for LLM responses — prevents duplicate API calls.

Cache key: MD5 hash of model name + prompt. Same pattern as P1's generator.py.
Cache location: data/cache/ as JSON files (one file per cached response).

Java/TS parallel: like a Redis cache or a Guava LoadingCache keyed on request hash.
The difference is we persist to disk (JSON files) instead of in-memory, because we
want cache to survive process restarts.

WHY MD5 for cache key: we only need collision-resistant keying, not cryptographic
security. MD5 is fast and produces a 32-char hex string — perfect for filenames.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)


def compute_cache_key(model: str, prompt: str) -> str:
    """Compute MD5 hash of model + prompt for cache keying.

    WHY include model in the hash: same prompt sent to different models
    produces different responses — must be cached separately.
    """
    combined = f"{model}\n---\n{prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


def _cache_path(cache_key: str) -> Path:
    """Return the path to a cache file for a given key."""
    return CACHE_DIR / f"{cache_key}.json"


def load_cached(cache_key: str) -> dict | None:
    """Load a cached LLM response if it exists.

    Returns the response dict on cache hit, None on miss or corruption.
    WHY graceful degradation: stale cache (e.g., after schema change) should
    never crash the pipeline — just log and re-call the LLM.
    """
    path = _cache_path(cache_key)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text())
        logger.info("Cache hit: %s", cache_key[:8])
        return data["response"]
    except (json.JSONDecodeError, KeyError) as exc:
        logger.warning("Cache hit but failed to load %s: %s", cache_key[:8], exc)
        return None


def save_cached(
    cache_key: str,
    response: dict,
    *,
    model: str = "",
    metadata: dict | None = None,
) -> None:
    """Save an LLM response to the JSON file cache.

    Creates cache directory lazily on first write (idempotent).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **(metadata or {}),
        "response": response,
    }
    _cache_path(cache_key).write_text(json.dumps(payload, indent=2))
    logger.info("Cached response: %s", cache_key[:8])
