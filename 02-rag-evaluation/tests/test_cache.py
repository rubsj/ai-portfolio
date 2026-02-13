"""Tests for cache.py — JSON file cache for LLM responses.

Verifies cache key computation, save/load roundtrip, graceful degradation
on corrupt data, and lazy directory creation.
"""

from __future__ import annotations

import json

import pytest

from src.cache import compute_cache_key, load_cached, save_cached


# ===========================================================================
# Cache Key Tests
# ===========================================================================

class TestCacheKey:
    """Verify MD5 cache key computation."""

    def test_cache_key_deterministic(self) -> None:
        """Same inputs always produce the same hash."""
        key1 = compute_cache_key("gpt-4o", "Hello world")
        key2 = compute_cache_key("gpt-4o", "Hello world")
        assert key1 == key2

    def test_cache_key_is_32_char_hex(self) -> None:
        """MD5 produces a 32-character hex string."""
        key = compute_cache_key("model", "prompt")
        assert len(key) == 32
        assert all(c in "0123456789abcdef" for c in key)

    def test_cache_key_differs_for_different_prompts(self) -> None:
        """Different prompts produce different hashes."""
        key1 = compute_cache_key("gpt-4o", "prompt A")
        key2 = compute_cache_key("gpt-4o", "prompt B")
        assert key1 != key2

    def test_cache_key_differs_for_different_models(self) -> None:
        """Same prompt with different models produces different hashes."""
        key1 = compute_cache_key("gpt-4o", "Hello")
        key2 = compute_cache_key("gpt-4o-mini", "Hello")
        assert key1 != key2


# ===========================================================================
# Save / Load Roundtrip Tests
# ===========================================================================

class TestSaveAndLoad:
    """Verify save/load roundtrip and cache miss behavior."""

    def test_cache_miss_returns_none(self, tmp_path, monkeypatch) -> None:
        """Non-existent key returns None."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        result = load_cached("nonexistent_key_abc123")
        assert result is None

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch) -> None:
        """Save a response, load it back — data matches."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        response = {"answer": "42", "tokens": 10}
        key = compute_cache_key("gpt-4o", "What is the answer?")

        save_cached(key, response, model="gpt-4o")
        loaded = load_cached(key)

        assert loaded == response

    def test_save_with_metadata(self, tmp_path, monkeypatch) -> None:
        """Metadata is stored in the cache file alongside the response."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        key = "test_meta_key"
        response = {"result": "ok"}
        metadata = {"question_id": "Q1"}

        save_cached(key, response, metadata=metadata)

        # Read raw JSON to verify metadata is present
        raw = json.loads((tmp_path / f"{key}.json").read_text())
        assert raw["question_id"] == "Q1"
        assert raw["response"] == response

    def test_cache_dir_created_on_save(self, tmp_path, monkeypatch) -> None:
        """Cache directory is created lazily on first write."""
        cache_dir = tmp_path / "nested" / "cache"
        monkeypatch.setattr("src.cache.CACHE_DIR", cache_dir)
        assert not cache_dir.exists()

        save_cached("key123", {"data": 1})

        assert cache_dir.exists()
        assert (cache_dir / "key123.json").exists()


# ===========================================================================
# Graceful Degradation Tests
# ===========================================================================

class TestGracefulDegradation:
    """Verify corrupt/stale cache returns None, never crashes."""

    def test_corrupt_json_returns_none(self, tmp_path, monkeypatch) -> None:
        """Malformed JSON file returns None."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        (tmp_path / "bad_key.json").write_text("not valid json {{{")

        result = load_cached("bad_key")
        assert result is None

    def test_missing_response_key_returns_none(self, tmp_path, monkeypatch) -> None:
        """Valid JSON but missing 'response' key returns None."""
        monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path)
        (tmp_path / "no_resp.json").write_text(json.dumps({"model": "gpt-4o"}))

        result = load_cached("no_resp")
        assert result is None
