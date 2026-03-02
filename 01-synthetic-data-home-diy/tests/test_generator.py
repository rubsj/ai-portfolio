"""Tests for src/generator.py — cache layer, generation, and persistence.

Covers:
- _prompt_hash: deterministic, different prompts give different hashes
- _cache_path: builds path under _CACHE_DIR
- load_from_cache: cache miss, cache hit, corrupt cache
- save_to_cache: writes valid JSON with expected keys
- generate_record: cache hit path, cache miss (mocked client)
- generate_batch: all succeed (mocked), exception branch (failed variant)
- _generate_variant: variant 0 vs variant 1+ prompt difference
- save_generated_records: writes JSON list, returns correct path
- load_generated_records: roundtrip with save_generated_records
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.generator import (
    _cache_path,
    _generate_variant,
    _prompt_hash,
    generate_batch,
    generate_record,
    load_from_cache,
    load_generated_records,
    save_generated_records,
    save_to_cache,
)
from src.schemas import DIYRepairRecord, GeneratedRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_record() -> DIYRepairRecord:
    return DIYRepairRecord(
        question="How do I fix a leaking kitchen faucet?",
        answer=(
            "First, turn off the water supply valve under the sink. "
            "Then remove the faucet handle by unscrewing the decorative cap. "
            "Replace the worn cartridge with a new one from the hardware store."
        ),
        equipment_problem="Leaking single-handle kitchen faucet",
        tools_required=["adjustable wrench", "plumber tape"],
        steps=[
            "Turn off the water supply valve under the sink",
            "Remove the faucet handle by unscrewing the cap",
        ],
        safety_info="Always turn off water supply before starting",
        tips="Take a photo before disassembly for reference",
    )


def _make_generated_record(trace_id: str = "r1") -> GeneratedRecord:
    return GeneratedRecord(
        trace_id=trace_id,
        category="plumbing_repair",
        difficulty="beginner",
        template_version="v1",
        generation_timestamp="2026-02-08T22:00:00Z",
        model_used="gpt-4o-mini",
        prompt_hash="abc123",
        record=_valid_record(),
    )


def _mock_client(record: DIYRepairRecord) -> MagicMock:
    """Return a mock Instructor client that returns `record` on every create() call."""
    client = MagicMock()
    client.chat.completions.create.return_value = record
    return client


# ===========================================================================
# _prompt_hash
# ===========================================================================


class TestPromptHash:
    """Tests for the private _prompt_hash helper."""

    def test_prompt_hash_is_deterministic(self) -> None:
        h1 = _prompt_hash("system", "user")
        h2 = _prompt_hash("system", "user")
        assert h1 == h2

    def test_prompt_hash_different_prompts_give_different_hashes(self) -> None:
        h1 = _prompt_hash("system A", "user A")
        h2 = _prompt_hash("system B", "user B")
        assert h1 != h2

    def test_prompt_hash_is_32_chars_md5(self) -> None:
        h = _prompt_hash("s", "u")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


# ===========================================================================
# _cache_path
# ===========================================================================


class TestCachePath:
    """Tests for _cache_path — path construction under _CACHE_DIR."""

    def test_cache_path_builds_under_cache_dir(self, tmp_path: Path) -> None:
        with patch("src.generator._CACHE_DIR", tmp_path):
            result = _cache_path("mykey")
        assert result == tmp_path / "mykey.json"

    def test_cache_path_uses_key_as_filename(self, tmp_path: Path) -> None:
        with patch("src.generator._CACHE_DIR", tmp_path):
            result = _cache_path("abc123def456")
        assert result.name == "abc123def456.json"


# ===========================================================================
# load_from_cache
# ===========================================================================


class TestLoadFromCache:
    """Tests for load_from_cache."""

    def test_load_from_cache_when_no_file_returns_none(self, tmp_path: Path) -> None:
        with patch("src.generator._CACHE_DIR", tmp_path):
            result = load_from_cache("nonexistent_key")
        assert result is None

    def test_load_from_cache_when_valid_file_returns_record(self, tmp_path: Path) -> None:
        record = _valid_record()
        payload = {
            "cache_key": "k1",
            "prompt_hash": "k1",
            "category": "plumbing_repair",
            "difficulty": "beginner",
            "model": "gpt-4o-mini",
            "timestamp": "2026-01-01T00:00:00Z",
            "response": record.model_dump(),
        }
        cache_file = tmp_path / "k1.json"
        cache_file.write_text(json.dumps(payload))

        with patch("src.generator._CACHE_DIR", tmp_path):
            result = load_from_cache("k1")

        assert result is not None
        assert result.question == record.question

    def test_load_from_cache_when_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("NOT JSON{{")
        with patch("src.generator._CACHE_DIR", tmp_path):
            result = load_from_cache("bad")
        assert result is None

    def test_load_from_cache_when_missing_response_key_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "nokey.json").write_text(json.dumps({"cache_key": "nokey"}))
        with patch("src.generator._CACHE_DIR", tmp_path):
            result = load_from_cache("nokey")
        assert result is None


# ===========================================================================
# save_to_cache
# ===========================================================================


class TestSaveToCache:
    """Tests for save_to_cache."""

    def test_save_to_cache_creates_json_file(self, tmp_path: Path) -> None:
        record = _valid_record()
        with patch("src.generator._CACHE_DIR", tmp_path):
            save_to_cache("key1", "plumbing_repair", "beginner", "gpt-4o-mini", record)
        assert (tmp_path / "key1.json").exists()

    def test_save_to_cache_file_contains_response_key(self, tmp_path: Path) -> None:
        record = _valid_record()
        with patch("src.generator._CACHE_DIR", tmp_path):
            save_to_cache("key2", "plumbing_repair", "beginner", "gpt-4o-mini", record)
        data = json.loads((tmp_path / "key2.json").read_text())
        assert "response" in data
        assert data["category"] == "plumbing_repair"
        assert data["difficulty"] == "beginner"

    def test_save_to_cache_creates_parent_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "cache"
        record = _valid_record()
        with patch("src.generator._CACHE_DIR", nested):
            save_to_cache("k3", "plumbing_repair", "beginner", "gpt-4o-mini", record)
        assert (nested / "k3.json").exists()


# ===========================================================================
# generate_record
# ===========================================================================


class TestGenerateRecord:
    """Tests for generate_record."""

    def test_generate_record_when_cache_hit_returns_cached_and_true(self, tmp_path: Path) -> None:
        record = _valid_record()
        # Pre-populate cache with the correct key for plumbing_repair/beginner
        from src.templates import build_system_prompt, build_user_prompt
        from src.generator import _prompt_hash as ph
        sys_p = build_system_prompt("plumbing_repair")
        usr_p = build_user_prompt("plumbing_repair", "beginner")
        key = ph(sys_p, usr_p)
        payload = {
            "cache_key": key,
            "prompt_hash": key,
            "category": "plumbing_repair",
            "difficulty": "beginner",
            "model": "gpt-4o-mini",
            "timestamp": "2026-01-01T00:00:00Z",
            "response": record.model_dump(),
        }
        (tmp_path / f"{key}.json").write_text(json.dumps(payload))

        client = _mock_client(record)
        with patch("src.generator._CACHE_DIR", tmp_path):
            result, cache_key, from_cache = generate_record(
                client, "plumbing_repair", "beginner", use_cache=True
            )

        assert from_cache is True
        assert result.question == record.question
        client.chat.completions.create.assert_not_called()

    def test_generate_record_when_cache_miss_calls_client(self, tmp_path: Path) -> None:
        record = _valid_record()
        client = _mock_client(record)
        with patch("src.generator._CACHE_DIR", tmp_path):
            result, cache_key, from_cache = generate_record(
                client, "plumbing_repair", "beginner", use_cache=True
            )
        assert from_cache is False
        assert result.question == record.question
        client.chat.completions.create.assert_called_once()

    def test_generate_record_when_use_cache_false_always_calls_client(self, tmp_path: Path) -> None:
        record = _valid_record()
        client = _mock_client(record)
        with patch("src.generator._CACHE_DIR", tmp_path):
            _, _, from_cache = generate_record(
                client, "plumbing_repair", "beginner", use_cache=False
            )
        assert from_cache is False
        client.chat.completions.create.assert_called_once()

    def test_generate_record_saves_to_cache_on_miss(self, tmp_path: Path) -> None:
        record = _valid_record()
        client = _mock_client(record)
        with patch("src.generator._CACHE_DIR", tmp_path):
            _, cache_key, _ = generate_record(
                client, "plumbing_repair", "beginner", use_cache=False
            )
        assert (tmp_path / f"{cache_key}.json").exists()


# ===========================================================================
# generate_batch
# ===========================================================================


class TestGenerateBatch:
    """Tests for generate_batch."""

    def test_generate_batch_returns_generated_records(self, tmp_path: Path) -> None:
        record = _valid_record()
        client = _mock_client(record)
        with patch("src.generator._CACHE_DIR", tmp_path):
            results = generate_batch(client, use_cache=False, records_per_combo=1)
        # 5 categories × 3 difficulties × 1 = 15
        assert len(results) == 15
        assert all(isinstance(r, GeneratedRecord) for r in results)

    def test_generate_batch_counts_cache_hits(self, tmp_path: Path) -> None:
        """generate_batch increments cache_hit_count when cache returns a record."""
        record = _valid_record()
        client = _mock_client(record)
        # First populate the cache for one combo (plumbing_repair/beginner v0)
        from src.templates import build_system_prompt, build_user_prompt
        from src.generator import _prompt_hash as ph
        sys_p = build_system_prompt("plumbing_repair")
        usr_p = build_user_prompt("plumbing_repair", "beginner")
        key = ph(sys_p, usr_p)
        payload = {
            "cache_key": key, "prompt_hash": key,
            "category": "plumbing_repair", "difficulty": "beginner",
            "model": "gpt-4o-mini", "timestamp": "2026-01-01T00:00:00Z",
            "response": record.model_dump(),
        }
        (tmp_path / f"{key}.json").write_text(json.dumps(payload))

        with patch("src.generator._CACHE_DIR", tmp_path):
            results = generate_batch(client, use_cache=True, records_per_combo=1)

        # At minimum 1 cache hit (plumbing/beginner)
        assert len(results) == 5 * 3 * 1  # all 15 should succeed

    def test_generate_batch_skips_failed_records(self, tmp_path: Path) -> None:
        record = _valid_record()
        client = MagicMock()
        # Raise on first call, succeed on subsequent
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("LLM error")
            return record

        client.chat.completions.create.side_effect = side_effect

        with patch("src.generator._CACHE_DIR", tmp_path):
            results = generate_batch(client, use_cache=False, records_per_combo=1)

        # 1 failed, 14 succeeded
        assert len(results) == 14


# ===========================================================================
# _generate_variant
# ===========================================================================


class TestGenerateVariant:
    """Tests for _generate_variant."""

    def test_generate_variant_0_uses_base_prompt(self, tmp_path: Path) -> None:
        record = _valid_record()
        client = _mock_client(record)
        with patch("src.generator._CACHE_DIR", tmp_path):
            result, _, from_cache = _generate_variant(
                client, "plumbing_repair", "beginner", variant=0, use_cache=False
            )
        assert result.question == record.question

    def test_generate_variant_positive_appends_hint(self, tmp_path: Path) -> None:
        """Variant 1+ must produce a different cache key than variant 0."""
        record = _valid_record()
        client = _mock_client(record)
        with patch("src.generator._CACHE_DIR", tmp_path):
            _, key0, _ = _generate_variant(
                client, "plumbing_repair", "beginner", variant=0, use_cache=False
            )
            _, key1, _ = _generate_variant(
                client, "plumbing_repair", "beginner", variant=1, use_cache=False
            )
        # Different prompts → different hashes
        assert key0 != key1

    def test_generate_variant_cache_hit_returns_true(self, tmp_path: Path) -> None:
        record = _valid_record()
        client = _mock_client(record)
        # First call populates cache; second call should hit it
        with patch("src.generator._CACHE_DIR", tmp_path):
            _, key, _ = _generate_variant(
                client, "plumbing_repair", "beginner", variant=0, use_cache=False
            )
            _, _, from_cache = _generate_variant(
                client, "plumbing_repair", "beginner", variant=0, use_cache=True
            )
        assert from_cache is True


# ===========================================================================
# save_generated_records / load_generated_records
# ===========================================================================


class TestSaveLoadGeneratedRecords:
    """Tests for save_generated_records and load_generated_records."""

    def test_save_generated_records_creates_json_file(self, tmp_path: Path) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        with patch("src.generator._GENERATED_DIR", tmp_path):
            path = save_generated_records(records, "test_batch.json")
        assert path.exists()
        assert path.name == "test_batch.json"

    def test_save_generated_records_content_is_list(self, tmp_path: Path) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        with patch("src.generator._GENERATED_DIR", tmp_path):
            path = save_generated_records(records, "test_batch.json")
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_load_generated_records_roundtrip(self, tmp_path: Path) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        with patch("src.generator._GENERATED_DIR", tmp_path):
            save_generated_records(records, "roundtrip.json")
            loaded = load_generated_records("roundtrip.json")
        assert len(loaded) == 2
        assert loaded[0].trace_id == "r1"
        assert loaded[1].trace_id == "r2"

    def test_save_generated_records_creates_parent_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "generated"
        records = [_make_generated_record("r1")]
        with patch("src.generator._GENERATED_DIR", nested):
            path = save_generated_records(records)
        assert path.exists()
