"""Tests for src/evaluator.py — LLM-as-Judge evaluation and agreement analysis.

Covers:
- _build_judge_user_prompt: includes category, difficulty, record JSON
- _judge_cache_key: starts with 'judge_', is deterministic
- _load_judge_cache: cache miss, hit, corrupt JSON
- _save_judge_cache: writes valid JSON to cache dir
- evaluate_record: cache hit returns cached result; cache miss calls client
- evaluate_batch: all succeed, exception is caught and skipped
- save_llm_labels: writes CSV with correct headers and one row per result
- save_llm_labels_json: writes JSON list
- load_manual_labels: returns [] when file missing; list of dicts when present
- compute_agreement: all agree, none agree, unmatched trace_ids, empty modes
- _load_llm_labels_csv: reads CSV into list of dicts
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluator import (
    _build_judge_user_prompt,
    _judge_cache_key,
    _load_judge_cache,
    _save_judge_cache,
    compute_agreement,
    evaluate_batch,
    evaluate_record,
    load_manual_labels,
    save_llm_labels,
    save_llm_labels_json,
)
from src.schemas import DIYRepairRecord, FailureLabel, GeneratedRecord, JudgeResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAILURE_MODES = [
    "incomplete_answer",
    "safety_violations",
    "unrealistic_tools",
    "overcomplicated_solution",
    "missing_context",
    "poor_quality_tips",
]


def _make_generated_record(trace_id: str = "r1") -> GeneratedRecord:
    record = DIYRepairRecord(
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
    return GeneratedRecord(
        trace_id=trace_id,
        category="plumbing_repair",
        difficulty="beginner",
        template_version="v1",
        generation_timestamp="2026-02-08T22:00:00Z",
        model_used="gpt-4o-mini",
        prompt_hash="abc123",
        record=record,
    )


def _make_judge_result(trace_id: str = "r1", fail_modes: list[str] | None = None) -> JudgeResult:
    """Build a JudgeResult; modes in fail_modes get label=1, others 0."""
    fail_set = set(fail_modes or [])
    labels = [
        FailureLabel(
            mode=m,
            label=1 if m in fail_set else 0,
            reason="test reason",
        )
        for m in _FAILURE_MODES
    ]
    return JudgeResult(trace_id=trace_id, labels=labels, overall_quality_score=3)


def _mock_client(judge_result: JudgeResult) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = judge_result
    return client


# ===========================================================================
# _build_judge_user_prompt
# ===========================================================================


class TestBuildJudgeUserPrompt:
    """Tests for _build_judge_user_prompt."""

    def test_build_judge_user_prompt_contains_category(self) -> None:
        record = _make_generated_record("r1")
        prompt = _build_judge_user_prompt(record)
        assert "plumbing_repair" in prompt

    def test_build_judge_user_prompt_contains_difficulty(self) -> None:
        record = _make_generated_record("r1")
        prompt = _build_judge_user_prompt(record)
        assert "beginner" in prompt

    def test_build_judge_user_prompt_contains_record_json(self) -> None:
        record = _make_generated_record("r1")
        prompt = _build_judge_user_prompt(record)
        assert "leaking" in prompt.lower()

    def test_build_judge_user_prompt_mentions_quality_score(self) -> None:
        record = _make_generated_record("r1")
        prompt = _build_judge_user_prompt(record)
        assert "quality" in prompt.lower() or "score" in prompt.lower()


# ===========================================================================
# _judge_cache_key
# ===========================================================================


class TestJudgeCacheKey:
    """Tests for _judge_cache_key."""

    def test_judge_cache_key_starts_with_judge_prefix(self) -> None:
        record = _make_generated_record("r1")
        key = _judge_cache_key(record)
        assert key.startswith("judge_")

    def test_judge_cache_key_is_deterministic(self) -> None:
        record = _make_generated_record("r1")
        key1 = _judge_cache_key(record)
        key2 = _judge_cache_key(record)
        assert key1 == key2

    def test_judge_cache_key_differs_for_different_record_content(self) -> None:
        r1 = _make_generated_record("r1")
        # r2 has a different category → different prompt content → different key
        import copy
        r2_data = r1.model_dump()
        r2_data["category"] = "electrical_repair"
        r2_data["trace_id"] = "r2"
        from src.schemas import GeneratedRecord as GR
        r2 = GR.model_validate(r2_data)
        assert _judge_cache_key(r1) != _judge_cache_key(r2)


# ===========================================================================
# _load_judge_cache
# ===========================================================================


class TestLoadJudgeCache:
    """Tests for _load_judge_cache."""

    def test_load_judge_cache_when_no_file_returns_none(self, tmp_path: Path) -> None:
        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = _load_judge_cache("nonexistent_key")
        assert result is None

    def test_load_judge_cache_when_valid_file_returns_judge_result(self, tmp_path: Path) -> None:
        judge = _make_judge_result("r1")
        payload = {"cache_key": "k1", "trace_id": "r1", "model": "gpt-4o", "response": judge.model_dump()}
        (tmp_path / "k1.json").write_text(json.dumps(payload))

        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = _load_judge_cache("k1")

        assert result is not None
        assert result.trace_id == "r1"

    def test_load_judge_cache_when_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("NOT JSON{{")
        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = _load_judge_cache("bad")
        assert result is None

    def test_load_judge_cache_when_missing_response_key_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "nokey.json").write_text(json.dumps({"cache_key": "nokey"}))
        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = _load_judge_cache("nokey")
        assert result is None


# ===========================================================================
# _save_judge_cache
# ===========================================================================


class TestSaveJudgeCache:
    """Tests for _save_judge_cache."""

    def test_save_judge_cache_creates_json_file(self, tmp_path: Path) -> None:
        judge = _make_judge_result("r1")
        with patch("src.evaluator._CACHE_DIR", tmp_path):
            _save_judge_cache("key1", "r1", judge)
        assert (tmp_path / "key1.json").exists()

    def test_save_judge_cache_file_has_response_key(self, tmp_path: Path) -> None:
        judge = _make_judge_result("r1")
        with patch("src.evaluator._CACHE_DIR", tmp_path):
            _save_judge_cache("key2", "r1", judge)
        data = json.loads((tmp_path / "key2.json").read_text())
        assert "response" in data
        assert data["trace_id"] == "r1"

    def test_save_judge_cache_creates_parent_dir_if_missing(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "cache"
        judge = _make_judge_result("r1")
        with patch("src.evaluator._CACHE_DIR", nested):
            _save_judge_cache("k3", "r1", judge)
        assert (nested / "k3.json").exists()


# ===========================================================================
# evaluate_record
# ===========================================================================


class TestEvaluateRecord:
    """Tests for evaluate_record."""

    def test_evaluate_record_when_cache_hit_returns_cached(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1")
        judge = _make_judge_result("r1")
        # Pre-populate cache with the correct key
        cache_key = _judge_cache_key(record)
        payload = {
            "cache_key": cache_key,
            "trace_id": "r1",
            "model": "gpt-4o",
            "response": judge.model_dump(),
        }
        (tmp_path / f"{cache_key}.json").write_text(json.dumps(payload))

        client = _mock_client(judge)
        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = evaluate_record(client, record, use_cache=True)

        assert result.trace_id == "r1"
        client.chat.completions.create.assert_not_called()

    def test_evaluate_record_when_cache_miss_calls_client(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1")
        judge = _make_judge_result("r1")
        client = _mock_client(judge)

        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = evaluate_record(client, record, use_cache=True)

        client.chat.completions.create.assert_called_once()
        assert result.trace_id == "r1"

    def test_evaluate_record_sets_trace_id_when_mismatched(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1")
        # Return judge result with wrong trace_id — evaluate_record should fix it
        judge_wrong_id = _make_judge_result("wrong_id")
        client = _mock_client(judge_wrong_id)

        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = evaluate_record(client, record, use_cache=False)

        assert result.trace_id == "r1"

    def test_evaluate_record_saves_to_cache_on_miss(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1")
        judge = _make_judge_result("r1")
        client = _mock_client(judge)

        with patch("src.evaluator._CACHE_DIR", tmp_path):
            evaluate_record(client, record, use_cache=False)
            cache_key = _judge_cache_key(record)

        assert (tmp_path / f"{cache_key}.json").exists()


# ===========================================================================
# evaluate_batch
# ===========================================================================


class TestEvaluateBatch:
    """Tests for evaluate_batch."""

    def test_evaluate_batch_returns_one_result_per_record(self, tmp_path: Path) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        # Each record needs its own judge result (trace_id must match)
        results = [_make_judge_result("r1"), _make_judge_result("r2")]
        client = MagicMock()
        client.chat.completions.create.side_effect = results

        with patch("src.evaluator._CACHE_DIR", tmp_path):
            batch_results = evaluate_batch(records, client, use_cache=False)

        assert len(batch_results) == 2

    def test_evaluate_batch_skips_failed_records(self, tmp_path: Path) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        client = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("LLM error")
            return _make_judge_result("r2")

        client.chat.completions.create.side_effect = side_effect

        with patch("src.evaluator._CACHE_DIR", tmp_path):
            batch_results = evaluate_batch(records, client, use_cache=False)

        assert len(batch_results) == 1

    def test_evaluate_batch_empty_list_returns_empty(self, tmp_path: Path) -> None:
        client = MagicMock()
        with patch("src.evaluator._CACHE_DIR", tmp_path):
            result = evaluate_batch([], client, use_cache=False)
        assert result == []


# ===========================================================================
# save_llm_labels
# ===========================================================================


class TestSaveLlmLabels:
    """Tests for save_llm_labels."""

    def test_save_llm_labels_creates_csv_file(self, tmp_path: Path) -> None:
        results = [_make_judge_result("r1"), _make_judge_result("r2")]
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            path = save_llm_labels(results, "labels.csv")
        assert path.exists()
        assert path.suffix == ".csv"

    def test_save_llm_labels_has_correct_headers(self, tmp_path: Path) -> None:
        results = [_make_judge_result("r1")]
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            path = save_llm_labels(results, "labels.csv")
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        assert "trace_id" in fieldnames
        assert "incomplete_answer" in fieldnames
        assert "overall_quality_score" in fieldnames

    def test_save_llm_labels_one_row_per_result(self, tmp_path: Path) -> None:
        results = [_make_judge_result("r1"), _make_judge_result("r2")]
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            path = save_llm_labels(results, "labels.csv")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_save_llm_labels_trace_id_correct(self, tmp_path: Path) -> None:
        results = [_make_judge_result("r1")]
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            path = save_llm_labels(results, "labels.csv")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["trace_id"] == "r1"


# ===========================================================================
# save_llm_labels_json
# ===========================================================================


class TestSaveLlmLabelsJson:
    """Tests for save_llm_labels_json."""

    def test_save_llm_labels_json_creates_json_file(self, tmp_path: Path) -> None:
        results = [_make_judge_result("r1")]
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            path = save_llm_labels_json(results, "labels.json")
        assert path.exists()

    def test_save_llm_labels_json_content_is_list(self, tmp_path: Path) -> None:
        results = [_make_judge_result("r1"), _make_judge_result("r2")]
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            path = save_llm_labels_json(results, "labels.json")
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_save_llm_labels_json_preserves_reasons(self, tmp_path: Path) -> None:
        results = [_make_judge_result("r1")]
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            path = save_llm_labels_json(results, "labels.json")
        data = json.loads(path.read_text())
        # Labels should have reason field
        assert "reason" in data[0]["labels"][0]


# ===========================================================================
# load_manual_labels
# ===========================================================================


class TestLoadManualLabels:
    """Tests for load_manual_labels."""

    def test_load_manual_labels_when_file_missing_returns_empty_list(self, tmp_path: Path) -> None:
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            result = load_manual_labels("nonexistent.csv")
        assert result == []

    def test_load_manual_labels_when_file_exists_returns_list_of_dicts(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "manual.csv"
        csv_path.write_text("trace_id,incomplete_answer\nr1,0\nr2,1\n")
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            result = load_manual_labels("manual.csv")
        assert len(result) == 2
        assert result[0]["trace_id"] == "r1"

    def test_load_manual_labels_returns_correct_values(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "manual.csv"
        csv_path.write_text("trace_id,incomplete_answer\nr1,1\n")
        with patch("src.evaluator._LABELS_DIR", tmp_path):
            result = load_manual_labels("manual.csv")
        assert result[0]["incomplete_answer"] == "1"


# ===========================================================================
# compute_agreement
# ===========================================================================


class TestComputeAgreement:
    """Tests for compute_agreement."""

    def _make_row(self, trace_id: str, mode_vals: dict[str, str]) -> dict:
        row = {"trace_id": trace_id}
        for m in _FAILURE_MODES:
            row[m] = mode_vals.get(m, "0")
        return row

    def test_compute_agreement_when_all_agree_returns_100_pct(self) -> None:
        manual = [self._make_row("r1", {"incomplete_answer": "1", "safety_violations": "0"})]
        llm = [self._make_row("r1", {"incomplete_answer": "1", "safety_violations": "0"})]
        result = compute_agreement(manual, llm)
        assert result["overall_agreement"] == "100.0%"
        assert result["matched_records"] == 1

    def test_compute_agreement_when_no_match_returns_zero(self) -> None:
        manual = [self._make_row("r1", {"incomplete_answer": "1"})]
        llm = [self._make_row("r1", {"incomplete_answer": "0"})]
        result = compute_agreement(manual, llm)
        # Only one mode disagreed — overall < 100%
        assert result["overall_agreement"] != "100.0%"

    def test_compute_agreement_when_trace_id_unmatched_skips(self) -> None:
        manual = [self._make_row("r1", {})]
        llm = [self._make_row("r2", {})]  # Different trace_id
        result = compute_agreement(manual, llm)
        assert result["matched_records"] == 0

    def test_compute_agreement_when_empty_mode_value_skips_mode(self) -> None:
        manual = [{"trace_id": "r1", "incomplete_answer": "", "safety_violations": "1",
                   **{m: "" for m in _FAILURE_MODES[2:]}}]
        llm = [{"trace_id": "r1", "incomplete_answer": "1", "safety_violations": "1",
                **{m: "0" for m in _FAILURE_MODES[2:]}}]
        result = compute_agreement(manual, llm)
        # Empty manual value for incomplete_answer → skipped; safety_violations agrees
        assert result["per_mode_agreement"]["incomplete_answer"] == "N/A"

    def test_compute_agreement_returns_required_keys(self) -> None:
        result = compute_agreement([], [])
        assert "matched_records" in result
        assert "per_mode_agreement" in result
        assert "overall_agreement" in result


# ===========================================================================
# _load_llm_labels_csv
# ===========================================================================


class TestLoadLlmLabelsCsv:
    """Tests for _load_llm_labels_csv."""

    def test_load_llm_labels_csv_returns_list_of_dicts(self, tmp_path: Path) -> None:
        from src.evaluator import _load_llm_labels_csv
        csv_path = tmp_path / "llm_labels.csv"
        csv_path.write_text("trace_id,incomplete_answer\nr1,0\nr2,1\n")

        with patch("src.evaluator._LABELS_DIR", tmp_path):
            result = _load_llm_labels_csv("llm_labels.csv")

        assert len(result) == 2
        assert result[0]["trace_id"] == "r1"
