"""Tests for src/corrector.py — correction pipeline helpers and orchestration.

Covers:
- _count_failures: per-mode counts + total + failure_rate
- build_comparison_metrics: 4-stage structure + experiment metadata
- correct_batch: mocked LLM correction with version_tag propagation
- analyze_failure_patterns: sorted modes per category
- run_full_pipeline: orchestration integration test (all LLM calls mocked)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.corrector import (
    _build_correction_prompt,
    _correction_cache_key,
    _count_failures,
    _load_correction_cache,
    _save_correction_cache,
    _v2_prompt_hash,
    analyze_failure_patterns,
    build_comparison_metrics,
    build_v2_system_prompt,
    build_v2_templates,
    build_v2_user_prompt,
    correct_batch,
    correct_record,
    generate_v2_batch,
    run_full_pipeline,
    save_corrected_records,
)
from src.schemas import DIYRepairRecord, GeneratedRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_judge_dict(
    trace_id: str = "test-001",
    failing_modes: list[str] | None = None,
) -> dict:
    """Build a JudgeResult-shaped dict with specified failures."""
    failing = set(failing_modes or [])
    modes = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
    ]
    labels = [
        {
            "mode": m,
            "label": 1 if m in failing else 0,
            "reason": f"{m} flagged" if m in failing else f"{m} passed",
        }
        for m in modes
    ]
    return {
        "trace_id": trace_id,
        "labels": labels,
        "overall_quality_score": 3,
    }


def _make_generated_record(trace_id: str = "test-001") -> GeneratedRecord:
    """Build a minimal valid GeneratedRecord for testing."""
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


# ===========================================================================
# _count_failures
# ===========================================================================


class TestCountFailures:
    """Tests for _count_failures helper."""

    def test_count_failures_when_mixed_results_returns_correct_counts(self) -> None:
        results = [
            _make_judge_dict("r1", ["incomplete_answer", "poor_quality_tips"]),
            _make_judge_dict("r2", ["incomplete_answer"]),
            _make_judge_dict("r3", []),
        ]
        counts = _count_failures(results, num_records=3)
        assert counts["incomplete_answer"] == 2
        assert counts["poor_quality_tips"] == 1
        assert counts["safety_violations"] == 0
        assert counts["total"] == 3
        # 3 failures / (3 records * 6 modes) = 3/18
        assert counts["failure_rate"] == pytest.approx(3 / 18)

    def test_count_failures_when_no_failures_returns_zeros(self) -> None:
        results = [_make_judge_dict("r1", []), _make_judge_dict("r2", [])]
        counts = _count_failures(results, num_records=2)
        assert counts["total"] == 0
        assert counts["failure_rate"] == 0.0

    def test_count_failures_when_empty_list_returns_zeros(self) -> None:
        counts = _count_failures([], num_records=0)
        assert counts["total"] == 0
        assert counts["failure_rate"] == 0.0


# ===========================================================================
# build_comparison_metrics
# ===========================================================================


class TestBuildComparisonMetrics:
    """Tests for build_comparison_metrics."""

    def test_build_comparison_metrics_when_valid_produces_all_stages(self) -> None:
        v1 = [_make_judge_dict("r1", ["incomplete_answer", "poor_quality_tips"])]
        v1c = [_make_judge_dict("r1", ["incomplete_answer"])]
        v2 = [_make_judge_dict("r1", ["poor_quality_tips"])]
        v2c = [_make_judge_dict("r1", [])]

        result = build_comparison_metrics(v1, v1c, v2, v2c, total_records=1)

        assert "v1_original" in result
        assert "corrected" in result
        assert "v2_generated" in result
        assert "v2_corrected" in result
        assert "target_met" in result
        assert result["v1_original"]["total_failures"] == 2
        assert result["corrected"]["total_failures"] == 1
        assert result["v2_generated"]["total_failures"] == 1
        assert result["v2_corrected"]["total_failures"] == 0

    def test_build_comparison_metrics_when_100pct_improvement_sets_target_met(self) -> None:
        v1 = [_make_judge_dict("r1", ["incomplete_answer"])]
        v1c = [_make_judge_dict("r1", [])]
        v2 = [_make_judge_dict("r1", [])]
        v2c = [_make_judge_dict("r1", [])]

        result = build_comparison_metrics(v1, v1c, v2, v2c, total_records=1)

        assert result["target_met"]["corrected_meets_80pct"] is True
        assert result["target_met"]["v2_meets_80pct"] is True
        assert result["target_met"]["v2_corrected_meets_80pct"] is True

    def test_build_comparison_metrics_includes_experiment_metadata(self) -> None:
        v1 = [_make_judge_dict("r1", ["incomplete_answer"])]
        empty = [_make_judge_dict("r1", [])]

        result = build_comparison_metrics(v1, empty, empty, empty, total_records=1)

        assert "generated_at" in result
        assert result["generator_model"] == "gpt-4o-mini"
        assert result["judge_model"] == "gpt-4o"
        assert result["pipeline_version"] == "1.0"


# ===========================================================================
# correct_batch — version_tag propagation
# ===========================================================================


class TestCorrectBatch:
    """Tests for correct_batch with mocked LLM client."""

    def test_correct_batch_when_clean_record_passes_through(self) -> None:
        record = _make_generated_record("r1")
        judge = [_make_judge_dict("r1", [])]

        result = correct_batch(
            [record], judge, client=MagicMock(), version_tag="v1_corrected"
        )

        assert len(result) == 1
        assert result[0].template_version == "v1"  # unchanged

    def test_correct_batch_when_failing_record_gets_corrected(self) -> None:
        record = _make_generated_record("r1")
        judge = [_make_judge_dict("r1", ["incomplete_answer"])]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = record.record

        result = correct_batch(
            [record], judge, client=mock_client, use_cache=False
        )

        assert len(result) == 1
        assert result[0].template_version == "v1_corrected"

    def test_correct_batch_when_no_judge_result_passes_through(self) -> None:
        """When no judge result exists for a record, it passes through unchanged."""
        record = _make_generated_record("r1")
        # Judge result is for a different trace_id
        judge = [_make_judge_dict("r99", ["incomplete_answer"])]

        result = correct_batch(
            [record], judge, client=MagicMock(), version_tag="v1_corrected"
        )

        assert len(result) == 1
        assert result[0].trace_id == "r1"
        assert result[0].template_version == "v1"  # unchanged

    def test_correct_batch_when_version_tag_set_propagates_to_corrected(self) -> None:
        record = _make_generated_record("r1")
        judge = [_make_judge_dict("r1", ["safety_violations"])]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = record.record

        result = correct_batch(
            [record], judge, client=mock_client,
            use_cache=False, version_tag="v2_corrected",
        )

        assert result[0].template_version == "v2_corrected"


# ===========================================================================
# analyze_failure_patterns
# ===========================================================================


class TestAnalyzeFailurePatterns:
    """Tests for analyze_failure_patterns."""

    def test_analyze_failure_patterns_when_no_judge_result_skips_record(self) -> None:
        records = [_make_generated_record("r1")]
        # Judge results have no matching trace_id
        judges = [_make_judge_dict("r99", ["incomplete_answer"])]

        patterns = analyze_failure_patterns(records, judges)
        # r1 was skipped — plumbing_repair has no data
        assert "plumbing_repair" not in patterns

    def test_analyze_failure_patterns_returns_sorted_modes_per_category(self) -> None:
        records = [
            _make_generated_record("r1"),
            _make_generated_record("r2"),
        ]
        # r1: 2 failures, r2: 1 failure — incomplete_answer most common
        judges = [
            _make_judge_dict("r1", ["incomplete_answer", "poor_quality_tips"]),
            _make_judge_dict("r2", ["incomplete_answer"]),
        ]

        patterns = analyze_failure_patterns(records, judges)

        assert "plumbing_repair" in patterns
        modes = patterns["plumbing_repair"]
        assert modes[0] == "incomplete_answer"  # most common first


# ===========================================================================
# run_full_pipeline — orchestration integration test
# ===========================================================================


class TestRunFullPipeline:
    """Integration test for run_full_pipeline with all LLM calls mocked."""

    def test_run_full_pipeline_when_inputs_missing_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        with patch("src.corrector._GENERATED_DIR", tmp_path), \
             patch("src.corrector._LABELS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="batch_v1.json not found"):
                run_full_pipeline()

    def test_run_full_pipeline_when_labels_missing_raises_file_not_found(
        self, tmp_path: Path
    ) -> None:
        # Create batch_v1.json but not llm_labels.json
        record = _make_generated_record("r1")
        (tmp_path / "batch_v1.json").write_text(
            json.dumps([record.model_dump()], indent=2)
        )

        with patch("src.corrector._GENERATED_DIR", tmp_path), \
             patch("src.corrector._LABELS_DIR", tmp_path):
            with pytest.raises(FileNotFoundError, match="llm_labels.json not found"):
                run_full_pipeline()

    def test_run_full_pipeline_when_all_mocked_produces_comparison(
        self, tmp_path: Path
    ) -> None:
        # Set up input files
        record = _make_generated_record("r1")
        judge = _make_judge_dict("r1", ["incomplete_answer"])

        generated_dir = tmp_path / "generated"
        generated_dir.mkdir()
        (generated_dir / "batch_v1.json").write_text(
            json.dumps([record.model_dump()], indent=2)
        )

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "llm_labels.json").write_text(
            json.dumps([judge], indent=2)
        )

        corrected_dir = tmp_path / "corrected"
        results_dir = tmp_path / "results"

        # Mock JudgeResult from evaluate_batch
        from src.schemas import FailureLabel, JudgeResult
        clean_judge = JudgeResult(
            trace_id="r1",
            labels=[
                FailureLabel(mode=m, label=0, reason=f"{m} passed")
                for m in [
                    "incomplete_answer", "safety_violations", "unrealistic_tools",
                    "overcomplicated_solution", "missing_context", "poor_quality_tips",
                ]
            ],
            overall_quality_score=4,
        )

        # Patch at the source modules — run_full_pipeline imports locally
        with patch("src.corrector._GENERATED_DIR", generated_dir), \
             patch("src.corrector._LABELS_DIR", labels_dir), \
             patch("src.corrector._CORRECTED_DIR", corrected_dir), \
             patch("src.corrector._PROJECT_ROOT", tmp_path), \
             patch("src.corrector.correct_batch", return_value=[record]) as mock_correct, \
             patch("src.corrector.generate_v2_batch", return_value=[record]) as mock_gen_v2, \
             patch("src.corrector.analyze_failure_patterns", return_value={}), \
             patch("src.evaluator.evaluate_batch", return_value=[clean_judge]) as mock_eval, \
             patch("src.evaluator.save_llm_labels"), \
             patch("src.evaluator.save_llm_labels_json"), \
             patch("src.corrector.save_corrected_records"), \
             patch("src.generator.save_generated_records"):

            result = run_full_pipeline()

        # Verify stages executed
        assert mock_correct.call_count == 2  # v1 correction + v2 correction
        assert mock_gen_v2.call_count == 1
        assert mock_eval.call_count == 3  # v1c, v2, v2c

        # Verify comparison file produced
        comparison_path = results_dir / "correction_comparison.json"
        assert comparison_path.exists()

        # Verify result structure
        assert "v1_original" in result
        assert "corrected" in result
        assert "v2_generated" in result
        assert "v2_corrected" in result
        assert "generated_at" in result
        assert result["pipeline_version"] == "1.0"


# ===========================================================================
# Cache helpers
# ===========================================================================


class TestCorrectionCacheKey:
    """Tests for _correction_cache_key."""

    def test_correction_cache_key_starts_with_correct_prefix(self) -> None:
        key = _correction_cache_key("some prompt")
        assert key.startswith("correct_")

    def test_correction_cache_key_is_deterministic(self) -> None:
        key1 = _correction_cache_key("prompt A")
        key2 = _correction_cache_key("prompt A")
        assert key1 == key2

    def test_correction_cache_key_differs_for_different_prompts(self) -> None:
        key1 = _correction_cache_key("prompt A")
        key2 = _correction_cache_key("prompt B")
        assert key1 != key2


class TestLoadCorrectionCache:
    """Tests for _load_correction_cache."""

    def test_load_correction_cache_when_no_file_returns_none(self, tmp_path: Path) -> None:
        with patch("src.corrector._CACHE_DIR", tmp_path):
            result = _load_correction_cache("nonexistent_key")
        assert result is None

    def test_load_correction_cache_when_valid_file_returns_record(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1").record
        payload = {
            "cache_key": "k1",
            "trace_id": "r1",
            "model": "gpt-4o-mini",
            "type": "correction",
            "timestamp": "2026-01-01T00:00:00Z",
            "response": record.model_dump(),
        }
        (tmp_path / "k1.json").write_text(json.dumps(payload))

        with patch("src.corrector._CACHE_DIR", tmp_path):
            result = _load_correction_cache("k1")

        assert result is not None
        assert result.question == record.question

    def test_load_correction_cache_when_corrupt_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "bad.json").write_text("NOT JSON")
        with patch("src.corrector._CACHE_DIR", tmp_path):
            result = _load_correction_cache("bad")
        assert result is None

    def test_load_correction_cache_when_missing_response_returns_none(self, tmp_path: Path) -> None:
        (tmp_path / "nokey.json").write_text(json.dumps({"cache_key": "nokey"}))
        with patch("src.corrector._CACHE_DIR", tmp_path):
            result = _load_correction_cache("nokey")
        assert result is None


class TestSaveCorrectionCache:
    """Tests for _save_correction_cache."""

    def test_save_correction_cache_creates_json_file(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1").record
        with patch("src.corrector._CACHE_DIR", tmp_path):
            _save_correction_cache("key1", "r1", record)
        assert (tmp_path / "key1.json").exists()

    def test_save_correction_cache_has_response_key(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1").record
        with patch("src.corrector._CACHE_DIR", tmp_path):
            _save_correction_cache("key2", "r1", record)
        data = json.loads((tmp_path / "key2.json").read_text())
        assert "response" in data
        assert data["trace_id"] == "r1"
        assert data["type"] == "correction"


# ===========================================================================
# _build_correction_prompt
# ===========================================================================


class TestBuildCorrectionPrompt:
    """Tests for _build_correction_prompt."""

    def test_build_correction_prompt_contains_category(self) -> None:
        record = _make_generated_record("r1")
        failures = [{"mode": "incomplete_answer", "reason": "too short"}]
        prompt = _build_correction_prompt(record, failures)
        assert "plumbing" in prompt.lower()

    def test_build_correction_prompt_contains_failure_mode(self) -> None:
        record = _make_generated_record("r1")
        failures = [{"mode": "incomplete_answer", "reason": "answer is too vague"}]
        prompt = _build_correction_prompt(record, failures)
        assert "incomplete_answer" in prompt

    def test_build_correction_prompt_contains_reason(self) -> None:
        record = _make_generated_record("r1")
        failures = [{"mode": "poor_quality_tips", "reason": "tips are generic"}]
        prompt = _build_correction_prompt(record, failures)
        assert "generic" in prompt


# ===========================================================================
# correct_record
# ===========================================================================


class TestCorrectRecord:
    """Tests for correct_record."""

    def test_correct_record_when_cache_hit_returns_cached(self, tmp_path: Path) -> None:
        from src.corrector import _CORRECTION_SYSTEM_PROMPT as _CSP
        record = _make_generated_record("r1")
        cached_diy = record.record
        failures = [{"mode": "incomplete_answer", "reason": "too short"}]

        user_prompt = _build_correction_prompt(record, failures)
        full_prompt = f"{_CSP}\n---\n{user_prompt}"
        cache_key = _correction_cache_key(full_prompt)

        payload = {
            "cache_key": cache_key,
            "trace_id": "r1",
            "model": "gpt-4o-mini",
            "type": "correction",
            "timestamp": "2026-01-01T00:00:00Z",
            "response": cached_diy.model_dump(),
        }
        (tmp_path / f"{cache_key}.json").write_text(json.dumps(payload))

        client = MagicMock()
        with patch("src.corrector._CACHE_DIR", tmp_path):
            result = correct_record(client, record, failures, use_cache=True)

        client.chat.completions.create.assert_not_called()
        assert result.question == cached_diy.question

    def test_correct_record_when_cache_miss_calls_client(self, tmp_path: Path) -> None:
        record = _make_generated_record("r1")
        failures = [{"mode": "incomplete_answer", "reason": "too short"}]

        client = MagicMock()
        client.chat.completions.create.return_value = record.record

        with patch("src.corrector._CACHE_DIR", tmp_path):
            result = correct_record(client, record, failures, use_cache=False)

        client.chat.completions.create.assert_called_once()
        assert result is not None

    def test_correct_batch_when_exception_falls_back_to_original(self, tmp_path: Path) -> None:
        """When correct_record raises, the original record should be kept."""
        record = _make_generated_record("r1")
        judge = [_make_judge_dict("r1", ["incomplete_answer"])]

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("LLM error")

        with patch("src.corrector._CACHE_DIR", tmp_path):
            result = correct_batch(
                [record], judge, client=client, use_cache=False
            )

        assert len(result) == 1
        assert result[0].trace_id == "r1"
        assert result[0].template_version == "v1"  # unchanged (not v1_corrected)


# ===========================================================================
# V2 template functions
# ===========================================================================


class TestBuildV2Templates:
    """Tests for build_v2_templates."""

    def test_build_v2_templates_returns_all_categories(self, tmp_path: Path) -> None:
        with patch("src.corrector._PROJECT_ROOT", tmp_path):
            v2 = build_v2_templates()

        expected = {
            "appliance_repair", "plumbing_repair", "electrical_repair",
            "hvac_maintenance", "general_home_repair",
        }
        assert set(v2.keys()) == expected

    def test_build_v2_templates_each_entry_has_required_keys(self, tmp_path: Path) -> None:
        with patch("src.corrector._PROJECT_ROOT", tmp_path):
            v2 = build_v2_templates()

        for cat, tmpl in v2.items():
            assert "persona" in tmpl
            assert "emphasis" in tmpl
            assert "v2_additions" in tmpl

    def test_build_v2_templates_when_metrics_exist_adds_instructions(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        metrics = {
            "per_mode_failures": {
                "incomplete_answer": {"count": 5, "rate": "16.7%"},
                "poor_quality_tips": {"count": 3, "rate": "10.0%"},
                "safety_violations": {"count": 0, "rate": "0.0%"},
                "unrealistic_tools": {"count": 0, "rate": "0.0%"},
                "overcomplicated_solution": {"count": 0, "rate": "0.0%"},
                "missing_context": {"count": 0, "rate": "0.0%"},
            }
        }
        (results_dir / "metrics.json").write_text(json.dumps(metrics))

        with patch("src.corrector._PROJECT_ROOT", tmp_path):
            v2 = build_v2_templates()

        any_additions = any(v["v2_additions"] for v in v2.values())
        assert any_additions


class TestBuildV2SystemPrompt:
    """Tests for build_v2_system_prompt."""

    def test_build_v2_system_prompt_contains_persona(self) -> None:
        tmpl = {"persona": "Expert plumber", "emphasis": "Safety", "v2_additions": ""}
        prompt = build_v2_system_prompt("plumbing_repair", tmpl)
        assert "Expert plumber" in prompt

    def test_build_v2_system_prompt_with_additions_includes_quality_section(self) -> None:
        tmpl = {
            "persona": "Expert plumber",
            "emphasis": "Safety",
            "v2_additions": "Add troubleshooting",
        }
        prompt = build_v2_system_prompt("plumbing_repair", tmpl)
        assert "QUALITY REQUIREMENTS" in prompt
        assert "troubleshooting" in prompt

    def test_build_v2_system_prompt_no_additions_excludes_quality_section(self) -> None:
        tmpl = {"persona": "Expert plumber", "emphasis": "Safety", "v2_additions": ""}
        prompt = build_v2_system_prompt("plumbing_repair", tmpl)
        assert "QUALITY REQUIREMENTS" not in prompt


class TestBuildV2UserPrompt:
    """Tests for build_v2_user_prompt."""

    def test_build_v2_user_prompt_variant_0_no_variation_hint(self) -> None:
        prompt = build_v2_user_prompt("plumbing_repair", "beginner", variant=0)
        assert "IMPORTANT: This is variation" not in prompt

    def test_build_v2_user_prompt_variant_1_has_variation_hint(self) -> None:
        prompt = build_v2_user_prompt("plumbing_repair", "beginner", variant=1)
        assert "variation" in prompt.lower()

    def test_build_v2_user_prompt_beginner_uses_article_a(self) -> None:
        prompt = build_v2_user_prompt("plumbing_repair", "beginner")
        assert "a beginner" in prompt

    def test_build_v2_user_prompt_advanced_uses_article_an(self) -> None:
        prompt = build_v2_user_prompt("plumbing_repair", "advanced")
        assert "an advanced" in prompt


class TestV2PromptHash:
    """Tests for _v2_prompt_hash."""

    def test_v2_prompt_hash_is_deterministic(self) -> None:
        h1 = _v2_prompt_hash("sys", "usr")
        h2 = _v2_prompt_hash("sys", "usr")
        assert h1 == h2

    def test_v2_prompt_hash_differs_from_v1_hash(self) -> None:
        from src.generator import _prompt_hash
        h_v1 = _prompt_hash("sys", "usr")
        h_v2 = _v2_prompt_hash("sys", "usr")
        assert h_v1 != h_v2  # "v2:" prefix makes them different


# ===========================================================================
# generate_v2_batch
# ===========================================================================


class TestGenerateV2Batch:
    """Tests for generate_v2_batch."""

    def test_generate_v2_batch_returns_generated_records(self, tmp_path: Path) -> None:
        diy_record = _make_generated_record("r1").record

        client = MagicMock()
        client.chat.completions.create.return_value = diy_record

        with patch("src.corrector._PROJECT_ROOT", tmp_path), \
             patch("src.generator._CACHE_DIR", tmp_path):
            results = generate_v2_batch(client, use_cache=False, records_per_combo=1)

        assert len(results) == 15

    def test_generate_v2_batch_records_have_v2_template_version(self, tmp_path: Path) -> None:
        diy_record = _make_generated_record("r1").record

        client = MagicMock()
        client.chat.completions.create.return_value = diy_record

        with patch("src.corrector._PROJECT_ROOT", tmp_path), \
             patch("src.generator._CACHE_DIR", tmp_path):
            results = generate_v2_batch(client, use_cache=False, records_per_combo=1)

        assert all(r.template_version == "v2" for r in results)


# ===========================================================================
# save_corrected_records
# ===========================================================================


class TestSaveCorrectedRecords:
    """Tests for save_corrected_records."""

    def test_save_corrected_records_creates_json_file(self, tmp_path: Path) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        with patch("src.corrector._CORRECTED_DIR", tmp_path):
            path = save_corrected_records(records, "test_corrected.json")
        assert path.exists()

    def test_save_corrected_records_content_is_list(self, tmp_path: Path) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        with patch("src.corrector._CORRECTED_DIR", tmp_path):
            path = save_corrected_records(records, "test_corrected.json")
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_save_corrected_records_creates_parent_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested" / "corrected"
        records = [_make_generated_record("r1")]
        with patch("src.corrector._CORRECTED_DIR", nested):
            path = save_corrected_records(records)
        assert path.exists()
