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
    _count_failures,
    analyze_failure_patterns,
    build_comparison_metrics,
    correct_batch,
    run_full_pipeline,
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
