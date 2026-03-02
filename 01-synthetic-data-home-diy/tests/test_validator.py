"""Tests for src/validator.py — validation tracking and rejection logging.

Covers:
- ValidationResult: dataclass creation
- ValidationReport: success_rate, success_rate_pct, summary
- validate_record: valid dict, invalid dict (multiple failure cases)
- validate_batch: all valid, some rejected, per-field error tracking
- save_validation_results: saves 3 files to patched _VALIDATED_DIR
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.validator import (
    ValidationReport,
    ValidationResult,
    save_validation_results,
    validate_batch,
    validate_record,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _valid_record_dict() -> dict:
    """Minimal valid DIYRepairRecord dict."""
    return {
        "question": "How do I fix a leaking kitchen faucet?",
        "answer": (
            "First, turn off the water supply valve under the sink. "
            "Then remove the faucet handle by unscrewing the decorative cap. "
            "Replace the worn cartridge with a new one from the hardware store."
        ),
        "equipment_problem": "Leaking single-handle kitchen faucet",
        "tools_required": ["adjustable wrench", "plumber tape"],
        "steps": [
            "Turn off the water supply valve under the sink",
            "Remove the faucet handle by unscrewing the cap",
        ],
        "safety_info": "Always turn off water supply before starting",
        "tips": "Take a photo before disassembly for reference",
    }


def _make_generated_record(trace_id: str = "r1"):
    """Build a minimal GeneratedRecord for batch validation tests."""
    from src.schemas import DIYRepairRecord, GeneratedRecord
    record = DIYRepairRecord(**_valid_record_dict())
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
# ValidationResult
# ===========================================================================


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_validation_result_when_valid_has_empty_errors(self) -> None:
        result = ValidationResult(trace_id="r1", is_valid=True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.failed_fields == []

    def test_validation_result_when_invalid_stores_errors(self) -> None:
        result = ValidationResult(
            trace_id="r1",
            is_valid=False,
            errors=["question: too short"],
            failed_fields=["question"],
        )
        assert result.is_valid is False
        assert "question" in result.failed_fields


# ===========================================================================
# ValidationReport
# ===========================================================================


class TestValidationReport:
    """Tests for ValidationReport aggregate metrics."""

    def test_validation_report_when_zero_records_success_rate_is_zero(self) -> None:
        report = ValidationReport()
        assert report.success_rate == 0.0
        assert report.success_rate_pct == "0.0%"

    def test_validation_report_success_rate_is_fraction(self) -> None:
        report = ValidationReport(total_records=4, valid_count=3, rejected_count=1)
        assert report.success_rate == pytest.approx(0.75)
        assert report.success_rate_pct == "75.0%"

    def test_validation_report_summary_has_all_keys(self) -> None:
        report = ValidationReport(total_records=2, valid_count=2, rejected_count=0)
        summary = report.summary()
        assert "total_records" in summary
        assert "valid_count" in summary
        assert "rejected_count" in summary
        assert "success_rate" in summary
        assert "field_error_frequency" in summary

    def test_validation_report_summary_field_error_frequency_sorted(self) -> None:
        from collections import Counter
        report = ValidationReport(
            total_records=3,
            valid_count=1,
            rejected_count=2,
            field_error_counts=Counter({"question": 2, "answer": 1}),
        )
        summary = report.summary()
        # most_common() order: question first
        keys = list(summary["field_error_frequency"].keys())
        assert keys[0] == "question"


# ===========================================================================
# validate_record
# ===========================================================================


class TestValidateRecord:
    """Tests for validate_record."""

    def test_validate_record_when_valid_returns_is_valid_true(self) -> None:
        result = validate_record(_valid_record_dict())
        assert result.is_valid is True
        assert result.errors == []

    def test_validate_record_when_no_trace_id_uses_unknown(self) -> None:
        data = _valid_record_dict()
        result = validate_record(data)
        # valid_record_dict has no trace_id key — defaults to "unknown"
        assert result.trace_id == "unknown"

    def test_validate_record_when_question_missing_mark_is_invalid(self) -> None:
        data = _valid_record_dict()
        data["question"] = "This is a statement without a question mark"
        result = validate_record(data)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "question" in result.failed_fields

    def test_validate_record_when_answer_too_short_is_invalid(self) -> None:
        data = _valid_record_dict()
        data["answer"] = "Too short."
        result = validate_record(data)
        assert result.is_valid is False
        assert "answer" in result.failed_fields

    def test_validate_record_when_tools_empty_is_invalid(self) -> None:
        data = _valid_record_dict()
        data["tools_required"] = []
        result = validate_record(data)
        assert result.is_valid is False

    def test_validate_record_when_steps_too_few_is_invalid(self) -> None:
        data = _valid_record_dict()
        data["steps"] = ["Only one step which is long enough to pass"]
        result = validate_record(data)
        assert result.is_valid is False


# ===========================================================================
# validate_batch
# ===========================================================================


class TestValidateBatch:
    """Tests for validate_batch."""

    def test_validate_batch_when_all_valid_returns_full_valid_list(self) -> None:
        records = [_make_generated_record("r1"), _make_generated_record("r2")]
        valid, rejected, report = validate_batch(records)
        assert len(valid) == 2
        assert len(rejected) == 0
        assert report.total_records == 2
        assert report.valid_count == 2
        assert report.rejected_count == 0

    def test_validate_batch_when_empty_list_returns_zero_counts(self) -> None:
        valid, rejected, report = validate_batch([])
        assert len(valid) == 0
        assert len(rejected) == 0
        assert report.total_records == 0

    def test_validate_batch_when_some_rejected_tracks_field_errors(self) -> None:
        from unittest.mock import patch
        from src.validator import ValidationResult

        good = _make_generated_record("r1")
        bad_record = _make_generated_record("r2")

        # Patch validate_record so the second call returns a failure
        side_effects = [
            ValidationResult(trace_id="r1", is_valid=True),
            ValidationResult(
                trace_id="r2",
                is_valid=False,
                errors=["question: no question mark"],
                failed_fields=["question"],
            ),
        ]
        with patch("src.validator.validate_record", side_effect=side_effects):
            valid, rejected, report = validate_batch([good, bad_record])

        assert len(valid) == 1
        assert len(rejected) == 1
        assert report.valid_count == 1
        assert report.rejected_count == 1
        assert "question" in report.field_error_counts

    def test_validate_batch_success_rate_correct(self) -> None:
        records = [_make_generated_record(f"r{i}") for i in range(5)]
        _, _, report = validate_batch(records)
        assert report.success_rate == pytest.approx(1.0)
        assert report.success_rate_pct == "100.0%"


# ===========================================================================
# save_validation_results
# ===========================================================================


class TestSaveValidationResults:
    """Tests for save_validation_results."""

    def test_save_validation_results_creates_three_files(
        self, tmp_path: Path
    ) -> None:
        valid = [_make_generated_record("r1")]
        rejected = [_make_generated_record("r2")]
        report = ValidationReport(total_records=2, valid_count=1, rejected_count=1)

        with patch("src.validator._VALIDATED_DIR", tmp_path):
            valid_path, rejected_path, report_path = save_validation_results(
                valid, rejected, report
            )

        assert valid_path.exists()
        assert rejected_path.exists()
        assert report_path.exists()

    def test_save_validation_results_valid_json_is_list(
        self, tmp_path: Path
    ) -> None:
        valid = [_make_generated_record("r1"), _make_generated_record("r2")]
        report = ValidationReport(total_records=2, valid_count=2, rejected_count=0)

        with patch("src.validator._VALIDATED_DIR", tmp_path):
            valid_path, _, _ = save_validation_results(valid, [], report)

        data = json.loads(valid_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 2

    def test_save_validation_results_report_has_success_rate(
        self, tmp_path: Path
    ) -> None:
        report = ValidationReport(total_records=3, valid_count=3, rejected_count=0)

        with patch("src.validator._VALIDATED_DIR", tmp_path):
            _, _, report_path = save_validation_results([], [], report)

        data = json.loads(report_path.read_text())
        assert "success_rate" in data
        assert data["total_records"] == 3
