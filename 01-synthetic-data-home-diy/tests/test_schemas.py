"""Tests for src/schemas.py — Pydantic model validation.

Covers all 4 models: DIYRepairRecord, GeneratedRecord, FailureLabel, JudgeResult.
Tests both happy paths and intentional validation failures per PRD Section 3.

Java/TS parallel: like JUnit @ParameterizedTest or Jest's test.each — pytest's
@parametrize decorator runs the same test with different inputs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    DIYRepairRecord,
    FailureLabel,
    GeneratedRecord,
    JudgeResult,
)


# ---------------------------------------------------------------------------
# Fixtures — reusable valid data dicts
# Java/TS parallel: like @BeforeEach setUp() or a test factory function.
# pytest fixtures are injected by name — the test declares a parameter
# matching the fixture name, and pytest passes the return value.
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_record_data() -> dict:
    """Minimal valid data for DIYRepairRecord."""
    return {
        "question": "How do I fix a leaking kitchen faucet?",
        "answer": (
            "First, turn off the water supply valve under the sink. "
            "Then remove the faucet handle by unscrewing the decorative cap."
        ),
        "equipment_problem": "Leaking single-handle kitchen faucet",
        "tools_required": ["adjustable wrench", "plumber tape"],
        "steps": [
            "Turn off the water supply valve under the sink",
            "Remove the faucet handle by unscrewing the cap",
        ],
        "safety_info": "Always turn off water supply before starting",
        "tips": "Take a photo before disassembly",
    }


@pytest.fixture
def valid_record(valid_record_data: dict) -> DIYRepairRecord:
    """A constructed valid DIYRepairRecord instance."""
    return DIYRepairRecord(**valid_record_data)


def _make_all_labels(*, skip: str | None = None, duplicate: str | None = None) -> list[dict]:
    """Helper to build a list of 6 FailureLabel dicts.

    Args:
        skip: mode to omit (replaced by duplicate if given).
        duplicate: mode to duplicate in place of skipped mode.
    """
    modes = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
    ]
    labels = []
    for mode in modes:
        if mode == skip:
            # Replace the skipped mode with a duplicate of another
            labels.append({"mode": duplicate or modes[0], "label": 0, "reason": "Duplicate entry for test"})
        else:
            labels.append({"mode": mode, "label": 0, "reason": f"{mode} check passed"})
    return labels


# ===========================================================================
# DIYRepairRecord — happy path
# ===========================================================================


class TestDIYRepairRecordValid:
    """Happy path tests for DIYRepairRecord."""

    def test_diy_record_when_valid_data_creates_successfully(
        self, valid_record: DIYRepairRecord
    ) -> None:
        assert valid_record.question.endswith("?")
        assert len(valid_record.tools_required) >= 1
        assert len(valid_record.steps) >= 2

    def test_diy_record_json_schema_has_all_fields(self) -> None:
        """Ensures Instructor will send all 7 fields to the LLM."""
        schema = DIYRepairRecord.model_json_schema()
        expected_fields = {
            "question",
            "answer",
            "equipment_problem",
            "tools_required",
            "steps",
            "safety_info",
            "tips",
        }
        assert set(schema["required"]) == expected_fields
        assert set(schema["properties"].keys()) == expected_fields

    def test_diy_record_roundtrip_json(self, valid_record: DIYRepairRecord) -> None:
        """model_dump_json → model_validate_json roundtrip preserves data."""
        json_str = valid_record.model_dump_json()
        restored = DIYRepairRecord.model_validate_json(json_str)
        assert restored == valid_record


# ===========================================================================
# DIYRepairRecord — validation failures
# ===========================================================================


class TestDIYRepairRecordInvalid:
    """Failure-case tests for DIYRepairRecord field validators."""

    def test_diy_record_question_when_no_question_mark_raises_error(
        self, valid_record_data: dict
    ) -> None:
        valid_record_data["question"] = "This is a statement not a question"
        with pytest.raises(ValidationError, match="must end with '\\?'"):
            DIYRepairRecord(**valid_record_data)

    def test_diy_record_question_when_too_short_raises_error(
        self, valid_record_data: dict
    ) -> None:
        valid_record_data["question"] = "Short?"
        with pytest.raises(ValidationError, match="at least 10"):
            DIYRepairRecord(**valid_record_data)

    def test_diy_record_answer_when_too_short_raises_error(
        self, valid_record_data: dict
    ) -> None:
        valid_record_data["answer"] = "Too short."
        with pytest.raises(ValidationError, match="at least 50"):
            DIYRepairRecord(**valid_record_data)

    def test_diy_record_tools_when_empty_list_raises_error(
        self, valid_record_data: dict
    ) -> None:
        valid_record_data["tools_required"] = []
        with pytest.raises(ValidationError, match="at least 1"):
            DIYRepairRecord(**valid_record_data)

    def test_diy_record_tools_when_item_too_short_raises_error(
        self, valid_record_data: dict
    ) -> None:
        valid_record_data["tools_required"] = ["x"]
        with pytest.raises(ValidationError, match="at least 2"):
            DIYRepairRecord(**valid_record_data)

    def test_diy_record_steps_when_only_one_raises_error(
        self, valid_record_data: dict
    ) -> None:
        valid_record_data["steps"] = ["Only one step but it is long enough"]
        with pytest.raises(ValidationError, match="at least 2"):
            DIYRepairRecord(**valid_record_data)

    def test_diy_record_steps_when_item_too_short_raises_error(
        self, valid_record_data: dict
    ) -> None:
        valid_record_data["steps"] = ["Too short", "Also short"]
        with pytest.raises(ValidationError, match="at least 10"):
            DIYRepairRecord(**valid_record_data)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("equipment_problem", "ab"),    # min_length=5
            ("safety_info", "careful"),      # min_length=10
            ("tips", "ok"),                  # min_length=5
        ],
        ids=["equipment_problem_too_short", "safety_info_too_short", "tips_too_short"],
    )
    def test_diy_record_field_when_too_short_raises_error(
        self, valid_record_data: dict, field: str, value: str
    ) -> None:
        valid_record_data[field] = value
        with pytest.raises(ValidationError):
            DIYRepairRecord(**valid_record_data)


# ===========================================================================
# GeneratedRecord
# ===========================================================================


class TestGeneratedRecord:
    """Tests for GeneratedRecord — metadata wrapper around DIYRepairRecord."""

    def test_generated_record_when_valid_creates_successfully(
        self, valid_record: DIYRepairRecord
    ) -> None:
        gr = GeneratedRecord(
            trace_id="test-uuid-001",
            category="plumbing_repair",
            difficulty="beginner",
            template_version="v1",
            generation_timestamp="2026-02-08T22:00:00Z",
            model_used="gpt-4o-mini",
            prompt_hash="abc123",
            record=valid_record,
        )
        assert gr.category == "plumbing_repair"
        assert gr.record.question.endswith("?")

    @pytest.mark.parametrize(
        "category",
        ["invalid_category", "plumbing", "APPLIANCE_REPAIR", ""],
        ids=["unknown", "partial_match", "wrong_case", "empty"],
    )
    def test_generated_record_when_invalid_category_raises_error(
        self, valid_record: DIYRepairRecord, category: str
    ) -> None:
        with pytest.raises(ValidationError):
            GeneratedRecord(
                trace_id="test",
                category=category,
                difficulty="beginner",
                template_version="v1",
                generation_timestamp="2026-02-08T22:00:00Z",
                model_used="gpt-4o-mini",
                prompt_hash="abc123",
                record=valid_record,
            )

    @pytest.mark.parametrize(
        "difficulty",
        ["expert", "easy", "BEGINNER", ""],
        ids=["expert", "easy", "wrong_case", "empty"],
    )
    def test_generated_record_when_invalid_difficulty_raises_error(
        self, valid_record: DIYRepairRecord, difficulty: str
    ) -> None:
        with pytest.raises(ValidationError):
            GeneratedRecord(
                trace_id="test",
                category="plumbing_repair",
                difficulty=difficulty,
                template_version="v1",
                generation_timestamp="2026-02-08T22:00:00Z",
                model_used="gpt-4o-mini",
                prompt_hash="abc123",
                record=valid_record,
            )


# ===========================================================================
# FailureLabel
# ===========================================================================


class TestFailureLabel:
    """Tests for FailureLabel — one failure mode evaluation."""

    def test_failure_label_when_valid_creates_successfully(self) -> None:
        fl = FailureLabel(
            mode="safety_violations",
            label=1,
            reason="Missing circuit breaker warning",
        )
        assert fl.label == 1

    def test_failure_label_when_invalid_mode_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            FailureLabel(mode="not_a_real_mode", label=0, reason="This should fail")

    @pytest.mark.parametrize("bad_label", [2, -1, 3])
    def test_failure_label_when_invalid_label_value_raises_error(
        self, bad_label: int
    ) -> None:
        with pytest.raises(ValidationError):
            FailureLabel(
                mode="incomplete_answer", label=bad_label, reason="This should fail"
            )


# ===========================================================================
# JudgeResult
# ===========================================================================


class TestJudgeResult:
    """Tests for JudgeResult — full evaluation of one record."""

    def test_judge_result_when_valid_creates_successfully(self) -> None:
        labels = _make_all_labels()
        jr = JudgeResult(
            trace_id="test-uuid-001",
            labels=labels,
            overall_quality_score=4,
        )
        assert len(jr.labels) == 6
        assert jr.overall_quality_score == 4

    def test_judge_result_when_missing_mode_raises_error(self) -> None:
        """Omit poor_quality_tips, replace with duplicate incomplete_answer."""
        labels = _make_all_labels(skip="poor_quality_tips", duplicate="incomplete_answer")
        with pytest.raises(ValidationError, match="Missing failure mode"):
            JudgeResult(
                trace_id="test",
                labels=labels,
                overall_quality_score=3,
            )

    def test_judge_result_when_wrong_label_count_raises_error(self) -> None:
        """Only 3 labels instead of 6."""
        labels = _make_all_labels()[:3]
        with pytest.raises(ValidationError):
            JudgeResult(
                trace_id="test",
                labels=labels,
                overall_quality_score=3,
            )

    @pytest.mark.parametrize("score", [0, 6])
    def test_judge_result_when_score_out_of_range_raises_error(
        self, score: int
    ) -> None:
        labels = _make_all_labels()
        with pytest.raises(ValidationError):
            JudgeResult(
                trace_id="test",
                labels=labels,
                overall_quality_score=score,
            )
