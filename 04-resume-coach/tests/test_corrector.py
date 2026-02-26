"""
test_corrector.py — Unit tests for src/corrector.py.

Mocks the Instructor LLM client so no API calls are made.
All tests are self-contained with inline builder helpers.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.corrector import (
    build_correction_prompt,
    correct_batch,
    correct_record,
    extract_validation_errors,
    generate_seeded_broken_records,
)
from src.schemas import (
    ContactInfo,
    Education,
    Experience,
    ProficiencyLevel,
    Resume,
    Skill,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_resume() -> Resume:
    """Build a minimal fully-valid Resume for use as mock LLM return values."""
    return Resume(
        contact_info=ContactInfo(
            name="John Smith",
            email="john@example.com",
            phone="555-123-4567",
            location="New York, NY",
        ),
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="MIT",
                graduation_date="2020-05",
            )
        ],
        experience=[
            Experience(
                company="Acme Corp",
                title="Software Engineer",
                start_date="2020-06",
                end_date="2023-01",
                responsibilities=["Built REST APIs", "Maintained systems"],
            )
        ],
        skills=[
            Skill(name="Python", proficiency_level=ProficiencyLevel.ADVANCED, years=3)
        ],
        summary="Experienced software engineer.",
    )


def _make_broken_dict(record_id: str = "test-001") -> dict:
    """
    Minimal resume dict with an invalid email — fails ContactInfo.validate_email.
    All other fields are valid so the error list is predictable (exactly 1 error).
    """
    return {
        "record_id": record_id,
        "contact_info": {
            "name": "Test User",
            "email": "invalid@nodomain",  # missing TLD → validation error
            "phone": "555-123-4567",
            "location": "Test City, CA",
        },
        "education": [
            {
                "degree": "B.S. CS",
                "institution": "Test University",
                "graduation_date": "2020-05",
            }
        ],
        "experience": [
            {
                "company": "TestCorp",
                "title": "Engineer",
                "start_date": "2020-06",
                "end_date": "2023-01",
                "responsibilities": ["Did engineering work"],
            }
        ],
        "skills": [
            {"name": "Python", "proficiency_level": "Advanced", "years": 3}
        ],
    }


def _make_mock_client(return_value: Resume | None = None, side_effect: Any = None) -> MagicMock:
    """
    Build a mock instructor client whose .chat.completions.create behaves as specified.

    WHY MagicMock: instructor patches client.chat.completions.create to return
    the response_model type directly (not a raw completion). Our mock mirrors this —
    returning a Resume object (or raising) when called.
    """
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    elif return_value is not None:
        client.chat.completions.create.return_value = return_value
    return client


# ---------------------------------------------------------------------------
# TestExtractValidationErrors
# ---------------------------------------------------------------------------


class TestExtractValidationErrors:
    def test_valid_record_returns_empty_list(self) -> None:
        valid_dict = _make_valid_resume().model_dump()
        errors = extract_validation_errors(valid_dict, Resume)
        assert errors == []

    def test_invalid_email_returns_error_with_path(self) -> None:
        broken = _make_broken_dict()
        errors = extract_validation_errors(broken, Resume)
        assert len(errors) >= 1
        email_error = next((e for e in errors if "email" in e["field_path"]), None)
        assert email_error is not None
        assert "email" in email_error["field_path"].lower()
        assert "error_message" in email_error
        assert "invalid_value" in email_error

    def test_multiple_errors_returned(self) -> None:
        # Compound: bad email + bad phone
        broken = _make_broken_dict()
        broken["contact_info"]["phone"] = "12345"  # too short (<10 digits)
        errors = extract_validation_errors(broken, Resume)
        assert len(errors) >= 2

    def test_gpa_out_of_range_returns_error(self) -> None:
        broken = _make_valid_resume().model_dump()
        broken["education"][0]["gpa"] = 85.0  # must be 0-4
        errors = extract_validation_errors(broken, Resume)
        assert len(errors) == 1
        assert "gpa" in errors[0]["field_path"]

    def test_error_dict_has_required_keys(self) -> None:
        broken = _make_broken_dict()
        errors = extract_validation_errors(broken, Resume)
        assert len(errors) >= 1
        for err in errors:
            assert "field_path" in err
            assert "error_message" in err
            assert "invalid_value" in err


# ---------------------------------------------------------------------------
# TestBuildCorrectionPrompt
# ---------------------------------------------------------------------------


class TestBuildCorrectionPrompt:
    def _sample_errors(self) -> list[dict]:
        return [
            {
                "field_path": "contact_info.email",
                "error_message": "Invalid email format: invalid@nodomain",
                "invalid_value": "invalid@nodomain",
            }
        ]

    def test_prompt_contains_error_field_path(self) -> None:
        errors = self._sample_errors()
        system, user = build_correction_prompt({"some": "data"}, errors)
        assert "contact_info.email" in user

    def test_prompt_contains_original_data(self) -> None:
        raw_data = {"contact_info": {"email": "invalid@nodomain"}}
        errors = self._sample_errors()
        _system, user = build_correction_prompt(raw_data, errors)
        assert "invalid@nodomain" in user
        assert "ORIGINAL DATA" in user

    def test_system_prompt_has_fix_instructions(self) -> None:
        errors = self._sample_errors()
        system, _user = build_correction_prompt({}, errors)
        assert "TLD" in system or "email" in system.lower()
        assert "Fix ONLY" in system or "fix only" in system.lower()

    def test_record_type_appears_in_system_prompt(self) -> None:
        errors = self._sample_errors()
        system, _user = build_correction_prompt({}, errors, record_type="JobDescription")
        assert "JobDescription" in system


# ---------------------------------------------------------------------------
# TestGenerateSeededBrokenRecords
# ---------------------------------------------------------------------------


class TestGenerateSeededBrokenRecords:
    def test_returns_eight_records(self) -> None:
        records = generate_seeded_broken_records()
        assert len(records) == 8

    def test_all_records_fail_validation(self) -> None:
        records = generate_seeded_broken_records()
        for record in records:
            with pytest.raises(ValidationError):
                Resume.model_validate(record)

    def test_each_record_has_unique_id(self) -> None:
        records = generate_seeded_broken_records()
        ids = [r["record_id"] for r in records]
        assert len(set(ids)) == 8

    def test_error_types_are_distinct(self) -> None:
        """Each record should have a distinct primary error field."""
        records = generate_seeded_broken_records()
        # Spot-check: record 1 has email error, record 3 has gpa error
        errors_1 = extract_validation_errors(records[0], Resume)
        assert any("email" in e["field_path"] for e in errors_1)

        errors_3 = extract_validation_errors(records[2], Resume)
        assert any("gpa" in e["field_path"] for e in errors_3)


# ---------------------------------------------------------------------------
# TestCorrectRecord
# ---------------------------------------------------------------------------


class TestCorrectRecord:
    def test_success_on_first_attempt(self) -> None:
        """Mock returns valid Resume → corrected_successfully=True, attempt_number=1."""
        valid_resume = _make_valid_resume()
        client = _make_mock_client(return_value=valid_resume)
        broken = _make_broken_dict("test-001")

        result = correct_record(client, broken, use_cache=False)

        assert result.corrected_successfully is True
        assert result.attempt_number == 1
        assert result.pair_id == "test-001"
        assert len(result.original_errors) >= 1
        assert result.remaining_errors == []
        client.chat.completions.create.assert_called_once()

    def test_success_on_second_attempt(self) -> None:
        """Mock fails once then succeeds → attempt_number=2, corrected_successfully=True."""
        valid_resume = _make_valid_resume()
        client = _make_mock_client(
            side_effect=[Exception("LLM timeout"), valid_resume]
        )
        broken = _make_broken_dict("test-002")

        result = correct_record(client, broken, max_attempts=3, use_cache=False)

        assert result.corrected_successfully is True
        assert result.attempt_number == 2
        assert client.chat.completions.create.call_count == 2

    def test_max_retries_exceeded_returns_failure(self) -> None:
        """Mock always raises → corrected_successfully=False, attempt_number=max_attempts."""
        client = _make_mock_client(side_effect=Exception("persistent LLM error"))
        broken = _make_broken_dict("test-003")

        result = correct_record(client, broken, max_attempts=3, use_cache=False)

        assert result.corrected_successfully is False
        assert result.attempt_number == 3
        assert client.chat.completions.create.call_count == 3
        assert result.remaining_errors is not None
        assert len(result.remaining_errors) >= 1

    def test_already_valid_record_skips_llm(self) -> None:
        """Valid record → immediate return without calling client."""
        client = _make_mock_client(return_value=_make_valid_resume())
        valid_dict = _make_valid_resume().model_dump()

        result = correct_record(client, valid_dict, use_cache=False)

        assert result.corrected_successfully is True
        assert result.attempt_number == 0  # 0 = no correction needed
        client.chat.completions.create.assert_not_called()

    def test_original_errors_captured(self) -> None:
        """original_errors must list the errors present before any correction attempt."""
        valid_resume = _make_valid_resume()
        client = _make_mock_client(return_value=valid_resume)
        broken = _make_broken_dict()

        result = correct_record(client, broken, use_cache=False)

        # Must capture at least the email error that was in the broken dict
        assert len(result.original_errors) >= 1
        assert any("email" in e.lower() or "invalid" in e.lower() for e in result.original_errors)


# ---------------------------------------------------------------------------
# TestCorrectBatch
# ---------------------------------------------------------------------------


class TestCorrectBatch:
    def _two_records(self) -> list[dict]:
        """One invalid record + one already-valid record (as raw dict)."""
        return [
            _make_broken_dict("batch-001"),  # will need correction
            _make_valid_resume().model_dump(),  # already valid, record_id="unknown"
        ]

    def test_summary_stats_are_correct(self) -> None:
        """Both records succeed → correction_rate=1.0, total_corrected=2."""
        valid_resume = _make_valid_resume()
        client = _make_mock_client(return_value=valid_resume)
        records = self._two_records()

        results, summary = correct_batch(records, "Resume", client, use_cache=False)

        assert summary.total_invalid == 2
        assert summary.total_corrected == 2
        assert summary.correction_rate == 1.0
        assert len(results) == 2

    def test_summary_failure_rate_when_some_fail(self) -> None:
        """LLM always fails → correction_rate=0, common_failure_reasons populated."""
        # Only the broken dict will try LLM; the valid dict skips
        client = _make_mock_client(side_effect=Exception("always fail"))
        records = [_make_broken_dict("fail-001"), _make_broken_dict("fail-002")]

        results, summary = correct_batch(records, "Resume", client, use_cache=False)

        assert summary.total_corrected == 0
        assert summary.correction_rate == 0.0
        # Both records failed → common_failure_reasons should have entries
        assert len(summary.common_failure_reasons) >= 1

    def test_results_length_matches_input(self) -> None:
        """One result returned per input record."""
        client = _make_mock_client(return_value=_make_valid_resume())
        records = [_make_broken_dict(f"r-{i}") for i in range(4)]

        results, summary = correct_batch(records, "Resume", client, use_cache=False)

        assert len(results) == 4
        assert summary.total_invalid == 4

    def test_avg_attempts_excludes_already_valid(self) -> None:
        """Already-valid records (attempt=0) must not skew avg_attempts_per_success."""
        valid_resume = _make_valid_resume()
        client = _make_mock_client(return_value=valid_resume)

        # Mix: 1 broken (fixed on attempt 1) + 1 already-valid (attempt 0)
        records = [_make_broken_dict("mix-001"), _make_valid_resume().model_dump()]
        _results, summary = correct_batch(records, "Resume", client, use_cache=False)

        # avg_attempts should be 1.0 (only the actually-corrected record counts)
        assert summary.avg_attempts_per_success == 1.0
