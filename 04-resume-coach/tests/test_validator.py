from __future__ import annotations

import json

import pytest

from src.schemas import ContactInfo
from src.validator import ValidationTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tracker() -> ValidationTracker:
    return ValidationTracker()


@pytest.fixture
def sample_errors() -> list[dict]:
    return [
        {"loc": ("education", 0, "gpa"), "msg": "GPA too high", "type": "value_error"},
        {"loc": ("contact_info", "email"), "msg": "Invalid email", "type": "value_error"},
    ]


def _make_contact(name: str = "Jane", idx: int = 0) -> ContactInfo:
    return ContactInfo(
        name=name,
        email=f"user{idx}@example.com",
        phone="+15551234567",
        location="Austin, TX",
    )


# ---------------------------------------------------------------------------
# record_success
# ---------------------------------------------------------------------------


class TestRecordSuccess:
    def test_records_single_success(self, tracker):
        tracker.record_success("Resume", "trace-1")
        stats = tracker.get_stats()
        assert stats["success_count"] == 1
        assert stats["failure_count"] == 0

    def test_records_multiple_successes_same_type(self, tracker):
        tracker.record_success("Resume", "trace-1")
        tracker.record_success("Resume", "trace-2")
        stats = tracker.get_stats()
        assert stats["success_count"] == 2

    def test_records_multiple_model_types(self, tracker):
        tracker.record_success("Resume", "trace-1")
        tracker.record_success("JobDescription", "trace-2")
        stats = tracker.get_stats()
        assert stats["success_count"] == 2
        assert stats["by_model_type"]["Resume"]["successes"] == 1
        assert stats["by_model_type"]["JobDescription"]["successes"] == 1

    def test_total_reflects_successes(self, tracker):
        for i in range(5):
            tracker.record_success("Resume", f"trace-{i}")
        assert tracker.get_stats()["total"] == 5


# ---------------------------------------------------------------------------
# record_failure
# ---------------------------------------------------------------------------


class TestRecordFailure:
    def test_records_single_failure(self, tracker, sample_errors):
        tracker.record_failure("Resume", "trace-1", sample_errors)
        stats = tracker.get_stats()
        assert stats["failure_count"] == 1
        assert stats["success_count"] == 0

    def test_failure_tracks_dot_separated_field_path(self, tracker, sample_errors):
        tracker.record_failure("Resume", "trace-1", sample_errors)
        error_fields = tracker.get_stats()["errors_by_field"]
        assert "education.0.gpa" in error_fields
        assert "contact_info.email" in error_fields

    def test_failure_by_model_type(self, tracker, sample_errors):
        tracker.record_failure("Resume", "trace-1", sample_errors)
        stats = tracker.get_stats()
        assert stats["by_model_type"]["Resume"]["failures"] == 1

    def test_error_without_loc_key(self, tracker):
        # WHY: Pydantic errors may occasionally omit 'loc'; must not crash
        tracker.record_failure("Resume", "trace-1", [{"msg": "bad", "type": "ve"}])
        stats = tracker.get_stats()
        assert stats["failure_count"] == 1
        assert "" in stats["errors_by_field"]


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_empty_tracker_returns_zeros(self, tracker):
        stats = tracker.get_stats()
        assert stats["total"] == 0
        assert stats["success_count"] == 0
        assert stats["failure_count"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["by_model_type"] == {}
        assert stats["errors_by_field"] == {}

    def test_success_rate_calculation(self, tracker, sample_errors):
        tracker.record_success("Resume", "t1")
        tracker.record_success("Resume", "t2")
        tracker.record_success("Resume", "t3")
        tracker.record_failure("Resume", "t4", sample_errors)
        stats = tracker.get_stats()
        assert stats["success_rate"] == pytest.approx(0.75)
        assert stats["total"] == 4

    def test_all_success_rate_is_1(self, tracker):
        tracker.record_success("Resume", "t1")
        tracker.record_success("Resume", "t2")
        assert tracker.get_stats()["success_rate"] == 1.0

    def test_all_failure_rate_is_0(self, tracker, sample_errors):
        tracker.record_failure("Resume", "t1", sample_errors)
        assert tracker.get_stats()["success_rate"] == 0.0

    def test_error_counts_most_common_first(self, tracker):
        for i in range(3):
            tracker.record_failure(
                "Resume", f"t{i}", [{"loc": ("email",), "msg": "bad", "type": "ve"}]
            )
        tracker.record_failure(
            "Resume", "t3", [{"loc": ("phone",), "msg": "bad", "type": "ve"}]
        )
        fields = list(tracker.get_stats()["errors_by_field"].keys())
        assert fields[0] == "email"  # most common first

    def test_by_model_type_includes_both_success_and_failure_keys(
        self, tracker, sample_errors
    ):
        tracker.record_success("Resume", "t1")
        tracker.record_failure("JobDescription", "t2", sample_errors)
        by_type = tracker.get_stats()["by_model_type"]
        assert "Resume" in by_type
        assert "JobDescription" in by_type
        assert by_type["Resume"]["failures"] == 0
        assert by_type["JobDescription"]["successes"] == 0


# ---------------------------------------------------------------------------
# save_stats
# ---------------------------------------------------------------------------


class TestSaveStats:
    def test_creates_output_file(self, tracker, tmp_path):
        tracker.record_success("Resume", "t1")
        output_file = tmp_path / "stats.json"
        tracker.save_stats(output_file)
        assert output_file.exists()

    def test_saved_file_contains_correct_stats(self, tracker, tmp_path):
        tracker.record_success("Resume", "t1")
        output_file = tmp_path / "stats.json"
        tracker.save_stats(output_file)
        data = json.loads(output_file.read_text())
        assert data["total"] == 1
        assert data["success_rate"] == 1.0

    def test_creates_parent_directories(self, tracker, tmp_path):
        output_file = tmp_path / "nested" / "deeply" / "stats.json"
        tracker.save_stats(output_file)
        assert output_file.exists()


# ---------------------------------------------------------------------------
# save_valid (static method)
# ---------------------------------------------------------------------------


class TestSaveValid:
    def test_creates_validated_jsonl(self, tmp_path):
        record = _make_contact("Jane", 0)
        ValidationTracker.save_valid(record, tmp_path)
        validated_file = tmp_path / "validated.jsonl"
        assert validated_file.exists()

    def test_record_written_as_valid_json(self, tmp_path):
        record = _make_contact("Alice", 1)
        ValidationTracker.save_valid(record, tmp_path)
        lines = (tmp_path / "validated.jsonl").read_text().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["name"] == "Alice"

    def test_multiple_records_append(self, tmp_path):
        for i in range(3):
            ValidationTracker.save_valid(_make_contact(f"Person{i}", i), tmp_path)
        lines = (tmp_path / "validated.jsonl").read_text().splitlines()
        assert len(lines) == 3

    def test_creates_output_directory_if_missing(self, tmp_path):
        nested = tmp_path / "new_dir"
        ValidationTracker.save_valid(_make_contact(), nested)
        assert (nested / "validated.jsonl").exists()


# ---------------------------------------------------------------------------
# save_invalid (static method)
# ---------------------------------------------------------------------------


class TestSaveInvalid:
    def test_creates_invalid_jsonl_from_string(self, tmp_path):
        raw = '{"name": "bad", "email": "not-email"}'
        errors = [{"loc": ("email",), "msg": "bad email", "type": "value_error"}]
        ValidationTracker.save_invalid(raw, errors, tmp_path)
        invalid_file = tmp_path / "invalid.jsonl"
        assert invalid_file.exists()

    def test_output_has_raw_data_and_errors_keys(self, tmp_path):
        raw = '{"name": "bad"}'
        errors = [{"loc": ("email",), "msg": "bad", "type": "ve"}]
        ValidationTracker.save_invalid(raw, errors, tmp_path)
        data = json.loads((tmp_path / "invalid.jsonl").read_text().strip())
        assert "raw_data" in data
        assert "errors" in data

    def test_accepts_dict_input(self, tmp_path):
        raw_dict = {"name": "bad", "email": "not-email"}
        errors = [{"loc": ("email",), "msg": "bad email", "type": "value_error"}]
        ValidationTracker.save_invalid(raw_dict, errors, tmp_path)
        data = json.loads((tmp_path / "invalid.jsonl").read_text().strip())
        assert data["raw_data"]["name"] == "bad"

    def test_creates_output_directory_if_missing(self, tmp_path):
        nested = tmp_path / "output"
        ValidationTracker.save_invalid({"a": 1}, [], nested)
        assert (nested / "invalid.jsonl").exists()

    def test_multiple_records_append(self, tmp_path):
        for i in range(2):
            ValidationTracker.save_invalid({"idx": i}, [], tmp_path)
        lines = (tmp_path / "invalid.jsonl").read_text().splitlines()
        assert len(lines) == 2
