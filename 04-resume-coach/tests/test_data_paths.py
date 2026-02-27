"""Tests for src/data_paths.py — path helpers and DataStore coverage.

Covers the lines missed in baseline:
- find_latest() returns None when directory is empty (line 46)
- load_feedback() body when file exists (lines 125-130)
- load_pipeline_summary() when file exists (line 137)
- DataStore.job_count / resume_count / pair_count properties (lines 189, 193, 197)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# find_latest — empty directory returns None
# ---------------------------------------------------------------------------


class TestFindLatest:
    def test_returns_none_when_no_matching_files(self, tmp_path: Path) -> None:
        from src.data_paths import find_latest

        result = find_latest(tmp_path, "jobs")
        assert result is None

    def test_returns_file_when_one_match(self, tmp_path: Path) -> None:
        from src.data_paths import find_latest

        f = tmp_path / "jobs_20260220.jsonl"
        f.write_text('{"trace_id": "j1"}\n')
        result = find_latest(tmp_path, "jobs")
        assert result == f

    def test_returns_largest_file_when_multiple_matches(self, tmp_path: Path) -> None:
        from src.data_paths import find_latest

        small = tmp_path / "jobs_dry.jsonl"
        small.write_text('{"trace_id": "j1"}\n')
        large = tmp_path / "jobs_full.jsonl"
        large.write_text('{"trace_id": "j1"}\n' * 50)

        result = find_latest(tmp_path, "jobs")
        assert result == large

    def test_ignores_non_matching_prefix(self, tmp_path: Path) -> None:
        from src.data_paths import find_latest

        (tmp_path / "resumes_20260220.jsonl").write_text("line\n")
        result = find_latest(tmp_path, "jobs")
        assert result is None


# ---------------------------------------------------------------------------
# load_feedback — body when file exists
# ---------------------------------------------------------------------------


class TestLoadFeedback:
    def test_returns_empty_dict_when_file_missing(self, tmp_path: Path) -> None:
        from src.data_paths import load_feedback

        result = load_feedback(tmp_path / "nonexistent.jsonl")
        assert result == {}

    def test_loads_feedback_entries_from_existing_file(self, tmp_path: Path) -> None:
        from src.data_paths import load_feedback

        fb_file = tmp_path / "feedback.jsonl"
        # FeedbackRequest needs pair_id and rating
        fb_file.write_text(
            json.dumps({"pair_id": "pair-001", "rating": "good", "comment": "Nice"}) + "\n"
        )
        result = load_feedback(fb_file)
        assert "pair-001" in result
        assert len(result["pair-001"]) == 1
        assert result["pair-001"][0].rating == "good"

    def test_groups_multiple_entries_by_pair_id(self, tmp_path: Path) -> None:
        from src.data_paths import load_feedback

        fb_file = tmp_path / "feedback.jsonl"
        fb_file.write_text(
            json.dumps({"pair_id": "pair-001", "rating": "good"}) + "\n"
            + json.dumps({"pair_id": "pair-001", "rating": "bad"}) + "\n"
            + json.dumps({"pair_id": "pair-002", "rating": "neutral"}) + "\n"
        )
        result = load_feedback(fb_file)
        assert len(result["pair-001"]) == 2
        assert len(result["pair-002"]) == 1

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        from src.data_paths import load_feedback

        fb_file = tmp_path / "feedback.jsonl"
        fb_file.write_text(
            "\n"
            + json.dumps({"pair_id": "pair-001", "rating": "good"}) + "\n"
            + "\n"
        )
        result = load_feedback(fb_file)
        assert len(result["pair-001"]) == 1


# ---------------------------------------------------------------------------
# load_pipeline_summary — when file exists
# ---------------------------------------------------------------------------


class TestLoadPipelineSummary:
    def test_returns_empty_dict_when_file_missing(self) -> None:
        from src.data_paths import RESULTS_DIR, load_pipeline_summary

        with patch("src.data_paths.RESULTS_DIR", Path("/nonexistent_dir_xyz")):
            result = load_pipeline_summary()
        assert result == {}

    def test_returns_parsed_json_when_file_exists(self, tmp_path: Path) -> None:
        from src.data_paths import load_pipeline_summary

        summary = {"generation": {"jobs_generated": 50}, "labeling": {"total_pairs": 250}}
        summary_file = tmp_path / "pipeline_summary.json"
        summary_file.write_text(json.dumps(summary))

        with patch("src.data_paths.RESULTS_DIR", tmp_path):
            result = load_pipeline_summary()

        assert result["generation"]["jobs_generated"] == 50
        assert result["labeling"]["total_pairs"] == 250


# ---------------------------------------------------------------------------
# DataStore properties — job_count, resume_count, pair_count
# ---------------------------------------------------------------------------


class TestDataStoreProperties:
    """Tests that exercise the @property accessors (lines 189, 193, 197)."""

    def _make_store_with_empty_data(self):
        """Create DataStore with all find_latest() calls returning None so no files load."""
        from src.data_paths import DataStore

        with (
            patch("src.data_paths.find_latest", return_value=None),
            patch("src.data_paths.load_pipeline_summary", return_value={}),
            patch.object(Path, "exists", return_value=False),
        ):
            return DataStore()

    def test_job_count_returns_zero_when_empty(self) -> None:
        store = self._make_store_with_empty_data()
        assert store.job_count == 0

    def test_resume_count_returns_zero_when_empty(self) -> None:
        store = self._make_store_with_empty_data()
        assert store.resume_count == 0

    def test_pair_count_returns_zero_when_empty(self) -> None:
        store = self._make_store_with_empty_data()
        assert store.pair_count == 0

    def test_job_count_reflects_loaded_jobs(self, tmp_path: Path) -> None:
        """job_count returns len(self.jobs)."""
        from src.data_paths import DataStore

        with (
            patch("src.data_paths.find_latest", return_value=None),
            patch("src.data_paths.load_pipeline_summary", return_value={}),
            patch.object(Path, "exists", return_value=False),
        ):
            store = DataStore()

        # Manually inject jobs to confirm property delegates to len()
        store.jobs = {"j1": object(), "j2": object()}  # type: ignore[assignment]
        assert store.job_count == 2

    def test_resume_count_reflects_loaded_resumes(self) -> None:
        store = self._make_store_with_empty_data()
        store.resumes = {"r1": object(), "r2": object(), "r3": object()}  # type: ignore[assignment]
        assert store.resume_count == 3

    def test_pair_count_reflects_loaded_pairs(self) -> None:
        store = self._make_store_with_empty_data()
        store.pairs = [object(), object()]  # type: ignore[assignment]
        assert store.pair_count == 2
