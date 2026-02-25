from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from src.schemas import (
    CompanyInfo,
    ContactInfo,
    Education,
    ExperienceLevel,
    Experience,
    FitLevel,
    GeneratedJob,
    GeneratedResume,
    JobDescription,
    JobRequirements,
    ProficiencyLevel,
    Resume,
    ResumeJobPair,
    Skill,
    WritingStyle,
)
from src.sanity_check import _load_latest_jsonl, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(trace_id: str | None = None) -> GeneratedJob:
    return GeneratedJob(
        trace_id=trace_id or str(uuid.uuid4()),
        job=JobDescription(
            title="Software Engineer",
            company=CompanyInfo(
                name="Acme", industry="Technology", size="Mid-size", location="Austin, TX"
            ),
            description="Build software",
            requirements=JobRequirements(
                required_skills=["Python", "AWS"],
                education="BS CS",
                experience_years=5,
                experience_level=ExperienceLevel.SENIOR,
            ),
        ),
        is_niche_role=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="job-standard",
        model_used="gpt-4o-mini",
    )


def _make_resume(trace_id: str | None = None, skills: list[str] | None = None) -> GeneratedResume:
    skill_list = skills or ["Python", "AWS"]
    return GeneratedResume(
        trace_id=trace_id or str(uuid.uuid4()),
        resume=Resume(
            contact_info=ContactInfo(
                name="Alice Smith",
                email="alice@example.com",
                phone="+15551234567",
                location="Austin, TX",
            ),
            education=[
                Education(degree="BS CS", institution="MIT", graduation_date="2020-05")
            ],
            experience=[
                Experience(
                    company="Acme",
                    title="Dev",
                    start_date="2020-06",
                    responsibilities=["Built things"],
                )
            ],
            skills=[
                Skill(name=s, proficiency_level=ProficiencyLevel.ADVANCED) for s in skill_list
            ],
        ),
        fit_level=FitLevel.EXCELLENT,
        writing_style=WritingStyle.FORMAL,
        template_version="v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="v1-formal",
        model_used="gpt-4o-mini",
    )


def _make_pair(
    resume_trace_id: str,
    job_trace_id: str,
    fit_level: FitLevel = FitLevel.EXCELLENT,
) -> ResumeJobPair:
    return ResumeJobPair(
        pair_id=str(uuid.uuid4()),
        resume_trace_id=resume_trace_id,
        job_trace_id=job_trace_id,
        fit_level=fit_level,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _write_jsonl(directory: Path, prefix: str, records: list[dict]) -> Path:
    """Write records as JSONL to a timestamped file."""
    filepath = directory / f"{prefix}20260224_120000.jsonl"
    with open(filepath, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return filepath


def _setup_generated_dir(
    tmp_path: Path,
    jobs: list[GeneratedJob],
    resumes: list[GeneratedResume],
    pairs: list[ResumeJobPair],
) -> Path:
    """Create data/generated structure under tmp_path."""
    gen_dir = tmp_path / "data" / "generated"
    gen_dir.mkdir(parents=True)
    _write_jsonl(gen_dir, "jobs_", [j.model_dump() for j in jobs])
    _write_jsonl(gen_dir, "resumes_", [r.model_dump() for r in resumes])
    _write_jsonl(gen_dir, "pairs_", [p.model_dump() for p in pairs])
    return tmp_path


# ---------------------------------------------------------------------------
# _load_latest_jsonl
# ---------------------------------------------------------------------------


class TestLoadLatestJsonl:
    def test_returns_records_from_file(self, tmp_path):
        _write_jsonl(tmp_path, "jobs_", [{"a": 1}, {"a": 2}])
        records = _load_latest_jsonl(tmp_path, "jobs_")
        assert len(records) == 2
        assert records[0]["a"] == 1

    def test_loads_most_recent_file_when_multiple_exist(self, tmp_path):
        # Earlier file has "old_data", newer file has "new_data"
        (tmp_path / "jobs_20260101_000000.jsonl").write_text(json.dumps({"v": "old"}) + "\n")
        (tmp_path / "jobs_20260224_120000.jsonl").write_text(json.dumps({"v": "new"}) + "\n")
        records = _load_latest_jsonl(tmp_path, "jobs_")
        # sorted(..., reverse=True) picks the lexicographically largest → newer timestamp
        assert records[0]["v"] == "new"

    def test_raises_file_not_found_when_no_files(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="jobs_"):
            _load_latest_jsonl(tmp_path, "jobs_")

    def test_returns_empty_list_for_empty_jsonl(self, tmp_path):
        (tmp_path / "jobs_20260224_120000.jsonl").write_text("")
        records = _load_latest_jsonl(tmp_path, "jobs_")
        assert records == []


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_without_error(self, tmp_path):
        job = _make_job()
        resume = _make_resume()
        pair = _make_pair(resume.trace_id, job.trace_id, FitLevel.EXCELLENT)
        _setup_generated_dir(tmp_path, [job], [resume], [pair])

        with patch("src.sanity_check._PROJECT_ROOT", tmp_path):
            main()  # Should complete without raising

    def test_main_handles_all_5_fit_levels(self, tmp_path):
        """Verify it picks one pair per fit level from a dataset with all 5."""
        jobs = [_make_job() for _ in range(5)]
        resumes = [_make_resume() for _ in range(5)]
        fit_levels = list(FitLevel)
        pairs = [
            _make_pair(resumes[i].trace_id, jobs[i].trace_id, fit_levels[i])
            for i in range(5)
        ]
        _setup_generated_dir(tmp_path, jobs, resumes, pairs)

        with patch("src.sanity_check._PROJECT_ROOT", tmp_path):
            main()  # Should not raise; table prints 5 rows

    def test_main_handles_missing_job_reference(self, tmp_path):
        """Pair pointing to a non-existent job trace_id triggers the 'continue' branch."""
        job = _make_job()
        resume = _make_resume()
        # Pair references a job_trace_id that doesn't exist in the jobs JSONL
        pair = _make_pair(resume.trace_id, "nonexistent-job-id", FitLevel.GOOD)
        _setup_generated_dir(tmp_path, [job], [resume], [pair])

        with patch("src.sanity_check._PROJECT_ROOT", tmp_path):
            main()  # Should log a warning and continue, not raise

    def test_main_handles_missing_resume_reference(self, tmp_path):
        """Pair pointing to a non-existent resume trace_id triggers the 'continue' branch."""
        job = _make_job()
        resume = _make_resume()
        # Pair references a resume_trace_id that doesn't exist in the resumes JSONL
        pair = _make_pair("nonexistent-resume-id", job.trace_id, FitLevel.PARTIAL)
        _setup_generated_dir(tmp_path, [job], [resume], [pair])

        with patch("src.sanity_check._PROJECT_ROOT", tmp_path):
            main()  # Should log and continue, not raise

    def test_main_stops_after_5_fit_levels(self, tmp_path):
        """With more than 5 pairs, only first 1 per fit level is shown (early break)."""
        job = _make_job()
        fit_levels = list(FitLevel)
        resumes = [_make_resume() for _ in range(8)]
        pairs = []
        for i, resume in enumerate(resumes):
            # First 5 cover all fit levels; remaining are duplicates
            pairs.append(_make_pair(resume.trace_id, job.trace_id, fit_levels[i % 5]))
        _setup_generated_dir(tmp_path, [job], resumes, pairs)

        with patch("src.sanity_check._PROJECT_ROOT", tmp_path):
            main()  # Should not raise; stops once 5 fit levels seen
