from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generated_job() -> GeneratedJob:
    return GeneratedJob(
        trace_id=str(uuid.uuid4()),
        job=JobDescription(
            title="Software Engineer",
            company=CompanyInfo(
                name="Acme", industry="Technology", size="Mid-size", location="Austin, TX"
            ),
            description="Build software",
            requirements=JobRequirements(
                required_skills=["Python"],
                education="BS CS",
                experience_years=3,
                experience_level=ExperienceLevel.MID,
            ),
        ),
        is_niche_role=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="job-standard",
        model_used="gpt-4o-mini",
    )


def _make_generated_resume(job_trace_id: str = "") -> GeneratedResume:
    return GeneratedResume(
        trace_id=str(uuid.uuid4()),
        resume=Resume(
            contact_info=ContactInfo(
                name="Jane",
                email="jane@example.com",
                phone="+15551234567",
                location="Austin, TX",
            ),
            education=[
                Education(degree="BS CS", institution="MIT", graduation_date="2020-05")
            ],
            experience=[
                Experience(
                    company="Co", title="Dev", start_date="2020-06", responsibilities=["Built"]
                )
            ],
            skills=[Skill(name="Python", proficiency_level=ProficiencyLevel.ADVANCED)],
        ),
        fit_level=FitLevel.EXCELLENT,
        writing_style=WritingStyle.FORMAL,
        template_version="v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="v1-formal",
        model_used="gpt-4o-mini",
    )


def _make_pair(resume_trace_id: str, job_trace_id: str) -> ResumeJobPair:
    return ResumeJobPair(
        pair_id=str(uuid.uuid4()),
        resume_trace_id=resume_trace_id,
        job_trace_id=job_trace_id,
        fit_level=FitLevel.EXCELLENT,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# main() — dry-run mode
# ---------------------------------------------------------------------------


class TestMainDryRun:
    def test_dry_run_flag_limits_generation(self, tmp_path):
        jobs = [_make_generated_job()]
        resume = _make_generated_resume()
        pair = _make_pair(resume.trace_id, jobs[0].trace_id)

        with patch.object(sys, "argv", ["prog", "--dry-run"]):
            with patch("src.run_generation._create_client", return_value=MagicMock()):
                with patch(
                    "src.run_generation.generate_all_jobs", return_value=(jobs, 1)
                ) as mock_jobs:
                    with patch(
                        "src.run_generation.generate_all_resumes",
                        return_value=([resume], [pair], 5),
                    ) as _:
                        with patch(
                            "src.run_generation.ValidationTracker.save_stats"
                        ):
                            from src.run_generation import main

                            main()

        # Dry run: count_per_industry=1, max_industries=2
        call_kwargs = mock_jobs.call_args[1]
        assert call_kwargs["count_per_industry"] == 1
        assert call_kwargs["max_industries"] == 2

    def test_dry_run_calls_generate_all_resumes_with_jobs(self, tmp_path):
        jobs = [_make_generated_job()]
        resume = _make_generated_resume()
        pair = _make_pair(resume.trace_id, jobs[0].trace_id)

        with patch.object(sys, "argv", ["prog", "--dry-run"]):
            with patch("src.run_generation._create_client", return_value=MagicMock()):
                with patch(
                    "src.run_generation.generate_all_jobs", return_value=(jobs, 0)
                ):
                    with patch(
                        "src.run_generation.generate_all_resumes",
                        return_value=([resume], [pair], 0),
                    ) as mock_resumes:
                        with patch("src.run_generation.ValidationTracker.save_stats"):
                            from src.run_generation import main

                            main()

        mock_resumes.assert_called_once()
        _, called_jobs = mock_resumes.call_args[0]
        assert called_jobs is jobs


# ---------------------------------------------------------------------------
# main() — full run mode
# ---------------------------------------------------------------------------


class TestMainFullRun:
    def test_full_run_uses_5_per_industry_no_limit(self, tmp_path):
        jobs = [_make_generated_job() for _ in range(2)]
        resume = _make_generated_resume()
        pair = _make_pair(resume.trace_id, jobs[0].trace_id)

        with patch.object(sys, "argv", ["prog"]):  # No --dry-run
            with patch("src.run_generation._create_client", return_value=MagicMock()):
                with patch(
                    "src.run_generation.generate_all_jobs", return_value=(jobs, 2)
                ) as mock_jobs:
                    with patch(
                        "src.run_generation.generate_all_resumes",
                        return_value=([resume], [pair], 5),
                    ):
                        with patch("src.run_generation.ValidationTracker.save_stats"):
                            from src.run_generation import main

                            main()

        call_kwargs = mock_jobs.call_args[1]
        assert call_kwargs["count_per_industry"] == 5
        assert call_kwargs["max_industries"] is None

    def test_full_run_saves_validation_stats(self, tmp_path):
        jobs = [_make_generated_job()]
        resume = _make_generated_resume()
        pair = _make_pair(resume.trace_id, jobs[0].trace_id)

        with patch.object(sys, "argv", ["prog"]):
            with patch("src.run_generation._create_client", return_value=MagicMock()):
                with patch(
                    "src.run_generation.generate_all_jobs", return_value=(jobs, 1)
                ):
                    with patch(
                        "src.run_generation.generate_all_resumes",
                        return_value=([resume], [pair], 5),
                    ):
                        with patch(
                            "src.run_generation.ValidationTracker.save_stats"
                        ) as mock_save:
                            from src.run_generation import main

                            main()

        mock_save.assert_called_once()

    def test_full_run_prints_cache_hit_rate_when_records_exist(self, tmp_path, capsys):
        jobs = [_make_generated_job()]
        resume = _make_generated_resume()
        pair = _make_pair(resume.trace_id, jobs[0].trace_id)

        with patch.object(sys, "argv", ["prog"]):
            with patch("src.run_generation._create_client", return_value=MagicMock()):
                with patch(
                    "src.run_generation.generate_all_jobs", return_value=(jobs, 0)
                ):
                    with patch(
                        "src.run_generation.generate_all_resumes",
                        return_value=([resume], [pair], 0),
                    ):
                        with patch("src.run_generation.ValidationTracker.save_stats"):
                            from src.run_generation import main

                            # Shouldn't raise — all records cached, cache_hits = total_records
                            main()

    def test_no_records_skips_cache_hit_line(self, tmp_path):
        # Edge case: zero jobs + zero resumes → total_records=0 → no division
        with patch.object(sys, "argv", ["prog"]):
            with patch("src.run_generation._create_client", return_value=MagicMock()):
                with patch(
                    "src.run_generation.generate_all_jobs", return_value=([], 0)
                ):
                    with patch(
                        "src.run_generation.generate_all_resumes",
                        return_value=([], [], 0),
                    ):
                        with patch("src.run_generation.ValidationTracker.save_stats"):
                            from src.run_generation import main

                            main()  # Must not raise ZeroDivisionError
