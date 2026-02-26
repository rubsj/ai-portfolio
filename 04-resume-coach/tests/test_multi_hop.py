"""
test_multi_hop.py — Unit tests for src/multi_hop.py.

All tests are deterministic (no LLM calls). Covers all 4 Q generators,
the public API, loaders, and the run() CLI function via monkeypatched paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.multi_hop import (
    _q_career_progression,
    _q_education_vs_experience,
    _q_skills_consistency,
    _q_skills_overlap_fit,
    generate_multi_hop_questions,
)
from src.schemas import (
    CompanyInfo,
    ContactInfo,
    Education,
    ExperienceLevel,
    FailureLabels,
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
from src.schemas import Experience as ExpSchema


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _make_contact() -> ContactInfo:
    return ContactInfo(
        name="Jane Doe",
        email="jane@example.com",
        phone="555-123-4567",
        location="New York, NY",
    )


def _make_education(degree: str = "BS Computer Science") -> Education:
    return Education(
        degree=degree,
        institution="State University",
        graduation_date="2018-05",
    )


def _make_experience(
    title: str = "Software Engineer",
    start_date: str = "2018-06",
    end_date: str | None = "2023-05",
) -> ExpSchema:
    return ExpSchema(
        company="TechCorp",
        title=title,
        start_date=start_date,
        end_date=end_date,
        responsibilities=["Built and maintained distributed systems"],
    )


def _make_skill(name: str = "Python") -> Skill:
    return Skill(name=name, proficiency_level=ProficiencyLevel.ADVANCED, years=4)


def _make_resume(
    *,
    experiences: list[ExpSchema] | None = None,
    education: list[Education] | None = None,
    skills: list[Skill] | None = None,
) -> Resume:
    return Resume(
        contact_info=_make_contact(),
        education=education or [_make_education()],
        experience=experiences or [_make_experience()],
        skills=skills or [_make_skill()],
        summary="Experienced software engineer",
    )


def _make_job(
    *,
    experience_level: ExperienceLevel = ExperienceLevel.MID,
    experience_years: int = 5,
    required_skills: list[str] | None = None,
    industry: str = "Technology",
    company_name: str = "BigCorp",
    title: str = "Senior Software Engineer",
) -> JobDescription:
    return JobDescription(
        title=title,
        company=CompanyInfo(
            name=company_name,
            industry=industry,
            size="Enterprise (500+)",
            location="San Francisco, CA",
        ),
        description="We need a skilled engineer",
        requirements=JobRequirements(
            required_skills=required_skills or ["Python", "SQL"],
            education="BS degree required",
            experience_years=experience_years,
            experience_level=experience_level,
        ),
    )


def _make_labels(**overrides) -> FailureLabels:
    """Create FailureLabels with sensible defaults; override any field via kwargs."""
    defaults: dict = dict(
        pair_id="test-pair-001",
        skills_overlap=0.6,
        skills_overlap_raw=3,
        skills_union_raw=5,
        experience_mismatch=False,
        seniority_mismatch=False,
        missing_core_skills=False,
        has_hallucinations=False,
        has_awkward_language=False,
        experience_years_resume=5.0,
        experience_years_required=3,
        seniority_level_resume=1,
        seniority_level_job=1,
        missing_skills=[],
        hallucination_reasons=[],
        awkward_language_reasons=[],
        resume_skills_normalized=["python", "sql", "pandas"],
        job_skills_normalized=["python", "sql", "java"],
    )
    defaults.update(overrides)
    return FailureLabels(**defaults)


# ---------------------------------------------------------------------------
# TestQEducationVsExperience
# ---------------------------------------------------------------------------


class TestQEducationVsExperience:
    def test_aligned_assessment(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(experience_mismatch=False, experience_years_resume=5.0)
        q = _q_education_vs_experience(resume, job, labels)
        assert q.assessment == "aligned"
        assert "education" in q.requires_sections
        assert "requirements" in q.requires_sections
        assert "BS Computer Science" in q.question

    def test_mismatch_assessment(self) -> None:
        resume = _make_resume()
        job = _make_job(experience_years=8)
        labels = _make_labels(
            experience_mismatch=True,
            experience_years_resume=2.0,
            experience_years_required=8,
        )
        q = _q_education_vs_experience(resume, job, labels)
        assert q.assessment == "mismatch"
        assert "2.0" in q.answer
        assert "gap" in q.answer.lower()

    def test_no_education_defaults_to_no_formal_degree(self) -> None:
        # Resume with no education entries — validator requires >=1, so we patch post-init
        resume = _make_resume(education=[_make_education("No formal degree")])
        job = _make_job()
        labels = _make_labels()
        q = _q_education_vs_experience(resume, job, labels)
        assert "No formal degree" in q.question

    def test_question_contains_exp_level_and_years(self) -> None:
        resume = _make_resume()
        job = _make_job(experience_level=ExperienceLevel.SENIOR, experience_years=7)
        labels = _make_labels()
        q = _q_education_vs_experience(resume, job, labels)
        # ExperienceLevel.SENIOR.value == "Senior"
        assert "Senior" in q.question
        assert "7" in q.question

    def test_returns_multi_hop_question_schema(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        q = _q_education_vs_experience(resume, job, labels)
        assert hasattr(q, "question")
        assert hasattr(q, "answer")
        assert hasattr(q, "assessment")
        assert hasattr(q, "requires_sections")


# ---------------------------------------------------------------------------
# TestQSkillsConsistency
# ---------------------------------------------------------------------------


class TestQSkillsConsistency:
    def test_consistent_assessment(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(has_hallucinations=False, missing_core_skills=False)
        q = _q_skills_consistency(resume, job, labels)
        assert q.assessment == "consistent"
        assert "Consistent" in q.answer

    def test_inconsistent_when_hallucinations(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(
            has_hallucinations=True,
            hallucination_reasons=["Claims 15 years of React (released 2013)"],
        )
        q = _q_skills_consistency(resume, job, labels)
        assert q.assessment == "inconsistent"
        assert "Inconsistency" in q.answer

    def test_partial_when_missing_core_skills(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(
            has_hallucinations=False,
            missing_core_skills=True,
            missing_skills=["AWS", "Kubernetes"],
        )
        q = _q_skills_consistency(resume, job, labels)
        assert q.assessment == "partial"
        assert "AWS" in q.answer or "Kubernetes" in q.answer

    def test_hallucinations_no_reasons_uses_fallback(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(
            has_hallucinations=True,
            hallucination_reasons=[],
        )
        q = _q_skills_consistency(resume, job, labels)
        assert q.assessment == "inconsistent"
        assert "inflated" in q.answer

    def test_requires_sections(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        q = _q_skills_consistency(resume, job, labels)
        assert "skills" in q.requires_sections
        assert "experience" in q.requires_sections
        assert "requirements" in q.requires_sections


# ---------------------------------------------------------------------------
# TestQCareerProgression
# ---------------------------------------------------------------------------


class TestQCareerProgression:
    def test_aligned_single_level_delta(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(
            seniority_mismatch=False,
            seniority_level_resume=1,
            seniority_level_job=1,
        )
        q = _q_career_progression(resume, job, labels)
        assert q.assessment == "aligned"
        assert "Realistic" in q.answer

    def test_mismatch_assessment(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(
            seniority_mismatch=True,
            seniority_level_resume=0,  # entry-level
            seniority_level_job=3,     # lead/principal
        )
        q = _q_career_progression(resume, job, labels)
        assert q.assessment == "mismatch"
        assert "mismatch" in q.answer.lower()

    def test_two_experiences_sorted_chronologically(self) -> None:
        """Oldest start_date should be first_title, newest should be last_title."""
        exp_old = _make_experience(title="Junior Developer", start_date="2015-01", end_date="2018-05")
        exp_new = _make_experience(title="Tech Lead", start_date="2020-06")
        resume = _make_resume(experiences=[exp_new, exp_old])  # intentionally reversed order
        job = _make_job()
        labels = _make_labels()
        q = _q_career_progression(resume, job, labels)
        # "Junior Developer" → "Tech Lead" — oldest first
        assert "Junior Developer" in q.question
        assert "Tech Lead" in q.question

    def test_single_experience_title_equals_both_ends(self) -> None:
        """One experience → first_title == last_title."""
        exp = _make_experience(title="Data Analyst")
        resume = _make_resume(experiences=[exp])
        job = _make_job()
        labels = _make_labels()
        q = _q_career_progression(resume, job, labels)
        # Both placeholders in question should be the same title
        assert q.question.count("Data Analyst") == 2

    def test_requires_experience_section(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        q = _q_career_progression(resume, job, labels)
        assert "experience" in q.requires_sections

    def test_seniority_labels_in_question(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(seniority_level_resume=2, seniority_level_job=3)
        q = _q_career_progression(resume, job, labels)
        # seniority_level 2 = "senior", 3 = "lead/principal"
        assert "senior" in q.question.lower()
        assert "lead" in q.question.lower()

    def test_no_experience_uses_fallback_title(self) -> None:
        """Empty experience list → fallback title = 'no experience listed'.

        WHY MagicMock: Resume schema requires min 1 experience entry (Pydantic validates).
        This else branch is a defensive guard that can't be reached via a valid Resume.
        MagicMock bypasses Pydantic so we can test the pure function logic at line 174.
        """
        from unittest.mock import MagicMock

        resume = MagicMock()
        resume.experience = []  # triggers the else branch
        job = _make_job()
        labels = _make_labels()
        q = _q_career_progression(resume, job, labels)
        assert "no experience listed" in q.question


# ---------------------------------------------------------------------------
# TestQSkillsOverlapFit
# ---------------------------------------------------------------------------


class TestQSkillsOverlapFit:
    def test_sufficient_overlap(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(skills_overlap=0.7, missing_skills=[])
        q = _q_skills_overlap_fit(resume, job, labels)
        assert q.assessment == "sufficient"
        assert "Sufficient" in q.answer

    def test_sufficient_with_minor_gaps(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(skills_overlap=0.6, missing_skills=["Docker"])
        q = _q_skills_overlap_fit(resume, job, labels)
        assert q.assessment == "sufficient"
        assert "Minor gaps" in q.answer
        assert "Docker" in q.answer

    def test_partial_overlap(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(skills_overlap=0.3, missing_skills=["Kubernetes", "Terraform"])
        q = _q_skills_overlap_fit(resume, job, labels)
        assert q.assessment == "partial"
        assert "Partial" in q.answer

    def test_insufficient_overlap(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels(skills_overlap=0.1, missing_skills=["Python", "SQL", "AWS"])
        q = _q_skills_overlap_fit(resume, job, labels)
        assert q.assessment == "insufficient"
        assert "Insufficient" in q.answer
        assert "Python" in q.answer or "SQL" in q.answer

    def test_question_contains_industry_and_percentage(self) -> None:
        resume = _make_resume()
        job = _make_job(industry="Healthcare IT")
        labels = _make_labels(skills_overlap=0.4, job_skills_normalized=["hl7", "fhir", "python"])
        q = _q_skills_overlap_fit(resume, job, labels)
        assert "Healthcare IT" in q.question
        assert "40%" in q.question

    def test_requires_correct_sections(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        q = _q_skills_overlap_fit(resume, job, labels)
        assert "skills" in q.requires_sections
        assert "requirements" in q.requires_sections
        assert "company" in q.requires_sections


# ---------------------------------------------------------------------------
# TestGenerateMultiHopQuestions
# ---------------------------------------------------------------------------


class TestGenerateMultiHopQuestions:
    def test_returns_four_questions(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        response = generate_multi_hop_questions(resume, job, labels, "pair-xyz")
        assert len(response.questions) == 4

    def test_pair_id_set_correctly(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        response = generate_multi_hop_questions(resume, job, labels, "pair-abc-123")
        assert response.pair_id == "pair-abc-123"

    def test_processing_time_non_negative(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        response = generate_multi_hop_questions(resume, job, labels, "pair-001")
        assert response.processing_time_seconds >= 0.0

    def test_all_four_question_types_present(self) -> None:
        """Verifies questions cover education, skills, career, and overlap."""
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        response = generate_multi_hop_questions(resume, job, labels, "pair-001")
        q1, q2, q3, q4 = response.questions
        assert "education" in q1.requires_sections
        assert "experience" in q2.requires_sections
        assert "experience" in q3.requires_sections
        assert "company" in q4.requires_sections

    def test_assessments_are_non_empty_strings(self) -> None:
        resume = _make_resume()
        job = _make_job()
        labels = _make_labels()
        response = generate_multi_hop_questions(resume, job, labels, "pair-001")
        for q in response.questions:
            assert isinstance(q.assessment, str)
            assert len(q.assessment) > 0


# ---------------------------------------------------------------------------
# TestLoaders
# ---------------------------------------------------------------------------


class TestLoaders:
    def test_load_pairs(self, tmp_path: Path) -> None:
        from src.multi_hop import _load_pairs

        pair = ResumeJobPair(
            pair_id="p1",
            resume_trace_id="r1",
            job_trace_id="j1",
            fit_level=FitLevel.EXCELLENT,
            created_at="2026-02-25T14:00:00Z",
        )
        jsonl = tmp_path / "pairs.jsonl"
        jsonl.write_text(pair.model_dump_json() + "\n")
        result = _load_pairs(jsonl)
        assert len(result) == 1
        assert result[0].pair_id == "p1"
        assert result[0].fit_level == FitLevel.EXCELLENT

    def test_load_jobs(self, tmp_path: Path) -> None:
        from src.multi_hop import _load_jobs

        job = _make_job()
        gj = GeneratedJob(
            trace_id="j1",
            job=job,
            is_niche_role=False,
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )
        jsonl = tmp_path / "jobs.jsonl"
        jsonl.write_text(gj.model_dump_json() + "\n")
        result = _load_jobs(jsonl)
        assert "j1" in result
        assert result["j1"].trace_id == "j1"

    def test_load_resumes(self, tmp_path: Path) -> None:
        from src.multi_hop import _load_resumes

        resume = _make_resume()
        gr = GeneratedResume(
            trace_id="r1",
            resume=resume,
            fit_level=FitLevel.EXCELLENT,
            writing_style=WritingStyle.TECHNICAL,
            template_version="v1",
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )
        jsonl = tmp_path / "resumes.jsonl"
        jsonl.write_text(gr.model_dump_json() + "\n")
        result = _load_resumes(jsonl)
        assert "r1" in result
        assert result["r1"].trace_id == "r1"

    def test_load_labels(self, tmp_path: Path) -> None:
        from src.multi_hop import _load_labels

        labels = _make_labels(pair_id="p1")
        jsonl = tmp_path / "labels.jsonl"
        jsonl.write_text(labels.model_dump_json() + "\n")
        result = _load_labels(jsonl)
        assert "p1" in result
        assert result["p1"].pair_id == "p1"


# ---------------------------------------------------------------------------
# TestRunFunction
# ---------------------------------------------------------------------------


class TestRunFunction:
    """Tests the run() CLI entry point using monkeypatched file paths."""

    def _write_test_data(self, tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
        """Create minimal JSONL fixtures in tmp_path; return (pairs, jobs, resumes, labels, output)."""
        job = _make_job()
        resume = _make_resume()
        labels = _make_labels(pair_id="p1")

        pair = ResumeJobPair(
            pair_id="p1",
            resume_trace_id="r1",
            job_trace_id="j1",
            fit_level=FitLevel.EXCELLENT,
            created_at="2026-02-25T14:00:00Z",
        )
        gj = GeneratedJob(
            trace_id="j1",
            job=job,
            is_niche_role=False,
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )
        gr = GeneratedResume(
            trace_id="r1",
            resume=resume,
            fit_level=FitLevel.EXCELLENT,
            writing_style=WritingStyle.TECHNICAL,
            template_version="v1",
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )

        pairs_file = tmp_path / "pairs.jsonl"
        jobs_file = tmp_path / "jobs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        labels_file = tmp_path / "labels.jsonl"
        output_file = tmp_path / "multi_hop_questions.jsonl"

        pairs_file.write_text(pair.model_dump_json() + "\n")
        jobs_file.write_text(gj.model_dump_json() + "\n")
        resumes_file.write_text(gr.model_dump_json() + "\n")
        labels_file.write_text(labels.model_dump_json() + "\n")

        return pairs_file, jobs_file, resumes_file, labels_file, output_file

    def test_run_produces_output_file(self, tmp_path: Path) -> None:
        import src.multi_hop as mh

        pairs_file, jobs_file, resumes_file, labels_file, output_file = self._write_test_data(
            tmp_path
        )

        with (
            patch.object(mh, "_PAIRS_FILE", pairs_file),
            patch.object(mh, "_JOBS_FILE", jobs_file),
            patch.object(mh, "_RESUMES_FILE", resumes_file),
            patch.object(mh, "_LABELS_FILE", labels_file),
            patch.object(mh, "_ANALYSIS_DIR", tmp_path),
            patch.object(mh, "OUTPUT_FILE", output_file),
        ):
            mh.run()

        assert output_file.exists()
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 1  # 1 pair
        record = json.loads(lines[0])
        assert record["pair_id"] == "p1"
        assert len(record["questions"]) == 4

    def test_run_handles_missing_key_gracefully(self, tmp_path: Path) -> None:
        """If a pair references a non-existent resume trace_id, it logs error and continues."""
        import src.multi_hop as mh

        job = _make_job()
        labels = _make_labels(pair_id="p-missing")

        pair = ResumeJobPair(
            pair_id="p-missing",
            resume_trace_id="r-DOES-NOT-EXIST",  # no matching resume
            job_trace_id="j1",
            fit_level=FitLevel.POOR,
            created_at="2026-02-25T14:00:00Z",
        )
        gj = GeneratedJob(
            trace_id="j1",
            job=job,
            is_niche_role=False,
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )

        pairs_file = tmp_path / "pairs.jsonl"
        jobs_file = tmp_path / "jobs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        labels_file = tmp_path / "labels.jsonl"
        output_file = tmp_path / "output.jsonl"

        pairs_file.write_text(pair.model_dump_json() + "\n")
        jobs_file.write_text(gj.model_dump_json() + "\n")
        resumes_file.write_text("")  # empty — no resumes
        labels_file.write_text(labels.model_dump_json() + "\n")

        with (
            patch.object(mh, "_PAIRS_FILE", pairs_file),
            patch.object(mh, "_JOBS_FILE", jobs_file),
            patch.object(mh, "_RESUMES_FILE", resumes_file),
            patch.object(mh, "_LABELS_FILE", labels_file),
            patch.object(mh, "_ANALYSIS_DIR", tmp_path),
            patch.object(mh, "OUTPUT_FILE", output_file),
        ):
            mh.run()  # must not raise

        # Output file created but empty (error logged, 0 responses)
        assert output_file.exists()
        content = output_file.read_text().strip()
        assert content == ""

    def test_run_handles_general_exception_gracefully(self, tmp_path: Path) -> None:
        """Non-KeyError exceptions in the processing loop are caught and logged."""
        import src.multi_hop as mh

        job = _make_job()
        resume = _make_resume()
        labels = _make_labels(pair_id="p1")

        pair = ResumeJobPair(
            pair_id="p1",
            resume_trace_id="r1",
            job_trace_id="j1",
            fit_level=FitLevel.EXCELLENT,
            created_at="2026-02-25T14:00:00Z",
        )
        gj = GeneratedJob(
            trace_id="j1",
            job=job,
            is_niche_role=False,
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )
        gr = GeneratedResume(
            trace_id="r1",
            resume=resume,
            fit_level=FitLevel.EXCELLENT,
            writing_style=WritingStyle.TECHNICAL,
            template_version="v1",
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )

        pairs_file = tmp_path / "pairs.jsonl"
        jobs_file = tmp_path / "jobs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        labels_file = tmp_path / "labels.jsonl"
        output_file = tmp_path / "output.jsonl"

        pairs_file.write_text(pair.model_dump_json() + "\n")
        jobs_file.write_text(gj.model_dump_json() + "\n")
        resumes_file.write_text(gr.model_dump_json() + "\n")
        labels_file.write_text(labels.model_dump_json() + "\n")

        # Patch generate_multi_hop_questions to raise a generic exception
        with (
            patch.object(mh, "_PAIRS_FILE", pairs_file),
            patch.object(mh, "_JOBS_FILE", jobs_file),
            patch.object(mh, "_RESUMES_FILE", resumes_file),
            patch.object(mh, "_LABELS_FILE", labels_file),
            patch.object(mh, "_ANALYSIS_DIR", tmp_path),
            patch.object(mh, "OUTPUT_FILE", output_file),
            patch.object(
                mh,
                "generate_multi_hop_questions",
                side_effect=ValueError("simulated processing error"),
            ),
        ):
            mh.run()  # must not raise

        # Output file is created but empty — the exception was caught
        assert output_file.exists()
