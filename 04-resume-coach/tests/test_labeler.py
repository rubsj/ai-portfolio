"""
Tests for src/labeler.py — all 6 failure mode detectors.

Each test class is fully self-contained via inline builder helpers.
No conftest needed — tests rely only on the public labeler API.
"""

from __future__ import annotations

import pytest

from src.labeler import (
    calculate_jaccard,
    calculate_total_experience,
    check_experience_mismatch,
    check_seniority_mismatch,
    detect_awkward_language,
    detect_hallucinations,
    infer_seniority,
    label_pair,
)
from src.normalizer import SkillNormalizer
from src.schemas import (
    CompanyInfo,
    ContactInfo,
    Education,
    Experience,
    ExperienceLevel,
    FailureLabels,
    JobDescription,
    JobRequirements,
    ProficiencyLevel,
    Resume,
    Skill,
)


# ---------------------------------------------------------------------------
# Inline builder helpers (keep tests self-contained)
# ---------------------------------------------------------------------------


def _make_experience(
    title: str = "Software Engineer",
    start_date: str = "2020-01",
    end_date: str | None = "2023-01",
    responsibilities: list[str] | None = None,
) -> Experience:
    return Experience(
        company="Acme Corp",
        title=title,
        start_date=start_date,
        end_date=end_date,
        responsibilities=responsibilities or ["Built features"],
    )


def _make_skill(
    name: str,
    proficiency_level: ProficiencyLevel = ProficiencyLevel.INTERMEDIATE,
    years: int | None = None,
) -> Skill:
    return Skill(name=name, proficiency_level=proficiency_level, years=years)


def _make_resume(
    skills: list[Skill] | None = None,
    experience: list[Experience] | None = None,
    summary: str | None = None,
) -> Resume:
    return Resume(
        contact_info=ContactInfo(
            name="Alice Smith",
            email="alice@example.com",
            phone="1234567890",
            location="NYC",
        ),
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="State University",
                graduation_date="2020-05",
            )
        ],
        experience=experience or [_make_experience()],
        skills=skills or [_make_skill("Python")],
        summary=summary,
    )


def _make_job(
    required_skills: list[str] | None = None,
    experience_years: int = 3,
    experience_level: ExperienceLevel = ExperienceLevel.MID,
    title: str = "Software Engineer",
) -> JobDescription:
    return JobDescription(
        title=title,
        company=CompanyInfo(
            name="TechCo",
            industry="Software",
            size="Mid-size (201-500)",
            location="NYC",
        ),
        description="We are hiring...",
        requirements=JobRequirements(
            required_skills=required_skills or ["Python", "SQL", "Git"],
            experience_years=experience_years,
            experience_level=experience_level,
            education="Bachelor's degree",
        ),
    )


# ---------------------------------------------------------------------------
# TestCalculateJaccard
# ---------------------------------------------------------------------------


class TestCalculateJaccard:
    """Tests for Jaccard similarity computation with skill normalization."""

    def setup_method(self) -> None:
        self.norm = SkillNormalizer()

    def test_perfect_overlap_returns_one(self) -> None:
        score, inter, union, _, _ = calculate_jaccard(["Python", "SQL"], ["Python", "SQL"], self.norm)
        assert score == 1.0
        assert inter == 2
        assert union == 2

    def test_no_overlap_returns_zero(self) -> None:
        score, inter, union, _, _ = calculate_jaccard(["Python", "SQL"], ["Java", "Oracle"], self.norm)
        assert score == 0.0
        assert inter == 0
        assert union == 4

    def test_partial_overlap(self) -> None:
        score, inter, union, _, _ = calculate_jaccard(["Python", "SQL", "Java"], ["Python", "Go"], self.norm)
        # intersection={"python"}, union={"python","sql","java","go"} → 1/4
        assert abs(score - 0.25) < 1e-9
        assert inter == 1
        assert union == 4

    def test_normalization_ignores_version_numbers(self) -> None:
        # "Python 3.10" and "python" should normalize to the same token
        score, inter, union, _, _ = calculate_jaccard(["Python 3.10"], ["python"], self.norm)
        assert score == 1.0
        assert inter == 1

    def test_alias_resolution_js_javascript(self) -> None:
        # "js" → "javascript" alias; both sides should match
        score, _, _, _, _ = calculate_jaccard(["js"], ["JavaScript"], self.norm)
        assert score == 1.0

    def test_both_empty_returns_zero(self) -> None:
        score, inter, union, _, _ = calculate_jaccard([], [], self.norm)
        assert score == 0.0
        assert inter == 0
        assert union == 0

    def test_one_side_empty(self) -> None:
        score, inter, union, _, _ = calculate_jaccard(["Python"], [], self.norm)
        assert score == 0.0
        assert inter == 0
        assert union == 1

    def test_returns_normalized_sets(self) -> None:
        # Returned sets are already normalized
        _, _, _, resume_set, job_set = calculate_jaccard(["Python 3.10"], ["python"], self.norm)
        assert "python" in resume_set
        assert "python" in job_set


# ---------------------------------------------------------------------------
# TestCalculateTotalExperience
# ---------------------------------------------------------------------------


class TestCalculateTotalExperience:
    """Tests for summing years across experience entries."""

    def test_single_job_three_years(self) -> None:
        exp = [_make_experience(start_date="2020-01", end_date="2023-01")]
        total = calculate_total_experience(exp)
        # 36 months / 12 = 3.0
        assert abs(total - 3.0) < 0.1

    def test_current_job_no_end_date(self) -> None:
        # end_date=None means job is ongoing — result should be > 0
        from datetime import datetime

        start_year = datetime.now().year - 2
        exp = [_make_experience(start_date=f"{start_year}-01", end_date=None)]
        total = calculate_total_experience(exp)
        assert total > 1.5  # At least ~2 years

    def test_multiple_jobs_summed(self) -> None:
        exps = [
            _make_experience(start_date="2015-01", end_date="2018-01"),  # 3yr
            _make_experience(start_date="2018-06", end_date="2020-06"),  # 2yr
        ]
        total = calculate_total_experience(exps)
        assert abs(total - 5.0) < 0.1

    def test_empty_list_returns_zero(self) -> None:
        assert calculate_total_experience([]) == 0.0

    def test_same_month_start_end_returns_zero(self) -> None:
        exp = [_make_experience(start_date="2023-01", end_date="2023-01")]
        total = calculate_total_experience(exp)
        assert total == 0.0


# ---------------------------------------------------------------------------
# TestInferSeniority
# ---------------------------------------------------------------------------


class TestInferSeniority:
    """Tests for title-based seniority level inference."""

    @pytest.mark.parametrize(
        "title, years, expected_level",
        [
            ("Junior Software Engineer", 1.0, 0),
            ("Software Engineer", 3.0, 1),
            ("Senior Software Engineer", 6.0, 2),
            ("Staff Engineer", 8.0, 3),
            ("VP of Engineering", 15.0, 4),
            ("Director of Engineering", 10.0, 4),
            ("Principal Engineer", 12.0, 3),
            ("Architect", 12.0, 3),
        ],
    )
    def test_title_based_inference(self, title: str, years: float, expected_level: int) -> None:
        assert infer_seniority(title, years) == expected_level

    def test_senior_wins_over_associate_in_compound_title(self) -> None:
        # "Senior Associate" — "senior" (2) must win over "associate" (0)
        assert infer_seniority("Senior Associate", 5.0) == 2

    def test_fallback_entry_by_years(self) -> None:
        # Generic title, < 2yr → entry
        assert infer_seniority("Software Developer", 1.0) == 0

    def test_fallback_mid_by_years(self) -> None:
        assert infer_seniority("Software Developer", 3.0) == 1

    def test_fallback_senior_by_years(self) -> None:
        assert infer_seniority("Software Developer", 7.0) == 2

    def test_fallback_lead_by_years(self) -> None:
        assert infer_seniority("Software Developer", 12.0) == 3

    def test_fallback_executive_by_years(self) -> None:
        assert infer_seniority("Software Developer", 20.0) == 4


# ---------------------------------------------------------------------------
# TestCheckExperienceMismatch
# ---------------------------------------------------------------------------


class TestCheckExperienceMismatch:
    """Tests for experience gap detection."""

    @pytest.mark.parametrize(
        "resume_years, job_years, expected",
        [
            (5.0, 5, False),    # Exact match
            (7.0, 5, False),    # Over-qualified
            (2.0, 5, True),     # < 50% of 5yr
            (1.0, 5, True),     # Severely under-qualified
            (0.0, 0, False),    # Entry-level job, no experience required
            (3.0, 0, False),    # Entry-level job, candidate has experience
            (7.0, 10, False),   # 7 < 10*0.5=5 → False; 7 < 10-3=7 → False (boundary)
            (6.5, 10, True),    # 6.5 < 10*0.5=5 → False, 6.5 < 10-3=7 → True
        ],
    )
    def test_parametrized(self, resume_years: float, job_years: int, expected: bool) -> None:
        assert check_experience_mismatch(resume_years, job_years) == expected


# ---------------------------------------------------------------------------
# TestCheckSeniorityMismatch
# ---------------------------------------------------------------------------


class TestCheckSeniorityMismatch:
    """Tests for seniority gap detection."""

    @pytest.mark.parametrize(
        "resume_level, job_level, expected",
        [
            (0, 0, False),   # Same level
            (0, 1, False),   # 1-level gap is acceptable
            (1, 0, False),   # Over-qualified by 1 is fine
            (0, 2, True),    # 2-level gap is a mismatch
            (4, 1, True),    # VP applying for mid-level
            (2, 2, False),   # Exact match
            (3, 1, True),    # Lead applying for junior
        ],
    )
    def test_parametrized(self, resume_level: int, job_level: int, expected: bool) -> None:
        assert check_seniority_mismatch(resume_level, job_level) == expected


# ---------------------------------------------------------------------------
# TestDetectHallucinations
# ---------------------------------------------------------------------------


class TestDetectHallucinations:
    """Tests for rule-based hallucination detection."""

    def test_normal_resume_no_flags(self) -> None:
        skills = [_make_skill("Python", ProficiencyLevel.INTERMEDIATE, years=3)]
        resume = _make_resume(skills=skills, experience=[_make_experience(start_date="2020-01", end_date="2023-01")])
        flagged, reasons = detect_hallucinations(resume, 3.0)
        assert not flagged
        assert reasons == []

    def test_entry_level_with_too_many_experts(self) -> None:
        # 0.5yr experience, 5 Expert skills → Rule 1
        skills = [
            _make_skill("Python", ProficiencyLevel.EXPERT),
            _make_skill("Java", ProficiencyLevel.EXPERT),
            _make_skill("AWS", ProficiencyLevel.EXPERT),
            _make_skill("React", ProficiencyLevel.EXPERT),
            _make_skill("Docker", ProficiencyLevel.EXPERT),
        ]
        resume = _make_resume(skills=skills)
        flagged, reasons = detect_hallucinations(resume, 0.5)
        assert flagged
        assert any("Entry-level" in r for r in reasons)

    def test_skill_years_exceed_total_experience(self) -> None:
        # 10yr skill years, 3yr total experience → Rule 3
        skills = [_make_skill("Python", ProficiencyLevel.ADVANCED, years=10)]
        resume = _make_resume(skills=skills)
        flagged, reasons = detect_hallucinations(resume, 3.0)
        assert flagged
        assert any("total experience" in r for r in reasons)

    def test_senior_title_with_little_experience(self) -> None:
        # VP title with 2yr experience → Rule 4
        exps = [
            _make_experience(title="VP of Engineering", start_date="2022-01", end_date="2024-01"),
            _make_experience(title="Software Engineer", start_date="2020-01", end_date="2022-01"),
        ]
        resume = _make_resume(experience=exps)
        flagged, reasons = detect_hallucinations(resume, 2.0)
        assert flagged
        assert any("Senior title" in r for r in reasons)

    def test_unrealistic_skill_count(self) -> None:
        # 25 skills, 12 Expert → Rule 2
        skills = [
            _make_skill(f"Skill{i}", ProficiencyLevel.EXPERT if i < 12 else ProficiencyLevel.INTERMEDIATE)
            for i in range(25)
        ]
        resume = _make_resume(skills=skills)
        flagged, reasons = detect_hallucinations(resume, 10.0)
        assert flagged
        assert any("Expert level" in r for r in reasons)


# ---------------------------------------------------------------------------
# TestDetectAwkwardLanguage
# ---------------------------------------------------------------------------


class TestDetectAwkwardLanguage:
    """Tests for buzzword and AI-pattern detection."""

    def test_clean_resume_no_flags(self) -> None:
        resume = _make_resume(
            summary="Built backend services using Python and PostgreSQL.",
            experience=[
                _make_experience(
                    responsibilities=["Implemented REST APIs", "Reviewed pull requests"],
                )
            ],
        )
        flagged, reasons = detect_awkward_language(resume)
        assert not flagged
        assert reasons == []

    def test_buzzword_density_trigger(self) -> None:
        # 7 buzzwords → Rule 1 (threshold is >5)
        summary = (
            "A results-driven self-starter who is passionate, motivated, and proactive. "
            "Leverages synergy to deliver scalable, robust, dynamic solutions."
        )
        resume = _make_resume(summary=summary)
        flagged, reasons = detect_awkward_language(resume)
        assert flagged
        assert any("buzzword" in r.lower() for r in reasons)

    def test_ai_patterns_trigger(self) -> None:
        # 3 AI patterns → Rule 2 (threshold is >2)
        responsibilities = [
            "As a seasoned engineer with a proven track record of success",
            "Demonstrated ability to deliver results in today's fast-paced environment",
        ]
        resume = _make_resume(
            experience=[_make_experience(responsibilities=responsibilities)]
        )
        flagged, reasons = detect_awkward_language(resume)
        assert flagged
        assert any("AI-generated" in r for r in reasons)

    def test_repeated_words_trigger(self) -> None:
        # WHY 20 repetitions: 20 × "implemented the feature" = 60 words.
        # The sliding window needs 51+ words to activate (range(len - 50) > 0).
        # "implemented" (11 chars, >4) appears every 3 words → 16+ times per window.
        resp = " ".join(["implemented the feature"] * 20)
        resume = _make_resume(experience=[_make_experience(responsibilities=[resp])])
        flagged, reasons = detect_awkward_language(resume)
        assert flagged
        assert any("Repeated words" in r for r in reasons)


# ---------------------------------------------------------------------------
# TestLabelPair
# ---------------------------------------------------------------------------


class TestLabelPair:
    """Integration tests for the label_pair orchestrator."""

    def test_label_pair_returns_valid_failure_labels(self) -> None:
        resume = _make_resume(
            skills=[_make_skill("Python"), _make_skill("SQL"), _make_skill("Git")],
            experience=[_make_experience(start_date="2020-01", end_date="2023-01")],
        )
        job = _make_job(required_skills=["Python", "SQL", "Git"])
        result = label_pair(resume, job, "test-pair-001")

        # Pydantic validates all 18 fields at construction — if we reach here, it's valid
        assert isinstance(result, FailureLabels)
        assert result.pair_id == "test-pair-001"
        assert 0.0 <= result.skills_overlap <= 1.0

    def test_excellent_fit_has_high_jaccard(self) -> None:
        # All 3 required skills present → Jaccard should be high
        skills = [_make_skill("Python"), _make_skill("SQL"), _make_skill("Docker")]
        resume = _make_resume(skills=skills)
        job = _make_job(required_skills=["Python", "SQL", "Docker"])
        result = label_pair(resume, job, "test-pair-excellent")
        assert result.skills_overlap > 0.5

    def test_poor_fit_has_low_jaccard_and_flags(self) -> None:
        # Resume has Java, job needs Python, SQL, Docker → no overlap
        resume = _make_resume(
            skills=[_make_skill("Java")],
            experience=[_make_experience(start_date="2022-01", end_date="2023-01")],
        )
        job = _make_job(
            required_skills=["Python", "SQL", "Docker"],
            experience_years=8,
            experience_level=ExperienceLevel.SENIOR,
            title="Senior Software Engineer",
        )
        result = label_pair(resume, job, "test-pair-poor")
        assert result.skills_overlap == 0.0
        assert result.missing_core_skills is True
        assert len(result.missing_skills) > 0
        assert result.experience_mismatch is True  # 1yr vs 8yr required

    def test_shared_normalizer_instance_accepted(self) -> None:
        norm = SkillNormalizer()
        resume = _make_resume()
        job = _make_job()
        result = label_pair(resume, job, "test-pair-shared-norm", normalizer=norm)
        assert isinstance(result, FailureLabels)
