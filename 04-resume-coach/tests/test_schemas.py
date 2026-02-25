from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    ContactInfo,
    Education,
    Experience,
    ExperienceLevel,
    FeedbackRequest,
    FitLevel,
    JobDescription,
    JobRequirements,
    JudgeResult,
    ProficiencyLevel,
    Resume,
    ReviewRequest,
    Skill,
    WritingStyle,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_contact() -> dict:
    return {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "phone": "+1-555-123-4567",
        "location": "Austin, TX",
    }


@pytest.fixture
def valid_education() -> dict:
    return {
        "degree": "BS Computer Science",
        "institution": "MIT",
        "graduation_date": "2020-05",
        "gpa": 3.8,
    }


@pytest.fixture
def valid_experience() -> dict:
    return {
        "company": "Acme Corp",
        "title": "Software Engineer",
        "start_date": "2020-06",
        "end_date": "2023-01",
        "responsibilities": ["Built REST APIs", "Led team of 3"],
    }


@pytest.fixture
def valid_skill() -> dict:
    return {"name": "Python", "proficiency_level": "Advanced", "years": 5}


@pytest.fixture
def valid_resume(valid_contact, valid_education, valid_experience, valid_skill) -> dict:
    return {
        "contact_info": valid_contact,
        "education": [valid_education],
        "experience": [valid_experience],
        "skills": [valid_skill],
    }


@pytest.fixture
def valid_job() -> dict:
    return {
        "title": "Senior Software Engineer",
        "company": {"name": "TechCorp", "industry": "Technology", "size": "500-1000", "location": "Austin, TX"},
        "description": "Build scalable systems",
        "requirements": {
            "required_skills": ["Python", "AWS", "Docker"],
            "education": "BS Computer Science",
            "experience_years": 5,
            "experience_level": "Senior",
        },
    }


# ---------------------------------------------------------------------------
# ContactInfo tests
# ---------------------------------------------------------------------------


def test_contact_info_valid_construction(valid_contact):
    c = ContactInfo(**valid_contact)
    assert c.name == "Jane Doe"
    assert c.email == "jane@example.com"


@pytest.mark.parametrize("email", [
    "invalid",
    "missing@tld",
    "@nodomain.com",
])
def test_contact_info_bad_email_raises(valid_contact, email):
    valid_contact["email"] = email
    with pytest.raises(ValidationError):
        ContactInfo(**valid_contact)


@pytest.mark.parametrize("email", [
    "user@example.com",
    "user.name+tag@domain.co.uk",
    "first.last@subdomain.example.org",
])
def test_contact_info_valid_emails(valid_contact, email):
    valid_contact["email"] = email
    c = ContactInfo(**valid_contact)
    assert c.email == email


def test_contact_info_short_phone_raises(valid_contact):
    valid_contact["phone"] = "123"
    with pytest.raises(ValidationError):
        ContactInfo(**valid_contact)


def test_contact_info_optional_fields_none(valid_contact):
    c = ContactInfo(**valid_contact)
    assert c.linkedin is None
    assert c.portfolio is None


# ---------------------------------------------------------------------------
# Education tests
# ---------------------------------------------------------------------------


def test_education_valid_construction(valid_education):
    e = Education(**valid_education)
    assert e.graduation_date == "2020-05"
    assert e.gpa == 3.8


def test_education_valid_full_date(valid_education):
    valid_education["graduation_date"] = "2020-05-15"
    e = Education(**valid_education)
    assert e.graduation_date == "2020-05-15"


def test_education_bad_date_raises(valid_education):
    valid_education["graduation_date"] = "May 2020"
    with pytest.raises(ValidationError):
        Education(**valid_education)


@pytest.mark.parametrize("date_str", [
    "05/2020",
    "2020",
    "May 2020",
    "20-05",
])
def test_education_invalid_date_formats(valid_education, date_str):
    valid_education["graduation_date"] = date_str
    with pytest.raises(ValidationError):
        Education(**valid_education)


@pytest.mark.parametrize("gpa,should_raise", [
    (0.0, False),
    (4.0, False),
    (3.5, False),
    (4.1, True),
    (-0.1, True),
])
def test_education_gpa_boundaries(valid_education, gpa, should_raise):
    valid_education["gpa"] = gpa
    if should_raise:
        with pytest.raises(ValidationError):
            Education(**valid_education)
    else:
        e = Education(**valid_education)
        assert e.gpa == gpa


def test_education_gpa_above_4_raises(valid_education):
    valid_education["gpa"] = 4.5
    with pytest.raises(ValidationError):
        Education(**valid_education)


def test_education_gpa_negative_raises(valid_education):
    valid_education["gpa"] = -1.0
    with pytest.raises(ValidationError):
        Education(**valid_education)


def test_education_gpa_none_is_valid(valid_education):
    valid_education["gpa"] = None
    e = Education(**valid_education)
    assert e.gpa is None


# ---------------------------------------------------------------------------
# Experience tests
# ---------------------------------------------------------------------------


def test_experience_valid_construction(valid_experience):
    e = Experience(**valid_experience)
    assert e.company == "Acme Corp"
    assert len(e.responsibilities) == 2


def test_experience_empty_responsibilities_raises(valid_experience):
    valid_experience["responsibilities"] = []
    with pytest.raises(ValidationError):
        Experience(**valid_experience)


@pytest.mark.parametrize("date_str", [
    "06/2020",
    "June 2020",
    "2020",
    "20-06",
])
def test_experience_bad_date_format_raises(valid_experience, date_str):
    valid_experience["start_date"] = date_str
    with pytest.raises(ValidationError):
        Experience(**valid_experience)


def test_experience_end_before_start_raises(valid_experience):
    valid_experience["start_date"] = "2023-01"
    valid_experience["end_date"] = "2020-01"
    with pytest.raises(ValidationError):
        Experience(**valid_experience)


def test_experience_current_job_no_end_date(valid_experience):
    valid_experience["end_date"] = None
    e = Experience(**valid_experience)
    assert e.end_date is None


def test_experience_same_start_end_valid(valid_experience):
    valid_experience["start_date"] = "2022-01"
    valid_experience["end_date"] = "2022-01"
    e = Experience(**valid_experience)
    assert e.start_date == e.end_date


# ---------------------------------------------------------------------------
# Skill tests
# ---------------------------------------------------------------------------


def test_skill_valid_construction(valid_skill):
    s = Skill(**valid_skill)
    assert s.name == "Python"
    assert s.proficiency_level == ProficiencyLevel.ADVANCED


@pytest.mark.parametrize("years,should_raise", [
    (0, False),
    (30, False),
    (15, False),
    (-1, True),
    (31, True),
    (35, True),
])
def test_skill_years_boundaries(valid_skill, years, should_raise):
    valid_skill["years"] = years
    if should_raise:
        with pytest.raises(ValidationError):
            Skill(**valid_skill)
    else:
        s = Skill(**valid_skill)
        assert s.years == years


def test_skill_years_negative_raises(valid_skill):
    valid_skill["years"] = -1
    with pytest.raises(ValidationError):
        Skill(**valid_skill)


def test_skill_years_above_30_raises(valid_skill):
    valid_skill["years"] = 35
    with pytest.raises(ValidationError):
        Skill(**valid_skill)


def test_skill_years_none_is_valid(valid_skill):
    valid_skill["years"] = None
    s = Skill(**valid_skill)
    assert s.years is None


# ---------------------------------------------------------------------------
# Resume tests
# ---------------------------------------------------------------------------


def test_resume_valid_construction(valid_resume):
    r = Resume(**valid_resume)
    assert r.contact_info.name == "Jane Doe"
    assert len(r.education) == 1
    assert len(r.experience) == 1
    assert len(r.skills) == 1


def test_resume_empty_education_raises(valid_resume):
    valid_resume["education"] = []
    with pytest.raises(ValidationError):
        Resume(**valid_resume)


def test_resume_empty_experience_raises(valid_resume):
    valid_resume["experience"] = []
    with pytest.raises(ValidationError):
        Resume(**valid_resume)


def test_resume_empty_skills_raises(valid_resume):
    valid_resume["skills"] = []
    with pytest.raises(ValidationError):
        Resume(**valid_resume)


def test_resume_summary_optional(valid_resume):
    r = Resume(**valid_resume)
    assert r.summary is None
    valid_resume["summary"] = "Experienced engineer"
    r2 = Resume(**valid_resume)
    assert r2.summary == "Experienced engineer"


# ---------------------------------------------------------------------------
# JobRequirements tests
# ---------------------------------------------------------------------------


def test_job_requirements_empty_skills_raises():
    with pytest.raises(ValidationError):
        JobRequirements(
            required_skills=[],
            education="BS CS",
            experience_years=3,
            experience_level=ExperienceLevel.MID,
        )


def test_job_requirements_years_negative_raises():
    with pytest.raises(ValidationError):
        JobRequirements(
            required_skills=["Python"],
            education="BS CS",
            experience_years=-1,
            experience_level=ExperienceLevel.MID,
        )


def test_job_requirements_years_above_30_raises():
    with pytest.raises(ValidationError):
        JobRequirements(
            required_skills=["Python"],
            education="BS CS",
            experience_years=31,
            experience_level=ExperienceLevel.MID,
        )


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


def test_proficiency_level_values():
    assert ProficiencyLevel.BEGINNER == "Beginner"
    assert ProficiencyLevel.INTERMEDIATE == "Intermediate"
    assert ProficiencyLevel.ADVANCED == "Advanced"
    assert ProficiencyLevel.EXPERT == "Expert"
    assert len(ProficiencyLevel) == 4


def test_experience_level_values():
    assert ExperienceLevel.ENTRY == "Entry"
    assert ExperienceLevel.MID == "Mid"
    assert ExperienceLevel.SENIOR == "Senior"
    assert ExperienceLevel.LEAD == "Lead"
    assert ExperienceLevel.EXECUTIVE == "Executive"
    assert len(ExperienceLevel) == 5


def test_fit_level_values():
    assert FitLevel.EXCELLENT == "excellent"
    assert FitLevel.GOOD == "good"
    assert FitLevel.PARTIAL == "partial"
    assert FitLevel.POOR == "poor"
    assert FitLevel.MISMATCH == "mismatch"
    assert len(FitLevel) == 5


def test_writing_style_values():
    assert WritingStyle.FORMAL == "formal"
    assert WritingStyle.CASUAL == "casual"
    assert WritingStyle.TECHNICAL == "technical"
    assert WritingStyle.ACHIEVEMENT == "achievement"
    assert WritingStyle.CAREER_CHANGER == "career_changer"
    assert len(WritingStyle) == 5


# ---------------------------------------------------------------------------
# JudgeResult tests
# ---------------------------------------------------------------------------


def test_judge_result_score_above_1_raises():
    with pytest.raises(ValidationError):
        JudgeResult(
            pair_id="pair-1",
            has_hallucinations=False,
            hallucination_details="none",
            has_awkward_language=False,
            awkward_language_details="none",
            overall_quality_score=1.5,
            fit_assessment="good",
            recommendations=[],
            red_flags=[],
        )


def test_judge_result_score_below_0_raises():
    with pytest.raises(ValidationError):
        JudgeResult(
            pair_id="pair-1",
            has_hallucinations=False,
            hallucination_details="none",
            has_awkward_language=False,
            awkward_language_details="none",
            overall_quality_score=-0.1,
            fit_assessment="good",
            recommendations=[],
            red_flags=[],
        )


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_judge_result_score_valid_boundaries(score):
    jr = JudgeResult(
        pair_id="pair-1",
        has_hallucinations=False,
        hallucination_details="none",
        has_awkward_language=False,
        awkward_language_details="none",
        overall_quality_score=score,
        fit_assessment="good",
        recommendations=[],
        red_flags=[],
    )
    assert jr.overall_quality_score == score


# ---------------------------------------------------------------------------
# ReviewRequest test
# ---------------------------------------------------------------------------


def test_review_request_valid(valid_resume, valid_job):
    rr = ReviewRequest(
        resume=Resume(**valid_resume),
        job_description=JobDescription(**valid_job),
    )
    assert rr.resume.contact_info.name == "Jane Doe"
    assert rr.job_description.title == "Senior Software Engineer"


# ---------------------------------------------------------------------------
# FeedbackRequest test
# ---------------------------------------------------------------------------


def test_feedback_request_valid():
    fr = FeedbackRequest(pair_id="pair-abc", rating="positive")
    assert fr.pair_id == "pair-abc"
    assert fr.rating == "positive"
    assert fr.comment is None


def test_feedback_request_with_comment():
    fr = FeedbackRequest(pair_id="pair-abc", rating="negative", comment="Too junior")
    assert fr.comment == "Too junior"
