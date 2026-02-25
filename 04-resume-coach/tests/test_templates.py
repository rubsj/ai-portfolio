from __future__ import annotations

import pytest

from src.schemas import (
    CompanyInfo,
    ExperienceLevel,
    FitLevel,
    JobDescription,
    JobRequirements,
    WritingStyle,
)
from src.templates import (
    FIT_LEVEL_INSTRUCTIONS,
    INDUSTRIES,
    STYLE_INSTRUCTIONS,
    TEMPLATE_VERSION,
    PromptTemplateLibrary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lib() -> PromptTemplateLibrary:
    return PromptTemplateLibrary()


@pytest.fixture
def dummy_job() -> JobDescription:
    return JobDescription(
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
    )


# ---------------------------------------------------------------------------
# PromptTemplateLibrary.get_job_prompt
# ---------------------------------------------------------------------------


class TestGetJobPrompt:
    def test_returns_two_non_empty_strings(self, lib):
        sys_p, usr_p = lib.get_job_prompt("Technology", False, ExperienceLevel.MID)
        assert len(sys_p) > 50
        assert len(usr_p) > 10

    def test_industry_appears_in_system_prompt(self, lib):
        sys_p, _ = lib.get_job_prompt("Healthcare", False, ExperienceLevel.SENIOR)
        assert "Healthcare" in sys_p

    def test_experience_level_in_system_prompt(self, lib):
        sys_p, _ = lib.get_job_prompt("Technology", False, ExperienceLevel.ENTRY)
        assert "Entry" in sys_p

    def test_standard_role_user_prompt_contains_standard(self, lib):
        _, usr_p = lib.get_job_prompt("Finance", False, ExperienceLevel.MID)
        assert "standard" in usr_p

    def test_niche_role_user_prompt_contains_niche(self, lib):
        _, usr_p = lib.get_job_prompt("Technology", True, ExperienceLevel.SENIOR)
        assert "niche/emerging" in usr_p

    def test_niche_role_adds_important_instruction_to_user_prompt(self, lib):
        # WHY: niche_instruction is appended to user_prompt, not system_prompt
        _, usr_p = lib.get_job_prompt("Technology", True, ExperienceLevel.SENIOR)
        assert "IMPORTANT" in usr_p

    def test_non_niche_prompt_has_no_important_instruction(self, lib):
        sys_p, usr_p = lib.get_job_prompt("Technology", False, ExperienceLevel.SENIOR)
        assert "IMPORTANT" not in sys_p
        assert "niche/emerging" not in usr_p

    @pytest.mark.parametrize("level", list(ExperienceLevel))
    def test_all_experience_levels_produce_prompts(self, lib, level):
        sys_p, usr_p = lib.get_job_prompt("Technology", False, level)
        assert level.value in sys_p
        assert level.value in usr_p

    @pytest.mark.parametrize("industry", INDUSTRIES)
    def test_all_industries_embed_in_prompt(self, lib, industry):
        sys_p, _ = lib.get_job_prompt(industry, False, ExperienceLevel.MID)
        assert industry in sys_p


# ---------------------------------------------------------------------------
# PromptTemplateLibrary.get_resume_prompt
# ---------------------------------------------------------------------------


class TestGetResumePrompt:
    def test_returns_two_non_empty_strings(self, lib, dummy_job):
        sys_p, usr_p = lib.get_resume_prompt(dummy_job, FitLevel.EXCELLENT, WritingStyle.FORMAL)
        assert len(sys_p) > 50
        assert len(usr_p) > 50

    def test_writing_style_value_in_system_prompt(self, lib, dummy_job):
        for style in WritingStyle:
            sys_p, _ = lib.get_resume_prompt(dummy_job, FitLevel.GOOD, style)
            assert style.value in sys_p

    def test_fit_level_value_in_user_prompt(self, lib, dummy_job):
        for fit in FitLevel:
            _, usr_p = lib.get_resume_prompt(dummy_job, fit, WritingStyle.FORMAL)
            assert fit.value in usr_p

    def test_job_title_embedded_in_user_prompt(self, lib, dummy_job):
        _, usr_p = lib.get_resume_prompt(dummy_job, FitLevel.EXCELLENT, WritingStyle.TECHNICAL)
        assert "Software Engineer" in usr_p

    def test_required_skills_embedded_in_user_prompt(self, lib, dummy_job):
        _, usr_p = lib.get_resume_prompt(dummy_job, FitLevel.EXCELLENT, WritingStyle.FORMAL)
        assert "Python" in usr_p

    @pytest.mark.parametrize("style", list(WritingStyle))
    def test_all_styles_produce_valid_prompts(self, lib, dummy_job, style):
        sys_p, usr_p = lib.get_resume_prompt(dummy_job, FitLevel.GOOD, style)
        assert len(sys_p) > 50
        assert len(usr_p) > 50

    @pytest.mark.parametrize("fit", list(FitLevel))
    def test_all_fit_levels_produce_valid_prompts(self, lib, dummy_job, fit):
        sys_p, usr_p = lib.get_resume_prompt(dummy_job, fit, WritingStyle.FORMAL)
        assert len(sys_p) > 50
        assert len(usr_p) > 50

    def test_style_instruction_content_differs_per_style(self, lib, dummy_job):
        prompts = [
            lib.get_resume_prompt(dummy_job, FitLevel.GOOD, style)[0]
            for style in WritingStyle
        ]
        # All 5 system prompts should be distinct because style instructions differ
        assert len(set(prompts)) == 5


# ---------------------------------------------------------------------------
# PromptTemplateLibrary.get_template_id
# ---------------------------------------------------------------------------


class TestGetTemplateId:
    @pytest.mark.parametrize(
        "style,expected",
        [
            (WritingStyle.FORMAL, "v1-formal"),
            (WritingStyle.CASUAL, "v1-casual"),
            (WritingStyle.TECHNICAL, "v1-technical"),
            (WritingStyle.ACHIEVEMENT, "v1-achievement"),
            (WritingStyle.CAREER_CHANGER, "v1-career_changer"),
        ],
    )
    def test_template_id_format(self, lib, style, expected):
        assert lib.get_template_id(style) == expected

    def test_template_id_starts_with_version(self, lib):
        for style in WritingStyle:
            assert lib.get_template_id(style).startswith(TEMPLATE_VERSION)


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_industries_has_10_entries(self):
        assert len(INDUSTRIES) == 10

    def test_template_version_is_v1(self):
        assert TEMPLATE_VERSION == "v1"

    def test_fit_level_instructions_covers_all_levels(self):
        for fit in FitLevel:
            assert fit in FIT_LEVEL_INSTRUCTIONS
            assert len(FIT_LEVEL_INSTRUCTIONS[fit]) > 20

    def test_style_instructions_covers_all_styles(self):
        for style in WritingStyle:
            assert style in STYLE_INSTRUCTIONS
            assert len(STYLE_INSTRUCTIONS[style]) > 10
