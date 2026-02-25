from __future__ import annotations

from src.schemas import ExperienceLevel, FitLevel, JobDescription, WritingStyle

INDUSTRIES: list[str] = [
    "Technology",
    "Healthcare",
    "Finance",
    "Education",
    "Manufacturing",
    "Retail",
    "Energy",
    "Legal",
    "Marketing",
    "Government",
]

TEMPLATE_VERSION = "v1"

FIT_LEVEL_INSTRUCTIONS: dict[FitLevel, str] = {
    FitLevel.EXCELLENT: (
        "Generate a resume that is an EXCELLENT match for this job. "
        "Include 80%+ of the required skills with matching proficiency levels. "
        "Experience years should meet or exceed the requirement. "
        "Seniority level should match the job level."
    ),
    FitLevel.GOOD: (
        "Generate a GOOD but not perfect candidate. "
        "Include most required skills but miss 1-2 key skills. "
        "Add some unrelated skills. Experience is close but slightly under."
    ),
    FitLevel.PARTIAL: (
        "Generate a candidate with PARTIAL, incomplete qualifications. "
        "Include roughly half of the required skills. "
        "Experience level may differ by one tier. Background is related but not ideal."
    ),
    FitLevel.POOR: (
        "Generate a POORLY matched candidate. "
        "Include few matching skills (20-40% overlap). Wrong seniority level. "
        "From a related but different domain. Clearly under-qualified."
    ),
    FitLevel.MISMATCH: (
        "Generate a completely MISMATCHED candidate from a different industry. "
        "Almost no skill overlap (<20%). Different seniority level entirely. "
        "Background is from a completely unrelated field."
    ),
}

STYLE_INSTRUCTIONS: dict[WritingStyle, str] = {
    WritingStyle.FORMAL: "Professional, corporate language. Standard chronological format. No first person.",
    WritingStyle.CASUAL: "Conversational, startup-friendly tone. May use first person. Project-focused.",
    WritingStyle.TECHNICAL: "Heavy on technical details, tool versions, and architecture specifics.",
    WritingStyle.ACHIEVEMENT: "Every bullet must have a quantified result (e.g., 'increased X by Y%').",
    WritingStyle.CAREER_CHANGER: (
        "Emphasize transferable skills from a different industry. Include narrative transitions."
    ),
}


class PromptTemplateLibrary:
    """Factory for job and resume generation prompts across 5 writing styles."""

    def get_job_prompt(
        self,
        industry: str,
        is_niche: bool,
        experience_level: ExperienceLevel,
    ) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt) for job description generation."""
        system_prompt = (
            "You are an expert HR professional and job description writer. "
            f"Generate a realistic job description for the {industry} industry. "
            f"The role should require {experience_level.value}-level experience.\n\n"
            "Requirements:\n"
            "- Include specific, measurable required skills (not generic buzzwords)\n"
            "- Required skills list should have 5-10 concrete technical or domain skills\n"
            "- Preferred skills should complement but not duplicate required skills\n"
            "- Experience years should match the seniority level realistically"
        )

        niche_instruction = ""
        if is_niche:
            niche_instruction = (
                "\n\nIMPORTANT: This should be an unusual or emerging role that combines "
                "skills from multiple traditional disciplines. The title should NOT be "
                "a standard job title found on most job boards."
            )

        user_prompt = (
            f"Generate a {'niche/emerging' if is_niche else 'standard'} "
            f"{experience_level.value}-level job description in the {industry} industry."
            f"{niche_instruction}"
        )

        return system_prompt, user_prompt

    def get_resume_prompt(
        self,
        job: JobDescription,
        fit_level: FitLevel,
        writing_style: WritingStyle,
    ) -> tuple[str, str]:
        """
        Returns (system_prompt, user_prompt) for resume generation.
        Template varies by writing_style; fit_level controls quality matching.
        """
        job_json = job.model_dump_json(indent=2)
        fit_instructions = FIT_LEVEL_INSTRUCTIONS[fit_level]
        style_instructions = STYLE_INSTRUCTIONS[writing_style]

        system_prompt = (
            "You are a professional resume writer creating realistic resumes "
            "for job applicants of varying qualification levels.\n\n"
            f"Writing Style: {writing_style.value}\n"
            f"Style Instructions: {style_instructions}\n\n"
            "IMPORTANT:\n"
            "- Skills should be specific technologies/tools, not generic buzzwords\n"
            "- Experience dates must be in ISO format (YYYY-MM-DD or YYYY-MM)\n"
            "- Each experience entry needs concrete responsibilities\n"
            "- Email must be a valid format (name@domain.tld)\n"
            "- Phone must have at least 10 digits"
        )

        user_prompt = (
            f"Generate a resume for someone applying to this job:\n\n"
            f"{job_json}\n\n"
            f"Fit Level: {fit_level.value}\n"
            f"{fit_instructions}"
        )

        return system_prompt, user_prompt

    def get_template_id(self, writing_style: WritingStyle) -> str:
        """Returns template ID string for tracking (e.g., 'v1-formal')."""
        return f"{TEMPLATE_VERSION}-{writing_style.value}"
