from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ProficiencyLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"


class ExperienceLevel(str, Enum):
    ENTRY = "Entry"
    MID = "Mid"
    SENIOR = "Senior"
    LEAD = "Lead"
    EXECUTIVE = "Executive"


class FitLevel(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    PARTIAL = "partial"
    POOR = "poor"
    MISMATCH = "mismatch"


class WritingStyle(str, Enum):
    FORMAL = "formal"
    CASUAL = "casual"
    TECHNICAL = "technical"
    ACHIEVEMENT = "achievement"
    CAREER_CHANGER = "career_changer"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str
    location: str
    linkedin: str | None = None
    portfolio: str | None = None

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        # WHY: LLMs generate plausible but invalid emails like "john@company" without TLD
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError(f"Invalid email format: {v}")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        # WHY: International formats vary but all have at least 10 digits
        digits = re.sub(r"\D", "", v)
        if len(digits) < 10:
            raise ValueError(f"Phone must have >=10 digits, got {len(digits)}")
        return v


class Education(BaseModel):
    degree: str
    institution: str
    graduation_date: str
    gpa: float | None = None
    coursework: list[str] | None = None

    @field_validator("graduation_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        # WHY: LLMs generate dates in dozens of formats. ISO is the spec requirement.
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            try:
                datetime.strptime(v, "%Y-%m")
            except ValueError:
                raise ValueError(
                    f"Date must be ISO format (YYYY-MM-DD or YYYY-MM), got: {v}"
                )
        return v

    @field_validator("gpa")
    @classmethod
    def validate_gpa(cls, v: float | None) -> float | None:
        # WHY: LLMs sometimes generate percentage-based GPAs (85.0) or negatives
        if v is not None and (v < 0.0 or v > 4.0):
            raise ValueError(f"GPA must be 0.0-4.0, got {v}")
        return v


class Experience(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: str | None = None
    responsibilities: list[str]
    achievements: list[str] | None = None

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            try:
                datetime.strptime(v, "%Y-%m")
            except ValueError:
                raise ValueError(f"Date must be ISO format, got: {v}")
        return v

    @field_validator("responsibilities")
    @classmethod
    def validate_responsibilities(cls, v: list[str]) -> list[str]:
        # WHY: Empty responsibilities = LLM generated a shell entry with no substance
        if len(v) < 1:
            raise ValueError("Experience must have at least 1 responsibility")
        return v

    @model_validator(mode="after")
    def validate_date_order(self) -> Experience:
        # WHY: LLMs sometimes generate impossible timelines (ended before starting)
        if self.end_date is not None:
            start = self.start_date[:7]
            end = self.end_date[:7]
            if end < start:
                raise ValueError(
                    f"end_date ({self.end_date}) must be after start_date ({self.start_date})"
                )
        return self


class Skill(BaseModel):
    name: str = Field(
        description=(
            "Individual skill or technology name as a short token (e.g., 'Python', 'React', 'PostgreSQL'). "
            "Must be 1-3 words. NOT a description. NOT 'Proficient in Python'. "
            "NOT 'Cloud Computing (AWS, Azure)' — instead list 'AWS' and 'Azure' as separate skills."
        )
    )
    proficiency_level: ProficiencyLevel
    years: int | None = None

    @field_validator("years")
    @classmethod
    def validate_years(cls, v: int | None) -> int | None:
        # WHY: 40 years of React experience is a hallucination (React released 2013)
        if v is not None and (v < 0 or v > 30):
            raise ValueError(f"Skill years must be 0-30, got {v}")
        return v


# ---------------------------------------------------------------------------
# Top-level domain models
# ---------------------------------------------------------------------------


class Resume(BaseModel):
    contact_info: ContactInfo
    education: list[Education]
    experience: list[Experience]
    skills: list[Skill]
    summary: str | None = None

    @field_validator("education")
    @classmethod
    def validate_education(cls, v: list[Education]) -> list[Education]:
        if len(v) < 1:
            raise ValueError("Resume must have at least 1 education entry")
        return v

    @field_validator("experience")
    @classmethod
    def validate_experience(cls, v: list[Experience]) -> list[Experience]:
        if len(v) < 1:
            raise ValueError("Resume must have at least 1 experience entry")
        return v

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, v: list[Skill]) -> list[Skill]:
        if len(v) < 1:
            raise ValueError("Resume must have at least 1 skill")
        return v


class CompanyInfo(BaseModel):
    name: str
    industry: str
    size: str = Field(
        description=(
            "Company size. Use EXACTLY one of these values: "
            "'Startup (1-50)', 'Small (51-200)', 'Mid-size (201-500)', 'Enterprise (500+)'"
        )
    )
    location: str


class JobRequirements(BaseModel):
    required_skills: list[str] = Field(
        description=(
            "List of individual skill/technology names as short tokens. "
            "Each skill must be 1-3 words max — a specific tool, language, or technology name. "
            "GOOD: ['Python', 'AWS', 'Docker', 'PostgreSQL', 'React', 'Git'] "
            "BAD: ['Experience with Python programming', 'Knowledge of cloud platforms (e.g., AWS)']"
        )
    )
    preferred_skills: list[str] | None = Field(
        None,
        description=(
            "List of individual skill/technology names as short tokens. "
            "Each skill must be 1-3 words max — a specific tool, language, or technology name. "
            "GOOD: ['TypeScript', 'Docker', 'CI/CD', 'GraphQL'] "
            "BAD: ['Experience with containerization tools', 'Knowledge of CI/CD pipelines (e.g., Jenkins)']"
        ),
    )
    education: str
    experience_years: int
    experience_level: ExperienceLevel

    @field_validator("required_skills")
    @classmethod
    def validate_required_skills(cls, v: list[str]) -> list[str]:
        if len(v) < 1:
            raise ValueError("Job must have at least 1 required skill")
        return v

    @field_validator("experience_years")
    @classmethod
    def validate_experience_years(cls, v: int) -> int:
        if v < 0 or v > 30:
            raise ValueError(f"Experience years must be 0-30, got {v}")
        return v


class JobDescription(BaseModel):
    title: str
    company: CompanyInfo
    description: str
    requirements: JobRequirements


# ---------------------------------------------------------------------------
# Pipeline metadata models
# ---------------------------------------------------------------------------


class GeneratedResume(BaseModel):
    trace_id: str
    resume: Resume
    fit_level: FitLevel
    writing_style: WritingStyle
    template_version: str
    generated_at: str
    prompt_template: str
    model_used: str


class GeneratedJob(BaseModel):
    trace_id: str
    job: JobDescription
    is_niche_role: bool
    generated_at: str
    prompt_template: str
    model_used: str


class ResumeJobPair(BaseModel):
    pair_id: str
    resume_trace_id: str
    job_trace_id: str
    fit_level: FitLevel
    created_at: str


# ---------------------------------------------------------------------------
# Failure / evaluation models
# ---------------------------------------------------------------------------


class FailureLabels(BaseModel):
    pair_id: str
    skills_overlap: float  # 0.0-1.0 Jaccard
    skills_overlap_raw: int
    skills_union_raw: int
    experience_mismatch: bool
    seniority_mismatch: bool
    missing_core_skills: bool
    has_hallucinations: bool
    has_awkward_language: bool
    experience_years_resume: float
    experience_years_required: int
    seniority_level_resume: int
    seniority_level_job: int
    missing_skills: list[str]
    hallucination_reasons: list[str]
    awkward_language_reasons: list[str]
    resume_skills_normalized: list[str]
    job_skills_normalized: list[str]


class JudgeResult(BaseModel):
    pair_id: str
    has_hallucinations: bool
    hallucination_details: str
    has_awkward_language: bool
    awkward_language_details: str
    overall_quality_score: float
    fit_assessment: str
    recommendations: list[str]
    red_flags: list[str]

    @field_validator("overall_quality_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        # WHY: Judge must use 0.0-1.0 scale for consistent comparison
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Quality score must be 0.0-1.0, got {v}")
        return v


class CorrectionResult(BaseModel):
    pair_id: str
    attempt_number: int
    original_errors: list[str]
    corrected_successfully: bool
    remaining_errors: list[str] | None = None


class CorrectionSummary(BaseModel):
    total_invalid: int
    total_corrected: int
    correction_rate: float
    avg_attempts_per_success: float
    common_failure_reasons: dict[str, int]


class FeedbackEntry(BaseModel):
    feedback_id: str
    pair_id: str
    rating: str
    comment: str | None = None
    timestamp: str


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    resume: Resume
    job_description: JobDescription


class ReviewResponse(BaseModel):
    pair_id: str
    failure_labels: FailureLabels
    judge_result: JudgeResult | None
    processing_time_seconds: float


class FailureRateResponse(BaseModel):
    total_pairs: int
    validation_success_rate: float
    failure_mode_rates: dict[str, float]
    correction_success_rate: float
    avg_jaccard_by_fit_level: dict[str, float]
    last_run_timestamp: str


class TemplateStats(BaseModel):
    template_id: str
    total_generated: int
    validation_success_rate: float
    failure_mode_rates: dict[str, float]
    avg_jaccard: float
    avg_judge_quality_score: float | None = None


class TemplateComparisonResponse(BaseModel):
    template_results: dict[str, TemplateStats]
    chi_squared_statistic: float
    chi_squared_p_value: float
    significant: bool
    best_template: str
    worst_template: str
    recommendation: str


class MultiHopQuestion(BaseModel):
    question: str
    requires_sections: list[str]
    answer: str
    assessment: str


class MultiHopRequest(BaseModel):
    resume: Resume
    job_description: JobDescription


class MultiHopResponse(BaseModel):
    pair_id: str
    questions: list[MultiHopQuestion]
    processing_time_seconds: float


class SimilarCandidate(BaseModel):
    resume_trace_id: str
    similarity_score: float
    name: str
    skills: list[str]
    experience_years: float
    fit_level: str


class SimilarCandidatesResponse(BaseModel):
    query: str
    results: list[SimilarCandidate]
    total_in_index: int
    filter_applied: str | None
    processing_time_seconds: float


class FeedbackRequest(BaseModel):
    pair_id: str
    rating: str
    comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    logged_to: list[str]
    timestamp: str


class JobSummary(BaseModel):
    trace_id: str
    title: str
    company_name: str
    industry: str
    experience_level: str
    is_niche: bool
    required_skills_count: int


class JobListResponse(BaseModel):
    jobs: list[JobSummary]
    total: int
    page: int
    page_size: int
    total_pages: int


class PairDetailResponse(BaseModel):
    pair_id: str
    resume: GeneratedResume
    job: GeneratedJob
    failure_labels: FailureLabels | None
    judge_result: JudgeResult | None
    correction_history: list[CorrectionResult]
    feedback: list[FeedbackRequest]
