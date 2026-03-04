# P4 Day 1 Implementation Plan: T1.1–T1.9

## Context

**Why this change:** P4 (Resume Coach) is an integration sprint combining P1's generation/validation/correction patterns, P2's evaluation rigor, with new capabilities (FastAPI, Jaccard similarity, skill normalization). Day 1 establishes the foundation: schemas, generation pipeline, and 250+ validated resume-job pairs.

**Current state:** `04-resume-coach/` has only CLAUDE.md, PRD.md, and p4-concepts-primer.html. All code, tests, data directories, and config need to be created from scratch.

**Target state:** 250+ generated pairs (50 jobs x 5 resumes each), validated and saved to JSONL, with sanity-checked quality.

---

## Dependency Graph

```
T1.1 (Project Setup)
  │
  ├──→ T1.2 (schemas.py)
  │      │
  │      ├──→ T1.3 (test_schemas.py)
  │      │
  │      ├──→ T1.4 (normalizer.py + test)  [parallel with T1.3]
  │      │
  │      ├──→ T1.5 (templates.py)
  │      │      │
  │      │      └──→ T1.6 (generator.py)
  │      │                │
  │      └──→ T1.7 (validator.py)
  │                │      │
  │                └──────┴──→ T1.8 (run_generation.py)
  │                                  │
  │                                  └──→ T1.9 (sanity_check.py)
```

**Parallelizable:** T1.3 + T1.4 can run simultaneously after T1.2. T1.5 + T1.7 can run simultaneously after T1.2.

---

## T1.1: Project Setup

**Est. time:** 20 min
**Files to create:** (all paths relative to `04-resume-coach/`)

```
pyproject.toml
.env
.gitignore
src/__init__.py
tests/__init__.py
data/cache/.gitkeep
data/generated/.gitkeep
data/validated/.gitkeep
data/labels/.gitkeep
data/corrected/.gitkeep
data/feedback/.gitkeep
data/analysis/.gitkeep
data/chromadb/.gitkeep
results/charts/.gitkeep
docs/adr/.gitkeep
```

### pyproject.toml

```toml
[project]
name = "resume-coach"
version = "0.1.0"
description = "AI-Powered Resume Coach: synthetic data pipeline with generation, validation, analysis, correction, and REST API"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "instructor",
    "openai",
    "pydantic>=2.0",
    "pandas",
    "matplotlib",
    "seaborn",
    "fastapi",
    "uvicorn",
    # "chromadb",                # Day 3: vector store
    # "sentence-transformers",   # Day 3: local embeddings
    "rich",
    "python-dotenv",
    "scipy",
]

[dependency-groups]
dev = [
    "pytest",
    "httpx",
    "ruff",
]

[tool.pytest.ini_options]
pythonpath = ["."]

[tool.ruff]
line-length = 100
```

### .env

```
OPENAI_API_KEY=sk-your-key-here
```

### .gitignore

```
data/cache/
data/chromadb/
.env
__pycache__/
*.pyc
.venv/
```

### Commands

```bash
cd 04-resume-coach
uv sync
```

### Validation

```bash
uv run python -c "import instructor; import fastapi; print('OK')"
# Expected: OK
```

---

## T1.2: schemas.py

**Est. time:** 60 min
**File:** `src/schemas.py`

### Imports

```python
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator
```

### Models — Exact Signatures (implement in this order)

#### Enums (4)

```python
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
```

#### Sub-models (4)

```python
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
                raise ValueError(f"Date must be ISO format (YYYY-MM-DD or YYYY-MM), got: {v}")
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
    name: str
    proficiency_level: ProficiencyLevel
    years: int | None = None

    @field_validator("years")
    @classmethod
    def validate_years(cls, v: int | None) -> int | None:
        # WHY: 40 years of React experience is a hallucination (React released 2013)
        if v is not None and (v < 0 or v > 30):
            raise ValueError(f"Skill years must be 0-30, got {v}")
        return v
```

#### Top-level domain models (2)

```python
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
    size: str
    location: str

class JobRequirements(BaseModel):
    required_skills: list[str]
    preferred_skills: list[str] | None = None
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
```

#### Pipeline metadata models (3)

```python
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
```

#### Failure/evaluation models (4)

```python
class FailureLabels(BaseModel):
    pair_id: str
    skills_overlap: float          # 0.0-1.0 Jaccard
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
```

#### API request/response models (14)

```python
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
```

### Validation

```bash
uv run python -c "from src.schemas import Resume, JobDescription, FailureLabels, FitLevel, WritingStyle; print('All models importable')"
# Expected: All models importable
```

---

## T1.3: test_schemas.py

**Est. time:** 30 min
**File:** `tests/test_schemas.py`

### Imports

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.schemas import (
    ContactInfo, Education, Experience, Skill, Resume,
    CompanyInfo, JobRequirements, JobDescription,
    ProficiencyLevel, ExperienceLevel, FitLevel, WritingStyle,
    GeneratedResume, GeneratedJob, ResumeJobPair,
    FailureLabels, JudgeResult, CorrectionResult,
    ReviewRequest, FeedbackRequest,
)
```

### Test Structure

**Helper fixtures:** Create reusable valid data fixtures at top of file.

```python
@pytest.fixture
def valid_contact() -> dict:
    return {"name": "Jane Doe", "email": "jane@example.com", "phone": "+1-555-123-4567", "location": "Austin, TX"}

@pytest.fixture
def valid_education() -> dict:
    return {"degree": "BS Computer Science", "institution": "MIT", "graduation_date": "2020-05", "gpa": 3.8}

@pytest.fixture
def valid_experience() -> dict:
    return {
        "company": "Acme Corp", "title": "Software Engineer",
        "start_date": "2020-06", "end_date": "2023-01",
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
```

**Tests to implement (30+ test functions):**

| Test Name | What it validates |
|-----------|-------------------|
| `test_contact_info_valid_construction` | Happy path |
| `test_contact_info_bad_email_raises` | `"invalid"` → ValidationError |
| `test_contact_info_short_phone_raises` | `"123"` → ValidationError |
| `test_education_valid_construction` | Happy path with YYYY-MM |
| `test_education_bad_date_raises` | `"May 2020"` → ValidationError |
| `test_education_gpa_above_4_raises` | `gpa=4.5` → ValidationError |
| `test_education_gpa_negative_raises` | `gpa=-1.0` → ValidationError |
| `test_education_gpa_none_is_valid` | `gpa=None` is OK |
| `test_experience_valid_construction` | Happy path |
| `test_experience_empty_responsibilities_raises` | `[]` → ValidationError |
| `test_experience_bad_date_format_raises` | `"06/2020"` → ValidationError |
| `test_experience_end_before_start_raises` | `start=2023, end=2020` → ValidationError |
| `test_experience_current_job_no_end_date` | `end_date=None` is OK |
| `test_skill_valid_construction` | Happy path |
| `test_skill_years_negative_raises` | `years=-1` → ValidationError |
| `test_skill_years_above_30_raises` | `years=35` → ValidationError |
| `test_resume_valid_construction` | Full nested resume |
| `test_resume_empty_education_raises` | `education=[]` → ValidationError |
| `test_resume_empty_experience_raises` | `experience=[]` → ValidationError |
| `test_resume_empty_skills_raises` | `skills=[]` → ValidationError |
| `test_job_requirements_empty_skills_raises` | `required_skills=[]` → ValidationError |
| `test_job_requirements_years_negative_raises` | `experience_years=-1` → ValidationError |
| `test_proficiency_level_values` | All 4 enum values valid |
| `test_experience_level_values` | All 5 enum values valid |
| `test_fit_level_values` | All 5 enum values valid |
| `test_writing_style_values` | All 5 enum values valid |
| `test_judge_result_score_above_1_raises` | `score=1.5` → ValidationError |
| `test_judge_result_score_below_0_raises` | `score=-0.1` → ValidationError |
| `test_review_request_valid` | Nested Resume + JobDescription |
| `test_feedback_request_valid` | `pair_id` + `rating` |

**Use `@pytest.mark.parametrize` for:**
- Email validation: valid emails + 3 invalid patterns
- Date format validation: YYYY-MM-DD, YYYY-MM, invalid formats
- GPA range: boundary values (0.0, 4.0, 4.1, -0.1)
- Skill years: boundary values (0, 30, 31, -1)

### Validation

```bash
cd 04-resume-coach && uv run pytest tests/test_schemas.py -v
# Expected: all tests pass
```

---

## T1.4: normalizer.py + test_normalizer.py

**Est. time:** 25 min
**Files:** `src/normalizer.py`, `tests/test_normalizer.py`

### src/normalizer.py

```python
from __future__ import annotations

import re
```

#### SKILL_ALIASES constant

```python
SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "react.js": "react",
    "reactjs": "react",
    "node.js": "node",
    "nodejs": "node",
    "vue.js": "vue",
    "vuejs": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "c++": "cpp",
    "c#": "csharp",
    "dot net": "dotnet",
    ".net": "dotnet",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "ci/cd": "cicd",
    "ci cd": "cicd",
}
```

#### SUFFIXES_TO_STRIP constant

```python
SUFFIXES_TO_STRIP: list[str] = [
    "developer", "engineer", "programming", "development",
    "framework", "language", "library",
]
```

#### SkillNormalizer class

```python
class SkillNormalizer:
    """Normalizes skill names for accurate Jaccard similarity calculation."""

    def __init__(self, aliases: dict[str, str] | None = None) -> None:
        # WHY: Allow custom aliases for domain-specific normalization
        self._aliases = aliases or SKILL_ALIASES

    def normalize(self, skill: str) -> str:
        """
        Normalize a single skill string.
        Pipeline: lowercase -> strip -> remove versions -> strip suffixes -> alias map
        """
        s = skill.lower().strip()
        if not s:
            return s

        # Step 1: Remove version numbers (e.g., "python 3.10" -> "python")
        s = re.sub(r"\s*\d+(\.\d+)*\s*$", "", s).strip()

        # Step 2: Strip suffixes (e.g., "python developer" -> "python")
        for suffix in SUFFIXES_TO_STRIP:
            if s.endswith(f" {suffix}"):
                s = s[: -(len(suffix) + 1)].strip()

        # Step 3: Alias mapping (e.g., "js" -> "javascript")
        s = self._aliases.get(s, s)

        return s

    def normalize_set(self, skills: list[str]) -> set[str]:
        """Normalize a list of skills, returning deduplicated set."""
        return {self.normalize(s) for s in skills if s.strip()}
```

### tests/test_normalizer.py

```python
from __future__ import annotations

import pytest
from src.normalizer import SkillNormalizer
```

**Tests to implement:**

```python
class TestSkillNormalizer:
    @pytest.fixture
    def normalizer(self) -> SkillNormalizer:
        return SkillNormalizer()

    @pytest.mark.parametrize("input_skill,expected", [
        ("Python 3.10", "python"),
        ("React 18.2", "react"),
        ("Node.js 20", "node"),
        ("Java 17", "java"),
    ])
    def test_version_removal(self, normalizer, input_skill, expected):
        assert normalizer.normalize(input_skill) == expected

    @pytest.mark.parametrize("input_skill,expected", [
        ("python developer", "python"),
        ("java engineer", "java"),
        ("react framework", "react"),
        ("go programming", "go"),
    ])
    def test_suffix_stripping(self, normalizer, input_skill, expected):
        assert normalizer.normalize(input_skill) == expected

    @pytest.mark.parametrize("input_skill,expected", [
        ("js", "javascript"),
        ("ts", "typescript"),
        ("k8s", "kubernetes"),
        ("py", "python"),
        ("c++", "cpp"),
        ("c#", "csharp"),
        (".net", "dotnet"),
        ("ci/cd", "cicd"),
    ])
    def test_alias_mapping(self, normalizer, input_skill, expected):
        assert normalizer.normalize(input_skill) == expected

    def test_empty_string(self, normalizer):
        assert normalizer.normalize("") == ""

    def test_already_normalized(self, normalizer):
        assert normalizer.normalize("python") == "python"

    def test_unknown_skill_passes_through(self, normalizer):
        assert normalizer.normalize("fortran") == "fortran"

    def test_normalize_set_deduplication(self, normalizer):
        result = normalizer.normalize_set(["Python 3.10", "python", "py"])
        assert result == {"python"}

    def test_normalize_set_empty_strings_filtered(self, normalizer):
        result = normalizer.normalize_set(["python", "", "  "])
        assert result == {"python"}

    def test_combined_version_and_alias(self, normalizer):
        # "React.js 18" -> lowercase "react.js 18" -> version removal "react.js" -> alias "react"
        assert normalizer.normalize("React.js 18") == "react"
```

### Validation

```bash
cd 04-resume-coach && uv run pytest tests/test_normalizer.py -v
# Expected: all tests pass
```

---

## T1.5: templates.py

**Est. time:** 40 min
**File:** `src/templates.py`

### Imports

```python
from __future__ import annotations

from src.schemas import ExperienceLevel, FitLevel, JobDescription, WritingStyle
```

### Key Constants

```python
INDUSTRIES: list[str] = [
    "Technology", "Healthcare", "Finance", "Education", "Manufacturing",
    "Retail", "Energy", "Legal", "Marketing", "Government",
]

TEMPLATE_VERSION = "v1"
```

### FIT_LEVEL_INSTRUCTIONS constant

```python
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
```

### STYLE_INSTRUCTIONS constant

```python
STYLE_INSTRUCTIONS: dict[WritingStyle, str] = {
    WritingStyle.FORMAL: "Professional, corporate language. Standard chronological format. No first person.",
    WritingStyle.CASUAL: "Conversational, startup-friendly tone. May use first person. Project-focused.",
    WritingStyle.TECHNICAL: "Heavy on technical details, tool versions, and architecture specifics.",
    WritingStyle.ACHIEVEMENT: "Every bullet must have a quantified result (e.g., 'increased X by Y%').",
    WritingStyle.CAREER_CHANGER: "Emphasize transferable skills from a different industry. Include narrative transitions.",
}
```

### PromptTemplateLibrary class

```python
class PromptTemplateLibrary:
    """Factory for job and resume generation prompts across 5 writing styles."""

    def get_job_prompt(
        self,
        industry: str,
        is_niche: bool,
        experience_level: ExperienceLevel,
    ) -> tuple[str, str]:
        """
        Returns (system_prompt, user_prompt) for job description generation.
        """
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
```

### Key Implementation Notes

- The `get_resume_prompt` method embeds the full job JSON in the user prompt so the LLM has context
- Each WritingStyle produces measurably different output (formal vs casual vs technical, etc.)
- Template IDs follow `v1-{style}` pattern for A/B tracking
- FIT_LEVEL_INSTRUCTIONS are the critical piece that controls quality variance

### Validation

```bash
uv run python -c "
from src.templates import PromptTemplateLibrary, INDUSTRIES
from src.schemas import FitLevel, WritingStyle, ExperienceLevel, JobDescription, CompanyInfo, JobRequirements

lib = PromptTemplateLibrary()

# Test job prompt
sys_p, user_p = lib.get_job_prompt('Technology', False, ExperienceLevel.SENIOR)
assert len(sys_p) > 50 and len(user_p) > 20, 'Job prompts too short'

# Test resume prompt with dummy job
job = JobDescription(
    title='Software Engineer',
    company=CompanyInfo(name='Acme', industry='Technology', size='Mid-size', location='Austin'),
    description='Build software',
    requirements=JobRequirements(
        required_skills=['Python', 'AWS'], education='BS CS',
        experience_years=5, experience_level=ExperienceLevel.SENIOR
    )
)
for style in WritingStyle:
    sys_p, user_p = lib.get_resume_prompt(job, FitLevel.EXCELLENT, style)
    assert len(sys_p) > 50, f'{style} system prompt too short'
    assert style.value in sys_p.lower(), f'{style} not mentioned in prompt'

print('All template tests passed')
"
```

---

## T1.6: generator.py

**Est. time:** 60 min
**File:** `src/generator.py`

### Imports

```python
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from rich.progress import Progress

from src.schemas import (
    ExperienceLevel, FitLevel, GeneratedJob, GeneratedResume,
    JobDescription, Resume, ResumeJobPair, WritingStyle,
)
from src.templates import INDUSTRIES, PromptTemplateLibrary, TEMPLATE_VERSION
```

### Module-level constants

```python
logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache"
_GENERATED_DIR = _PROJECT_ROOT / "data" / "generated"

_MODEL = "gpt-4o-mini"
_TEMPERATURE = 0.8
_MAX_RETRIES = 5
```

### Functions — Signatures and Key Logic

```python
def _create_client() -> instructor.Instructor:
    """Create Instructor-wrapped OpenAI client."""
    import os
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
    # WHY: Fail fast with clear message instead of cryptic OpenAI auth error
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Check .env file in 04-resume-coach/")
    return instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON)


def _prompt_hash(system_prompt: str, user_prompt: str, model_name: str) -> str:
    """MD5 hash of prompt combination for cache key."""
    combined = f"{model_name}\n{system_prompt}\n---\n{user_prompt}"
    return hashlib.md5(combined.encode()).hexdigest()


def _load_cache(cache_key: str, response_model: type[BaseModel]) -> BaseModel | None:
    """Load cached response; return None on miss or corruption."""
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        return response_model.model_validate(data["response"])
    except (json.JSONDecodeError, KeyError, ValidationError) as exc:
        logger.warning("Cache corruption for %s: %s", cache_key[:8], exc)
        return None


def _save_cache(
    cache_key: str,
    prompt_key: str,
    result: BaseModel,
) -> None:
    """Save validated result to cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": cache_key,
        "prompt_key": prompt_key,
        "model": _MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response": result.model_dump(),
    }
    (_CACHE_DIR / f"{cache_key}.json").write_text(json.dumps(payload, indent=2))


def generate_with_cache(
    client: instructor.Instructor,
    prompt_key: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
) -> tuple[BaseModel, bool]:
    """
    Generate structured output with cache-first pattern.
    Returns: (result, from_cache)
    """
    cache_key = _prompt_hash(system_prompt, user_prompt, _MODEL)
    cached = _load_cache(cache_key, response_model)
    if cached is not None:
        return cached, True

    result = client.chat.completions.create(
        model=_MODEL,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=_TEMPERATURE,
        max_retries=_MAX_RETRIES,
    )

    _save_cache(cache_key, prompt_key, result)
    return result, False


def generate_job(
    client: instructor.Instructor,
    industry: str,
    is_niche: bool,
    experience_level: ExperienceLevel,
    templates: PromptTemplateLibrary,
) -> tuple[GeneratedJob, bool]:
    """Generate a single job description with metadata wrapper. Returns (job, from_cache)."""
    system_prompt, user_prompt = templates.get_job_prompt(
        industry, is_niche, experience_level
    )
    prompt_key = f"job_{industry}_{is_niche}_{experience_level.value}"

    job, from_cache = generate_with_cache(
        client, prompt_key, system_prompt, user_prompt, JobDescription
    )

    return GeneratedJob(
        trace_id=str(uuid.uuid4()),
        job=job,
        is_niche_role=is_niche,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="job-standard",
        model_used=_MODEL,
    ), from_cache


def generate_resume(
    client: instructor.Instructor,
    job: GeneratedJob,
    fit_level: FitLevel,
    writing_style: WritingStyle,
    templates: PromptTemplateLibrary,
) -> tuple[GeneratedResume, bool]:
    """Generate a single resume matched to a job at specified fit level. Returns (resume, from_cache)."""
    system_prompt, user_prompt = templates.get_resume_prompt(
        job.job, fit_level, writing_style
    )
    template_id = templates.get_template_id(writing_style)
    prompt_key = f"resume_{job.trace_id}_{fit_level.value}_{writing_style.value}"

    resume, from_cache = generate_with_cache(
        client, prompt_key, system_prompt, user_prompt, Resume
    )

    return GeneratedResume(
        trace_id=str(uuid.uuid4()),
        resume=resume,
        fit_level=fit_level,
        writing_style=writing_style,
        template_version=TEMPLATE_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template=template_id,
        model_used=_MODEL,
    ), from_cache


def _append_jsonl(record: BaseModel, filepath: Path) -> None:
    """Append a single record to JSONL file (crash-resilient writes)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a") as f:
        f.write(json.dumps(record.model_dump()) + "\n")


def generate_all_jobs(
    client: instructor.Instructor,
    count_per_industry: int = 5,
    max_industries: int | None = None,
) -> tuple[list[GeneratedJob], int]:
    """
    Generate jobs across industries.

    Strategy: count_per_industry per industry (default 5 × 10 = 50 total).
    ~20% are niche roles (1 per industry).
    Experience levels rotate: Entry, Mid, Senior, Lead, Executive.
    max_industries: limit number of industries (for --dry-run).
    """
    templates = PromptTemplateLibrary()
    industries = INDUSTRIES[:max_industries] if max_industries else INDUSTRIES

    experience_levels = list(ExperienceLevel)
    jobs: list[GeneratedJob] = []
    total = len(industries) * count_per_industry
    failed = 0
    api_calls = 0  # WHY: Track non-cached calls for smart rate limiting

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = _GENERATED_DIR / f"jobs_{timestamp}.jsonl"

    with Progress() as progress:
        task = progress.add_task("Generating jobs...", total=total)

        for industry in industries:
            for i in range(count_per_industry):
                is_niche = (i == 0)  # First job per industry is niche
                exp_level = experience_levels[i % len(experience_levels)]

                try:
                    job, from_cache = generate_job(client, industry, is_niche, exp_level, templates)
                    jobs.append(job)
                    _append_jsonl(job, output_file)
                    if not from_cache:
                        api_calls += 1
                except (ValidationError, Exception) as exc:
                    failed += 1
                    api_calls += 1  # Assume API was called on failure
                    logger.error("Failed job %s/%d: %s", industry, i, exc)

                progress.advance(task)

                # WHY: Only rate-limit when actually hitting the API, not on cache hits
                if api_calls > 0 and api_calls % 10 == 0:
                    time.sleep(2)

    logger.info("Jobs generated: %d succeeded, %d failed, %d API calls", len(jobs), failed, api_calls)
    return jobs, api_calls


def generate_all_resumes(
    client: instructor.Instructor,
    jobs: list[GeneratedJob],
) -> tuple[list[GeneratedResume], list[ResumeJobPair], int]:
    """
    Generate 5 resumes per job (one per fit level).
    Writing styles rotate evenly across all resumes.

    Returns: (resumes, pairs)
    """
    templates = PromptTemplateLibrary()

    writing_styles = list(WritingStyle)
    resumes: list[GeneratedResume] = []
    pairs: list[ResumeJobPair] = []
    total = len(jobs) * len(FitLevel)
    failed = 0
    api_calls = 0  # WHY: Track non-cached calls for smart rate limiting
    style_index = 0  # Rotate through styles

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_file = _GENERATED_DIR / f"resumes_{timestamp}.jsonl"
    pair_file = _GENERATED_DIR / f"pairs_{timestamp}.jsonl"

    with Progress() as progress:
        task = progress.add_task("Generating resumes...", total=total)

        for job in jobs:
            for fit_level in FitLevel:
                style = writing_styles[style_index % len(writing_styles)]
                style_index += 1

                try:
                    resume, from_cache = generate_resume(
                        client, job, fit_level, style, templates
                    )
                    resumes.append(resume)
                    _append_jsonl(resume, resume_file)
                    if not from_cache:
                        api_calls += 1

                    pair = ResumeJobPair(
                        pair_id=str(uuid.uuid4()),
                        resume_trace_id=resume.trace_id,
                        job_trace_id=job.trace_id,
                        fit_level=fit_level,
                        created_at=datetime.now(timezone.utc).isoformat(),
                    )
                    pairs.append(pair)
                    _append_jsonl(pair, pair_file)

                except (ValidationError, Exception) as exc:
                    failed += 1
                    api_calls += 1  # Assume API was called on failure
                    logger.error(
                        "Failed resume for job %s, fit=%s: %s",
                        job.trace_id[:8], fit_level.value, exc,
                    )

                progress.advance(task)

                # WHY: Only rate-limit when actually hitting the API, not on cache hits
                if api_calls > 0 and api_calls % 10 == 0:
                    time.sleep(2)

    logger.info(
        "Resumes generated: %d succeeded, %d failed (%d pairs, %d API calls)",
        len(resumes), failed, len(pairs), api_calls,
    )
    return resumes, pairs, api_calls
```

### Key Implementation Notes

- **Crash resilience:** `_append_jsonl` writes each record immediately. If pipeline crashes at record 200, records 1-199 are saved.
- **Style rotation:** `style_index` increments globally across all jobs, giving even distribution (50 per style for 250 resumes).
- **Niche detection:** First job per industry (`i==0`) is niche. That's 10/50 = 20%.
- **Smart rate limiting:** 2-second pause every 10 **API calls** (not cache hits). Re-runs from cache skip rate limiting entirely.
- **Cache is per-prompt:** Same (industry, niche, level) or (job, fit_level, style) combo always hits cache on re-run.
- **Return values:** Both `generate_all_jobs` and `generate_all_resumes` return `api_calls` count alongside data for cache hit reporting.

### Validation

```bash
# Quick smoke test: generate 1 job + 1 resume (uses real API)
uv run python -c "
from src.generator import _create_client, generate_job, generate_resume
from src.schemas import ExperienceLevel, FitLevel, WritingStyle
from src.templates import PromptTemplateLibrary

client = _create_client()
templates = PromptTemplateLibrary()
job, cached = generate_job(client, 'Technology', False, ExperienceLevel.MID, templates)
print(f'Job: {job.job.title} at {job.job.company.name} (cached={cached})')

resume, cached = generate_resume(client, job, FitLevel.EXCELLENT, WritingStyle.FORMAL, templates)
print(f'Resume: {resume.resume.contact_info.name}, {len(resume.resume.skills)} skills (cached={cached})')
print('Generator smoke test passed')
"
```

---

## T1.7: validator.py

**Est. time:** 25 min
**File:** `src/validator.py`

### Imports

```python
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
```

### Classes

```python
@dataclass
class ValidationTracker:
    """
    Tracks validation outcomes across the pipeline.
    Reuse of P1's ValidationReport pattern with per-model-type granularity.
    """
    _successes: dict[str, list[str]] = field(default_factory=lambda: {})
    _failures: dict[str, list[dict]] = field(default_factory=lambda: {})
    _error_counts: Counter = field(default_factory=Counter)

    def record_success(self, model_type: str, trace_id: str) -> None:
        """Record a successful validation."""
        self._successes.setdefault(model_type, []).append(trace_id)

    def record_failure(
        self,
        model_type: str,
        trace_id: str,
        errors: list[dict],
    ) -> None:
        """
        Record a validation failure with structured error info.
        errors: list of dicts from Pydantic's ValidationError.errors()
        """
        self._failures.setdefault(model_type, []).append({
            "trace_id": trace_id,
            "errors": errors,
        })
        # WHY: Track field-level error frequency for debugging
        for error in errors:
            field_path = ".".join(str(loc) for loc in error.get("loc", []))
            self._error_counts[field_path] += 1

    def get_stats(self) -> dict:
        """Return aggregate validation statistics."""
        total_success = sum(len(v) for v in self._successes.values())
        total_failure = sum(len(v) for v in self._failures.values())
        total = total_success + total_failure

        return {
            "total": total,
            "success_count": total_success,
            "failure_count": total_failure,
            "success_rate": total_success / total if total > 0 else 0.0,
            "by_model_type": {
                model_type: {
                    "successes": len(self._successes.get(model_type, [])),
                    "failures": len(self._failures.get(model_type, [])),
                }
                for model_type in set(
                    list(self._successes.keys()) + list(self._failures.keys())
                )
            },
            "errors_by_field": dict(self._error_counts.most_common()),
        }

    def save_stats(self, output_path: Path) -> None:
        """Save validation stats to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.get_stats(), indent=2))
        logger.info("Saved validation stats to %s", output_path)

    @staticmethod
    def save_valid(record: BaseModel, output_dir: Path) -> None:
        """Append a valid record to the validated JSONL file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "validated.jsonl"
        with open(filepath, "a") as f:
            f.write(json.dumps(record.model_dump()) + "\n")

    @staticmethod
    def save_invalid(
        record_json: str,
        errors: list[dict],
        output_dir: Path,
    ) -> None:
        """Append an invalid record with errors to the invalid JSONL file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / "invalid.jsonl"
        entry = {
            "raw_data": json.loads(record_json) if isinstance(record_json, str) else record_json,
            "errors": errors,
        }
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

### Validation

```bash
uv run python -c "
from src.validator import ValidationTracker

tracker = ValidationTracker()
tracker.record_success('Resume', 'trace-1')
tracker.record_success('Resume', 'trace-2')
tracker.record_success('Resume', 'trace-3')
tracker.record_failure('Resume', 'trace-4', [{'loc': ('education', 0, 'gpa'), 'msg': 'GPA too high', 'type': 'value_error'}])
tracker.record_failure('Resume', 'trace-5', [{'loc': ('contact_info', 'email'), 'msg': 'Invalid email', 'type': 'value_error'}])

stats = tracker.get_stats()
assert stats['total'] == 5
assert stats['success_count'] == 3
assert stats['failure_count'] == 2
assert stats['success_rate'] == 0.6
print('Validator tests passed:', stats)
"
```

---

## T1.8: Run Generation Pipeline

**Est. time:** 90 min (mostly waiting for API calls)
**File:** `src/run_generation.py`

### Imports

```python
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console

from src.generator import generate_all_jobs, generate_all_resumes, _create_client
from src.validator import ValidationTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)
console = Console()
```

### Main Function

```python
_PROJECT_ROOT = Path(__file__).parent.parent

# WHY: 2 industries × 1 job = 2 jobs × 5 resumes = 10 pairs — validates full pipeline fast
_DRY_RUN_INDUSTRIES = 2
_DRY_RUN_JOBS_PER_INDUSTRY = 1


def main() -> None:
    """
    End-to-end generation pipeline:
    1. Generate 50 jobs (5 per industry × 10 industries)
    2. Generate 250 resumes (5 per job: 1 each of excellent/good/partial/poor/mismatch)
    3. Track validation outcomes
    4. Save validation stats

    Use --dry-run for quick validation: 2 jobs × 5 resumes = 10 pairs.
    """
    dry_run = "--dry-run" in sys.argv

    console.print("[bold green]P4 Resume Coach — Generation Pipeline[/bold green]")
    if dry_run:
        console.print("[yellow]DRY RUN: 2 industries × 1 job = 2 jobs × 5 resumes = 10 pairs[/yellow]")
    console.print(f"Started at {datetime.now().isoformat()}")

    client = _create_client()
    tracker = ValidationTracker()

    # Phase 1: Generate jobs
    count_per_industry = _DRY_RUN_JOBS_PER_INDUSTRY if dry_run else 5
    max_industries = _DRY_RUN_INDUSTRIES if dry_run else None
    expected_jobs = (_DRY_RUN_INDUSTRIES if dry_run else 10) * count_per_industry
    console.print(f"\n[bold]Phase 1: Generating {expected_jobs} job descriptions...[/bold]")
    jobs, job_api_calls = generate_all_jobs(client, count_per_industry=count_per_industry, max_industries=max_industries)
    console.print(f"  Jobs generated: {len(jobs)}")

    for job in jobs:
        tracker.record_success("JobDescription", job.trace_id)

    # Phase 2: Generate resumes
    expected_resumes = len(jobs) * 5
    console.print(f"\n[bold]Phase 2: Generating {expected_resumes} resumes ({len(jobs)} jobs × 5 fit levels)...[/bold]")
    resumes, pairs, resume_api_calls = generate_all_resumes(client, jobs)
    console.print(f"  Resumes generated: {len(resumes)}")
    console.print(f"  Pairs created: {len(pairs)}")

    for resume in resumes:
        tracker.record_success("Resume", resume.trace_id)

    # Phase 3: Save validation stats
    stats_path = _PROJECT_ROOT / "data" / "validated" / "validation_stats.json"
    tracker.save_stats(stats_path)

    # Summary
    stats = tracker.get_stats()
    total_api_calls = job_api_calls + resume_api_calls
    total_records = len(jobs) + len(resumes)
    cache_hits = total_records - total_api_calls

    console.print("\n[bold]Pipeline Summary[/bold]")
    console.print(f"  Total records: {stats['total']}")
    console.print(f"  Success rate: {stats['success_rate']:.1%}")
    console.print(f"  Jobs: {len(jobs)}")
    console.print(f"  Resumes: {len(resumes)}")
    console.print(f"  Pairs: {len(pairs)}")
    console.print(f"  Cache hits: {cache_hits}/{total_records} ({cache_hits/total_records:.0%})" if total_records > 0 else "")
    console.print(f"  API calls: {total_api_calls}")
    console.print(f"\nCompleted at {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
```

### Key Implementation Notes

- **Instructor handles retries internally.** If a record fails after 5 retries, Instructor raises `ValidationError`, which `generate_all_resumes` catches and logs. The failed record is simply skipped (counted in `failed`).
- **JSONL is written incrementally** inside `generate_all_jobs` and `generate_all_resumes` via `_append_jsonl`.
- **Smart rate limiting** is inside the generator functions (2s delay every 10 **API calls**, skipped on cache hits).
- **--dry-run mode:** 2 industries × 1 job = 2 jobs × 5 resumes = 10 pairs. Validates full pipeline in ~1 min before committing to 300 API calls.
- **Cache hit reporting:** Summary prints `Cache hits: X/Y (Z%)` so you know if re-runs are hitting cache.
- **Expected runtime:** ~20-30 min for 300 LLM calls (50 jobs + 250 resumes). Second run should be near-instant from cache (100% cache hits).
- **Memory is fine:** 300 Pydantic models in RAM is ~50MB, well within 8GB.

### Run Command

```bash
# Dry run first (validates pipeline, ~1 min)
cd 04-resume-coach && uv run python -m src.run_generation --dry-run

# Full run (300 LLM calls, ~20-30 min)
cd 04-resume-coach && uv run python -m src.run_generation
```

### Validation

After completion, verify:
```bash
# Check file counts
wc -l data/generated/jobs_*.jsonl       # Expected: 50 lines
wc -l data/generated/resumes_*.jsonl    # Expected: ~250 lines
wc -l data/generated/pairs_*.jsonl      # Expected: ~250 lines
cat data/validated/validation_stats.json # Check success rate
```

---

## T1.9: Sanity Check

**Est. time:** 10 min
**File:** `src/sanity_check.py`

### Imports

```python
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.normalizer import SkillNormalizer
from src.schemas import GeneratedJob, GeneratedResume, ResumeJobPair
```

### Main Function

```python
_PROJECT_ROOT = Path(__file__).parent.parent
console = Console()


def _load_latest_jsonl(directory: Path, prefix: str) -> list[dict]:
    """Load the most recent JSONL file matching prefix."""
    files = sorted(directory.glob(f"{prefix}*.jsonl"), reverse=True)
    if not files:
        raise FileNotFoundError(f"No {prefix}*.jsonl files in {directory}")
    records = []
    with open(files[0]) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def main() -> None:
    """Spot-check 5 resume-job pairs for quality."""
    gen_dir = _PROJECT_ROOT / "data" / "generated"
    normalizer = SkillNormalizer()

    # Load data
    jobs_raw = _load_latest_jsonl(gen_dir, "jobs_")
    resumes_raw = _load_latest_jsonl(gen_dir, "resumes_")
    pairs_raw = _load_latest_jsonl(gen_dir, "pairs_")

    # Build lookup maps
    jobs = {j["trace_id"]: GeneratedJob.model_validate(j) for j in jobs_raw}
    resumes = {r["trace_id"]: GeneratedResume.model_validate(r) for r in resumes_raw}

    # Pick 5 pairs: 1 of each fit level
    fit_levels_seen: dict[str, dict] = {}
    for pair_data in pairs_raw:
        fl = pair_data["fit_level"]
        if fl not in fit_levels_seen:
            fit_levels_seen[fl] = pair_data
        if len(fit_levels_seen) >= 5:
            break

    # Display
    table = Table(title="Sanity Check: 5 Sample Pairs")
    table.add_column("Fit Level", style="bold")
    table.add_column("Job Title")
    table.add_column("Candidate Name")
    table.add_column("Resume Skills")
    table.add_column("Job Required Skills")
    table.add_column("Skill Overlap")

    for fit_level, pair_data in fit_levels_seen.items():
        job = jobs.get(pair_data["job_trace_id"])
        resume = resumes.get(pair_data["resume_trace_id"])

        if not job or not resume:
            continue

        resume_skills = normalizer.normalize_set(
            [s.name for s in resume.resume.skills]
        )
        job_skills = normalizer.normalize_set(
            job.job.requirements.required_skills
        )
        overlap = resume_skills & job_skills
        union = resume_skills | job_skills
        jaccard = len(overlap) / len(union) if union else 0.0

        table.add_row(
            fit_level,
            job.job.title,
            resume.resume.contact_info.name,
            str(len(resume_skills)),
            str(len(job_skills)),
            f"{len(overlap)}/{len(union)} ({jaccard:.0%})",
        )

    console.print(table)
    console.print("\n[bold]Expected:[/bold] Excellent > Good > Partial > Poor > Mismatch in overlap")


if __name__ == "__main__":
    main()
```

### Run Command

```bash
cd 04-resume-coach && uv run python -m src.sanity_check
```

### Validation

Visual inspection: the Rich table should show decreasing skill overlap from excellent → mismatch.

---

## Execution Order

```
1. T1.1  Project Setup           [20 min]  — directories, pyproject.toml, uv sync
2. T1.2  schemas.py              [60 min]  — all 30+ Pydantic models
3. T1.3  test_schemas.py         [30 min]  — pytest, commit
   T1.4  normalizer.py + tests   [25 min]  — can overlap with T1.3
4. T1.5  templates.py            [40 min]  — 5 prompt templates
5. T1.7  validator.py            [25 min]  — can overlap with T1.5
6. T1.6  generator.py            [60 min]  — Instructor + caching + batch
7. T1.8  run_generation.py       [90 min]  — run pipeline (API wait time)
8. T1.9  sanity_check.py         [10 min]  — spot-check output
                                 --------
                          Total:  ~6 hours (incl. API wait time)
```

### Git Checkpoints

1. After T1.1: `chore(p4): project setup with pyproject.toml and directory structure`
2. After T1.3+T1.4: `feat(p4): schemas, normalizer with tests`
3. After T1.5+T1.7: `feat(p4): templates and validator`
4. After T1.6 smoke test: `feat(p4): generator with caching and batch processing`
5. After T1.8+T1.9: `feat(p4): generation pipeline — 250+ pairs generated`

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Instructor fails >30% on deeply nested Resume schema | Medium | Delays T1.8 by 30+ min | Strategic Optional fields already in schema. If >30% fail: add `Field(description="...")` to guide LLM, or increase `max_retries` to 7 temporarily. |
| OpenAI rate limiting during 300-call batch | Medium | Adds 10-20 min to T1.8 | 2s delay every 10 records. Cache means re-runs are free. |
| `uv sync` dependency conflicts (chromadb + sentence-transformers) | Mitigated | N/A | Proactively commented out in pyproject.toml. Will uncomment on Day 3. |
| Generated resumes don't correlate with fit levels | Medium | Sanity check fails (T1.9) | Strengthen FIT_LEVEL_INSTRUCTIONS with more explicit skill overlap targets. This is a prompt engineering iteration, not a code fix. |
| Cache key collisions (different prompts → same MD5) | Very Low | Wrong cached results returned | Include model name in hash. MD5 collision probability at 300 records is negligible. |

---

## Verification Checklist (End of Day 1)

- [ ] `uv sync` succeeds with all dependencies
- [ ] `uv run pytest tests/test_schemas.py -v` — all pass
- [ ] `uv run pytest tests/test_normalizer.py -v` — all pass
- [ ] `data/generated/jobs_*.jsonl` has ~50 lines
- [ ] `data/generated/resumes_*.jsonl` has ~250 lines
- [ ] `data/generated/pairs_*.jsonl` has ~250 lines
- [ ] `data/validated/validation_stats.json` shows >90% success rate
- [ ] Sanity check table shows decreasing overlap excellent → mismatch
- [ ] All code committed to feature branch, ready for Day 2
