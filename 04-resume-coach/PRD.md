# PRD: P4 — AI-Powered Resume Coach: Synthetic Data Pipeline

> **This is the implementation contract.** Claude Code: read this + both CLAUDE.md files before starting.
> Do NOT re-debate architecture decisions. They are final. If something is ambiguous, ask the user.

**Project:** P4 — AI-Powered Resume Coach: Synthetic Data Pipeline
**Timeline:** Feb 24–26, 2026 (Mon–Wed, 3 sessions, no hard time cap)
**Owner:** Developer (Java/TS background, completed P1 + P2 + P3)
**Source of Truth:** [Notion Requirements](https://www.notion.so/Mini_Project_4_requirements-2ffdb630640a81ac8eb7d35a8772a8e4)
**Concepts Primer:** `p4-concepts-primer.html` in project root — read for Jaccard similarity, skill normalization, FastAPI, correction loops, and correlation analysis theory
**PRD Version:** v1

---

## 1. Objective

Build a **production-grade synthetic data pipeline** that generates, validates, analyzes, corrects, and serves resume-job description pairs through a REST API. This is P4's core identity: an **integration sprint** that combines P1's generation + validation + correction patterns, P2's evaluation rigor, and introduces FastAPI + Jaccard similarity + skill normalization as new capabilities.

The pipeline:

1. **Generates** 50+ job descriptions across diverse industries with niche role detection
2. **Generates** 5–10 resumes per job with controlled fit levels (excellent → mismatch)
3. **Validates** all data with deeply nested Pydantic schemas (Resume → Experience[] → responsibilities[])
4. **Labels failures** using 6 rule-based metrics (Jaccard overlap, seniority mismatch, hallucination detection, etc.)
5. **Judges** subtle quality issues via LLM-as-Judge (hallucinations, awkward language, holistic fit)
6. **Analyzes** failure patterns via correlation heatmaps and cross-dimensional breakdowns
7. **Corrects** invalid records through iterative LLM feedback (>50% success rate, max 3 retries)
8. **Serves** intelligence via FastAPI REST API with 8 endpoints (<2s without judge, <10s with judge)
9. **Stores** resumes in ChromaDB vector database for semantic "find similar candidates" search
10. **A/B tests** prompt templates to measure which produces highest validation rates (chi-squared significance)
11. **Generates** multi-hop evaluation questions requiring cross-section reasoning
12. **Collects feedback** via thumbs up/down API endpoint for continuous improvement

**The output is a COMPLETE SYSTEM, not a script.** The deliverable is the sentence: *"My pipeline generated 250+ resume-job pairs across 5 fit levels, detected 6 failure modes with Jaccard similarity, corrected 65% of invalid records, serves real-time analysis via 8 FastAPI endpoints with ChromaDB semantic search — and prompt A/B testing proved Template X produces 15% fewer validation failures with p<0.05 significance."* — backed by data.

**Success Criteria:**

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Generation volume | 250+ pairs (50 jobs × 5+ resumes) | Count in output JSONL files |
| Validation success rate | >90% | valid_count / total_count |
| Skill overlap accuracy | Jaccard correctly calculated | Manual spot-check on 10 pairs |
| Mismatch detection | All 6 failure modes calculated | Verify in failure_labels.jsonl |
| Correction success rate | >50% | corrected_count / invalid_count within 3 retries |
| API endpoints | 9 functional endpoints | All return valid JSON, /docs renders correctly |
| API response time | <2s (no judge), <10s (with judge) | Timed endpoint calls |
| Visualization quality | 5+ meaningful charts | Visual inspection, publication-ready |
| LLM-as-Judge | Holistic quality scores for all pairs | Judge output in judge_results.jsonl |
| A/B template testing | Chi-squared significance test | p-value reported, best/worst templates identified |
| Multi-hop evaluation | 10+ cross-section questions | Questions + assessment in output files |
| Vector store | ChromaDB index with all resumes | /search/similar-candidates returns relevant results |
| Feedback mechanism | POST /feedback logs entries | feedback.jsonl accumulates entries |

---

## 2. Architecture Decisions (FINAL — Do Not Re-Debate)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **LLM provider** | OpenAI GPT-4o-mini (generation + correction), GPT-4o (judge) | Proven from P1. Structured output reliability matters more than cost savings from Groq. ~$0.50–$1.00 total budget. |
| **Structured output** | Instructor for ALL generation (resumes, jobs, corrections, judge) | Auto-retry on Pydantic validation failures. `max_retries=5` (up from P1's 3) because nested schemas have more failure points. `mode=instructor.Mode.JSON` forces JSON mode at API level. |
| **Schema depth** | Fully nested — Resume has Education[], Experience[], Skills[] sub-models | Spec requires it. Demonstrates Pydantic mastery with complex real-world schemas. Strategic use of `Optional` fields to separate structural parsing from content quality validation. |
| **Validation strategy** | Two-phase: Instructor handles structural parsing (retry on malformed JSON), labeling pipeline handles semantic quality (Jaccard, hallucination, etc.) | Separation of concerns. Instructor answers "is it valid JSON matching the schema?" Labeling answers "is the content good?" Same principle as Java's `@Valid` for structure vs business rule validation. |
| **Skill normalization** | Custom normalizer: lowercase → version removal → suffix stripping → alias mapping | Without normalization, Jaccard similarity is artificially low ("Python 3.10" ≠ "python"). This is the #1 data quality issue the spec warns about. |
| **Correction loop** | Core feature. Feed Pydantic validation errors + field path + expected format back to LLM. Max 3 retries. Track success rate. | Spec requires >50% correction success rate. P1 proved the pattern (20% → 0% failures). P4 extends it with structured error context for nested schemas. |
| **LLM-as-Judge** | Core feature. GPT-4o evaluates hallucinations, awkward language, holistic fit, red flags. | Catches subtle quality issues rule-based systems miss. The judge output schema is itself a Pydantic model — Instructor handles it. Makes the API's `/review-resume` endpoint dramatically more useful. |
| **FastAPI** | Core feature. 3 endpoints minimum: POST /review-resume, GET /health, GET /analysis/failure-rates | Spec requirement. New technology for the portfolio. Demonstrates the pipeline isn't just a script — it's a service. Auto-generates OpenAPI docs at /docs. |
| **Vector store** | ChromaDB with `PersistentClient` for resume embeddings, semantic search | New tool for portfolio breadth (FAISS used in P2 for benchmarking). ChromaDB provides: (1) built-in persistence (survives API restarts), (2) native metadata filtering (`where={"fit_level": "excellent"}`), (3) simpler API for production integration. Shows you know when to use each: FAISS for benchmarking, ChromaDB for production. |
| **Prompt A/B testing** | Core feature. 5+ templates, track validation rate + failure modes per template, chi-squared test for significance | Quantifies which templates produce better data. Strong eval signal for interviews. Chi-squared proves differences aren't random noise. |
| **Multi-hop evaluation** | Core feature. Generate questions requiring cross-section reasoning (education + experience + skills alignment) | Tests whether the labeling system handles complex assessments. Adds depth to the evaluation story. Directly maps to spec's bonus challenges. |
| **Feedback mechanism** | Core feature. POST /feedback endpoint logs thumbs up/down on analysis results to JSON (Braintrust if time permits) | Spec explicitly mentions "thumbs up/down feedback mechanism." Even without Braintrust, logged feedback demonstrates continuous improvement thinking. |
| **Caching** | JSON file cache keyed on MD5 of (model + prompt). Same pattern as P1/P2. | Cache ALL LLM responses. Re-run pipeline without re-calling APIs. Essential for iterative development. |
| **Observability** | Braintrust for experiment tracking (if time permits), otherwise JSON metrics files | P2 introduced Braintrust. Consistent tracking across projects. Fallback to structured JSON if time is tight. |
| **Demo** | Streamlit Cloud + Loom video | Interactive demo: input job description → see resume generation + analysis + match scores. Loom for walkthrough. |

---

## 3. Data Schemas (Pydantic Specifications)

> Claude Code: implement these as Pydantic `BaseModel` classes in `src/schemas.py`. Use `@field_validator` with `@classmethod` (Pydantic v2). Add "WHY" comments on every validator.

### 3a. Core Domain Models

#### ContactInfo (Nested sub-model)

```python
class ContactInfo(BaseModel):
    name: str                          # Full name
    email: str                         # Must be valid email format
    phone: str                         # ≥10 characters
    location: str                      # City, State or City, Country
    linkedin: str | None = None        # Optional — LinkedIn URL
    portfolio: str | None = None       # Optional — portfolio/GitHub URL

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        # WHY: LLMs frequently generate plausible but invalid emails
        # like "john@company" without TLD. Regex catches structural issues.
        if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", v):
            raise ValueError(f"Invalid email format: {v}")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: str) -> str:
        # WHY: Spec requires ≥10 characters. International formats vary
        # but all have at least 10 digits. Stripping non-digits for counting.
        digits = re.sub(r"\D", "", v)
        if len(digits) < 10:
            raise ValueError(f"Phone must have ≥10 digits, got {len(digits)}")
        return v
```

#### Education (Nested sub-model)

```python
class Education(BaseModel):
    degree: str                        # e.g., "Bachelor of Science in Computer Science"
    institution: str                   # University/college name
    graduation_date: str               # ISO date format (YYYY-MM-DD or YYYY-MM)
    gpa: float | None = None           # Optional — 0.0 to 4.0
    coursework: list[str] | None = None  # Optional — relevant courses

    @field_validator("graduation_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        # WHY: LLMs generate dates in dozens of formats ("May 2020", "2020-05-15",
        # "05/2020"). ISO format is the spec requirement. We validate strictly
        # during schema validation, then the correction loop fixes non-ISO dates.
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
        # WHY: Spec says 0.0-4.0. LLMs sometimes generate percentage-based
        # GPAs (85.0) or negative values. This catches both.
        if v is not None and (v < 0.0 or v > 4.0):
            raise ValueError(f"GPA must be 0.0-4.0, got {v}")
        return v
```

#### Experience (Nested sub-model)

```python
class Experience(BaseModel):
    company: str                       # Company name
    title: str                         # Job title
    start_date: str                    # ISO date format
    end_date: str | None = None        # None = current position
    responsibilities: list[str]        # ≥1 responsibility
    achievements: list[str] | None = None  # Optional — quantified achievements

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
        # WHY: Empty responsibilities list means the LLM generated a shell
        # experience entry with no substance. This is a data quality red flag.
        if len(v) < 1:
            raise ValueError("Experience must have at least 1 responsibility")
        return v

    @model_validator(mode="after")
    def validate_date_order(self) -> "Experience":
        # WHY: Spec requires end_date > start_date. LLMs sometimes
        # generate impossible timelines (ended before starting).
        # This is a logical consistency check, not just format.
        if self.end_date is not None:
            start = self.start_date[:7]  # YYYY-MM for comparison
            end = self.end_date[:7]
            if end < start:
                raise ValueError(
                    f"end_date ({self.end_date}) must be after start_date ({self.start_date})"
                )
        return self
```

#### Skill (Nested sub-model)

```python
class ProficiencyLevel(str, Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"
    EXPERT = "Expert"

class Skill(BaseModel):
    name: str                          # Skill name (raw — normalization happens in labeling)
    proficiency_level: ProficiencyLevel
    years: int | None = None           # Optional — years of experience with this skill

    @field_validator("years")
    @classmethod
    def validate_years(cls, v: int | None) -> int | None:
        # WHY: Spec says 0-30. A claim of 40 years of React experience
        # is a hallucination signal (React was released in 2013).
        if v is not None and (v < 0 or v > 30):
            raise ValueError(f"Skill years must be 0-30, got {v}")
        return v
```

#### Resume (Top-level model)

```python
class Resume(BaseModel):
    contact_info: ContactInfo
    education: list[Education]         # ≥1 education entry
    experience: list[Experience]       # ≥1 experience entry
    skills: list[Skill]                # ≥1 skill
    summary: str | None = None         # Optional — professional summary

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
```

### 3b. Job Description Schema

```python
class ExperienceLevel(str, Enum):
    ENTRY = "Entry"
    MID = "Mid"
    SENIOR = "Senior"
    LEAD = "Lead"
    EXECUTIVE = "Executive"

class CompanyInfo(BaseModel):
    name: str
    industry: str
    size: str                          # e.g., "Startup (1-50)", "Mid-size (51-500)", "Enterprise (500+)"
    location: str

class JobRequirements(BaseModel):
    required_skills: list[str]         # ≥1 required skill
    preferred_skills: list[str] | None = None
    education: str                     # e.g., "Bachelor's in Computer Science or related field"
    experience_years: int              # 0-30
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
    title: str                         # Job title
    company: CompanyInfo
    description: str                   # Full job description text
    requirements: JobRequirements
```

### 3c. Metadata & Pipeline Models

```python
class FitLevel(str, Enum):
    """Controlled fit levels for resume generation."""
    EXCELLENT = "excellent"     # 80%+ skill overlap
    GOOD = "good"               # 60-80%
    PARTIAL = "partial"         # 40-60%
    POOR = "poor"               # 20-40%
    MISMATCH = "mismatch"       # <20%

class WritingStyle(str, Enum):
    """Prompt template styles for diverse generation."""
    FORMAL = "formal"           # Corporate tone
    CASUAL = "casual"           # Startup-friendly
    TECHNICAL = "technical"     # Detail-heavy
    ACHIEVEMENT = "achievement" # Metrics-driven
    CAREER_CHANGER = "career_changer"  # Transferable skills

class GeneratedResume(BaseModel):
    """Wraps Resume with pipeline metadata. Trace IDs link to job + pair."""
    trace_id: str               # UUID — unique across entire pipeline run
    resume: Resume
    fit_level: FitLevel
    writing_style: WritingStyle
    template_version: str       # "v1", "v2" (for A/B testing)
    generated_at: str           # ISO datetime
    prompt_template: str        # Which template produced this
    model_used: str             # "gpt-4o-mini"

class GeneratedJob(BaseModel):
    """Wraps JobDescription with pipeline metadata."""
    trace_id: str               # UUID
    job: JobDescription
    is_niche_role: bool         # Flag for niche/unusual job titles
    generated_at: str
    prompt_template: str
    model_used: str

class ResumeJobPair(BaseModel):
    """Links a resume to a job with analysis metadata."""
    pair_id: str                # UUID
    resume_trace_id: str        # FK to GeneratedResume
    job_trace_id: str           # FK to GeneratedJob
    fit_level: FitLevel         # The intended fit level
    created_at: str
```

### 3d. Failure Labeling Models

```python
class FailureLabels(BaseModel):
    """Rule-based failure metrics for a single resume-job pair."""
    pair_id: str

    # Jaccard similarity: |A ∩ B| / |A ∪ B| where A=resume skills, B=job required skills
    skills_overlap: float              # 0.0 to 1.0
    skills_overlap_raw: int            # Size of intersection (for debugging)
    skills_union_raw: int              # Size of union (for debugging)

    # Binary flags
    experience_mismatch: bool          # Years gap or <50% of required
    seniority_mismatch: bool           # |resume_level - job_level| > 1
    missing_core_skills: bool          # Absence of top-3 required skills
    has_hallucinations: bool           # Rule-based: unrealistic claims
    has_awkward_language: bool          # Excessive buzzwords, AI patterns

    # Supporting detail for debugging/analysis
    experience_years_resume: float     # Calculated from experience entries
    experience_years_required: int     # From job requirements
    seniority_level_resume: int        # Mapped: Entry=0, Mid=1, Senior=2, Lead=3, Exec=4
    seniority_level_job: int
    missing_skills: list[str]          # Which top-3 skills are missing
    hallucination_reasons: list[str]   # Why hallucination was flagged
    awkward_language_reasons: list[str] # Why awkward language was flagged

    # Normalized skill sets used for Jaccard (post-normalization)
    resume_skills_normalized: list[str]
    job_skills_normalized: list[str]

class JudgeResult(BaseModel):
    """LLM-as-Judge evaluation of a resume-job pair."""
    pair_id: str
    has_hallucinations: bool
    hallucination_details: str
    has_awkward_language: bool
    awkward_language_details: str
    overall_quality_score: float       # 0.0 to 1.0
    fit_assessment: str                # Narrative assessment
    recommendations: list[str]         # Actionable suggestions
    red_flags: list[str]               # Concerns identified

class CorrectionResult(BaseModel):
    """Tracks a single correction attempt."""
    pair_id: str
    attempt_number: int                # 1, 2, or 3
    original_errors: list[str]         # Pydantic validation errors from the failed record
    corrected_successfully: bool
    remaining_errors: list[str] | None = None  # Errors after correction (if still failing)

class CorrectionSummary(BaseModel):
    """Aggregate correction statistics."""
    total_invalid: int
    total_corrected: int
    correction_rate: float             # corrected / invalid
    avg_attempts_per_success: float
    common_failure_reasons: dict[str, int]  # reason → count

class FeedbackEntry(BaseModel):
    """Tracks user feedback on analysis results."""
    feedback_id: str                   # UUID
    pair_id: str                       # Which analysis was rated
    rating: str                        # "thumbs_up" or "thumbs_down"
    comment: str | None = None         # Optional free-text
    timestamp: str                     # ISO datetime
```

### 3e. Schema Design Rationale (Java/TS Parallel)

**Why deeply nested models instead of flat?**

In Java, this is the difference between a single `HashMap<String, Object>` and a properly typed DTO hierarchy (`ResumeDTO` → `ExperienceDTO[]` → `AchievementDTO[]`). The nested approach gives you:

1. **Type safety at every level** — Pydantic validates each sub-model independently
2. **Granular error messages** — "experience[2].end_date is before start_date" vs "invalid resume"
3. **Composability** — `ContactInfo` can be reused in other models
4. **Self-documenting API** — FastAPI auto-generates OpenAPI docs from nested Pydantic models

**Why Optional fields for some nested attributes?**

This is the **strategic leniency** that makes Instructor + nested schemas work:

- `coursework: list[str] | None = None` — LLMs often omit this. Not worth a retry.
- `achievements: list[str] | None = None` — Nice to have, not structural.
- `summary: str | None = None` — Many real resumes don't have summaries.

**The principle:** Make fields Optional when their absence is a data quality issue (caught by labeling), not a structural issue (caught by Instructor retries). This reduces Instructor retries from ~5 per record to ~2, saving cost and latency.

---

## 4. Generation Pipeline

### 4a. Job Description Generation

**Strategy:** Generate 50+ jobs across 10 industries with mix of standard and niche roles.

**Industry Distribution (5 jobs per industry):**

| Industry | Example Roles | Niche Role Example |
|----------|--------------|-------------------|
| Technology | Software Engineer, Data Scientist | MLOps Platform Engineer |
| Healthcare | Nurse Practitioner, Clinical Data Analyst | Bioethics Compliance Officer |
| Finance | Financial Analyst, Risk Manager | Quantitative ESG Strategist |
| Education | Curriculum Developer, EdTech PM | Learning Experience Designer |
| Manufacturing | Process Engineer, QA Manager | Digital Twin Architect |
| Retail | Merchandising Analyst, Supply Chain Lead | Omnichannel Experience Strategist |
| Energy | Renewable Energy Engineer, Grid Analyst | Carbon Credit Trading Analyst |
| Legal | Paralegal, Compliance Officer | AI Ethics & Policy Counsel |
| Marketing | Brand Manager, SEO Specialist | Growth Hacking Lead |
| Government | Policy Analyst, Program Manager | Civic Tech Innovation Director |

**Niche role detection:** ~20% of generated jobs should be niche roles (flagged with `is_niche_role=True`). A niche role is one where standard skill matching may fail because the role title doesn't map to common skill taxonomies. This tests whether the labeling system handles unusual roles gracefully.

**Generation prompt structure:**

```
System: You are an expert HR professional and job description writer.
Generate a realistic job description for the {industry} industry.
The role should be {standard|niche} and require {experience_level}-level experience.

Requirements:
- Include specific, measurable required skills (not generic buzzwords)
- Required skills list should have 5-10 concrete technical or domain skills
- Preferred skills should complement but not duplicate required skills
- Experience years should match the seniority level realistically

{IF niche: "This should be an unusual or emerging role that combines
skills from multiple traditional disciplines. The title should NOT be
a standard job title found on most job boards."}
```

### 4b. Resume Generation with Controlled Fit Levels

**The core challenge:** Generating a "poor fit" resume is harder than generating a "good fit" one. The LLM naturally wants to produce relevant content. You must explicitly instruct it to introduce specific deficiencies.

**Fit Level Generation Strategy:**

| Fit Level | Skill Overlap Target | Generation Strategy |
|-----------|---------------------|-------------------|
| Excellent (80%+) | Include 80%+ of required skills with matching proficiency | "Generate a resume that is an excellent match for this job" |
| Good (60-80%) | Include most required skills, miss 1-2, add some unrelated | "Generate a strong but not perfect candidate — missing 1-2 key skills" |
| Partial (40-60%) | Include ~half of required skills, different experience level | "Generate a candidate with related but incomplete qualifications" |
| Poor (20-40%) | Few matching skills, wrong seniority, wrong domain | "Generate a candidate from a related field who lacks most required skills" |
| Mismatch (<20%) | Almost no matching skills, completely different domain | "Generate a candidate from a completely different industry" |

**Fit level distribution per job:** Generate 5 resumes per job with this distribution:
- 1 Excellent, 1 Good, 1 Partial, 1 Poor, 1 Mismatch

For the target of 250+ pairs: 50 jobs × 5 resumes = 250 pairs minimum.

**Resume generation prompt structure:**

```
System: You are a professional resume writer creating realistic resumes
for job applicants of varying qualification levels.

Generate a resume for someone applying to this job:
{job_description_json}

Fit Level: {fit_level}
Writing Style: {writing_style}

{FIT_LEVEL_INSTRUCTIONS based on table above}

Style Instructions:
- formal: Professional, corporate language. Standard chronological format.
- casual: Conversational, startup-friendly. May use first person.
- technical: Heavy on technical details, tools, and specific technologies.
- achievement: Metrics-driven. Every bullet has a quantified result.
- career_changer: Emphasizes transferable skills from a different industry.

IMPORTANT:
- Skills should be specific technologies/tools, not generic buzzwords
- Experience dates must be in ISO format (YYYY-MM-DD or YYYY-MM)
- Each experience entry needs concrete responsibilities, not vague descriptions
- Email must be a valid format (name@domain.tld)
- Phone must have at least 10 digits
```

### 4c. Prompt Templates (5 Templates for A/B Testing)

Each template has a distinct personality that affects the generated resume's characteristics:

| Template | ID | Style | Characteristics | Expected Failure Profile |
|----------|-------|-------|----------------|------------------------|
| Formal Corporate | `v1-formal` | FORMAL | Conservative language, standard structure, minimal buzzwords | Low awkward language, may be generic |
| Startup Casual | `v1-casual` | CASUAL | Informal, first-person, project-focused | Higher awkward language, more authentic feel |
| Technical Deep | `v1-technical` | TECHNICAL | Specification-heavy, tool versions, architecture details | Lower hallucination (concrete claims), may overcomplicate |
| Achievement Metrics | `v1-achievement` | ACHIEVEMENT | Every bullet quantified ("increased X by Y%") | Higher hallucination risk (fabricated metrics) |
| Career Changer | `v1-career-changer` | CAREER_CHANGER | Transferable skills emphasis, narrative transitions | Higher skill mismatch, lower core skill coverage |

**A/B Testing Protocol:**
1. Generate an equal number of resumes per template (50 each = 250 total)
2. Track validation success rate per template
3. Track failure mode distribution per template
4. Run chi-squared test for statistical significance on failure rate differences
5. Identify best and worst templates
6. Create `v2` improved versions of the worst-performing templates

### 4d. Instructor Integration (Nested Schema Strategy)

```python
import instructor
from openai import OpenAI

# WHY instructor.Mode.JSON: Forces OpenAI API to return valid JSON
# BEFORE Pydantic validation. Without this, the LLM might return
# markdown-wrapped JSON (```json...```) which breaks parsing.
# This is a two-layer defense: JSON mode ensures structure,
# Pydantic ensures content validity.
client = instructor.from_openai(OpenAI(), mode=instructor.Mode.JSON)

resume = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Resume,           # Instructor injects JSON schema + validates
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.8,                 # WHY 0.8: Higher than P1's 0.7 because we
                                     # want MORE diversity across resumes. P1
                                     # generated repair guides (consistency matters).
                                     # P4 generates people (diversity matters).
    max_retries=5,                   # WHY 5: Nested schemas have ~30 failure points
                                     # vs P1's ~7. More retries = higher success rate.
                                     # Cost: ~5× more tokens on retry path, but
                                     # only ~10% of records need >3 retries.
)
```

**Why Instructor over raw OpenAI + manual Pydantic parsing:**

In P1, Instructor reduced generation boilerplate by ~60%. For P4's nested schemas, the savings are even larger:

- **Without Instructor:** You'd need to: (1) call OpenAI API, (2) parse JSON from response, (3) handle markdown wrapping, (4) call `Resume.model_validate_json()`, (5) catch `ValidationError`, (6) extract error details, (7) build a retry prompt with error context, (8) retry. That's ~50 lines of retry logic per model.
- **With Instructor:** `client.chat.completions.create(response_model=Resume, max_retries=5)` — Instructor does all 8 steps internally. One line.

The tradeoff: Instructor's retry mechanism is opaque (you can't customize the retry prompt). But for P4, the default behavior (feed `ValidationError` back to LLM) is exactly what we want.

### 4e. Caching Layer

Same pattern as P1/P2. Cache AROUND the Instructor call:

```python
def generate_with_cache(
    prompt_key: str,
    system_prompt: str,
    user_prompt: str,
    response_model: type[BaseModel],
    cache_dir: Path = Path("data/cache"),
) -> BaseModel:
    """Check cache before calling LLM. Cache validated result after."""
    cache_key = hashlib.md5(
        f"{system_prompt}{user_prompt}{response_model.__name__}".encode()
    ).hexdigest()
    cache_file = cache_dir / f"{cache_key}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        return response_model.model_validate(data["response"])

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=response_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_retries=5,
    )

    cache_file.write_text(json.dumps({
        "cache_key": cache_key,
        "prompt_key": prompt_key,
        "model": "gpt-4o-mini",
        "timestamp": datetime.now().isoformat(),
        "response": result.model_dump(),
    }, indent=2))

    return result
```

### 4f. Batch Processing with Progress Tracking

```python
# WHY batch processing instead of generating all at once:
# 1. If the pipeline crashes at record 200, you don't lose records 1-199 (cached)
# 2. Progress tracking shows estimated completion time
# 3. Rate limiting: OpenAI has per-minute token limits
# 4. Memory: 250 nested Pydantic models in RAM is fine (~50MB), but
#    generating them all simultaneously would overwhelm the API

from rich.progress import Progress, TaskID

async def generate_all_pairs(jobs: list[GeneratedJob]) -> list[ResumeJobPair]:
    pairs = []
    with Progress() as progress:
        task = progress.add_task("Generating resumes...", total=len(jobs) * 5)
        for job in jobs:
            for fit_level in FitLevel:
                style = random.choice(list(WritingStyle))
                resume = generate_resume(job, fit_level, style)
                pair = create_pair(resume, job, fit_level)
                pairs.append(pair)
                progress.advance(task)
    return pairs
```

---

## 5. Skill Normalization

**First principle:** Jaccard similarity is only as good as the sets you feed it. "Python 3.10", "python", and "Python Developer" are all the same skill. Without normalization, your Jaccard scores are artificially low, and your failure labeling is wrong.

### 5a. Normalization Pipeline

```
Raw skill → lowercase → strip whitespace → remove version numbers → 
strip suffixes → apply alias mapping → normalized skill
```

**Step-by-step:**

| Step | Input | Output | Regex/Logic |
|------|-------|--------|-------------|
| Lowercase | "Python 3.10" | "python 3.10" | `.lower()` |
| Strip whitespace | "  python 3.10  " | "python 3.10" | `.strip()` |
| Remove versions | "python 3.10" | "python" | `re.sub(r'\s*\d+(\.\d+)*\s*$', '', s)` |
| Strip suffixes | "python developer" | "python" | Remove: developer, engineer, programming, development, framework, language, library |
| Alias mapping | "js" | "javascript" | Dict lookup: `{"js": "javascript", "ts": "typescript", "k8s": "kubernetes", ...}` |

### 5b. Alias Map (Extensible)

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

### 5c. Jaccard Similarity Calculation

```python
def calculate_jaccard(
    resume_skills: list[str],
    job_skills: list[str],
    normalizer: SkillNormalizer,
) -> tuple[float, int, int]:
    """
    Calculate Jaccard similarity between normalized skill sets.

    Returns: (jaccard_score, intersection_size, union_size)

    WHY return raw sizes too: For debugging. If Jaccard is 0.33,
    knowing it's 2/6 vs 5/15 tells different stories.
    2/6 = resume has few relevant skills.
    5/15 = resume has many skills but job requires many more.
    """
    resume_set = {normalizer.normalize(s) for s in resume_skills}
    job_set = {normalizer.normalize(s) for s in job_skills}

    intersection = resume_set & job_set
    union = resume_set | job_set

    if len(union) == 0:
        return 0.0, 0, 0  # Edge case: both have no skills

    jaccard = len(intersection) / len(union)
    return jaccard, len(intersection), len(union)
```

---

## 6. Failure Labeling Pipeline

### 6a. The 6 Failure Modes

| # | Metric | Calculation | Threshold | Why It Matters |
|---|--------|-------------|-----------|----------------|
| 1 | **Skills Overlap** | Jaccard similarity (normalized) | Continuous 0.0–1.0 | Core match metric — is this candidate qualified? |
| 2 | **Experience Mismatch** | `resume_years < job_years * 0.5` OR `resume_years < job_years - 3` | Binary flag | Under-experienced candidates waste interviewer time |
| 3 | **Seniority Mismatch** | `|resume_level - job_level| > 1` (Entry=0, Mid=1, Senior=2, Lead=3, Exec=4) | Binary flag | A Junior applying for Staff is a red flag |
| 4 | **Missing Core Skills** | Absence of ≥1 of top-3 required skills (after normalization) | Binary flag | Top-3 skills are non-negotiable dealbreakers |
| 5 | **Hallucinated Skills** | Rule-based: entry-level claiming 10+ "Expert" skills; 30+ skills total; impossible timelines | Binary flag | LLMs hallucinate credentials; real resumes don't |
| 6 | **Awkward Language** | Buzzword density >5 in summary/description; repeated words 3+ times in proximity; AI-pattern phrases | Binary flag | Distinguishes authentic from low-quality AI-generated |

### 6b. Experience Calculation

```python
def calculate_total_experience(experiences: list[Experience]) -> float:
    """
    Sum duration of all experience entries in years.

    WHY this is non-trivial:
    - Current jobs (end_date=None) use today as end date
    - Overlapping jobs (two jobs at same time) should NOT double-count
      SPECULATION: For P4 scope, we DO double-count overlapping jobs.
      Production systems would merge overlapping date ranges.
      Flag this in ADR as a known simplification.
    """
    total_months = 0
    today = datetime.now()

    for exp in experiences:
        start = datetime.strptime(exp.start_date[:7], "%Y-%m")
        if exp.end_date:
            end = datetime.strptime(exp.end_date[:7], "%Y-%m")
        else:
            end = today

        months = (end.year - start.year) * 12 + (end.month - start.month)
        total_months += max(0, months)  # Guard against negative

    return total_months / 12.0
```

### 6c. Seniority Level Mapping

```python
SENIORITY_MAP: dict[str, int] = {
    "entry": 0, "junior": 0, "intern": 0, "associate": 0,
    "mid": 1, "intermediate": 1, "regular": 1,
    "senior": 2, "sr": 2,
    "lead": 3, "principal": 3, "staff": 3, "architect": 3,
    "executive": 4, "director": 4, "vp": 4, "chief": 4, "head": 4, "c-level": 4,
}

def infer_seniority(title: str, years_experience: float) -> int:
    """
    Infer seniority level from job title and experience.

    WHY title + experience: A "Software Engineer" with 15 years
    is probably Senior/Lead level even if the title doesn't say so.
    Title-based mapping is primary, experience is fallback.
    """
    title_lower = title.lower()
    for keyword, level in SENIORITY_MAP.items():
        if keyword in title_lower:
            return level

    # Fallback: infer from experience years
    if years_experience < 2:
        return 0
    elif years_experience < 5:
        return 1
    elif years_experience < 10:
        return 2
    elif years_experience < 15:
        return 3
    else:
        return 4
```

### 6d. Hallucination Detection (Rule-Based)

```python
def detect_hallucinations(resume: Resume, experience_years: float) -> tuple[bool, list[str]]:
    """
    Rule-based hallucination detection.

    WHY rule-based first, LLM-judge second:
    - Rules are fast (microseconds), deterministic, and testable
    - LLM judge is slow (~5s), non-deterministic, and expensive
    - Rules catch obvious hallucinations; judge catches subtle ones
    - Having both lets us compare agreement (portfolio talking point)
    """
    reasons = []

    # Rule 1: Entry-level (<2 years) claiming Expert in many skills
    expert_count = sum(1 for s in resume.skills if s.proficiency_level == ProficiencyLevel.EXPERT)
    if experience_years < 2 and expert_count > 3:
        reasons.append(
            f"Entry-level ({experience_years:.1f} years) claims Expert in {expert_count} skills"
        )

    # Rule 2: Unrealistic total skill count with high proficiency
    if len(resume.skills) > 20 and expert_count > 10:
        reasons.append(
            f"{len(resume.skills)} total skills with {expert_count} at Expert level"
        )

    # Rule 3: Skill years exceed total experience
    for skill in resume.skills:
        if skill.years and skill.years > experience_years + 1:
            reasons.append(
                f"Claims {skill.years}yr in {skill.name} but total experience is {experience_years:.1f}yr"
            )

    # Rule 4: Impossible career progression
    # Check if experience entries suggest impossible timeline
    if len(resume.experience) > 1:
        titles = [exp.title.lower() for exp in resume.experience]
        # Director/VP before 5 years of total experience
        senior_titles = ["director", "vp", "vice president", "chief", "head of"]
        for i, title in enumerate(titles):
            if any(st in title for st in senior_titles) and experience_years < 5:
                reasons.append(
                    f"Senior title '{resume.experience[i].title}' with only {experience_years:.1f} years experience"
                )

    return len(reasons) > 0, reasons
```

### 6e. Awkward Language Detection

```python
BUZZWORDS = [
    "synergy", "synergize", "leverage", "paradigm", "disrupt", "innovative",
    "thought leader", "move the needle", "circle back", "deep dive",
    "bleeding edge", "cutting edge", "game changer", "rockstar", "ninja",
    "guru", "wizard", "unicorn", "best-of-breed", "world-class",
    "results-driven", "self-starter", "detail-oriented", "team player",
    "proactive", "dynamic", "passionate", "motivated",
    "thinking outside the box", "low-hanging fruit", "bandwidth",
    "pivot", "ecosystem", "holistic", "scalable", "robust",
]

AI_PATTERNS = [
    "as a seasoned", "in today's fast-paced", "proven track record of",
    "demonstrated ability to", "spearheaded initiatives",
    "orchestrated cross-functional", "championed the adoption",
    "leveraged cutting-edge", "passionate about driving",
]

def detect_awkward_language(resume: Resume) -> tuple[bool, list[str]]:
    """
    Detect AI-generated or buzzword-heavy text.

    WHY pattern matching over LLM classification:
    Same reasoning as hallucination detection — fast, deterministic,
    testable. The LLM judge catches subtle cases we miss here.
    """
    reasons = []

    # Combine all text for analysis
    all_text = " ".join([
        resume.summary or "",
        *[r for exp in resume.experience for r in exp.responsibilities],
        *[a for exp in resume.experience for a in (exp.achievements or [])],
    ]).lower()

    # Rule 1: Buzzword density
    buzzword_count = sum(1 for bw in BUZZWORDS if bw in all_text)
    if buzzword_count > 5:
        reasons.append(f"High buzzword density: {buzzword_count} buzzwords detected")

    # Rule 2: AI-pattern phrases
    ai_pattern_count = sum(1 for pattern in AI_PATTERNS if pattern in all_text)
    if ai_pattern_count > 2:
        reasons.append(f"AI-generated patterns detected: {ai_pattern_count} matches")

    # Rule 3: Repeated words in proximity (3+ times in 50-word window)
    words = all_text.split()
    for i in range(len(words) - 50):
        window = words[i:i+50]
        word_counts = {}
        for w in window:
            if len(w) > 4:  # Skip short words (the, and, etc.)
                word_counts[w] = word_counts.get(w, 0) + 1
        repetitions = {w: c for w, c in word_counts.items() if c >= 3}
        if repetitions:
            reasons.append(f"Repeated words in close proximity: {repetitions}")
            break  # One finding is enough

    return len(reasons) > 0, reasons
```

---

## 7. LLM-as-Judge

### 7a. Judge Prompt

```
System: You are an expert hiring manager and resume quality evaluator.
Analyze this resume-job pair for quality issues that rule-based systems miss.

Be specific and evidence-based in your assessments. Don't flag issues
without citing the specific part of the resume that triggered your concern.

User:
Job Description:
{job_description_json}

Resume:
{resume_json}

Evaluate for:
1. Hallucinations: Unverifiable claims, timeline inconsistencies, impossible credentials
2. Awkward Language: Excessive jargon, unnatural phrasing, obvious AI-generated patterns
3. Fit Assessment: Holistic skills/experience alignment (beyond just skill keyword matching)
4. Red Flags: Employment gaps, inconsistent career progression, other concerns

Provide an overall quality score (0.0 = terrible, 1.0 = excellent).
```

### 7b. Judge Response Model

```python
class JudgeResult(BaseModel):
    """LLM-as-Judge output — parsed via Instructor with max_retries=3."""
    pair_id: str
    has_hallucinations: bool
    hallucination_details: str         # Empty string if no hallucinations
    has_awkward_language: bool
    awkward_language_details: str
    overall_quality_score: float       # 0.0 to 1.0
    fit_assessment: str                # Narrative assessment
    recommendations: list[str]         # Actionable suggestions
    red_flags: list[str]               # Concerns identified

    @field_validator("overall_quality_score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Quality score must be 0.0-1.0, got {v}")
        return v
```

### 7c. Judge Integration

- **Model:** GPT-4o (higher quality for evaluation, same as P1's judge)
- **Instructor:** `max_retries=3` (judge schema is simpler than resume schema)
- **Cost:** ~250 pairs × ~$0.003/pair = ~$0.75 for judge pass
- **Timing:** Judge runs AFTER rule-based labeling. Results stored separately in `judge_results.jsonl`
- **API integration:** Judge is OPTIONAL per request (query param `?use_judge=true`). Default is rule-based only (<2s). With judge, <10s.

---

## 8. Correction Loop

### 8a. Strategy

When schema validation fails (Instructor exhausts max_retries), the record is saved to `invalid.jsonl`. The correction loop processes these records:

```
For each invalid record:
  1. Extract: field path, error type, invalid value, expected format
  2. Build correction prompt with error context
  3. Send to GPT-4o-mini via Instructor (same model, corrective prompt)
  4. Re-validate
  5. If still failing: retry up to 3 total attempts
  6. Track: attempt count, success/failure, remaining errors
```

### 8b. Correction Prompt

```
System: You are a data correction specialist. Your job is to fix
invalid resume/job data based on specific validation errors.

Fix ONLY the flagged errors. Do not change valid fields.

User:
The following {resume|job_description} failed validation with these errors:

{for each error:}
  - Field: {field_path} (e.g., "experience[2].end_date")
  - Error: {error_message} (e.g., "end_date must be after start_date")
  - Current Value: {invalid_value}
  - Expected: {format_description}

Original data:
{record_json}

Generate a corrected version that fixes these specific issues.
```

### 8c. Success Metrics

```
correction_rate = successfully_corrected / total_invalid
Target: > 50%

Track per record:
- attempts_needed (1, 2, or 3)
- which_errors_fixed (list of field paths)
- which_errors_remain (if permanently failed)
- time_per_attempt
```

---

## 9. Analysis & Visualization

### 9a. Pandas DataFrame Structure

```
| pair_id | job_trace_id | resume_trace_id | fit_level | template | is_niche |
| skills_overlap | exp_mismatch | seniority_mismatch | missing_core |
| hallucinations | awkward_language | judge_quality_score | total_flags |
```

### 9b. Required Visualizations (5+ Charts)

| # | Chart | What It Reveals | Library |
|---|-------|----------------|---------|
| 1 | **Failure mode correlation matrix** | Which failures co-occur? (e.g., seniority mismatch + experience mismatch) | seaborn `heatmap()` on `df[failure_cols].corr()` |
| 2 | **Failure rates by fit level** | Do "poor fit" resumes actually fail more? (validates controlled generation) | seaborn `barplot()` grouped by fit_level |
| 3 | **Failure rates by template** | Which writing styles cause issues? (A/B test results) | seaborn `barplot()` grouped by template |
| 4 | **Niche vs standard roles** | Do niche jobs have different failure patterns? | seaborn `barplot()` with hue=is_niche |
| 5 | **Schema validation heatmap** | Which fields fail most often during generation? | seaborn `heatmap()` on field error counts |
| 6 | **Skills overlap distribution** | Jaccard score distribution by fit level (box plot) | seaborn `boxplot()` by fit_level |
| 7 | **Hallucination by seniority** | Do entry-level resumes hallucinate more? | seaborn `barplot()` grouped by inferred_seniority |
| 8 | **Correction success by error type** | Which errors are easiest/hardest to correct? | matplotlib `barh()` |
| 9 | **Judge vs rule-based agreement** | Where does the LLM judge disagree with rules? | seaborn `heatmap()` confusion matrix |

**Quality Standard:** Publication-ready with clear labels, appropriate color schemes (diverging for correlations, sequential for rates), readable font sizes, legends, and proper titles.

### 9c. Key Analysis Questions

1. Does Jaccard similarity correlate with intended fit level? (Validates controlled generation)
2. Do failure modes cluster? (Skills overlap + missing core skills should co-occur)
3. Which template produces the most/fewest hallucinations?
4. Are niche roles harder to match? (More missing core skills expected)
5. Does the correction loop actually improve data quality? (Before/after comparison)
6. Where does the LLM judge disagree with rule-based detection? (Agreement analysis)

---

## 10. FastAPI REST API

### 10a. Endpoints (8 Total)

```python
from fastapi import FastAPI, Query, Path, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="AI Resume Coach",
    description="Analyze resume-job fit with rule-based metrics, LLM judge, and semantic search",
    version="1.0.0",
)

# --- Endpoint 1: Health Check ---
@app.get("/health")
def health_check():
    """Simple health check. Returns service status and pipeline run info."""
    return {"status": "healthy", "version": "1.0.0", "last_pipeline_run": "..."}

# --- Endpoint 2: Resume Review (Core) ---
@app.post("/review-resume")
def review_resume(
    request: ReviewRequest,
    use_judge: bool = Query(False, description="Enable LLM-as-Judge (slower, more thorough)"),
) -> ReviewResponse:
    """
    Analyze a resume against a job description.

    Without judge (~1-2s): Rule-based metrics (Jaccard, seniority, hallucinations)
    With judge (~5-10s): Adds LLM holistic evaluation

    WHY POST not GET: Request body contains full resume + job JSON.
    GET would require URL-encoding complex nested objects.
    Same reasoning as REST API design in Java/Spring.
    """
    pass

# --- Endpoint 3: Failure Rate Statistics (Core) ---
@app.get("/analysis/failure-rates")
def get_failure_rates() -> FailureRateResponse:
    """
    Aggregate statistics from the most recent pipeline run.
    Reads from pipeline_summary.json on disk.
    """
    pass

# --- Endpoint 4: Template Comparison / A/B Test Results (Core) ---
@app.get("/analysis/template-comparison")
def get_template_comparison() -> TemplateComparisonResponse:
    """
    A/B test results: validation rates and failure modes per template.
    Includes chi-squared p-value for statistical significance.

    WHY a separate endpoint from /failure-rates: Separation of concerns.
    /failure-rates gives overall pipeline health.
    /template-comparison gives generation optimization insights.
    Different consumers care about different things.
    """
    pass

# --- Endpoint 5: Multi-Hop Evaluation (Core) ---
@app.post("/evaluate/multi-hop")
def evaluate_multi_hop(
    request: MultiHopRequest,
) -> MultiHopResponse:
    """
    Run multi-hop evaluation questions on a resume-job pair.
    Questions require cross-section reasoning (education + experience + skills).

    WHY POST: The request includes full resume + job for evaluation.
    Returns evaluation results with explanations.
    """
    pass

# --- Endpoint 6: Similar Candidates (Core — Vector Store) ---
@app.get("/search/similar-candidates")
def find_similar_candidates(
    job_description: str = Query(..., description="Job description text to match against"),
    top_k: int = Query(5, ge=1, le=20, description="Number of similar candidates to return"),
    fit_level: str | None = Query(None, description="Filter by fit level (excellent, good, partial, poor, mismatch)"),
) -> SimilarCandidatesResponse:
    """
    Semantic search for resumes similar to a job description.
    Uses ChromaDB vector store populated during pipeline run.

    WHY ChromaDB over FAISS: ChromaDB provides native metadata filtering.
    The fit_level query parameter translates directly to ChromaDB's
    where={"fit_level": value} — zero custom code for filtering.
    FAISS would require post-retrieval filtering (retrieve more, filter down).
    """
    pass

# --- Endpoint 7: Feedback (Core) ---
@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
) -> FeedbackResponse:
    """
    Submit thumbs up/down feedback on an analysis result.
    Logs to data/feedback/feedback.jsonl for continuous improvement.
    If Braintrust is configured, also logs there.

    WHY this matters: Demonstrates continuous improvement loop thinking.
    Production ML systems need feedback mechanisms. Even without Braintrust,
    the logged JSON is auditable evidence of user satisfaction tracking.
    """
    pass

# --- Endpoint 8: List Generated Jobs (Utility) ---
@app.get("/jobs")
def list_jobs(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=50, description="Items per page"),
    industry: str | None = Query(None, description="Filter by industry"),
    is_niche: bool | None = Query(None, description="Filter niche roles only"),
) -> JobListResponse:
    """
    Browse generated jobs with pagination and filtering.
    Reads from data/generated/jobs_{timestamp}.jsonl.

    WHY this endpoint: Makes the /docs page a complete demo.
    A recruiter can: browse jobs → pick one → review resume against it →
    search for similar candidates → submit feedback. Full user journey.
    """
    pass

# --- Endpoint 9: Get Pair Details (Utility) ---
@app.get("/pairs/{pair_id}")
def get_pair_details(
    pair_id: str = Path(..., description="UUID of the resume-job pair"),
) -> PairDetailResponse:
    """
    Retrieve a specific pair with all labels, judge result, and correction history.
    Single endpoint for debugging and deep-dive analysis.

    WHY Path parameter not Query: The pair_id uniquely identifies one resource.
    RESTful convention: /resource/{id} for specific items.
    Same as Spring's @PathVariable.
    """
    pass
```

### 10b. Request/Response Models

```python
class ReviewRequest(BaseModel):
    """Input for POST /review-resume."""
    resume: Resume                     # Full resume (nested Pydantic model)
    job_description: JobDescription     # Full job description

class ReviewResponse(BaseModel):
    """Output for POST /review-resume."""
    pair_id: str                       # Generated UUID for this analysis
    failure_labels: FailureLabels      # All 6 rule-based metrics
    judge_result: JudgeResult | None   # Only if use_judge=True
    processing_time_seconds: float

class FailureRateResponse(BaseModel):
    """Output for GET /analysis/failure-rates."""
    total_pairs: int
    validation_success_rate: float
    failure_mode_rates: dict[str, float]  # mode_name → rate
    correction_success_rate: float
    avg_jaccard_by_fit_level: dict[str, float]
    last_run_timestamp: str

class TemplateComparisonResponse(BaseModel):
    """Output for GET /analysis/template-comparison."""
    template_results: dict[str, TemplateStats]  # template_id → stats
    chi_squared_statistic: float
    chi_squared_p_value: float
    significant: bool                  # p_value < 0.05
    best_template: str
    worst_template: str
    recommendation: str                # e.g., "v1-technical produces 15% fewer failures"

class TemplateStats(BaseModel):
    """Per-template statistics for A/B comparison."""
    template_id: str
    total_generated: int
    validation_success_rate: float
    failure_mode_rates: dict[str, float]
    avg_jaccard: float
    avg_judge_quality_score: float | None  # None if judge hasn't run

class MultiHopRequest(BaseModel):
    """Input for POST /evaluate/multi-hop."""
    resume: Resume
    job_description: JobDescription

class MultiHopResponse(BaseModel):
    """Output for POST /evaluate/multi-hop."""
    pair_id: str
    questions: list[MultiHopQuestion]
    processing_time_seconds: float

class MultiHopQuestion(BaseModel):
    """A single multi-hop evaluation question with answer."""
    question: str
    requires_sections: list[str]       # e.g., ["education", "experience", "skills"]
    answer: str                        # Assessed answer
    assessment: str                    # Pass/Fail with reasoning

class SimilarCandidatesResponse(BaseModel):
    """Output for GET /search/similar-candidates."""
    query: str
    results: list[SimilarCandidate]
    total_in_index: int
    filter_applied: str | None         # e.g., "fit_level=excellent"
    processing_time_seconds: float

class SimilarCandidate(BaseModel):
    resume_trace_id: str
    similarity_score: float
    name: str
    skills: list[str]
    experience_years: float
    fit_level: str

class FeedbackRequest(BaseModel):
    """Input for POST /feedback."""
    pair_id: str                       # Which analysis to rate
    rating: str                        # "thumbs_up" or "thumbs_down"
    comment: str | None = None         # Optional free-text feedback

class FeedbackResponse(BaseModel):
    """Output for POST /feedback."""
    feedback_id: str                   # UUID for this feedback entry
    logged_to: list[str]               # ["json"] or ["json", "braintrust"]
    timestamp: str

class JobListResponse(BaseModel):
    """Output for GET /jobs."""
    jobs: list[JobSummary]
    total: int
    page: int
    page_size: int
    total_pages: int

class JobSummary(BaseModel):
    """Lightweight job representation for listing."""
    trace_id: str
    title: str
    company_name: str
    industry: str
    experience_level: str
    is_niche: bool
    required_skills_count: int

class PairDetailResponse(BaseModel):
    """Output for GET /pairs/{pair_id}."""
    pair_id: str
    resume: GeneratedResume
    job: GeneratedJob
    failure_labels: FailureLabels | None
    judge_result: JudgeResult | None
    correction_history: list[CorrectionResult]
    feedback: list[FeedbackRequest]    # All feedback for this pair
```

### 10c. FastAPI Integration Notes (Java/TS Parallel)

- **FastAPI ≈ Spring Boot + Jackson + Swagger combined.** It auto-generates OpenAPI docs from Pydantic models (like Swagger from Spring DTOs), handles JSON serialization (like Jackson), and provides dependency injection (like Spring @Autowired).
- **`@app.post` ≈ `@PostMapping`.** Route decorators work the same way.
- **`Query(...)` ≈ `@RequestParam`.** FastAPI query parameters with validation.
- **`Path(...)` ≈ `@PathVariable`.** Path parameters for resource IDs.
- **Pydantic models in routes ≈ `@RequestBody @Valid MyDTO`.** FastAPI validates the request body against the Pydantic model automatically. Invalid requests get 422 responses with detailed error messages.
- **Run with:** `uvicorn src.api:app --reload` (like `mvn spring-boot:run`)
- **9 endpoints** gives a complete user journey in `/docs`: browse jobs → review resume → search similar → evaluate multi-hop → submit feedback. A recruiter can test the full system without curl.

---

## 11. ChromaDB Vector Store Integration

### 11a. Why ChromaDB (Not FAISS)

FAISS was the right choice for P2 — benchmarking requires low-level control over index types, distance metrics, and brute-force vs approximate search. P4 has different needs: production integration with persistence and metadata filtering.

| Need | FAISS | ChromaDB |
|------|-------|----------|
| Persist across API restarts | Manual save/load files | Built-in `PersistentClient` |
| Filter by fit_level, industry | Build custom post-filter | `where={"fit_level": "excellent"}` |
| Portfolio story | Redundant with P2 | New tool = "I know when to use each" |

Using both FAISS (P2) and ChromaDB (P4) across the portfolio demonstrates you choose tools based on requirements, not habit.

### 11b. ChromaDB Setup

```python
import chromadb
from sentence_transformers import SentenceTransformer

# WHY ChromaDB over FAISS (which we used in P2):
# - ChromaDB provides persistence (saves to disk, survives restarts)
# - Built-in metadata filtering (filter by fit_level, industry, etc.)
# - Simpler API for insert + query
# - FAISS is better for benchmarking (raw performance). ChromaDB is
#   better for production integration (persistence + filtering).
# - Using both across projects shows you know when to use each.

def build_resume_index(resumes: list[GeneratedResume]) -> chromadb.Collection:
    """
    Embed all resumes and store in ChromaDB for semantic search.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="data/chromadb")
    collection = client.get_or_create_collection(
        name="resumes",
        metadata={"hnsw:space": "cosine"},  # Cosine similarity
    )

    # Build text representation for embedding
    texts = []
    ids = []
    metadatas = []
    for resume in resumes:
        text = resume_to_text(resume)  # Flatten resume to searchable text
        texts.append(text)
        ids.append(resume.trace_id)
        metadatas.append({
            "fit_level": resume.fit_level.value,
            "writing_style": resume.writing_style.value,
            "skills": ", ".join(s.name for s in resume.resume.skills),
        })

    # Batch embed + insert
    embeddings = model.encode(texts, show_progress_bar=True).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    del model
    gc.collect()  # Free model memory (P2/P3 pattern)
    return collection
```

---

## 12. Multi-Hop Evaluation Questions

Generate test questions that require reasoning across multiple resume sections:

```python
MULTI_HOP_TEMPLATES = [
    "Does this candidate's education level ({education}) align with the job's required experience level ({experience_level})?",
    "Are the claimed skills ({top_skills}) consistent with the job titles ({job_titles}) and years of experience ({years})?",
    "Given the career progression from {first_title} to {last_title} over {total_years} years, is the claimed seniority level realistic?",
    "Does the candidate's industry background ({industries}) provide transferable skills for the target role in {target_industry}?",
]
```

These questions test whether the labeling system can handle complex assessments that span multiple resume sections, not just individual field checks.

---

## 13. File Structure

```
04-resume-coach/
├── CLAUDE.md                          # Project-specific Claude Code memory
├── PRD.md                             # THIS FILE — implementation contract
├── pyproject.toml                     # Dependencies
├── src/
│   ├── __init__.py
│   ├── schemas.py                     # All Pydantic models (Resume, Job, FailureLabels, etc.)
│   ├── normalizer.py                  # Skill normalization (lowercase, versions, aliases)
│   ├── templates.py                   # 5+ prompt templates (v1 and v2 for A/B testing)
│   ├── generator.py                   # Instructor-based generation + caching + batch processing
│   ├── validator.py                   # Schema validation tracking + error categorization
│   ├── labeler.py                     # 6 failure modes: Jaccard, seniority, hallucination, etc.
│   ├── judge.py                       # LLM-as-Judge (GPT-4o via Instructor)
│   ├── corrector.py                   # Correction loop: error extraction → re-prompt → re-validate
│   ├── analyzer.py                    # Pandas analysis + all visualization generation
│   ├── vector_store.py                # ChromaDB integration for semantic search (bonus)
│   ├── api.py                         # FastAPI app with all endpoints
│   └── pipeline.py                    # Orchestrator: runs entire pipeline end-to-end
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py                # Schema validation tests (valid + invalid data)
│   ├── test_normalizer.py             # Skill normalization tests (aliases, versions, suffixes)
│   ├── test_labeler.py                # Failure labeling tests (Jaccard, seniority, hallucination)
│   ├── test_corrector.py              # Correction loop tests
│   └── test_api.py                    # FastAPI endpoint tests (TestClient)
├── data/
│   ├── cache/                         # LLM response cache (JSON files)
│   ├── generated/                     # Raw generated records (JSONL)
│   │   ├── jobs_{timestamp}.jsonl
│   │   ├── resumes_{timestamp}.jsonl
│   │   └── pairs_{timestamp}.jsonl
│   ├── validated/                     # Validated + invalid records
│   │   ├── validated_{timestamp}.json
│   │   └── invalid_{timestamp}.jsonl
│   ├── labels/                        # Failure labels + judge results
│   │   ├── failure_labels_{timestamp}.jsonl
│   │   ├── failure_labels_{timestamp}.csv   # CSV export for pandas/Excel
│   │   └── judge_results_{timestamp}.jsonl
│   ├── corrected/                     # Correction results
│   │   └── corrected_{timestamp}.jsonl
│   ├── feedback/                      # User feedback logs
│   │   └── feedback.jsonl
│   ├── analysis/                      # Analysis outputs
│   │   ├── schema_failure_modes_{timestamp}.json
│   │   ├── template_comparison_{timestamp}.json
│   │   ├── template_comparison_{timestamp}.csv  # CSV export for pandas/Excel
│   │   └── pipeline_summary_{timestamp}.json
│   └── chromadb/                      # Vector store (persistent)
├── results/
│   └── charts/                        # All PNG visualizations
│       ├── failure_correlation.png
│       ├── failure_by_fit_level.png
│       ├── failure_by_template.png
│       ├── niche_vs_standard.png
│       ├── schema_validation_heatmap.png
│       ├── skills_overlap_distribution.png
│       ├── hallucination_by_seniority.png
│       ├── correction_success.png
│       └── judge_vs_rules_agreement.png
├── docs/
│   └── adr/                           # Architecture Decision Records
│       ├── ADR-001-instructor-nested-schemas.md
│       ├── ADR-002-skill-normalization-strategy.md
│       ├── ADR-003-two-phase-validation.md
│       ├── ADR-004-fastapi-over-flask.md
│       └── ADR-005-chromadb-over-faiss.md
├── streamlit_app.py                   # Demo app
└── README.md                          # Project documentation
```

---

## 14. Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "instructor",              # Structured LLM output via Pydantic (auto-retry on validation failure)
    "openai",                  # LLM API client
    "pydantic>=2.0",           # Schema validation with nested models
    "pandas",                  # Data manipulation and analysis
    "matplotlib",              # Static visualization
    "seaborn",                 # Statistical visualization (heatmaps, box plots)
    "fastapi",                 # REST API framework (auto-generates OpenAPI docs)
    "uvicorn",                 # ASGI server for FastAPI (like Tomcat for Spring Boot)
    "chromadb",                # Vector database for semantic search (bonus)
    "sentence-transformers",   # Local embeddings for ChromaDB (all-MiniLM-L6-v2)
    "rich",                    # Console progress bars and formatted output
    "python-dotenv",           # Environment variable management (.env file)
    "pytest",                  # Testing framework
    "httpx",                   # TestClient for FastAPI tests (async HTTP client)
    "ruff",                    # Linting + formatting
    "scipy",                   # Chi-squared test for A/B template comparison
]
```

---

## 15. Day-by-Day Execution Plan

> **Schedule:** 3 sessions (Feb 24–26), no hard time cap per session. Stretch as needed.
> **Total estimated effort:** 18–24 hours across 3 sessions.
> **Philosophy:** Front-load generation + validation (Day 1), analysis + correction (Day 2), API + bonus + polish (Day 3).

### Day 1 (Monday Feb 24) — Schemas + Generation + Validation

**Goal:** End the night with 250+ generated pairs, validated, and saved to disk.

| Task | Est. Time | Description | Output |
|------|-----------|-------------|--------|
| T1.1 | 30min | Project setup: directory structure, pyproject.toml, `uv sync`, .env with OPENAI_API_KEY | Working environment |
| T1.2 | 60min | `src/schemas.py` — ALL Pydantic models: ContactInfo, Education, Experience, Skill, Resume, JobDescription, CompanyInfo, JobRequirements, FitLevel, WritingStyle, GeneratedResume, GeneratedJob, ResumeJobPair, FailureLabels, JudgeResult, CorrectionResult | Complete model hierarchy |
| T1.3 | 30min | `tests/test_schemas.py` — Valid/invalid data tests for every model. Parametrized tests for edge cases (empty lists, invalid dates, GPA out of range, etc.) | All schema tests pass |
| T1.4 | 30min | `src/normalizer.py` — SkillNormalizer class: lowercase, version removal, suffix stripping, alias mapping. `tests/test_normalizer.py` | Normalizer + tests pass |
| T1.5 | 45min | `src/templates.py` — 5 prompt templates (v1-formal, v1-casual, v1-technical, v1-achievement, v1-career-changer) for both job and resume generation | Template library |
| T1.6 | 60min | `src/generator.py` — Instructor-based generation with caching, batch processing, progress tracking. Generate 50 jobs first, then 5 resumes per job. | Generator module |
| T1.7 | 30min | `src/validator.py` — Track validation attempts vs successes, error categorization by field path, save valid/invalid records separately | Validator module |
| T1.8 | 90min | **Run generation pipeline:** Generate 50 jobs + 250 resumes. Monitor Instructor retry rates. Save to data/generated/ | 250+ pairs in JSONL |
| T1.9 | 15min | Quick sanity check: spot-check 5 pairs manually. Are fit levels reflected in the generated data? | Visual confirmation |

**Git checkpoint after T1.9.** Commit message: `feat(p4): schemas, generation, validation — 250+ pairs generated`

**End of Day 1 state:** 250+ pairs generated and validated. Instructor retry rate logged. Invalid records saved separately. Ready for analysis.

### Day 2 (Tuesday Feb 25) — Labeling + Judge + Correction + Analysis

**Goal:** All 6 failure modes calculated, LLM judge run, correction loop complete, all visualizations generated.

| Task | Est. Time | Description | Output |
|------|-----------|-------------|--------|
| T2.1 | 60min | `src/labeler.py` — Implement all 6 failure modes: Jaccard (using normalizer), experience mismatch, seniority mismatch, missing core skills, hallucination detection, awkward language detection | Labeler module |
| T2.2 | 30min | `tests/test_labeler.py` — Unit tests for each failure mode with known-good and known-bad inputs | All labeler tests pass |
| T2.3 | 30min | **Run labeling pipeline** on all 250+ pairs. Save to failure_labels.jsonl | Labels for every pair |
| T2.4 | 45min | `src/judge.py` — LLM-as-Judge (GPT-4o via Instructor). Run on all pairs. Save to judge_results.jsonl | Judge results |
| T2.5 | 60min | `src/corrector.py` — Correction loop: extract errors → build correction prompt → re-validate → retry up to 3×. Track success rates. | Corrector module |
| T2.6 | 30min | `tests/test_corrector.py` — Correction tests with mock LLM responses | Tests pass |
| T2.7 | 30min | **Run correction pipeline** on all invalid records. Log results. | Correction results |
| T2.8 | 90min | `src/analyzer.py` — Build analysis DataFrame, generate all 9+ visualizations. Calculate A/B template comparison with chi-squared test. | All charts in results/charts/ |
| T2.9 | 30min | Generate `pipeline_summary.json` — total records, validation rate, failure distribution, correction rate, processing times | Pipeline summary |
| T2.10 | 30min | Multi-hop evaluation questions (bonus): generate 10+ questions, verify labeling system handles them | Multi-hop questions |

**Git checkpoint after T2.10.** Commit message: `feat(p4): labeling, judge, correction, analysis — all metrics calculated`

**End of Day 2 state:** Complete analysis pipeline. All failure modes calculated. Visualizations generated. Correction loop metrics tracked. Ready for API + polish.

### Day 3 (Wednesday Feb 26) — API + Vector Store + Demo + Documentation

**Goal:** FastAPI running with all 9 endpoints, ChromaDB integrated, Streamlit demo, README, ADRs, Loom.

| Task | Est. Time | Description | Output |
|------|-----------|-------------|--------|
| T3.1 | 90min | `src/api.py` — FastAPI with all 9 endpoints: /health, /review-resume, /analysis/failure-rates, /analysis/template-comparison, /evaluate/multi-hop, /search/similar-candidates, /feedback, /jobs, /pairs/{pair_id} | Working API with 9 endpoints |
| T3.2 | 45min | `tests/test_api.py` — FastAPI TestClient tests for all endpoints (including edge cases: empty skills, missing fields, invalid pair_id, pagination) | API tests pass |
| T3.3 | 45min | `src/vector_store.py` — ChromaDB integration: embed resumes with `all-MiniLM-L6-v2`, build persistent index, semantic search with metadata filtering | Vector store module |
| T3.4 | 15min | Wire vector store into /search/similar-candidates + wire feedback into /feedback | Endpoints connected |
| T3.5 | 15min | `src/pipeline.py` — Orchestrator that runs entire pipeline end-to-end (generate → validate → label → judge → correct → analyze → index) | Pipeline script |
| T3.6 | 60min | `streamlit_app.py` — Demo: browse jobs → select → generate/analyze resume → show charts + match scores → search similar candidates. Full user journey. | Streamlit demo |
| T3.7 | 30min | ADRs: ADR-001 through ADR-005 (Instructor nested schemas, skill normalization, two-phase validation, FastAPI over Flask, ChromaDB over FAISS) | 5 ADRs |
| T3.8 | 45min | `README.md` — Problem, architecture (Mermaid diagram), results summary, API endpoint table, demo link, setup instructions | Complete README |
| T3.9 | 15min | Loom recording (2 min walkthrough) | Loom video |
| T3.10 | 15min | Final git push, update Notion Project Tracker to "Done" | Project complete |

**Git checkpoint after T3.10.** Commit message: `feat(p4): API (9 endpoints), ChromaDB, demo, documentation — P4 complete`

---

## 16. ADRs to Write

| ADR | Title | Key Point |
|-----|-------|-----------|
| ADR-001 | Instructor with max_retries=5 for nested schemas | Why 5 retries, JSON mode, and strategic Optional fields. Trade-off: more retries = higher cost but higher success rate. |
| ADR-002 | Skill normalization strategy | Why custom normalizer over library (spaCy, fuzzywuzzy). Trade-off: simpler but less flexible. |
| ADR-003 | Two-phase validation (structural vs semantic) | Why separate Instructor parsing from failure labeling. Java parallel: @Valid vs business rules. |
| ADR-004 | FastAPI over Flask | Why FastAPI: auto-generated OpenAPI docs, Pydantic integration, async support. Flask is simpler but lacks native Pydantic. |
| ADR-005 | ChromaDB over FAISS for P4 | Why ChromaDB: persistence + metadata filtering. FAISS (used in P2) is better for benchmarking. Shows knowledge of when to use each. |

---

## 17. What NOT to Build

- No database (Postgres, SQLite) — JSON/JSONL files + ChromaDB are sufficient for P4's volume
- No authentication on API — save for P5 production RAG
- No CI/CD — manual deployment only
- No frontend beyond Streamlit — no React/Vue
- No LangChain — Instructor handles structured output, ChromaDB handles vectors, no need for LangChain's abstractions
- No overlapping job deduplication — generated jobs are diverse by design
- No resume PDF rendering — JSON/text only
- No Braintrust integration unless time permits after core is complete — feedback logs to JSON as baseline
- No Logfire observability — optional, add if time permits. Logfire is OpenTelemetry-based tracing built by the Pydantic team. Valuable for production pipelines with high request volume and latency debugging. P4's batch pipeline doesn't benefit enough to justify setup time in a 3-day sprint. JSON logging is sufficient for P4's debugging needs.

### Storage Format Strategy (from Spec)

- **JSONL** for generated data (streaming-friendly, line-by-line processing)
- **JSON** for summaries and analysis results
- **CSV** for tabular exports — `failure_labels.csv` and `template_comparison.csv` for easy loading in pandas/Excel. Generated with `df.to_csv()` in analyzer.py.
- **PNG** for visualizations (widely compatible)

---

## 18. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Instructor fails too often on nested schemas | Delays generation phase | Strategic Optional fields reduce failure points. max_retries=5 provides margin. If >30% fail after 5 retries, flatten the most problematic sub-model. |
| OpenAI rate limiting during batch generation | Slows pipeline | Cache every response. Implement exponential backoff. Batch in groups of 10 with 2s delays. |
| LLM judge is slow (5-10s per pair) | Extends Day 2 significantly | Run judge in parallel batches with ThreadPoolExecutor. 250 pairs × 5s ÷ 8 threads ≈ 2.5 minutes. |
| Jaccard similarity doesn't correlate with fit level | Invalidates analysis story | This would actually be an interesting finding — document it. But likely caused by insufficient skill normalization. |
| ChromaDB dependency conflict with other packages | Blocks bonus feature | ChromaDB is a bonus. If it conflicts, skip it. Core pipeline doesn't depend on it. |
| 3-day timeline too tight | Incomplete deliverables | Day 1 and Day 2 cover all spec requirements. Day 3 is API + polish + bonus. Worst case: skip Streamlit polish and Loom, ship API + README. |

---

## 19. Pipeline Summary Output Format

```json
{
  "pipeline_run_id": "uuid",
  "timestamp": "2026-02-26T01:30:00Z",
  "generation": {
    "total_jobs": 50,
    "total_resumes": 250,
    "total_pairs": 250,
    "validation_success_rate": 0.92,
    "avg_instructor_retries": 1.8,
    "generation_time_seconds": 1200
  },
  "labeling": {
    "total_labeled": 250,
    "failure_mode_rates": {
      "skills_overlap_below_40pct": 0.35,
      "experience_mismatch": 0.28,
      "seniority_mismatch": 0.22,
      "missing_core_skills": 0.40,
      "hallucinations": 0.15,
      "awkward_language": 0.20
    },
    "avg_jaccard_by_fit_level": {
      "excellent": 0.85,
      "good": 0.68,
      "partial": 0.48,
      "poor": 0.30,
      "mismatch": 0.12
    }
  },
  "judge": {
    "total_judged": 250,
    "avg_quality_score": 0.65,
    "hallucination_rate": 0.18,
    "awkward_language_rate": 0.22,
    "judge_time_seconds": 300
  },
  "correction": {
    "total_invalid": 20,
    "total_corrected": 12,
    "correction_rate": 0.60,
    "avg_attempts_per_success": 1.8
  },
  "ab_testing": {
    "template_validation_rates": {
      "v1-formal": 0.94,
      "v1-casual": 0.88,
      "v1-technical": 0.96,
      "v1-achievement": 0.86,
      "v1-career-changer": 0.90
    },
    "chi_squared_statistic": 8.42,
    "chi_squared_p_value": 0.03,
    "significant": true,
    "best_template": "v1-technical",
    "worst_template": "v1-achievement"
  },
  "vector_store": {
    "total_indexed": 250,
    "embedding_model": "all-MiniLM-L6-v2",
    "index_build_time_seconds": 15,
    "storage_backend": "ChromaDB (persistent)"
  },
  "multi_hop": {
    "total_questions_generated": 12,
    "sections_covered": ["education", "experience", "skills", "seniority"],
    "pass_rate": 0.75
  },
  "feedback": {
    "total_feedback_entries": 0,
    "endpoint_active": true
  },
  "api": {
    "total_endpoints": 9,
    "endpoints": [
      "GET /health",
      "POST /review-resume",
      "GET /analysis/failure-rates",
      "GET /analysis/template-comparison",
      "POST /evaluate/multi-hop",
      "GET /search/similar-candidates",
      "POST /feedback",
      "GET /jobs",
      "GET /pairs/{pair_id}"
    ]
  },
  "total_processing_time_seconds": 2400,
  "estimated_api_cost_usd": 0.85
}
```

---

## 20. Interview Talking Points

After completing P4, you should be able to answer:

1. **"How did you handle deeply nested schema validation with LLMs?"** → Two-phase strategy: Instructor for structural parsing with strategic Optional fields (reduce retry cost), then semantic validation in labeling pipeline. Like Java's `@Valid` for deserialization vs business rule validation.

2. **"Why Jaccard similarity and not cosine similarity for skill matching?"** → Jaccard is set-based (binary: skill present or not). Cosine similarity is vector-based (continuous). For skill matching, the question is "does the resume have this skill?" (binary), not "how similar are these skill descriptions?" (continuous). Jaccard is the right tool. But I also use cosine similarity via ChromaDB for semantic candidate search — different question, different metric.

3. **"How do you detect hallucinations without ground truth?"** → Two layers: rule-based (impossible timelines, unrealistic claims, skill-year mismatches) catches obvious cases. LLM-as-Judge catches subtle ones (unverifiable claims, inconsistent career narratives). Comparing agreement between layers is itself a quality metric.

4. **"What did your A/B testing reveal?"** → Template X produced Y% fewer validation failures because [specific reason — e.g., achievement-focused templates generate more hallucinated metrics]. Chi-squared test confirmed this with p<0.05, so it's not random noise. This informed v2 templates.

5. **"Why FastAPI over Flask?"** → Native Pydantic integration (request validation for free), auto-generated OpenAPI docs (Swagger equivalent), async support for future scaling, dependency injection. Flask requires manual validation and separate Swagger setup. The 9 endpoints at `/docs` become a complete interactive demo.

6. **"Why ChromaDB for P4 but FAISS for P2?"** → Different requirements. P2 needed low-level control for benchmarking (IndexFlatIP vs HNSW, cosine vs L2 distance). P4 needs production features: persistence across API restarts, native metadata filtering (`where={"fit_level": "excellent"}`). Choosing the right tool for the job is the signal, not loyalty to one library.

7. **"How does this connect to your other projects?"** → P1's correction loop pattern reused for nested schemas. P2's embedding + retrieval knowledge informed ChromaDB choice. P3's SentenceTransformer lifecycle management (load → encode → del → gc.collect()) applied to vector indexing. P4 integrates all three into a production pipeline with a REST API — the first project that's a *service*, not just a *script*.
