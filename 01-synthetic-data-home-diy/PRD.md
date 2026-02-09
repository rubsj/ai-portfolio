# PRD: P1 — Synthetic Data Generation for Home DIY Repair

> **This is the implementation contract.** Claude Code: read this + both CLAUDE.md files before starting.
> Do NOT re-debate architecture decisions. They are final. If something is ambiguous, ask the user.

**Project:** P1 — Synthetic Data, Home DIY Repair
**Timeline:** Feb 8–11, 2026 (Sun–Wed)
**Owner:** Developer (Java/TS background, learning Python)
**Source of Truth:** [Notion Requirements](https://www.notion.so/Mini_Project_1_Requirements-2ffdb630640a8109a98cfae432d4d1e9)

---

## 1. Objective

Build an end-to-end pipeline that:
1. **Generates** 30 synthetic Home DIY Repair Q&A records using GPT-4o-mini via Instructor
2. **Validates** them structurally with Pydantic
3. **Labels failures** using both manual review and LLM-as-Judge (GPT-4o), then compares agreement
4. **Analyzes** failure patterns via heatmap and correlation analysis
5. **Corrects** failed records via second-pass prompts AND improves the original prompt templates
6. **Re-evaluates** to demonstrate >80% failure rate reduction

**Success Criteria:** Correction loop reduces failure rate by >80%.

---

## 2. Architecture Decisions (FINAL — Do Not Re-Debate)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Instructor mode** | `instructor.from_openai()` | Cleanest API; Instructor manages the OpenAI client. Current recommended approach. |
| **Data schema** | Flat model — exactly 7 fields | Matches spec. No nested sub-models. Keep P1 simple. |
| **Generation model** | GPT-4o-mini | Cheap ($0.15/1M input). Good enough for generation. |
| **Judge model** | GPT-4o | Expensive ($2.50/1M input). Higher quality evaluation. |
| **Labeling approach** | Both manual + LLM-as-Judge | Manual label first 10, LLM labels all 30, compare inter-rater agreement. Strongest portfolio artifact. |
| **Generation volume** | 30 records (5 categories × 2 per category × 3 difficulties) | Richer analysis than spec minimum of 20. Clean matrix. |
| **Caching** | JSON file cache in `data/cache/` | Simple, zero-dependency, inspectable. Keyed on MD5 of prompt. |
| **Correction scope** | Full loop — fix individual records AND improve templates | Individual correction meets spec. Template improvement is the portfolio differentiator. |

---

## 3. Data Schema

One flat Pydantic model. **7 required fields**, exactly as the requirements spec defines:

```python
class DIYRepairRecord(BaseModel):
    question: str           # A clear question a homeowner would ask
    answer: str             # Detailed step-by-step answer
    equipment_problem: str  # Specific problem/equipment description
    tools_required: list[str]  # List of tools needed
    steps: list[str]        # Ordered repair steps
    safety_info: str        # Safety warnings and precautions
    tips: str               # Helpful practical tips
```

**Additional fields to add for tracking (not sent to LLM, populated by pipeline):**

```python
class GeneratedRecord(BaseModel):
    """Wraps DIYRepairRecord with metadata for tracking."""
    trace_id: str              # Auto-generated UUID
    category: str              # Which of 5 categories
    difficulty: str            # beginner | intermediate | advanced
    template_version: str      # "v1" initially, "v2" after template improvement
    generation_timestamp: str  # ISO datetime
    model_used: str            # "gpt-4o-mini"
    prompt_hash: str           # Cache key
    record: DIYRepairRecord    # The actual generated data
```

### Pydantic Validators to Implement

| Field | Validation | Why |
|-------|-----------|-----|
| `question` | `min_length=10`, must end with `?` | LLM sometimes generates statements not questions |
| `answer` | `min_length=50` | Catches truncated or incomplete answers |
| `equipment_problem` | `min_length=5` | Must be specific, not empty/generic |
| `tools_required` | `min_length=1`, each tool `min_length=2` | Must have at least one tool, no empty strings |
| `steps` | `min_length=2`, each step `min_length=10` | A repair needs at least 2 steps with real content |
| `safety_info` | `min_length=10` | Safety can't be empty or trivial |
| `tips` | `min_length=5` | Must have substance |

---

## 4. Generation Pipeline

### 4a. Prompt Templates

**5 templates**, one per category. Each template has:
- A **system prompt** with the expert persona
- A **user prompt** requesting a Q&A pair for a specific difficulty level
- The Pydantic model's JSON schema embedded (Instructor handles this automatically)

```
Template Structure:
┌─────────────────────────────────────────────────────┐
│ SYSTEM: You are a {persona} with {experience}.      │
│ You specialize in {focus_area}.                     │
│ Emphasis: {emphasis_points}                         │
│                                                     │
│ USER: Generate a {difficulty}-level DIY repair Q&A  │
│ for {category}. The homeowner should be able to     │
│ follow your instructions safely.                    │
│                                                     │
│ Include realistic tools, clear steps, and safety    │
│ warnings. For {difficulty} level, adjust complexity │
│ accordingly.                                        │
└─────────────────────────────────────────────────────┘
```

**Category × Persona mapping:**

| Category | System Persona | Emphasis |
|----------|---------------|----------|
| `appliance_repair` | "Expert home appliance repair technician with 20+ years of experience" | Technical details and practical homeowner solutions |
| `plumbing_repair` | "Professional plumber with extensive residential experience" | Safety for homeowner attempts and realistic solutions |
| `electrical_repair` | "Licensed electrician specializing in safe home electrical repairs" | Critical safety warnings and when to call professionals |
| `hvac_maintenance` | "HVAC technician specializing in homeowner maintenance" | Seasonal considerations and maintenance best practices |
| `general_home_repair` | "Skilled handyperson with general home repair expertise" | Material specifications and practical DIY solutions |

**Difficulty modifiers:**
- `beginner`: Simple tasks, common tools, minimal risk. Steps should be very detailed and assume no prior experience.
- `intermediate`: Moderate complexity, may need specialty tools. Some prior DIY experience assumed.
- `advanced`: Complex tasks, specialized tools, higher risk. Experienced DIY-ers only. Must emphasize when to call a professional.

### 4b. Generation Matrix

| Category | Beginner | Intermediate | Advanced | Total |
|----------|----------|-------------|----------|-------|
| appliance_repair | 2 | 2 | 2 | 6 |
| plumbing_repair | 2 | 2 | 2 | 6 |
| electrical_repair | 2 | 2 | 2 | 6 |
| hvac_maintenance | 2 | 2 | 2 | 6 |
| general_home_repair | 2 | 2 | 2 | 6 |
| **Total** | 10 | 10 | 10 | **30** |

### 4c. Instructor Integration

```python
import instructor
from openai import OpenAI

# instructor.from_openai() wraps the client — it handles:
# - Injecting the Pydantic model's JSON schema into the API call
# - Parsing the response into the Pydantic model
# - Automatic retries (default 3) when validation fails
client = instructor.from_openai(OpenAI())

record = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=DIYRepairRecord,  # Instructor uses this for schema + validation
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    temperature=0.7,
    max_retries=3,  # Instructor retries on ValidationError automatically
)
```

**WHY Instructor over raw OpenAI:**
- Instructor calls `model_validate_json()` internally — no manual parsing
- Auto-retries with the validation error fed back to the LLM (self-healing)
- `response_model` parameter generates and sends the JSON schema automatically
- Less boilerplate = less bugs in P1, more time for analysis

### 4d. Caching Layer

```
Before each LLM call:
1. Build full prompt string (system + user)
2. Hash it: cache_key = hashlib.md5(prompt.encode()).hexdigest()
3. Check data/cache/{cache_key}.json
4. If exists → load and return (skip API call)
5. If not → call API, save response to data/cache/{cache_key}.json, return
```

Cache file format:
```json
{
  "cache_key": "abc123...",
  "prompt_hash": "abc123...",
  "category": "plumbing_repair",
  "difficulty": "beginner",
  "model": "gpt-4o-mini",
  "timestamp": "2026-02-08T10:30:00Z",
  "response": { ...the DIYRepairRecord as dict... }
}
```

**IMPORTANT:** Instructor manages the API call internally. You'll need to implement caching AROUND the Instructor call — check cache before calling `client.chat.completions.create()`, and cache the validated result after.

---

## 5. Validation Pipeline

### 5a. Structural Validation (Pydantic)

Instructor handles this during generation (auto-retries on failure). But we still need to:
1. **Track** validation attempts vs successes (how many retries did Instructor need?)
2. **Log** any records that fail after max retries
3. **Report** overall generation success rate

Output: `data/validated/validated_records.json` (list of valid GeneratedRecord) and `data/validated/rejected_records.json` (any that failed after 3 retries).

### 5b. Metrics to Track

| Metric | Target | How |
|--------|--------|-----|
| Generation success rate | >90% | Valid records / total attempts |
| First-attempt success rate | Track but no target | Records that passed without retry |
| Per-field error frequency | Report | Which fields cause most retries |

---

## 6. Failure Labeling

### 6a. The 6 Failure Modes

| Code | Name | Detection Criteria |
|------|------|-------------------|
| `incomplete_answer` | Incomplete Answer | Answer < 3 sentences, lacks key information for the repair |
| `safety_violations` | Safety Violations | Missing power-off/water-off warning for relevant repairs, no PPE mention for hazardous tasks |
| `unrealistic_tools` | Unrealistic Tools | Tools a typical homeowner wouldn't own (oscilloscope, pipe threading machine, etc.) |
| `overcomplicated_solution` | Overcomplicated Solution | Beginner-tagged repair requiring professional skills, too many steps for stated difficulty |
| `missing_context` | Missing Context | Answer doesn't reference the specific equipment_problem, generic advice not tied to scenario |
| `poor_quality_tips` | Poor Quality Tips | Tips are obvious ("be careful"), contradicts safety_info, or dangerous |

### 6b. Dual Labeling Approach

**Step 1: Manual Labeling (first 10 records)**
- Developer reviews 10 records manually
- Labels each with 6 binary failure modes (0/1)
- Stores in `data/labels/manual_labels.csv`
- Purpose: establishes ground truth for calibration

**Step 2: LLM-as-Judge (all 30 records)**
- Send each record to GPT-4o with a structured evaluation prompt
- GPT-4o returns binary labels for all 6 failure modes + reasoning
- Stores in `data/labels/llm_labels.csv`

**Step 3: Agreement Analysis**
- For the 10 records that have both manual and LLM labels:
  - Compute **per-mode agreement rate** (% of matching labels)
  - Compute **Cohen's Kappa** if possible (accounts for chance agreement)
  - Report which failure modes the LLM judges well vs poorly
- This becomes a portfolio talking point: "I validated my LLM evaluation against human labels and found X% agreement"

### 6c. LLM-as-Judge Prompt Structure

```
System: You are a quality evaluator for DIY home repair guides.
Evaluate the following repair guide for these 6 failure modes.
For each mode, respond with 0 (pass) or 1 (fail) and a brief reason.

User: 
Category: {category}
Difficulty: {difficulty}

Record:
{record as JSON}

Evaluate for:
1. incomplete_answer (0/1): Is the answer detailed enough?
2. safety_violations (0/1): Are safety precautions adequate?
3. unrealistic_tools (0/1): Would a homeowner have these tools?
4. overcomplicated_solution (0/1): Is complexity appropriate for {difficulty}?
5. missing_context (0/1): Does the answer address the specific problem?
6. poor_quality_tips (0/1): Are the tips actually helpful?
```

Response model (Pydantic, via Instructor):
```python
class FailureLabel(BaseModel):
    mode: str
    label: Literal[0, 1]
    reason: str

class JudgeResult(BaseModel):
    trace_id: str
    labels: list[FailureLabel]  # exactly 6 items
    overall_quality_score: int  # 1-5 scale
```

---

## 7. Analysis

### 7a. Pandas DataFrame Structure

```
| trace_id | category | difficulty | incomplete_answer | safety_violations | unrealistic_tools | overcomplicated | missing_context | poor_tips | total_failures |
```

### 7b. Visualizations to Produce

| Chart | Library | File |
|-------|---------|------|
| **Failure mode heatmap** — records × failure modes | seaborn `heatmap()` | `results/charts/failure_heatmap.png` |
| **Failure frequency bar chart** — count per mode | matplotlib `bar()` | `results/charts/failure_frequency.png` |
| **Correlation matrix** — failure mode co-occurrence | seaborn `heatmap()` on `df.corr()` | `results/charts/failure_correlation.png` |
| **Category breakdown** — failures by repair category | seaborn `countplot()` or grouped bar | `results/charts/category_failures.png` |
| **Difficulty breakdown** — failures by difficulty level | matplotlib grouped bar | `results/charts/difficulty_failures.png` |
| **Agreement matrix** — manual vs LLM labels (10 records) | seaborn heatmap or simple table | `results/charts/agreement_matrix.png` |

### 7c. Key Questions the Analysis Should Answer

1. Which failure mode is most common across all records?
2. Do certain categories have higher failure rates? (e.g., does electrical_repair have more safety_violations?)
3. Do certain difficulties have different failure profiles? (e.g., does advanced have more overcomplicated_solution?)
4. Which failure modes co-occur? (e.g., does missing_context correlate with incomplete_answer?)
5. How well does GPT-4o-as-judge agree with manual labels?

---

## 8. Correction Loop

### 8a. Individual Record Correction (Required by Spec)

For each record that has ≥1 failure:
1. Build a correction prompt that includes:
   - The original record
   - The specific failure modes flagged
   - The judge's reasoning for each failure
2. Send to GPT-4o-mini via Instructor (same model, corrective prompt)
3. Validate the corrected record
4. Re-evaluate with GPT-4o judge
5. Track: did the failure modes get resolved?

```
Correction Prompt:
System: You are a quality improvement specialist for DIY repair guides.

User: The following repair guide was flagged with quality issues.
Fix ONLY the flagged issues while preserving the overall structure.

Original record: {record JSON}

Flagged issues:
- safety_violations: "Missing warning to turn off circuit breaker before electrical work"
- overcomplicated_solution: "Steps require wire fishing tool, unusual for beginner task"

Generate a corrected version that addresses these specific issues.
```

### 8b. Template Improvement (Portfolio Enhancement)

After analyzing which failure modes are most common per category:
1. Identify the top 2-3 failure patterns per template
2. Add explicit instructions to the template to prevent those failures
3. Mark improved templates as `v2`
4. Re-generate a batch using `v2` templates
5. Compare v1 vs v2 failure rates

Example: If `electrical_repair` consistently triggers `safety_violations`:
- v1 template: "Generate a DIY electrical repair guide"
- v2 template: "Generate a DIY electrical repair guide. CRITICAL: Always include a warning to turn off the circuit breaker at the panel before starting. Always specify when the homeowner should call a licensed electrician instead."

### 8c. Success Measurement

```
initial_failure_rate = total_failures_v1 / (30 records × 6 modes)
corrected_failure_rate = total_failures_after_correction / (same denominator)
improvement = (initial - corrected) / initial × 100

Target: improvement > 80%
```

Track this at two levels:
- **Per-record correction:** How many individual records went from failing to passing?
- **Per-template improvement:** How much did v2 templates reduce failures vs v1?

---

## 9. File Structure

```
01-synthetic-data-home-diy/
├── CLAUDE.md                          # Project-specific Claude Code memory
├── PRD.md                             # THIS FILE — implementation contract
├── pyproject.toml                     # Dependencies (add: instructor)
├── src/
│   ├── __init__.py
│   ├── schemas.py                     # DIYRepairRecord, GeneratedRecord, JudgeResult models
│   ├── templates.py                   # 5 prompt templates (v1 and v2)
│   ├── generator.py                   # Instructor-based generation + caching
│   ├── validator.py                   # Validation tracking + rejection logging
│   ├── evaluator.py                   # LLM-as-Judge + manual label loading + agreement
│   ├── corrector.py                   # Individual correction + template improvement
│   └── analysis.py                    # Pandas analysis + chart generation
├── tests/
│   ├── __init__.py
│   ├── test_schemas.py                # Schema validation tests
│   ├── test_generator.py              # Generation tests (with mocked API)
│   └── test_evaluator.py              # Evaluator tests
├── data/
│   ├── cache/                         # LLM response cache (JSON files)
│   ├── generated/                     # Raw generated records
│   ├── validated/                     # Validated + rejected records
│   ├── labels/                        # manual_labels.csv + llm_labels.csv
│   └── corrected/                     # Corrected records (v1 corrections + v2 generations)
├── results/
│   ├── charts/                        # All PNG visualizations
│   └── metrics.json                   # Summary metrics (success rate, failure rates, improvement %)
├── docs/
│   └── adr/                           # Architecture Decision Records
├── streamlit_app.py                   # Demo app (Wednesday)
└── README.md                          # Project documentation (Wednesday)
```

### New file vs original scaffolding:
- **Added:** `templates.py` (separates prompt templates from generator logic), `corrector.py` (correction loop is its own module)
- **Renamed:** None
- **Removed:** None

---

## 10. Dependencies

Add to `pyproject.toml`:
```
instructor    # Structured LLM output via Pydantic (THE key library for this project)
```

Already installed: `openai`, `pydantic`, `pandas`, `matplotlib`, `seaborn`, `python-dotenv`, `streamlit`, `pytest`, `ruff`

---

## 11. Implementation Order (for Claude Code to break into tasks)

> **Schedule note:** Sunday session started late (10:30 PM). Only 2.5h available tonight.
> Remaining work redistributed across Mon–Wed with extended sessions.
> Total budget: 2.5h (Sun) + 5h (Mon) + 5h (Tue) + 5h (Wed) = 17.5h

**Sunday Feb 8 (2.5h — 10:30 PM to 1:00 AM):**
1. Add `instructor` dependency: `uv add instructor`
2. `src/schemas.py` — DIYRepairRecord, GeneratedRecord, JudgeResult, FailureLabel models with validators
3. `src/templates.py` — 5 prompt templates (v1) with category/persona/emphasis
4. `tests/test_schemas.py` — Valid/invalid data tests

**Monday Feb 9 (5h — 9:00 PM to 2:00 AM):**
5. `src/generator.py` — Instructor-based generation + JSON cache + batch generation (30 records)
6. Run first batch: generate 30 records, save to `data/generated/`
7. `src/validator.py` — Validation tracking, success rate calculation, rejection logging
8. `src/evaluator.py` — LLM-as-Judge with GPT-4o, JudgeResult parsing
9. Manual labeling: developer labels first 10 records (CSV)
10. Run LLM judge on all 30 records

**Tuesday Feb 10 (5h — 9:00 PM to 2:00 AM):**
11. `src/analysis.py` — DataFrame construction, heatmap, correlation matrix, all charts
12. Agreement analysis: manual vs LLM labels comparison
13. `src/corrector.py` — Individual record correction loop
14. Template v2 creation based on failure patterns
15. Re-generation + re-evaluation, measure improvement %
16. Write ADRs: ADR-001 (Instructor), ADR-002 (flat schema), ADR-003 (dual labeling), ADR-004 (template improvement)

**Wednesday Feb 11 (5h — 9:00 PM to 2:00 AM):**
17. `streamlit_app.py` — Demo app
18. `README.md` — Problem, architecture, results, demo link
19. Loom recording (2 min)
20. Final git push, update Notion Project Tracker to "Done"

---

## 12. ADRs to Write

| ADR | Title | When |
|-----|-------|------|
| ADR-001 | Why Instructor over raw OpenAI API | Sunday (after implementing) |
| ADR-002 | Why flat schema matching spec over nested models | Sunday (after schemas) |
| ADR-003 | Dual labeling — manual + LLM agreement as evaluation strategy | Monday (after labeling) |
| ADR-004 | Template improvement as correction strategy | Tuesday (after correction loop) |

---

## 13. What NOT to Build

- No database (SQLite, Postgres) — JSON files are sufficient for P1
- No API endpoint (FastAPI) — save for P5
- No CLI (Click) — save for P5
- No Braintrust integration — save for P2
- No nested Pydantic sub-models — flat schema per decision
- No deployment beyond Streamlit Community Cloud
