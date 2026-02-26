# P4 Day 2 Plan — Labeling + Judge + Correction + Analysis (T2.1–T2.10)

## Context

Day 1 + Day 1.5 are complete and merged (PR #34). We have 50 jobs, 250 resumes, 250 pairs with 100% validation rate and confirmed Jaccard gradient (Excellent=73%, Good=73%, Partial=75%, Poor=8%, Mismatch=0%). Day 2 builds the full analysis pipeline: rule-based failure labeling, LLM-as-Judge, correction loop, 9 charts, and pipeline summary.

**Branch**: `feat/p4-day2-analysis` (create from main)

**Two decisions resolved:**
1. **Rule-based hallucination/awkward detection**: Implement fully per PRD 6d-6e. Chart #9 (judge vs rule-based agreement) requires both independent signals.
2. **Corrector with 0 invalid records**: Seed 8 deliberately broken resume dicts to demonstrate the loop and generate Chart #8 data.

---

## T2.1: `src/labeler.py` — All 6 Failure Modes

**New file.** ~250 lines. No API calls — pure computation.

### Constants

```python
SENIORITY_MAP: dict[str, int] = {
    "entry": 0, "junior": 0, "intern": 0, "associate": 0,
    "mid": 1, "intermediate": 1, "regular": 1,
    "senior": 2, "sr": 2,
    "lead": 3, "principal": 3, "staff": 3, "architect": 3,
    "executive": 4, "director": 4, "vp": 4, "chief": 4, "head": 4, "c-level": 4,
}
```

BUZZWORDS (35 entries) and AI_PATTERNS (9 entries) — copy verbatim from PRD lines 881-897.

### Functions

| Function | Signature | Key Logic |
|----------|-----------|-----------|
| `calculate_jaccard` | `(resume_skills: list[str], job_skills: list[str], normalizer: SkillNormalizer) -> tuple[float, int, int, set[str], set[str]]` | `normalize_set()` both, `len(intersection)/len(union)`, guard empty union=0.0. Return (score, \|intersect\|, \|union\|, resume_set, job_set) |
| `calculate_total_experience` | `(experiences: list[Experience]) -> float` | Parse `start_date[:7]` as `%Y-%m`, `end_date` or `datetime.now()`, sum months/12. `max(0, months)` guard. |
| `infer_seniority` | `(title: str, years_experience: float) -> int` | Iterate SENIORITY_MAP sorted by **descending level** (so "senior" matches before "associate" for "Senior Associate"). Fallback: <2yr→0, <5yr→1, <10yr→2, <15yr→3, 15+→4. |
| `check_experience_mismatch` | `(resume_years: float, job_years: int) -> bool` | `resume_years < job_years * 0.5 or resume_years < job_years - 3`. Return False if `job_years == 0`. |
| `check_seniority_mismatch` | `(resume_level: int, job_level: int) -> bool` | `abs(resume_level - job_level) > 1` |
| `check_missing_core_skills` | `(resume_skills_norm: set[str], job_required: list[str], normalizer: SkillNormalizer) -> tuple[bool, list[str]]` | Top-3 required (first 3 in list), normalize each, check membership. Return missing list. |
| `detect_hallucinations` | `(resume: Resume, experience_years: float) -> tuple[bool, list[str]]` | 4 rules from PRD 6d: entry+expert, unrealistic count, skill years>exp+1, senior title<5yr. |
| `detect_awkward_language` | `(resume: Resume) -> tuple[bool, list[str]]` | 3 rules from PRD 6e: buzzword>5, AI patterns>2, repeated words (>4 chars, 3+ times in 50-word window). |
| `label_pair` | `(resume: Resume, job: JobDescription, pair_id: str, normalizer: SkillNormalizer \| None = None) -> FailureLabels` | Orchestrator — calls all above, returns populated FailureLabels (18 fields). Resume seniority: latest title + total years. Job seniority: `job.title` + `job.requirements.experience_years`. |

### Reuses
- `SkillNormalizer` from `src/normalizer.py`
- `FailureLabels`, `Resume`, `JobDescription`, `Experience`, `ProficiencyLevel` from `src/schemas.py`
- Jaccard pattern from `src/sanity_check.py:70-74`

### Validation
- `label_pair()` returns valid `FailureLabels` (Pydantic validates all 18 fields)
- Jaccard for excellent-fit pairs > 0.5
- Experience years non-negative, seniority levels 0-4

---

## T2.2: `tests/test_labeler.py` — Unit Tests

**New file.** ~42 tests across 8 test classes.

### Helpers (inline, no conftest)
- `_make_experience(title, start_date, end_date, responsibilities)` → Experience
- `_make_skill(name, proficiency_level, years)` → Skill
- `_make_resume(skills, experience, summary)` → Resume
- `_make_job(required_skills, experience_years, experience_level, title)` → JobDescription

### Test Classes

| Class | Tests | Key Cases |
|-------|-------|-----------|
| `TestCalculateJaccard` | ~8 | Perfect overlap=1.0, no overlap=0.0, normalization ("Python 3.10" vs "python"), aliases ("js" vs "JavaScript"), both empty=0.0 |
| `TestCalculateTotalExperience` | ~5 | Single job 3yr, current job (end=None), multiple jobs sum, negative guarded |
| `TestInferSeniority` | ~8 | Parametrize: "Junior Dev"→0, "Senior Engineer"→2, "VP Engineering"→4, "Senior Associate"→2 (senior wins), fallback by years |
| `TestCheckExperienceMismatch` | ~5 | Parametrize: (5,5)→False, (2,5)→True, (0,0)→False |
| `TestCheckSeniorityMismatch` | ~4 | Parametrize: (0,0)→False, (0,1)→False, (0,2)→True, (4,1)→True |
| `TestDetectHallucinations` | ~5 | Normal resume→no flag, entry+5 Expert→flag, skill.years>exp→flag, VP+2yr→flag |
| `TestDetectAwkwardLanguage` | ~4 | Clean→no flag, 7 buzzwords→flag, 3 AI patterns→flag, repeated words→flag |
| `TestLabelPair` | ~3 | Excellent fit pair, poor fit pair, verify returns valid FailureLabels |

**Run**: `uv run pytest tests/test_labeler.py -v`

---

## T2.3: Run Labeling Pipeline — `src/run_labeling.py`

**New file.** ~80 lines. No API calls.

### Logic
1. Load exact files: `jobs_20260225_141615.jsonl`, `resumes_20260225_142052.jsonl`, `pairs_20260225_142052.jsonl`
2. Build lookup dicts: `{trace_id: GeneratedJob}`, `{trace_id: GeneratedResume}`
3. One shared `SkillNormalizer()` instance
4. For each of 250 pairs: `label_pair(resume.resume, job.job, pair["pair_id"], normalizer)`
5. Write all labels at once to `data/labeled/failure_labels.jsonl` (no streaming needed — no API calls to crash)
6. Print summary: failure mode counts, avg Jaccard per fit level (Rich table)

### Validation
- 250 records in `failure_labels.jsonl`
- Jaccard gradient: excellent > poor > mismatch
- No exceptions

**Run**: `uv run python -m src.run_labeling`

---

## T2.4: `src/judge.py` — LLM-as-Judge (GPT-4o)

**New file.** ~200 lines. **Makes API calls** (~$0.75 for 250 pairs).

### Key Design
- **Model**: GPT-4o (not 4o-mini) — higher quality evaluation
- **Instructor**: `max_retries=3`, `temperature=0.3` (consistent judgments)
- **Cache**: MD5 hash of `gpt-4o\n{system}\n---\n{user}` → `data/cache/{hash}.json`. Same pattern as `src/generator.py:52-117`
- **Threading**: `ThreadPoolExecutor(max_workers=4)`, rate limit every 10 calls with `time.sleep(2)`
- **pair_id injection**: Set after LLM returns (LLM doesn't know pair_id)

### Functions

| Function | Signature | Notes |
|----------|-----------|-------|
| `_create_judge_client` | `() -> instructor.Instructor` | Same pattern as generator's `_create_client()` |
| `_build_judge_prompt` | `(job: JobDescription, resume: Resume) -> tuple[str, str]` | PRD 7a prompt. User prompt includes `job.model_dump_json(indent=2)` and `resume.model_dump_json(indent=2)` |
| `judge_pair` | `(client, job, resume, pair_id, *, use_cache=True) -> JudgeResult` | Cache check → API call → inject pair_id → cache save |
| `judge_batch` | `(pairs_data, jobs, resumes, *, max_workers=4) -> list[JudgeResult]` | ThreadPoolExecutor, rate limiting |
| `save_judge_results` | `(results: list[JudgeResult]) -> Path` | Write to `data/labeled/judge_results.jsonl` |

### Validation
- 250 records in `judge_results.jsonl`
- All `overall_quality_score` in [0.0, 1.0]
- Cost ~$0.75

---

## T2.5: `src/corrector.py` — Correction Loop

**New file.** ~200 lines. **Makes API calls** for seeded broken records only.

### Seeded Broken Records (8 records)

| # | Error Type | What's Wrong | Schema Validator That Catches It |
|---|-----------|-------------|----------------------------------|
| 1 | Invalid email | `"john@company"` (no TLD) | `ContactInfo.validate_email` |
| 2 | Malformed date | `"March 2020"` not `"2020-03"` | `Education.validate_date_format` |
| 3 | GPA out of range | `85.0` (percentage) | `Education.validate_gpa` (0-4) |
| 4 | Phone too short | `"123456"` (6 digits) | `ContactInfo.validate_phone` (>=10) |
| 5 | End before start | `start="2023-01", end="2020-01"` | `Experience.validate_date_order` |
| 6 | Empty responsibilities | `[]` | `Experience` min_length=1 |
| 7 | Skill years out of range | `years=35` | `Skill` (0-30) |
| 8 | Compound: bad email + bad date + high GPA | Multiple errors | Tests harder correction case |

### Functions

| Function | Signature | Notes |
|----------|-----------|-------|
| `extract_validation_errors` | `(raw_data: dict, model_class: type) -> list[dict]` | Try `model_validate()`, catch `ValidationError`, return `[{field_path, error_message, invalid_value, expected_format}]` |
| `build_correction_prompt` | `(raw_data: dict, errors: list[dict], record_type: str) -> tuple[str, str]` | PRD 8b prompt template |
| `correct_record` | `(client, raw_data, record_type, max_attempts=3) -> CorrectionResult` | Loop: extract errors → prompt → Instructor(Resume) → re-validate → retry |
| `generate_seeded_broken_records` | `() -> list[dict]` | Returns 8 deliberately broken resume dicts |
| `correct_batch` | `(invalid_records, record_type, client) -> tuple[list[CorrectionResult], CorrectionSummary]` | Process all, build summary |
| `save_correction_results` | `(results, summary) -> tuple[Path, Path]` | `data/corrected/correction_results.jsonl` + `data/corrected/correction_summary.json` |

### Correction Loop (per record)
1. `extract_validation_errors(raw, Resume)` → error list
2. `build_correction_prompt(raw, errors)` → system + user prompts
3. `client.chat.completions.create(model="gpt-4o-mini", response_model=Resume, messages=...)`
4. If Instructor succeeds → `corrected_successfully=True`
5. If fails → increment attempt, re-extract remaining errors, retry (up to 3)
6. Track as `CorrectionResult`

### Validation
- Runs on 8 seeded records without crashing
- Correction rate > 50% (target: 6+/8)
- Output in `data/corrected/`

**Run**: `uv run python -m src.corrector`

---

## T2.6: `tests/test_corrector.py` — Correction Tests

**New file.** ~15 tests.

| Class | Tests | Notes |
|-------|-------|-------|
| `TestExtractValidationErrors` | 4 | Valid→empty, invalid email→error with path, multiple errors, GPA out of range |
| `TestBuildCorrectionPrompt` | 3 | Contains error details, contains original data, has fix instruction |
| `TestGenerateSeededBrokenRecords` | 2 | Returns 8 records, all fail `Resume.model_validate()` |
| `TestCorrectRecord` | 4 | Mock LLM: success 1st try, success 2nd try, max retries exceeded, preserves valid fields |
| `TestCorrectBatch` | 2 | Summary stats correct, common_failure_reasons populated |

**Mocking**: `unittest.mock.patch` on Instructor's create method. Return pre-built valid `Resume` for success, raise `ValidationError` for failure.

**Run**: `uv run pytest tests/test_corrector.py -v`

---

## T2.7: Run Correction Pipeline

Not a separate file — `src/corrector.py` has `if __name__ == "__main__"` block.

1. `generate_seeded_broken_records()` → 8 broken dicts
2. `correct_batch(broken)` → results + summary
3. `save_correction_results(results, summary)`
4. Print summary via Rich

**Run**: `uv run python -m src.corrector`

---

## T2.8: `src/analyzer.py` — DataFrame + 9 Charts + A/B Chi-Squared

**New file.** ~450 lines. Largest module of Day 2.

### DataFrame Construction

`build_analysis_dataframe() -> pd.DataFrame`

Join 5 data sources on `pair_id`:
- `data/generated/pairs_*.jsonl` → pair_id, fit_level, resume_trace_id, job_trace_id
- `data/generated/resumes_*.jsonl` → trace_id → writing_style (for template column)
- `data/generated/jobs_*.jsonl` → trace_id → is_niche_role, company.industry
- `data/labeled/failure_labels.jsonl` → pair_id → all 6 failure flags + metrics
- `data/labeled/judge_results.jsonl` → pair_id → quality_score, judge hallucination/awkward flags

Computed columns: `template` (from `prompt_template`), `is_niche`, `industry`, `total_flags` (sum of booleans).

### 9 Charts

| # | Function | Output File | Chart Type |
|---|----------|-------------|------------|
| 1 | `plot_failure_correlation(df)` | `failure_correlation.png` | seaborn heatmap, `df[failure_cols].corr()`, coolwarm colormap |
| 2 | `plot_failure_rates_by_fit(df)` | `failure_by_fit_level.png` | seaborn barplot, grouped by fit_level (ordered: excellent→mismatch) |
| 3 | `plot_failure_rates_by_template(df)` | `failure_by_template.png` | seaborn barplot, grouped by template (5 templates) |
| 4 | `plot_niche_vs_standard(df)` | `niche_vs_standard.png` | seaborn barplot with hue=is_niche |
| 5 | `plot_validation_summary()` | `validation_summary.png` | Annotated bar chart: "300/300 records valid — 0 field errors". Single bar at 100% with bold annotation. No heatmap (0 errors = empty). |
| 6 | `plot_skills_overlap_distribution(df)` | `skills_overlap_distribution.png` | seaborn boxplot, Jaccard by fit_level — most important chart |
| 7 | `plot_hallucination_by_seniority(df)` | `hallucination_by_seniority.png` | seaborn barplot grouped by seniority_level_resume (0-4) |
| 8 | `plot_correction_success()` | `correction_success.png` | matplotlib barh from `correction_summary.json` |
| 9 | `plot_judge_vs_rules_agreement(df)` | `judge_vs_rules_agreement.png` | seaborn heatmap, 2x2 confusion matrices for hallucination + awkward language |

### Chi-Squared A/B Test

`compute_template_ab_test(df) -> dict`

- Contingency table: rows=templates, columns=[failed, not_failed]
- "Failed" = any of 6 failure flags True
- `scipy.stats.chi2_contingency(table)` → chi2, p_value, dof, expected
- Return: `{chi_squared_statistic, p_value, significant (p<0.05), best_template, worst_template}`

### Styling
- `sns.set_theme(style="whitegrid", font_scale=1.1)`
- `figure.dpi=150`, `savefig.bbox="tight"`
- `matplotlib.use("Agg")` for non-interactive backend

### Orchestrator
`generate_all_charts() -> list[Path]` — calls all 9 chart functions, returns paths.

**Run**: `uv run python -m src.analyzer`

---

## T2.9: Pipeline Summary — in `src/analyzer.py`

`generate_pipeline_summary(df, ab_results) -> dict`

Sections from PRD Section 19:
- `generation`: totals from generated data + `validation_stats.json`
- `labeling`: failure_mode_rates + avg_jaccard_by_fit_level from DataFrame
- `judge`: avg_quality_score, hallucination_rate, awkward_language_rate from judge results
- `correction`: from `correction_summary.json`
- `ab_testing`: from chi-squared results
- `vector_store`, `multi_hop`, `feedback`, `api`: placeholders (Day 3)

Save to: `results/pipeline_summary.json`

---

## T2.10: Multi-Hop Evaluation Questions

**Add to bottom of `src/labeler.py`** (or separate `src/multi_hop.py` if labeler gets too long).

```python
MULTI_HOP_TEMPLATES = [
    "Does this candidate's education level ({education}) align with the job's required experience level ({experience_level})?",
    "Are the claimed skills ({top_skills}) consistent with the job titles ({job_titles}) and years of experience ({years})?",
    "Given the career progression from {first_title} to {last_title} over {total_years} years, is the claimed seniority level realistic?",
    "Does the candidate's industry background ({industries}) provide transferable skills for the target role in {target_industry}?",
]
```

`generate_multi_hop_questions(resume, job, labels) -> list[MultiHopQuestion]`

Generate for 10+ pairs (2-3 per fit level). Save to `data/analysis/multi_hop_questions.jsonl`.

---

## Execution Order & Git Commits

```
1. Create branch feat/p4-day2-analysis from main
2. T2.1 → T2.2 → run tests
   Commit: feat(p4): labeler with 6 failure modes + tests
3. T2.3 → run labeling
   Commit: feat(p4): run labeling pipeline on 250 pairs
4. T2.5 → T2.6 → run tests → T2.7
   Commit: feat(p4): corrector with seeded broken records + tests
5. T2.4 → run judge ($0.75 API cost)
   Commit: feat(p4): LLM-as-Judge with caching and threading
6. T2.8 → T2.9
   Commit: feat(p4): analyzer with 9 charts + A/B chi-squared + pipeline summary
7. T2.10
   Commit: feat(p4): multi-hop evaluation questions
8. Update CLAUDE.md Day 2 checkboxes → [x]
   Commit: chore(p4): update CLAUDE.md current state after Day 2
9. Push branch → create PR
```

---

## Files Modified/Created

| File | Action | Lines (est) |
|------|--------|-------------|
| `src/labeler.py` | CREATE | ~280 |
| `src/judge.py` | CREATE | ~200 |
| `src/corrector.py` | CREATE | ~220 |
| `src/analyzer.py` | CREATE | ~450 |
| `src/run_labeling.py` | CREATE | ~80 |
| `tests/test_labeler.py` | CREATE | ~350 |
| `tests/test_corrector.py` | CREATE | ~200 |
| `CLAUDE.md` | EDIT | Update Day 2 checkboxes |
| `results/charts/*.png` | CREATE | 9 charts |
| `data/labeled/failure_labels.jsonl` | CREATE | 250 records |
| `data/labeled/judge_results.jsonl` | CREATE | 250 records |
| `data/corrected/correction_results.jsonl` | CREATE | 8 records |
| `data/corrected/correction_summary.json` | CREATE | 1 file |
| `data/analysis/multi_hop_questions.jsonl` | CREATE | 10+ records |
| `results/pipeline_summary.json` | CREATE | 1 file |

**Day 1 files (DO NOT MODIFY):** schemas.py, normalizer.py, templates.py, generator.py, validator.py, sanity_check.py, run_generation.py

---

## Verification

1. `uv run pytest tests/test_labeler.py tests/test_corrector.py -v` — all pass
2. `uv run pytest tests/ -v` — full suite still passes (no regressions)
3. `uv run python -m src.run_labeling` — 250 labels, Jaccard gradient visible
4. `uv run python -m src.corrector` — 8 seeded records, >50% correction rate
5. `uv run python -m src.judge` (or integrated into run script) — 250 judge results
6. `uv run python -m src.analyzer` — 9 PNGs in results/charts/, pipeline_summary.json
7. Spot-check: open 2-3 charts to verify legibility and correct labels
8. `ruff check src/ tests/` — no lint errors
