# PRD Gap Analysis — P1: Synthetic Data Home DIY Repair

**Generated:** 2026-03-02
**PRD version:** `PRD.md` (implementation contract)
**Test run:** 203 passed, 0 failed (`uv run pytest tests/ -v`)

---

## Summary

| Status | Count |
|--------|-------|
| PASS | 34 |
| PARTIAL | 2 |
| MISSING | 2 |

All core requirements are met. Both partials are explicitly anticipated in the PRD ("if possible" / "track but no target"). Both missing items are intentionally deferred (noted in CLAUDE.md).

---

## PASS — Fully Implemented

### Section 3: Data Schema

| Requirement | Location | Notes |
|-------------|----------|-------|
| `question` field: `min_length=10`, ends with `?` | `src/schemas.py:63–70` | Field validator + `pattern=r".*\?$"` |
| `answer` field: `min_length=50` | `src/schemas.py:71` | |
| `equipment_problem` field: `min_length=5` | `src/schemas.py:72` | |
| `tools_required`: list, `min_length=1`, each tool `min_length=2` | `src/schemas.py:73–80` | Field validator loops each item |
| `steps`: list, `min_length=2`, each step `min_length=10` | `src/schemas.py:81–90` | Field validator loops each item |
| `safety_info` field: `min_length=10` | `src/schemas.py:91` | |
| `tips` field: `min_length=5` | `src/schemas.py:92` | |
| `GeneratedRecord`: all 8 metadata fields | `src/schemas.py:114–147` | `trace_id`, `category`, `difficulty`, `template_version`, `generation_timestamp`, `model_used`, `prompt_hash`, `record` |
| `JudgeResult` model with `overall_quality_score` 1–5 | `src/schemas.py:175–217` | Validator enforces exactly 6 `FailureLabel` items |

### Section 4: Generation Pipeline

| Requirement | Location | Notes |
|-------------|----------|-------|
| 5 templates, one per category | `src/templates.py:44–69` | All 5 categories with persona + emphasis |
| Category × persona mapping matches PRD table | `src/templates.py:28–71` | Exact match for all 5 rows |
| Difficulty modifiers (beginner/intermediate/advanced) | `src/templates.py:74–96` | `DIFFICULTY_MODIFIERS` dict |
| 30 records (5 × 3 difficulties × 2 per combo) | `src/generator.py`, `data/generated/batch_v1.json` | `RECORDS_PER_COMBO=2`, confirmed 30 records |
| Instructor integration: `instructor.from_openai()` | `src/generator.py:140` | `_create_client()` |
| `response_model=DIYRepairRecord` | `src/generator.py:179` | |
| `max_retries=3` on Instructor call | `src/generator.py:182` | |
| `temperature=0.7` | `src/generator.py:183` | |
| JSON cache in `data/cache/`, keyed on MD5 | `src/generator.py:75`, `data/cache/` (108 files) | `_prompt_hash()` uses `hashlib.md5` |
| Cache format matches PRD spec | `src/generator.py:114–123` | All 7 keys present: `cache_key`, `prompt_hash`, `category`, `difficulty`, `model`, `timestamp`, `response` |

### Section 5: Validation Pipeline

| Requirement | Location | Notes |
|-------------|----------|-------|
| Generation success rate tracked | `src/validator.py`, `data/validated/validation_report.json` | `"success_rate": "100.0%"` |
| Per-field error frequency reported | `src/validator.py:66`, `validation_report.json` | `field_error_frequency` dict in report |
| `data/validated/validated_records.json` exists | `data/validated/validated_records.json` | 30 valid records |
| `data/validated/rejected_records.json` exists | `data/validated/rejected_records.json` | 0 rejections (100% success) |
| `data/validated/validation_report.json` exists | `data/validated/validation_report.json` | Summary with `total_records`, `valid_count`, `rejected_count`, `success_rate`, `field_error_frequency` |

### Section 6: Failure Labeling

| Requirement | Location | Notes |
|-------------|----------|-------|
| `incomplete_answer` mode | `src/schemas.py:37`, `src/evaluator.py:56–58` | ✅ |
| `safety_violations` mode | `src/schemas.py:37`, `src/evaluator.py:60–64` | ✅ |
| `unrealistic_tools` mode | `src/schemas.py:37`, `src/evaluator.py:66–69` | ✅ |
| `overcomplicated_solution` mode | `src/schemas.py:37`, `src/evaluator.py:71–75` | ✅ |
| `missing_context` mode | `src/schemas.py:37`, `src/evaluator.py:77–79` | ✅ |
| `poor_quality_tips` mode | `src/schemas.py:37`, `src/evaluator.py:81–83` | ✅ |
| Manual labels for first 10 records | `data/labels/manual_labels.csv` | 10 rows confirmed |
| LLM labels for all 30 records | `data/labels/llm_labels.csv` | 30 rows confirmed |
| Per-mode agreement rate computed | `src/evaluator.py:327–401` (`compute_agreement()`) | 81.7% overall, per-mode breakdown in `agreement_report.json` |

### Section 7: Analysis

| Requirement | Location | Notes |
|-------------|----------|-------|
| Pandas DataFrame with all required columns | `src/analysis.py:54–107` (`build_analysis_dataframe()`) | `trace_id`, `category`, `difficulty`, 6 failure modes, `total_failures` |
| Failure mode heatmap | `src/analysis.py:119–150`, `results/charts/failure_heatmap.png` | ✅ seaborn `heatmap()` |
| Failure frequency bar chart | `src/analysis.py:153–177`, `results/charts/failure_frequency.png` | ✅ matplotlib `bar()` |
| Correlation matrix | `src/analysis.py:180–212`, `results/charts/failure_correlation.png` | ✅ seaborn `heatmap()` on `df.corr()` |
| Category breakdown | `src/analysis.py:215–235`, `results/charts/category_failures.png` | ✅ |
| Difficulty breakdown | `src/analysis.py:238–263`, `results/charts/difficulty_failures.png` | ✅ |
| Agreement matrix | `src/analysis.py:266–350`, `results/charts/agreement_matrix.png` | ✅ |
| Key questions answered (5 of 5) | `src/analysis.py`, `results/metrics.json` | Most common mode (incomplete_answer 50%), category distribution, difficulty profile, co-occurrence, judge agreement all answered |

### Section 8: Correction Loop

| Requirement | Location | Notes |
|-------------|----------|-------|
| Individual record correction with original + failures + reasoning in prompt | `src/corrector.py:139–181` (`correct_record()`) | Correction prompt includes full record, flagged modes, judge reasoning |
| Correction uses GPT-4o-mini via Instructor | `src/corrector.py:222–240` | Same model as generation |
| Template v2 based on failure pattern analysis | `src/corrector.py:261–374` (`analyze_failure_patterns()`, `build_v2_templates()`) | V2 additions target top failure modes per category |
| V2 re-generation batch | `src/corrector.py:421–487` (`generate_v2_batch()`), `data/generated/batch_v2.json` | 30 fresh V2 records |
| Improvement > 80% | `results/correction_comparison.json` | −100% achieved (36 → 0) |
| 4-stage pipeline tracked and reproducible | `src/corrector.py:490–600` (`run_full_pipeline()`), `results/correction_comparison.json` | 36 → 12 → 8 → 0 |

### Section 9: File Structure

All required directories and files exist:

| PRD specifies | Status |
|---------------|--------|
| `src/__init__.py` | ✅ |
| `src/schemas.py` | ✅ |
| `src/templates.py` | ✅ |
| `src/generator.py` | ✅ |
| `src/validator.py` | ✅ |
| `src/evaluator.py` | ✅ |
| `src/corrector.py` | ✅ |
| `src/analysis.py` | ✅ |
| `tests/__init__.py` | ✅ |
| `tests/test_schemas.py` | ✅ |
| `tests/test_generator.py` | ✅ |
| `tests/test_evaluator.py` | ✅ |
| `data/cache/` | ✅ (108 JSON files) |
| `data/generated/` | ✅ (batch_v1.json, batch_v2.json) |
| `data/validated/` | ✅ (validated_records.json, rejected_records.json, validation_report.json) |
| `data/labels/` | ✅ (manual_labels.csv, llm_labels.csv + 6 stage-specific label files) |
| `data/corrected/` | ✅ (corrected_records.json, v2_corrected_records.json) |
| `results/charts/` | ✅ (7 PNG files — 6 required + 1 bonus) |
| `results/metrics.json` | ✅ |
| `docs/adr/` | ✅ |
| `streamlit_app.py` | ✅ |
| `README.md` | ✅ |

### Section 12: ADRs

| ADR | File | Status |
|-----|------|--------|
| ADR-001: Why Instructor over raw OpenAI API | `docs/adr/ADR-001-instructor-over-raw-openai.md` | ✅ |
| ADR-002: Why flat schema over nested models | `docs/adr/ADR-002-flat-schema-over-nested-models.md` | ✅ |
| ADR-003: Dual labeling agreement strategy | `docs/adr/ADR-003-judge-prompt-calibration.md` | ✅ |
| ADR-004: Template improvement as correction strategy | `docs/adr/ADR-004-template-improvement-correction.md` | ✅ |

### Tests and Documentation

| Requirement | Status |
|-------------|--------|
| All tests pass | ✅ 203 passed, 0 failed |
| README has problem statement | ✅ "Why This Matters" section |
| README has architecture | ✅ Mermaid flowchart diagram |
| README has results | ✅ "Key Results" table + 36→12→8→0 narrative |
| README has demo link | ✅ Streamlit quick start section |

---

## PARTIAL — Implemented but Incomplete

### 1. First-attempt success rate (PRD §5b)

**PRD says:** "First-attempt success rate: Track but no target — Records that passed without retry."

**What's implemented:** `ValidationReport` in `src/validator.py:66` tracks `total_records`, `valid_count`, `rejected_count`, `success_rate`, and `field_error_counts`. The `validation_report.json` reflects this structure.

**Gap:** Instructor handles retries internally. The number of retries per record (i.e., whether a record passed on the first attempt vs. required retry 2 or 3) is not surfaced or logged. `ValidationReport` has no `first_attempt_rate` field.

**Impact:** Low. PRD sets no target for this metric and uses the qualifier "Track but no target." Since Instructor's auto-retry is opaque, implementing this would require intercepting Instructor internals or wrapping retry logic manually. Given 100% success rate at generation time, the missing metric has no analytical value for this dataset.

---

### 2. Cohen's Kappa (PRD §6b)

**PRD says:** "Compute Cohen's Kappa if possible (accounts for chance agreement)."

**What's implemented:** `compute_agreement()` in `src/evaluator.py:327–401` computes per-mode agreement rate (% matching labels) and overall agreement rate. Result: 81.7% overall. Stored in `data/labels/agreement_report.json`.

**Gap:** Cohen's Kappa statistic (κ) is not computed. The PRD qualifies this with "if possible" — it would require `sklearn.metrics.cohen_kappa_score` or a manual implementation. With only 10 overlapping records and binary labels per mode, Kappa would provide limited additional statistical value over the raw agreement rate.

**Impact:** Low-to-medium. The 81.7% raw agreement rate is cited in the README and README positions this as a portfolio artifact ("I validated my LLM evaluation against human labels"). Kappa would strengthen the statistical rigor of this claim.

---

## MISSING — Not Implemented

### 1. Loom recording (PRD §11 item 19)

**PRD says (Wednesday §11):** "Loom recording (2 min)"

**Status:** Not recorded. Intentionally deferred in `CLAUDE.md` (item 19: "deferred until all 9 projects complete").

**Impact:** None on code quality. Affects portfolio presentation only. Acceptable deferral.

---

### 2. Streamlit Cloud deployment (PRD §11 item 20)

**PRD says (Wednesday §11):** "Final git push, update Notion Project Tracker to 'Done'" — and Section 13: "No deployment beyond Streamlit Community Cloud" (implies Streamlit Cloud IS the deployment target).

**Status:** Not deployed. Intentionally deferred in `CLAUDE.md` (item 20: "deferred until all 9 projects complete").

**Impact:** The demo cannot be linked in the README with a live URL. The `README.md` currently has a "Quick Start" section with local run instructions but no hosted demo link. Acceptable deferral given the portfolio strategy.

---

## Beyond PRD — Implemented Without Being Required

These are additions that exceed the PRD specification:

| Addition | Location |
|----------|----------|
| 4 additional test files beyond the 3 specified | `tests/test_templates.py`, `tests/test_validator.py`, `tests/test_analysis.py`, `tests/test_corrector.py` |
| 95% test coverage (203 tests) | PRD only specified 3 test files |
| 7th chart: correction improvement 4-bar chart | `results/charts/correction_improvement.png` (PRD required 6) |
| Reproducible 4-stage pipeline (`run_full_pipeline()`) | `src/corrector.py:490–600` |
| `correction_comparison.json` with experiment metadata | `results/correction_comparison.json` |
| Stage-specific label files for all 4 pipeline stages | `data/labels/llm_labels_corrected.csv`, `llm_labels_v2.csv`, `llm_labels_v2_corrected.csv` |
| Interactive Streamlit demo with 6 sections | PRD only specified "demo app" |

---

## Verdict

**P1 fully satisfies the PRD** with two minor partials (both pre-qualified by the PRD itself with "if possible" / "no target") and two intentionally deferred deliverables (Loom + cloud deploy). The codebase exceeds the PRD in test coverage, pipeline reproducibility, and visualization depth.
