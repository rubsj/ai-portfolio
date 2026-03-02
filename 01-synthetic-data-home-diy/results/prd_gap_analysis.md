# PRD Gap Analysis — P1: Synthetic Data, Home DIY Repair

**Generated:** 2026-03-02
**Test run:** `uv run pytest tests/ -v` → **209 passed, 0 failed (5.23s)**

---

## PASS — Fully implemented and verifiable

### Section 3: Data Schema

**DIYRepairRecord — all 7 fields present with all validators**

| Field | PRD Spec | Implementation | Location |
|-------|----------|----------------|----------|
| `question` | `min_length=10`, ends with `?` | `Field(min_length=10)` + `@field_validator` | `src/schemas.py:63-108` |
| `answer` | `min_length=50` | `Field(min_length=50)` | `src/schemas.py:67-69` |
| `equipment_problem` | `min_length=5` | `Field(min_length=5)` | `src/schemas.py:71-73` |
| `tools_required` | list `min_length=1`, each item `min_length=2` | `list[Annotated[str, Field(min_length=2)]]` with `Field(min_length=1)` | `src/schemas.py:78-80` |
| `steps` | list `min_length=2`, each item `min_length=10` | `list[Annotated[str, Field(min_length=10)]]` with `Field(min_length=2)` | `src/schemas.py:83-85` |
| `safety_info` | `min_length=10` | `Field(min_length=10)` | `src/schemas.py:87-89` |
| `tips` | `min_length=5` | `Field(min_length=5)` | `src/schemas.py:91-93` |

**GeneratedRecord — all 8 metadata fields:** `trace_id`, `category` (typed `Category` Literal), `difficulty` (typed `Difficulty` Literal), `template_version`, `generation_timestamp`, `model_used`, `prompt_hash`, `record`. (`src/schemas.py:114-147`)

**FailureLabel and JudgeResult models:** `FailureLabel` has `mode` (`FailureMode` Literal), `label` (`Literal[0, 1]`), `reason` (`min_length=5`). `JudgeResult` has `trace_id`, `labels` (exactly 6 via `min_length`/`max_length` + `@field_validator` ensuring all modes present), `overall_quality_score` (1-5 via `ge=1, le=5`). (`src/schemas.py:153-217`)

---

### Section 4: Generation Pipeline

**5 templates (v1) with correct personas and emphasis:** All 5 categories mapped with PRD-specified personas and emphasis strings. (`src/templates.py:43-70`)

**30 records generated (5 × 3 × 2 matrix):** `data/generated/batch_v1.json` confirmed 30 records. Every (category, difficulty) combo has exactly 2 records. `data/generated/batch_v2.json` also 30 records.

**JSON file cache (MD5 keyed):** `_prompt_hash()` uses MD5 on `system + "---" + user`. `load_from_cache()` / `save_to_cache()` manage `data/cache/{key}.json`. Cache format matches PRD Section 4d (cache_key, prompt_hash, category, difficulty, model, timestamp, response). 176 cache files present. (`src/generator.py:68-123`)

**Instructor integration:** `instructor.from_openai(OpenAI())`, `response_model=DIYRepairRecord`, `max_retries=3`, `temperature=0.7`, model `gpt-4o-mini`. (`src/generator.py:130-183`)

**Batch generation with variant diversity:** `generate_batch()` loops 5 × 3 × 2. `_generate_variant()` appends variation hint for index > 0. (`src/generator.py:194-320`)

---

### Section 5: Validation Pipeline

**Success rate tracking:** `ValidationReport` with `success_rate` property, `success_rate_pct` string. (`src/validator.py:56-88`)

**Per-field error frequency:** `field_error_counts: Counter` tracks per-field errors. (`src/validator.py:66`)

**validated_records.json and rejected_records.json:** Both present in `data/validated/`. `validation_report.json` confirms 30/30 valid, 0 rejected, 100.0% success rate. (`src/validator.py:181-207`)

---

### Section 6: Failure Labeling

**All 6 failure modes:** Defined in `FailureMode` Literal type (`src/schemas.py:36-43`). Detailed judge prompt criteria for each mode (`src/evaluator.py:49-88`).

**Manual labels for 10 records:** `data/labels/manual_labels.csv` — 10 data rows with all 6 binary failure modes.

**LLM labels for all 30 records:** `data/labels/llm_labels.csv` and `llm_labels.json` — 30 entries each. Additional label files for all pipeline stages: `llm_labels_corrected`, `llm_labels_v2`, `llm_labels_v2_corrected` (both CSV and JSON).

**Per-mode agreement rate:** Computed in `compute_agreement()` (`src/evaluator.py:328-440`). Results in `data/labels/agreement_report.json`:
- `incomplete_answer`: 80.0%, `safety_violations`: 80.0%, `unrealistic_tools`: 70.0%, `overcomplicated_solution`: 100.0%, `missing_context`: 100.0%, `poor_quality_tips`: 60.0%
- **Overall: 81.7%**

**Cohen's Kappa:** `cohen_kappa_score` from sklearn (`src/evaluator.py:29`). Per-mode kappa computed with degenerate-case handling. Results: `incomplete_answer=0.545`, `safety_violations=0.412`, `unrealistic_tools=-0.154`, `poor_quality_tips=0.000`, `overcomplicated_solution=N/A`, `missing_context=N/A`. **Overall kappa: 0.201**.

---

### Section 7: Analysis

**All 6 PRD-specified charts + 1 bonus:**

| Chart | File | Code |
|-------|------|------|
| Failure mode heatmap | `results/charts/failure_heatmap.png` | `src/analysis.py:119-150` |
| Failure frequency bar | `results/charts/failure_frequency.png` | `src/analysis.py:153-177` |
| Correlation matrix | `results/charts/failure_correlation.png` | `src/analysis.py:180-212` |
| Category breakdown | `results/charts/category_failures.png` | `src/analysis.py:215-235` |
| Difficulty breakdown | `results/charts/difficulty_failures.png` | `src/analysis.py:238-263` |
| Agreement matrix | `results/charts/agreement_matrix.png` | `src/analysis.py:266-361` |
| Correction improvement (bonus) | `results/charts/correction_improvement.png` | `src/analysis.py:364-421` |

**Per-category and per-difficulty breakdowns:** `compute_metrics()` produces both. Saved to `results/metrics.json`. (`src/analysis.py:447-467`)

**Key analysis questions (Section 7c) all answered:**
1. Most common failure mode: `incomplete_answer` at 50.0% (15/30)
2. Category differences: `electrical_repair` = 0.0%, `general_home_repair` = 30.6%
3. Difficulty profiles: `intermediate` = 23.3%, `beginner` = 16.7%, `advanced` = 20.0%
4. Co-occurrence: correlation matrix chart
5. Manual vs LLM agreement: 81.7% with kappa

---

### Section 8: Correction Loop

**Strategy A — Individual record correction:** `correct_record()` sends original record + judge feedback to GPT-4o-mini via Instructor. `correct_batch()` processes all records with ≥1 failure. (`src/corrector.py:97-254`)

**Strategy B — Template v2 improvement:** `analyze_failure_patterns()` identifies top modes per category. `_V2_MODE_INSTRUCTIONS` dict provides 6 failure-prevention instruction strings. `build_v2_templates()` assembles per-category templates. `generate_v2_batch()` regenerates 30 records. (`src/corrector.py:261-487`)

**Full 4-stage pipeline:** `run_full_pipeline()` orchestrates all stages end-to-end. (`src/corrector.py:624-740`)

**Per-record and per-template correction tracking:** `correct_batch()` counts corrected vs unchanged. `build_comparison_metrics()` computes `improvement_vs_v1` for each stage. (`src/corrector.py:553-617`)

---

### Section 9: File Structure

All required files and directories present:

| Required Path | Status |
|---------------|--------|
| `CLAUDE.md` | Present |
| `PRD.md` | Present |
| `pyproject.toml` (with `instructor` dep) | Present |
| `src/__init__.py` | Present |
| `src/schemas.py` | Present |
| `src/templates.py` | Present |
| `src/generator.py` | Present |
| `src/validator.py` | Present |
| `src/evaluator.py` | Present |
| `src/corrector.py` | Present |
| `src/analysis.py` | Present |
| `tests/__init__.py` | Present |
| `tests/test_schemas.py` | Present |
| `tests/test_generator.py` | Present |
| `tests/test_evaluator.py` | Present |
| `data/cache/` | 176 files |
| `data/generated/` | `batch_v1.json` (30 records), `batch_v2.json` (30 records) |
| `data/validated/` | `validated_records.json`, `rejected_records.json`, `validation_report.json` |
| `data/labels/` | `manual_labels.csv` (10 rows), `llm_labels.csv`, `llm_labels.json`, + corrected/v2 variants, `agreement_report.json` |
| `data/corrected/` | `corrected_records.json`, `v2_corrected_records.json` |
| `results/charts/` | 7 PNG files (6 required + 1 bonus) |
| `results/metrics.json` | Present |
| `docs/adr/` | 5 ADR files (4 required + 1 bonus) |
| `streamlit_app.py` | Present (633 lines) |
| `README.md` | Present (portfolio-quality) |

**Bonus test files beyond PRD minimum:** `test_templates.py`, `test_validator.py`, `test_corrector.py`, `test_analysis.py`. Total: 209 tests across 7 test files.

---

### Section 12: ADRs

| ADR | PRD Title | Actual Title | Status |
|-----|-----------|-------------|--------|
| ADR-001 | Why Instructor over raw OpenAI API | "Instructor over Raw OpenAI API" | PASS |
| ADR-002 | Flat schema matching spec over nested models | "Flat Schema over Nested Models" | PASS |
| ADR-004 | Template improvement as correction strategy | "Template Improvement as Correction Strategy" | PASS |
| ADR-005 | (not in PRD — bonus) | "Dual Labeling Strategy — Manual + LLM Inter-Rater Agreement" | PASS |

---

### Streamlit App

`streamlit_app.py` exists (633 lines), loads from `data/` and `results/` with `@st.cache_data`, displays categories, failure modes, charts, correction pipeline metrics.

### README

`README.md` has: problem statement ("Why This Matters"), architecture (Mermaid flowchart), key results table (20.0% → 0.0%), engineering practices, key insights, embedded screenshot.

### All Tests Pass

```
209 passed in 5.23s
```

---

## PARTIAL — Implemented but incomplete or deviating from PRD

### 1. First-attempt success rate not tracked (Section 5b)

**PRD says:** "First-attempt success rate — Track but no target — Records that passed without retry"

**Actual:** Not tracked. `src/validator.py` re-validates saved records but has no concept of retry history. Instructor's `client.chat.completions.create()` either succeeds (possibly after internal retries) or raises — the return value contains no retry count.

**Why:** Instructor does not expose per-call retry count in its standard return path. Tracking this would require Instructor's `hooks` mechanism or wrapping each call with `max_retries=0` as a probe.

**Impact:** Low. PRD explicitly says "no target" for this metric. All 30 records passed validation, so the data is complete — the gap is in observability, not correctness.

---

### 2. correction_comparison.json disagrees with metrics.json on final result (Section 8c)

**PRD says:** "Target: improvement > 80%"

**Actual:** Two conflicting data files exist:
- **`results/correction_comparison.json`** (newer, 2026-03-02T06:21): V2 Corrected = **8 failures** (77.8% improvement). All three `target_met` flags are `false`.
- **`results/metrics.json`** (older, 2026-03-02T06:08): `correction_pipeline.v2_corrected` = **0 failures** (100% improvement).

The README and CLAUDE.md document the "36 → 12 → 8 → 0" narrative which matched the earlier successful run, but the latest `correction_comparison.json` shows "36 → 12 → 8 → 8" — the V2 correction stage failed to reduce failures further.

**Root cause:** Non-deterministic LLM outputs between pipeline runs. The v2_corrected stage in the latest run produced the same 8 failures as v2_generated, suggesting either stale cache entries or the corrections didn't resolve the remaining failures.

**Impact:** Medium. The >80% target was met in a prior run (per `metrics.json`) but is not reproducible in the most recent pipeline execution. The `correction_comparison.json` and `metrics.json` are inconsistent with each other. An interviewer examining `correction_comparison.json` would see all `target_met` flags as `false`.

---

### 3. ADR-003 topic mismatches PRD spec (Section 12)

**PRD says:** `ADR-003 — Dual labeling — manual + LLM agreement as evaluation strategy`

**Actual:** ADR-003 is titled "Judge Prompt Calibration (0% → 20%)" and documents the prompt engineering decision for judge strictness — not the dual labeling design choice. The dual labeling topic was written as ADR-005 instead.

Additionally, ADR-003 contains the outdated line: _"Cohen's Kappa not computed — acknowledged gap"_ — but Kappa IS now computed (`agreement_report.json` has per-mode and overall kappa values).

**Impact:** Low. All dual labeling content exists (ADR-005 covers it thoroughly). The numbering doesn't match the PRD, and ADR-003 has a stale statement about Kappa.

---

### 4. V2 templates not in templates.py (Section 9)

**PRD says:** `src/templates.py — 5 prompt templates (v1 and v2)`

**Actual:** `src/templates.py` contains v1 only (`TEMPLATE_VERSION = "v1"` at line 96). V2 templates are built programmatically at runtime in `src/corrector.py`:
- `_V2_MODE_INSTRUCTIONS` dict (`corrector.py:296-331`) — 6 failure-prevention strings
- `build_v2_templates()` (`corrector.py:334-374`) — reads metrics to assemble per-category templates
- `build_v2_system_prompt()` (`corrector.py:377-387`) — generates the system prompt

**Impact:** Low. V2 templates function correctly (`batch_v2.json` exists with 30 records, 4.4% failure rate). The gap is organizational — a reader of `templates.py` does not see v2 content.

---

### 5. No live Streamlit deployment link in README

**PRD says:** Section 11 item 20 calls for Streamlit Cloud deployment.

**Actual:** README has embedded screenshot but no live deployment URL. CLAUDE.md explicitly marks this as "deferred until all 9 projects complete."

**Impact:** Low. Intentionally deferred — not a gap in implementation, just deployment.

---

## MISSING — Not implemented at all

**None.** All PRD sections have implementations. Every requirement has either a PASS or a PARTIAL.

---

## Summary

| PRD Section | Requirement | Status |
|-------------|-------------|--------|
| §3 | DIYRepairRecord — 7 fields + all validators | PASS |
| §3 | GeneratedRecord — 8 metadata fields | PASS |
| §3 | FailureLabel + JudgeResult models | PASS |
| §4 | 5 v1 templates (category × persona × emphasis) | PASS |
| §4 | v2 templates in `templates.py` | PARTIAL — in `corrector.py` instead |
| §4 | 30 records (5 × 3 × 2 matrix) | PASS |
| §4 | JSON cache (MD5 keyed, PRD format) | PASS |
| §4 | Instructor integration | PASS |
| §5 | Generation success rate >90% | PASS (100%) |
| §5 | First-attempt success rate tracking | PARTIAL — not trackable via Instructor |
| §5 | Per-field error frequency | PASS |
| §5 | validated_records.json + rejected_records.json | PASS |
| §6 | All 6 failure modes | PASS |
| §6 | Manual labels — 10 records | PASS |
| §6 | LLM labels — 30 records | PASS |
| §6 | Per-mode agreement rate | PASS (81.7%) |
| §6 | Cohen's Kappa | PASS (0.201 overall) |
| §7 | All 6 charts | PASS |
| §7 | Per-category breakdown | PASS |
| §7 | Per-difficulty breakdown | PASS |
| §7 | Key analysis questions answered | PASS |
| §8 | Strategy A — individual correction | PASS |
| §8 | Strategy B — template v2 | PASS |
| §8 | 4-stage pipeline | PASS |
| §8 | >80% improvement target (final stage) | PARTIAL — met in prior run (100%) but latest `correction_comparison.json` shows 77.8% |
| §9 | All files and directories | PASS |
| §12 | ADR-001, ADR-002, ADR-004 | PASS |
| §12 | ADR-003 (dual labeling topic) | PARTIAL — covers judge calibration instead; dual labeling is ADR-005 |
| — | Streamlit app exists and works | PASS |
| — | README with problem + architecture + results | PASS |
| — | Live demo link | PARTIAL — deferred |
| — | All tests pass | PASS (209/209) |

**Final tally: 27 PASS, 5 PARTIAL, 0 MISSING**
