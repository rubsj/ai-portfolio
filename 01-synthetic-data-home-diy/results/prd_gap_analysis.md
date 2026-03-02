# PRD Gap Analysis — P1: Synthetic Data, Home DIY Repair

**Generated:** 2026-03-02
**Branch:** feat/p1-cohen-kappa
**Test run:** `uv run pytest tests/ -v` → **209 passed, 0 failed (6.90s)**

---

## PASS — Fully implemented and verifiable

### Section 3: Data Schema

**DIYRepairRecord — all 7 fields present**
[src/schemas.py:52–94](../src/schemas.py) — `question`, `answer`, `equipment_problem`, `tools_required`, `steps`, `safety_info`, `tips` all defined as required fields.

**All validators from PRD table**
| Field | PRD Spec | Implementation | Location |
|-------|----------|----------------|----------|
| `question` | `min_length=10`, ends with `?` | `Field(min_length=10)` + `@field_validator` | `schemas.py:63–108` |
| `answer` | `min_length=50` | `Field(min_length=50)` | `schemas.py:67` |
| `equipment_problem` | `min_length=5` | `Field(min_length=5)` | `schemas.py:71` |
| `tools_required` | list `min_length=1`, each item `min_length=2` | `list[Annotated[str, Field(min_length=2)]]` with `Field(min_length=1)` | `schemas.py:78–81` |
| `steps` | list `min_length=2`, each item `min_length=10` | `list[Annotated[str, Field(min_length=10)]]` with `Field(min_length=2)` | `schemas.py:83–86` |
| `safety_info` | `min_length=10` | `Field(min_length=10)` | `schemas.py:87` |
| `tips` | `min_length=5` | `Field(min_length=5)` | `schemas.py:91` |

**GeneratedRecord — all 8 metadata fields present**
[src/schemas.py:114–148](../src/schemas.py) — `trace_id`, `category`, `difficulty`, `template_version`, `generation_timestamp`, `model_used`, `prompt_hash`, `record` all defined.

**FailureLabel and JudgeResult models**
[src/schemas.py:153–218](../src/schemas.py) — `FailureLabel` has `mode` (typed `FailureMode` Literal), `label` (`Literal[0, 1]`), `reason`. `JudgeResult` has `trace_id`, `labels` (exactly 6), `overall_quality_score` (1–5). `@field_validator` ensures all 6 modes are present.

---

### Section 4: Generation Pipeline

**5 templates (v1) with correct personas and emphasis**
[src/templates.py:43–70](../src/templates.py) — all 5 categories (`appliance_repair`, `plumbing_repair`, `electrical_repair`, `hvac_maintenance`, `general_home_repair`) mapped with PRD-specified personas and emphasis strings.

**30 records generated (5 × 3 × 2 matrix)**
`data/generated/batch_v1.json` — confirmed 30 records. `data/generated/batch_v2.json` — confirmed 30 records.

**JSON file cache (MD5 keyed)**
[src/generator.py:68–123](../src/generator.py) — `_prompt_hash()` uses MD5 on `system + "---" + user`. `load_from_cache()` / `save_to_cache()` check/write `data/cache/{key}.json`. Cache format matches PRD Section 4d spec (cache_key, prompt_hash, category, difficulty, model, timestamp, response).

**Instructor integration**
[src/generator.py:177–183](../src/generator.py) — `instructor.from_openai(OpenAI())` wrapping, `response_model=DIYRepairRecord`, `max_retries=3`, `temperature=0.7`.

**Batch generation matrix**
[src/generator.py:194–268](../src/generator.py) — `generate_batch()` loops 5 categories × 3 difficulties × 2 variants = 30 records. `_generate_variant()` appends variation hint for index > 0 to ensure unique prompts and diverse outputs.

---

### Section 5: Validation Pipeline

**Success rate tracking**
[src/validator.py:69–73](../src/validator.py) — `ValidationReport.success_rate` property returns `valid_count / total_records`. `success_rate_pct` is human-readable string.

**Per-field error frequency**
[src/validator.py:66](../src/validator.py) — `field_error_counts: Counter` tracks field name → count. Exposed via `ValidationReport.summary()` at `validator.py:80–88`.

**validated_records.json and rejected_records.json**
Both files present: `data/validated/validated_records.json`, `data/validated/rejected_records.json`. Code at [src/validator.py:181–207](../src/validator.py).

---

### Section 6: Failure Labeling

**All 6 failure modes implemented**
[src/schemas.py:36–43](../src/schemas.py) — `FailureMode` Literal type defines all 6: `incomplete_answer`, `safety_violations`, `unrealistic_tools`, `overcomplicated_solution`, `missing_context`, `poor_quality_tips`. Judge prompt criteria at [src/evaluator.py:56–88](../src/evaluator.py).

**Manual labels for exactly 10 records**
`data/labels/manual_labels.csv` — confirmed 10 rows.

**LLM labels for all 30 records**
`data/labels/llm_labels.csv` and `data/labels/llm_labels.json` — both present with full 30-record coverage. LLM labels also exist for all pipeline stages: `llm_labels_corrected.{csv,json}`, `llm_labels_v2.{csv,json}`, `llm_labels_v2_corrected.{csv,json}`.

**Per-mode agreement rate**
[src/evaluator.py:391–397](../src/evaluator.py) — `per_mode_agreement` dict computed for each failure mode. Results in `data/labels/agreement_report.json`:
- `incomplete_answer`: 80.0%
- `safety_violations`: 80.0%
- `unrealistic_tools`: 70.0%
- `overcomplicated_solution`: 100.0%
- `missing_context`: 100.0%
- `poor_quality_tips`: 60.0%
- **Overall: 81.7%**

**Cohen's Kappa**
[src/evaluator.py:403–431](../src/evaluator.py) — `cohen_kappa_score` from sklearn, computed per-mode and aggregated. Results in `data/labels/agreement_report.json`:
- Per-mode: `incomplete_answer=0.545`, `safety_violations=0.412`, `unrealistic_tools=-0.154`, `poor_quality_tips=0.000`, `overcomplicated_solution=N/A` (degenerate — all zeros in both raters), `missing_context=N/A` (degenerate)
- **Overall kappa: 0.201** (mean of valid modes)

---

### Section 7: Analysis

**All 6 PRD-specified charts generated**
All files present in `results/charts/`:
| Chart | PRD Spec | File | Code |
|-------|----------|------|------|
| Failure mode heatmap | `failure_heatmap.png` | ✅ exists | `analysis.py:119–150` |
| Failure frequency bar | `failure_frequency.png` | ✅ exists | `analysis.py:153–177` |
| Correlation matrix | `failure_correlation.png` | ✅ exists | `analysis.py:180–212` |
| Category breakdown | `category_failures.png` | ✅ exists | `analysis.py:215–235` |
| Difficulty breakdown | `difficulty_failures.png` | ✅ exists | `analysis.py:238–263` |
| Agreement matrix | `agreement_matrix.png` | ✅ exists | `analysis.py:266–361` |

**Extra chart not in PRD** (bonus deliverable): `results/charts/correction_improvement.png` — 4-bar red-to-green pipeline chart. [src/analysis.py:364–421](../src/analysis.py).

**Per-category and per-difficulty breakdowns**
[src/analysis.py:447–467](../src/analysis.py) — `compute_metrics()` produces per-category and per-difficulty failure rate dicts. Saved to `results/metrics.json`.

**Key analysis questions answered (Section 7c)**
[src/analysis.py:428–508](../src/analysis.py) — `compute_metrics()` computes per-mode failures (Q1), per-category rates (Q2), per-difficulty profiles (Q3); correlation matrix answers Q4 visually; agreement report answers Q5.

---

### Section 8: Correction Loop

**Strategy A — Individual record correction**
[src/corrector.py:97–181](../src/corrector.py) — `correct_record()` builds correction prompt with original record + flagged issues + judge reasons. Sends to GPT-4o-mini via Instructor. `correct_batch()` at `corrector.py:184–254` processes all records with ≥1 failure; clean records pass through unchanged.

**Strategy B — Template v2 improvement**
[src/corrector.py:261–487](../src/corrector.py) — `analyze_failure_patterns()` identifies top modes per category; `build_v2_templates()` adds targeted instructions per failure mode (6 instruction strings in `_V2_MODE_INSTRUCTIONS` at `corrector.py:296–331`); `generate_v2_batch()` regenerates full 30-record matrix with v2 system prompts.

**Full 4-stage pipeline (36 → 12 → 8 → 0)**
[src/corrector.py:624–741](../src/corrector.py) — `run_full_pipeline()` orchestrates all 9 sub-stages end-to-end. `results/correction_comparison.json` confirms:
- V1 Original: **36 failures** (20.0%)
- V1 Corrected: **12 failures** (−66.7%)
- V2 Generated: **8 failures** (−77.8%)
- V2 Corrected: **0 failures** (−100%)

**>80% improvement target**
`results/correction_comparison.json` → `target_met.v2_corrected_meets_80pct: true`. Final stage achieves 100% reduction, meeting the >80% criterion.

---

### Section 9: File Structure

All required directories and files present:

| Required Path | Status |
|---------------|--------|
| `CLAUDE.md` | ✅ |
| `PRD.md` | ✅ |
| `src/__init__.py` | ✅ |
| `src/schemas.py` | ✅ |
| `src/templates.py` | ✅ (v1 only — see PARTIAL) |
| `src/generator.py` | ✅ |
| `src/validator.py` | ✅ |
| `src/evaluator.py` | ✅ |
| `src/corrector.py` | ✅ |
| `src/analysis.py` | ✅ |
| `tests/__init__.py` | ✅ |
| `tests/test_schemas.py` | ✅ |
| `tests/test_generator.py` | ✅ |
| `tests/test_evaluator.py` | ✅ |
| `data/cache/` | ✅ |
| `data/generated/batch_v1.json` | ✅ 30 records |
| `data/generated/batch_v2.json` | ✅ 30 records |
| `data/validated/validated_records.json` | ✅ |
| `data/validated/rejected_records.json` | ✅ |
| `data/labels/manual_labels.csv` | ✅ 10 rows |
| `data/labels/llm_labels.csv` | ✅ |
| `data/corrected/corrected_records.json` | ✅ |
| `data/corrected/v2_corrected_records.json` | ✅ |
| `results/charts/` (6 required PNGs) | ✅ all 6 present + 1 bonus |
| `results/metrics.json` | ✅ |
| `docs/adr/` (4 ADRs) | ✅ |
| `streamlit_app.py` | ✅ |
| `README.md` | ✅ |

**Extra test files beyond PRD minimum:** `tests/test_templates.py`, `tests/test_validator.py`, `tests/test_corrector.py`, `tests/test_analysis.py` — all passing. Test count: 209 (PRD did not specify a count).

---

### Section 12: ADRs

**ADR-001** (`docs/adr/ADR-001-instructor-over-raw-openai.md`): "Why Instructor over raw OpenAI API" — matches PRD spec exactly. Full template: context, decision (with code snippet), alternatives table, quantified validation (30/30 records, ~120 LOC saved), consequences, Java/TS parallel, interview signal.

**ADR-002** (`docs/adr/ADR-002-flat-schema-over-nested-models.md`): "Flat Schema over Nested Models" — matches PRD spec.

**ADR-004** (`docs/adr/ADR-004-template-improvement-correction.md`): "Template Improvement as Correction Strategy" — matches PRD spec.

---

### Streamlit App

`streamlit_app.py` — exists, loads from `data/` and `results/` dirs with `@st.cache_data`. Displays all 6 failure modes, 5 categories, 3 difficulties. Loads generated records, labels, metrics, charts. PRD Section 9 requirement met.

### README

`README.md` — has problem statement ("36 → 0 story"), architecture (Mermaid flowchart), results table (20.0% → 0.0%), engineering practices (ADRs, caching, dual labeling), key insights, demo link section. PRD requirement met.

### All Tests Pass

```
209 passed in 6.90s
```

All 7 test files pass.

---

## PARTIAL — Implemented but incomplete or deviating from PRD

### §4 / §9: v2 Templates Not in templates.py

**PRD spec** (`Section 9`): `src/templates.py — 5 prompt templates (v1 and v2)`

**Actual**: `src/templates.py` contains v1 only (`TEMPLATE_VERSION = "v1"` at `templates.py:96`). V2 templates are built **programmatically at runtime** in `src/corrector.py`:
- `_V2_MODE_INSTRUCTIONS` dict at `corrector.py:296–331` — 6 failure-mode-specific instruction strings
- `build_v2_templates()` at `corrector.py:334–374` — reads `results/metrics.json` to determine which modes to address, then assembles per-category v2 templates
- `build_v2_system_prompt()` at `corrector.py:377–387` — builds the actual system prompt string

**Gap**: V2 prompt content is not a static, reviewable artifact in `templates.py` as the PRD implies. It is dynamically assembled from failure metrics at runtime.

**Impact**: Low. The v2 templates function correctly — `batch_v2.json` exists with 30 records, v2 failure rate is 4.4% (−77.8%). The gap is organizational: a reader of `templates.py` does not see v2 content.

---

### §5: First-Attempt Success Rate Not Tracked

**PRD spec** (`Section 5b`): "First-attempt success rate — Track but no target — Records that passed without retry"

**Actual**: Not tracked anywhere. `src/validator.py` only re-validates saved records; it has no concept of retry history. `src/generator.py:177–183` calls `client.chat.completions.create()`, which either succeeds after Instructor's internal retries or raises after exhausting `max_retries=3`. The return value contains no retry count.

**Gap**: Instructor does not expose per-call retry count in its standard return path. Tracking this would require: (a) Instructor's `hooks` mechanism (not currently used), or (b) wrapping `client.chat.completions.create()` with `max_retries=0` as a probe then escalating — a significant complexity addition.

**Impact**: Low. PRD explicitly says "no target" for this metric. The practical information value is minimal since Instructor's auto-retry ensures near-100% first-attempt success for well-structured schemas like `DIYRepairRecord`.

---

### §8: V1 Correction Alone Does Not Meet >80%

**PRD spec** (`Section 8c`): "Target: improvement > 80%"

**Actual**: The >80% target is met only by the **final stage** (V2 Corrected, −100%). Intermediate stages fall short:
- V1 Corrected: **−66.7%** (36 → 12 failures) — below target
- V2 Generated: **−77.8%** (36 → 8 failures) — below target
- V2 Corrected: **−100%** ✅ — meets target

`correction_comparison.json` correctly reports `corrected_meets_80pct: false` and `v2_meets_80pct: false`.

**Gap**: The PRD's "Target: improvement > 80%" is semantically ambiguous about which stage must meet it. The implementation meets the spirit (final result is 100%) but the individually-labeled intermediate stage ("corrected") is documented as not meeting the threshold, and `target_met.corrected_meets_80pct: false` is visible in the artifact.

**Impact**: Medium. An interviewer examining `correction_comparison.json` will see two `false` target flags. The README correctly emphasizes the 36→0 end result, but the per-stage story requires explanation: Strategy A alone is insufficient; Strategy B (template improvement) is what pushes past 80%.

---

### §12: ADR-003 Topic Mismatches PRD Spec

**PRD spec**: `ADR-003 — Dual labeling — manual + LLM agreement as evaluation strategy`

**Actual** (`docs/adr/ADR-003-judge-prompt-calibration.md`): "Judge Prompt Calibration (0% → 20%)" — documents the prompt engineering decision that moved failure rate from 0% to 20%. The dual labeling approach and its 81.7% agreement result are mentioned in the "Quantified Validation" section, but the ADR's central thesis is prompt strictness calibration, not the dual labeling design choice.

**Note**: ADR-003 also contains this line: _"Cohen's Kappa not computed — acknowledged gap, though 81.7% on strict binary labels indicates concordance well above chance"_ — which is now outdated. Kappa **is** computed (overall: 0.201, per-mode in `agreement_report.json`) as of the current branch.

**Gap**: The decision "why use both manual labels AND LLM labels" — the portfolio signal about validating LLM evaluation against human ground truth — has no dedicated ADR. The data exists; the architectural rationale is not documented with PRD template structure.

**Impact**: Low-to-medium. All the evidence is present (`agreement_report.json`, `agreement_matrix.png`, Kappa scores). The missing piece is the documented reasoning for *why* dual labeling was chosen and what it proves about judge reliability — a strong EM interview talking point.

---

### §9: correction_comparison.json Artifact Missing Metadata Fields

**PRD spec**: Not explicitly required, but `build_comparison_metrics()` at [src/corrector.py:553–617](../src/corrector.py) is designed to emit experiment metadata: `generated_at`, `generator_model`, `judge_model`, `pipeline_version`.

**Actual** (`results/correction_comparison.json`): The file does **not** contain these four fields. The artifact on disk was generated before the metadata fields were added to `build_comparison_metrics()`. Current file structure:
```json
{
  "v1_original": {...},
  "corrected": {...},
  "v2_generated": {...},
  "v2_corrected": {...},
  "target_met": {...}
}
```

**Gap**: The code and the artifact are out of sync on metadata. Running `uv run python -m src.corrector` would regenerate the file with correct metadata (since all pipeline inputs are cached, this would be a cache-only run with no API cost).

**Impact**: Very low. Failure counts and improvement percentages are correct. Only experiment provenance is missing from the saved file.

---

## MISSING — Not implemented at all

**None found.** All PRD sections have implementations. Every requirement has either a PASS or a PARTIAL.

---

## Summary Table

| PRD Section | Requirement | Status |
|-------------|-------------|--------|
| §3 | DIYRepairRecord — 7 fields + all validators | ✅ PASS |
| §3 | GeneratedRecord — 8 metadata fields | ✅ PASS |
| §3 | FailureLabel + JudgeResult models | ✅ PASS |
| §4 | 5 v1 templates (category × persona × emphasis) | ✅ PASS |
| §4 | v2 templates in `templates.py` | ⚠️ PARTIAL — v2 in `corrector.py` instead |
| §4 | 30 records (5 × 3 × 2 matrix) | ✅ PASS |
| §4 | JSON cache (MD5 keyed, PRD format) | ✅ PASS |
| §4 | Instructor integration (response_model, max_retries) | ✅ PASS |
| §5 | Generation success rate tracking | ✅ PASS |
| §5 | First-attempt success rate | ⚠️ PARTIAL — architecturally untrackable via Instructor |
| §5 | Per-field error frequency | ✅ PASS |
| §5 | validated_records.json + rejected_records.json | ✅ PASS |
| §6 | All 6 failure modes defined | ✅ PASS |
| §6 | Manual labels — 10 records | ✅ PASS |
| §6 | LLM labels — all 30 records | ✅ PASS |
| §6 | Per-mode agreement rate | ✅ PASS (81.7% overall) |
| §6 | Cohen's Kappa | ✅ PASS (0.201 overall, per-mode) |
| §7 | All 6 specified charts | ✅ PASS |
| §7 | Per-category breakdown | ✅ PASS |
| §7 | Per-difficulty breakdown | ✅ PASS |
| §8 | Strategy A — individual record correction | ✅ PASS |
| §8 | Strategy B — template v2 | ✅ PASS |
| §8 | 4-stage pipeline (36→12→8→0) | ✅ PASS |
| §8 | >80% improvement (final stage) | ✅ PASS (100%) |
| §8 | >80% improvement (V1 corrected alone) | ⚠️ PARTIAL (66.7%) |
| §9 | All required files and directories | ✅ PASS |
| §9 | correction_comparison.json metadata fields | ⚠️ PARTIAL — stale artifact |
| §12 | ADR-001 (Instructor) | ✅ PASS |
| §12 | ADR-002 (flat schema) | ✅ PASS |
| §12 | ADR-003 (dual labeling topic) | ⚠️ PARTIAL — written as "Judge Prompt Calibration"; Kappa note is now outdated |
| §12 | ADR-004 (template improvement) | ✅ PASS |
| — | Streamlit app | ✅ PASS |
| — | README (problem + architecture + results + demo) | ✅ PASS |
| — | All tests pass | ✅ PASS (209/209) |
