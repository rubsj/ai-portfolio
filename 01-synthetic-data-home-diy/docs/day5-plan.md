# Plan: Close P1 Narrative Gap — Reproducible 4-Stage Correction Pipeline

## Context

The P1 project claims a quality improvement narrative: **V1 (36 failures) → V1 corrected (12) → V2 (8) → V2 corrected (0)**. The data artifacts exist from manual ad-hoc runs, but the code doesn't reproduce this programmatically:

- `corrector.py __main__` only runs Strategy A on v1 + generates v2 — stops there
- `analysis.py` only analyzes v1 data — no comparison chart
- The improvement story is claimed but not provable from running the code

**Goal:** Make `uv run python -m src.corrector` run the full 4-stage pipeline and produce `correction_comparison.json`. Add a comparison chart to `analysis.py`.

---

## Changes

### 1. `src/corrector.py` — Wire up full pipeline

**a) Add `_count_failures()` helper** (before `__main__`, ~line 504)
- Takes `list[dict]` of JudgeResult dicts + `num_records: int`, returns `dict[str, int | float]` with per-mode counts, `total` key, and `failure_rate: float` (total / num_records)
- Single source of truth for derived metrics — `build_comparison_metrics()` consumes this directly instead of re-deriving rates
- Pure function, no side effects

**b) Add `build_comparison_metrics()` function**
- Takes 4 lists of JudgeResult dicts (v1, v1_corrected, v2, v2_corrected) + `total_records`
- Calls `_count_failures()` for each stage — consumes its `failure_rate` directly instead of re-deriving
- Returns the comparison dict structure (matches existing `correction_comparison.json` format)
- Adds experiment metadata at top level: `generated_at` (ISO timestamp via `datetime.now().isoformat()`), `generator_model` (from `src.generator._MODEL`), `judge_model` (from `src.evaluator._JUDGE_MODEL`), `pipeline_version: "1.0"`
- Calculates improvement percentages between stages

**c) Add `version_tag` parameter to `correct_batch()`** (line 184)
- New keyword arg: `version_tag: str = "v1_corrected"`
- Use it on line 238 instead of hardcoded `"v1_corrected"`
- Enables calling `correct_batch(..., version_tag="v2_corrected")` for v2 correction

**d) Add `run_full_pipeline()` function**
- **Guard clause at top**: Check `batch_v1.json` and `llm_labels.json` exist, raise `FileNotFoundError` with actionable message (e.g., `"batch_v1.json not found — run 'uv run python -m src.generator' first"`, `"llm_labels.json not found — run 'uv run python -m src.evaluator' first"`). Fail fast before any LLM calls.
- Orchestrates all 4 stages:
  1. Load v1 + v1 labels (existing)
  2. Strategy A: `correct_batch(v1)` → save `corrected_records.json` (existing)
  3. **Evaluate corrected v1** → save `llm_labels_corrected.csv/.json` (NEW)
  4. Strategy B: `generate_v2_batch()` → save `batch_v2.json` (existing)
  5. **Evaluate v2** → save `llm_labels_v2.csv/.json` (NEW)
  6. **Strategy A on v2**: `correct_batch(v2, version_tag="v2_corrected")` → save `v2_corrected_records.json` (NEW)
  7. **Evaluate corrected v2** → save `llm_labels_v2_corrected.csv/.json` (NEW)
  8. **Build comparison metrics** → save `correction_comparison.json` (NEW) — include experiment metadata at top level: `generated_at` (ISO timestamp), `generator_model`, `judge_model`, `pipeline_version: "1.0"`
  9. Print summary table
- Imports `evaluate_batch`, `save_llm_labels`, `save_llm_labels_json` from `src.evaluator` locally (no circular dep — evaluator only imports from generator)
- Note: `evaluate_batch` returns `list[JudgeResult]`, but `correct_batch` expects `list[dict]` — call `.model_dump()` on each before passing. Comment at the call site as intentional technical debt: `# TODO: accept JudgeResult directly — mismatch exists because correct_batch was written to consume raw JSON from cache files`

**e) Replace `__main__` block** (lines 509–540)
- Simply calls `run_full_pipeline()`

### 2. `src/analysis.py` — Add comparison chart

**a) Add `plot_correction_improvement()` function** (~line 350)
- Reads `results/correction_comparison.json`
- Creates a bar chart: 4 bars (V1 Original, V1 Corrected, V2 Generated, V2 Corrected) with failure counts
- Color gradient: red → orange → blue → green (showing improvement)
- Labels with count on top, failure rate below
- Saves to `results/charts/correction_improvement.png`
- Raises `FileNotFoundError` if comparison JSON doesn't exist

**b) Add chart to `run_full_analysis()`** (line 443–459)
- Append `("correction_improvement", plot_correction_improvement)` to `charts` list
- Add to the existing conditional: `if name in ("agreement_matrix", "correction_improvement"): path = fn()` (these charts don't take a DataFrame)

**c) Add pipeline data to `compute_metrics()`** (after line 417)
- Load `correction_comparison.json` if it exists
- Add `"correction_pipeline"` key to returned dict with summary of all 4 stages

### 3. `tests/test_corrector.py` — New test file

Tests for:
- `_count_failures`: mixed results, no failures, empty list
- `build_comparison_metrics`: valid inputs produce all 4 stages, 100% improvement sets target_met, experiment metadata present (`generated_at`, `generator_model`, `judge_model`, `pipeline_version`)
- `correct_batch` (mocked client): one failing record gets corrected, clean records pass through, version_tag propagates
- `analyze_failure_patterns`: returns sorted modes per category
- `run_full_pipeline` (orchestration integration test): mock all LLM-calling functions (`evaluate_batch`, `correct_batch`, `generate_v2_batch`), verify stages execute in correct order, correct data shapes pass between stages, `correction_comparison.json` is produced with experiment metadata, and guard clause raises `FileNotFoundError` with helpful message when inputs missing

### 4. `tests/test_analysis.py` — New test file

Tests for:
- `plot_correction_improvement`: creates PNG when comparison JSON exists, raises FileNotFoundError when missing
- `compute_metrics`: includes `correction_pipeline` key when comparison JSON exists

---

## Files Modified

| File | Action |
|------|--------|
| `src/corrector.py` | Add `_count_failures`, `build_comparison_metrics`, `run_full_pipeline`; add `version_tag` to `correct_batch`; replace `__main__` |
| `src/analysis.py` | Add `plot_correction_improvement`; update `run_full_analysis` chart list; update `compute_metrics` |
| `tests/test_corrector.py` | New file |
| `tests/test_analysis.py` | New file |

---

## Verification

1. **Unit tests**: `uv run pytest tests/test_corrector.py tests/test_analysis.py -v` — all pass
2. **Existing tests**: `uv run pytest tests/ -v` — no regressions
3. **Dry run** (uses cache): `uv run python -m src.corrector` — produces all output files + prints summary table with 36 → 12 → 8 → 0
4. **Chart generation**: `uv run python -m src.analysis` — produces `correction_improvement.png` alongside existing 6 charts
5. **Check outputs exist**: `ls results/correction_comparison.json results/charts/correction_improvement.png data/labels/llm_labels_v2.csv data/labels/llm_labels_v2_corrected.csv`
