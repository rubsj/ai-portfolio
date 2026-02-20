# P3 Day 3 Plan: Post-Training Evaluation + Documentation

## Context

Day 2 completed: standard model (Spearman 0.853) and LoRA model (Spearman 0.827) trained and saved. Baseline metrics at `eval/baseline_metrics.json` show inverted embeddings (Spearman -0.219, margin -0.083). Day 3 must evaluate both fine-tuned models using the same 8-metric pipeline, generate comparison charts, and write portfolio documentation.

User priority: T3.1–T3.4 (evaluation + charts + report + FP analysis) first, then T3.6 (README), T3.5 (ADRs).

---

## Hard Constraints

- **Memory**: Load ONE model at a time. `del model; gc.collect()` between each. Never have baseline + standard + LoRA loaded simultaneously.
- **Chart colors** (enforce everywhere): Baseline `#9E9E9E` (gray), Standard `#2196F3` (blue), LoRA `#FF9800` (orange)
- **Metric reuse**: Import and call existing functions from `src/metrics.py` and `src/baseline_analysis.py` — do NOT re-implement any metric calculation.
- **Baseline path resolution**: Check `eval/baseline_metrics.json` first, fall back to `data/baseline/baseline_metrics.json`. Same pattern for `.npz` embeddings.

## Sanity Check Gates (STOP if these fail)

| Gate | When | Condition |
|------|------|-----------|
| G1 | After `evaluate` | Standard margin > 0, LoRA margin > 0 |
| G2 | After `evaluate` | AUC-ROC > 0.85 for both models |
| G3 | After `compare` | 8 PNG files exist in `eval/visualizations/comparison/` |
| G4 | After `compare` | `comparison_report.html` file size > 100KB (charts embedded) |

---

## Files to Create/Modify

| # | File | Action | ~Lines |
|---|------|--------|--------|
| 1 | `src/models.py` | Add `EvaluationBundle` dataclass | +15 |
| 2 | `src/post_training_eval.py` | NEW — evaluation pipeline | ~200 |
| 3 | `src/comparison.py` | NEW — 8 charts + HTML report + FP analysis | ~550 |
| 4 | `src/cli.py` | Add `evaluate` and `compare` commands | +60 |
| 5 | `README.md` | NEW — portfolio README with Mermaid diagram | ~150 |
| 6 | `docs/adr/adr-001-lora-vs-standard.md` | NEW | ~80 |
| 7 | `docs/adr/adr-002-qlora-skip.md` | NEW | ~40 |
| 8 | `docs/adr/adr-003-cosine-similarity-loss.md` | NEW | ~60 |
| 9 | `tests/test_post_training_eval.py` | NEW | ~80 |
| 10 | `tests/test_comparison.py` | NEW | ~120 |

---

## Step 1: Add `EvaluationBundle` to `src/models.py`

Add a dataclass (not Pydantic — no validation needed for numpy arrays) at the bottom of `src/models.py`:

```python
@dataclass
class EvaluationBundle:
    metrics: BaselineMetrics
    similarities: np.ndarray      # (295,)
    projections: np.ndarray       # (590, 2) — UMAP
    cluster_labels: np.ndarray    # (295,)
    labels: list[int]
    categories: list[str]
    pair_types: list[str]
```

Add `import numpy as np` and `from dataclasses import dataclass` to imports.

**Commit**: `feat(p3): add EvaluationBundle dataclass to models`

---

## Step 2: Create `src/post_training_eval.py` (T3.1)

**Purpose:** Load each fine-tuned model ONE AT A TIME, generate embeddings, compute all 8 metrics, return `EvaluationBundle`.

### Imports to reuse (Correction #3 — exact signatures from existing code):

```python
# From src/metrics.py — all 8 metric functions:
from src.metrics import (
    compute_cosine_similarities,    # (emb1: ndarray, emb2: ndarray) → ndarray shape (N,)
    compute_margin,                 # (similarities: ndarray, labels: list[int]) → (compat_mean, incompat_mean, margin)
    compute_cohens_d,               # (group1: ndarray, group2: ndarray) → float
    compute_welch_ttest,            # (group1: ndarray, group2: ndarray) → (t_stat, p_value)
    compute_spearman,               # (similarities: ndarray, labels: list[int]) → float
    compute_roc_metrics,            # (similarities: ndarray, labels: list[int]) → (auc_roc, best_thresh, best_f1, acc, prec, rec)
    compute_false_positive_analysis,# (similarities: ndarray, labels: list[int], pair_types: list[str], threshold: float) → dict[str,int]
    compute_category_metrics,       # (similarities: ndarray, labels: list[int], groups: list[str]) → list[CategoryMetrics]
)

# From src/baseline_analysis.py — embedding I/O + dimensionality reduction:
from src.baseline_analysis import (
    load_embeddings,                # (path: Path) → (text1_emb: ndarray, text2_emb: ndarray)
    run_umap_projection,            # (emb1: ndarray, emb2: ndarray, random_state=42) → ndarray shape (2N, 2)
    run_hdbscan_clustering,         # (projections: ndarray, labels: list[int]) → (purity, n_clusters, noise_ratio, cluster_labels)
)

# From src/data_loader.py:
from src.data_loader import load_pairs, pairs_to_texts, get_categories, get_pair_types
```

### Key functions:

**`resolve_baseline_path(filename: str) → Path`** (Correction #2)
```python
def resolve_baseline_path(filename: str) -> Path:
    for parent in [Path("eval"), Path("data/baseline")]:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find {filename} in eval/ or data/baseline/")
```
Apply the same pattern for baseline embeddings (check `data/embeddings/baseline_eval.npz`).

**`generate_finetuned_embeddings(model_path: str, eval_pairs, output_path: Path, is_lora: bool = False)`**
- Standard: `SentenceTransformer(model_path)`
- LoRA (Correction #1 — try/except fallback):
```python
try:
    # Try loading as merged SentenceTransformer first
    model = SentenceTransformer(model_path)
except Exception:
    # Fallback: load base + apply LoRA adapter + merge
    base = SentenceTransformer("all-MiniLM-L6-v2")
    from peft import PeftModel
    base[0].auto_model = PeftModel.from_pretrained(
        base[0].auto_model, model_path
    )
    base[0].auto_model = base[0].auto_model.merge_and_unload()
    model = base
```
- Encode eval pairs via `pairs_to_texts()`, save to `.npz`, `del model; gc.collect()`

**`evaluate_from_embeddings(embeddings_path: Path, eval_pairs) → EvaluationBundle`**
- Load `.npz` via `load_embeddings()`
- Compute cosine similarities via `compute_cosine_similarities()`
- Compute all 8 metrics by calling the imported functions (NOT re-implementing)
- Run UMAP via `run_umap_projection()`, HDBSCAN via `run_hdbscan_clustering()`
- Assemble `BaselineMetrics` instance (same schema as baseline — enables apples-to-apples comparison)
- Return `EvaluationBundle` with metrics + intermediate arrays for charts

**`run_post_training_evaluation() → (EvaluationBundle, EvaluationBundle)`**
Orchestrator:
1. Load eval pairs from `data/raw/eval_pairs.jsonl`
2. Generate standard embeddings → save `data/embeddings/finetuned_eval.npz` → `del model; gc.collect()`
3. Evaluate standard → save `eval/finetuned_metrics.json`
4. Generate LoRA embeddings → save `data/embeddings/lora_eval.npz` → `del model; gc.collect()`
5. Evaluate LoRA → save `eval/lora_metrics.json`
6. Print summary table with key metrics side-by-side

**Commit**: `feat(p3): add post-training evaluation pipeline`
**Tests**: Write `tests/test_post_training_eval.py` alongside — mock SentenceTransformer, test `evaluate_from_embeddings` with fake NPZ data, verify EvaluationBundle shapes and metric ranges.
**Commit**: `test(p3): add post-training evaluation tests`

---

## Step 3: RUN evaluation + verify (Gate G1, G2)

```bash
cd 03-fine-tuning-guardrails
uv run python -m src.cli evaluate --mode all
```

**STOP AND VERIFY before continuing:**
- Standard Spearman should be ~0.85, margin MUST be positive
- LoRA Spearman should be ~0.83, margin MUST be positive
- AUC-ROC > 0.85 for both
- If margin is negative → model loading failed, debug before continuing

---

## Step 4: Create `src/comparison.py` — Split into 3 commits (Correction #4)

**Purpose:** Generate all comparison charts, HTML report, and false positive analysis.

### Color constants (top of file):
```python
COLOR_BASELINE = "#9E9E9E"  # gray
COLOR_STANDARD = "#2196F3"  # blue
COLOR_LORA = "#FF9800"      # orange
```

### Commit 4a: Charts 1–4 + tests

| # | Function | Chart Type | Input |
|---|----------|-----------|-------|
| 1 | `plot_comparison_cosine_distributions` | 2-row subplot: top=compatible KDEs, bottom=incompatible KDEs, 3 overlaid per row | 3x similarities + labels |
| 2 | `plot_comparison_umap` | 1x3 scatter panels (baseline, standard, LoRA) | 3x projections + labels |
| 3 | `plot_comparison_cluster_purity` | 3-bar chart | 3x purity floats |
| 4 | `plot_comparison_roc_curves` | 3 ROC curves on same axes + AUC in legend | 3x similarities + labels |

Run `ruff check src/comparison.py` and `uv run pytest tests/test_comparison.py -v` after.

**Commit**: `feat(p3): add comparison charts 1-4 (distributions, UMAP, purity, ROC)`

### Commit 4b: Charts 5–8 + FP analysis + tests

| # | Function | Chart Type | Input |
|---|----------|-----------|-------|
| 5 | `plot_comparison_category_heatmap` | seaborn heatmap: rows=categories, cols=models, cells=margin | 3x category_metrics |
| 6 | `plot_comparison_cohens_d` | 3 bars, color-coded by magnitude (red<0.2, yellow 0.2-0.8, green>0.8) | 3x d values |
| 7 | `plot_comparison_classification_metrics` | Grouped bar: Accuracy, Precision, Recall, F1 for 3 models side by side (PRD T3.2 #6) | 3x BaselineMetrics |
| 8 | `plot_comparison_false_positives` | Grouped horizontal bar: pair_types x 3 models | 3x FP count dicts |

**`write_false_positive_analysis(baseline_metrics, standard_metrics, lora_metrics)`**
- Markdown text file comparing FP counts per pair_type across 3 models
- Shows reduction percentages
- Save to `eval/false_positive_analysis.txt`

Run `ruff check` and `pytest` after.

**Commit**: `feat(p3): add comparison charts 5-8, FP analysis`

### Commit 4c: HTML report + orchestrator + integration test

**`generate_comparison_report_html(baseline_metrics, standard_metrics, lora_metrics, chart_paths)`**
- Self-contained HTML with base64-embedded PNGs (follow `visualizations.generate_baseline_report_html()` pattern)
- Summary table with 3 columns (baseline, standard, LoRA) x all metrics
- Each chart in its own section

**`build_comparison_result(baseline, standard, lora) → ComparisonResult`**
- Compute improvement deltas (standard vs baseline) using existing `ComparisonResult` schema from `src/models.py`
- Save to `eval/comparison_result.json`

**`generate_all_comparison_charts(baseline_bundle, standard_bundle, lora_bundle) → dict[str, Path]`**
- Orchestrator: calls all 8 chart functions, returns path dict for HTML report

**`run_comparison()`**
- Main orchestrator for the `compare` CLI command
- Loads 3 metrics JSONs + 3 embedding NPZs (NO model loading — fast re-run)
- Rebuilds EvaluationBundles from saved data
- Generates charts, HTML, FP analysis, ComparisonResult

Run `ruff check` and `pytest` after.

**Commit**: `feat(p3): add HTML comparison report and orchestrator`

All charts saved to `eval/visualizations/comparison/` as PNG at 150+ DPI.

---

## Step 5: Modify `src/cli.py`

Add two commands to existing Click CLI:
- `evaluate --mode standard|lora|all` — runs `post_training_eval.run_post_training_evaluation()`
- `compare` — runs `comparison.run_comparison()` (works from saved files, no model loading)

**Commit**: `feat(p3): add evaluate and compare CLI commands`

---

## Step 6: RUN comparison + verify (Gate G3, G4)

```bash
uv run python -m src.cli compare
```

**STOP AND VERIFY:**
- 8 PNG files exist in `eval/visualizations/comparison/`
- `eval/comparison_report.html` file size > 100KB (charts embedded as base64)
- `eval/false_positive_analysis.txt` shows FP reduction per pair_type
- `eval/comparison_result.json` has correct improvement deltas

---

## Step 7: Write README.md (T3.6)

Portfolio README with:
- Problem statement (1-2 paragraphs)
- Results table: baseline vs standard vs LoRA (fill with actual metrics from Step 3/6)
- Mermaid architecture diagram (flowchart LR)
- ADR links
- Quick start commands
- Deliverables checklist
- Project structure tree

**Commit**: `docs(p3): add README with results and architecture diagram`

---

## Step 8: Write ADRs (T3.5)

Three files in `docs/adr/`:
1. **adr-001-lora-vs-standard.md** — Actual results table, when to use each, Java/TS parallel (Decorator pattern)
2. **adr-002-qlora-skip.md** — CUDA dependency, model too small, revisit at 7B+
3. **adr-003-cosine-similarity-loss.md** — Why not ContrastiveLoss/TripletLoss, training-eval metric alignment

Fill ADR-001 results table with actual metrics from Step 3/6.

**Commit**: `docs(p3): add ADR-001, ADR-002, ADR-003`

---

## Step 9: Final verification

```bash
uv run pytest tests/ -v          # all tests pass
ruff check src/ tests/           # no lint errors
```

**Commit** (if any fixes needed): `fix(p3): address lint/test issues`

---

## Step 10: Git workflow

```bash
git checkout -b feat/p3-day3-evaluation
# Commits happen incrementally per steps above (total ~8-10 commits)
git push origin feat/p3-day3-evaluation:feat/p3-day3-evaluation
gh pr create --title "feat(p3): Day 3 — post-training evaluation + documentation"
```

---

## Execution Order Summary

| # | Action | Gate |
|---|--------|------|
| 1 | `src/models.py` — add EvaluationBundle → commit | — |
| 2 | `src/post_training_eval.py` + `tests/test_post_training_eval.py` → 2 commits | — |
| 3 | **RUN** `evaluate --mode all` | G1: margin>0, G2: AUC>0.85 |
| 4a | `src/comparison.py` charts 1–4 + tests → commit | — |
| 4b | `src/comparison.py` charts 5–8 + FP → commit | — |
| 4c | `src/comparison.py` HTML + orchestrator → commit | — |
| 5 | `src/cli.py` — add evaluate + compare → commit | — |
| 6 | **RUN** `compare` | G3: 8 PNGs, G4: HTML>100KB |
| 7 | `README.md` → commit | — |
| 8 | `docs/adr/` (3 files) → commit | — |
| 9 | `pytest` + `ruff` final check | all pass |
| 10 | Push branch, create PR | — |

---

## Verification Checklist

1. `eval/finetuned_metrics.json` exists with positive Spearman (~0.85)
2. `eval/lora_metrics.json` exists with positive Spearman (~0.83)
3. `eval/visualizations/comparison/` has 8 PNG files
4. `eval/comparison_report.html` opens in browser with all charts visible, size > 100KB
5. `eval/false_positive_analysis.txt` shows FP reduction per pair_type
6. `eval/comparison_result.json` has correct improvement deltas
7. All 3 ADRs in `docs/adr/` with actual metrics
8. `README.md` has Mermaid diagram rendering, results table with actual values
9. `uv run pytest tests/ -v` passes
10. Memory stayed under 8GB (one model loaded at a time)
