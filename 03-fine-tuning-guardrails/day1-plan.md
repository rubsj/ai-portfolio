# P3 Day 1 Implementation Plan — Data Evaluation + Baseline Analysis

## Context

**Why:** P3 fine-tunes `all-MiniLM-L6-v2` on dating compatibility data using contrastive loss. Day 1 establishes the "before" baseline — every metric needs a pre-training number so Day 3 can show the improvement. Without a solid baseline, the fine-tuning story has no anchor.

**What prompted it:** P3 is Week 2 of the portfolio sprint. Data files (1,195 train + 295 eval pairs) are already in `data/raw/`. No code exists yet — fresh start.

**Intended outcome:** By end of Day 1, `eval/baseline_metrics.json` exists with all 8 metrics, `eval/baseline_report.html` is viewable, and every chart is saved. Ready for Day 2 training.

**Hardware constraint:** 8GB M2 — never load two models simultaneously, `batch_size=16`, `del model; gc.collect()` after every use.

---

## File Creation Order

```
1. pyproject.toml                    (Day 0 — deps)
2. src/__init__.py                   (T1.1 — empty)
3. src/models.py                     (T1.2 — all Pydantic schemas)
4. src/data_loader.py                (T1.2 — JSONL loading)
5. tests/test_data_loader.py         (T1.2 — validation tests)
6. src/data_evaluator.py             (T1.3 — 5-dimension scoring)
7. tests/test_evaluator.py           (T1.3 — evaluator tests)
8. src/metrics.py                    (T1.5 — reusable metric functions)
9. tests/test_metrics.py             (T1.5 — metric correctness tests)
10. src/baseline_analysis.py         (T1.4–T1.7 — embeddings + UMAP + HDBSCAN)
11. src/visualizations.py            (T1.8 — all chart generation)
12. Run baseline pipeline            (T1.4–T1.8 — execute end-to-end)
```

---

## Day 0: Project Setup

### D0.1: Directory Structure
```bash
cd /Users/prathamjha/Ruby/projects/ai-portfolio/03-fine-tuning-guardrails
mkdir -p src tests data/evaluation data/embeddings
mkdir -p training/model/standard_model training/model/lora_model
mkdir -p eval/visualizations/baseline eval/visualizations/comparison
mkdir -p docs
```

### D0.2: pyproject.toml
File: `03-fine-tuning-guardrails/pyproject.toml`

```toml
[project]
name = "fine-tuning-guardrails"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "sentence-transformers>=3.0",
    "torch>=2.0",
    "peft>=0.12",
    "umap-learn>=0.5",
    "hdbscan>=0.8",
    "scikit-learn>=1.4",
    "scipy>=1.12",
    "matplotlib>=3.8",
    "seaborn>=0.13",
    "plotly>=5.18",
    "pydantic>=2.5",
    "click>=8.1",
    "rich>=13.0",
    "numpy>=1.26",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.4",
]

[tool.pytest.ini_options]
pythonpath = ["."]
```

### D0.3: Run `uv sync`
```bash
cd /Users/prathamjha/Ruby/projects/ai-portfolio/03-fine-tuning-guardrails
uv sync
```

### D0.4: Verify data files exist
Confirm all 4 files in `data/raw/`: `dating_pairs.jsonl`, `eval_pairs.jsonl`, `dating_pairs_metadata.json`, `eval_pairs_metadata.json`.

### D0.5: Create P3 card in Notion Project Tracker
Use Notion MCP `notion-create-pages` to create a page in data source `collection://4eb4a0f8-83c5-4a78-af3a-10491ba75327` with:
- Title: "P3 — Contrastive Embedding Fine-Tuning"
- Status: "In Progress"
- Timeline: "Feb 18–20"

---

## T1.1: Project Setup

Create `src/__init__.py` (empty file). Verify `uv sync` completed and `uv run python -c "import sentence_transformers; print('OK')"` works.

---

## T1.2: Data Loading + Pydantic Models

### File: `src/models.py` — ALL Pydantic schemas

```python
from __future__ import annotations
from enum import IntEnum
from pydantic import BaseModel, Field, field_validator

class CompatibilityLabel(IntEnum):
    INCOMPATIBLE = 0
    COMPATIBLE = 1

class DatingPair(BaseModel):
    text_1: str = Field(..., description="First person's statement (boy/girl: ...)")
    text_2: str = Field(..., description="Second person's statement (boy/girl: ...)")
    label: CompatibilityLabel
    category: str
    subcategory: str
    pair_type: str

    @field_validator("text_1", "text_2")
    @classmethod
    def validate_gender_prefix(cls, v: str) -> str:
        if ":" not in v:
            raise ValueError(f"Text must follow 'gender: statement' format, got: {v}")
        gender = v.split(":", 1)[0].strip()
        if gender not in ("boy", "girl"):
            raise ValueError(f"Gender prefix must be 'boy' or 'girl', got: {gender}")
        return v

class DataQualityScore(BaseModel):
    data_quality: float = Field(..., ge=0, le=100)
    diversity: float = Field(..., ge=0, le=100)
    bias: float = Field(..., ge=0, le=100)
    linguistic_quality: float = Field(..., ge=0, le=100)
    overall: float = Field(..., ge=0, le=100)

class DimensionDetail(BaseModel):
    """Per-dimension breakdown with sub-scores."""
    sub_scores: dict[str, float]   # e.g. {"completeness": 100.0, "consistency": 100.0, ...}
    dimension_score: float

class DataQualityReport(BaseModel):
    scores: DataQualityScore
    details: dict[str, DimensionDetail]  # keys: "data_quality", "diversity", "bias", "linguistic_quality"
    record_count: int
    timestamp: str  # ISO format

class CategoryMetrics(BaseModel):
    category: str
    compatible_mean: float
    incompatible_mean: float
    margin: float
    count: int
    cohens_d: float | None = None  # None if group too small

class BaselineMetrics(BaseModel):
    compatible_mean_cosine: float
    incompatible_mean_cosine: float
    compatibility_margin: float
    cohens_d: float
    t_statistic: float
    p_value: float
    auc_roc: float
    best_threshold: float
    best_f1: float
    accuracy_at_best_threshold: float
    precision_at_best_threshold: float
    recall_at_best_threshold: float
    cluster_purity: float
    n_clusters: int
    noise_ratio: float
    spearman_correlation: float
    false_positive_counts: dict[str, int] = Field(default_factory=dict)  # PRD Metric #4: FP count per pair_type
    category_metrics: list[CategoryMetrics] = Field(default_factory=list)
    pair_type_metrics: list[CategoryMetrics] = Field(default_factory=list)

class ComparisonResult(BaseModel):
    baseline: BaselineMetrics
    standard_finetuned: BaselineMetrics
    lora_finetuned: BaselineMetrics | None = None
    margin_improvement: float
    margin_improvement_pct: float
    cohens_d_improvement: float
    spearman_improvement: float
```

### File: `src/data_loader.py`

```python
def load_pairs(path: Path) -> list[DatingPair]:
    """Load JSONL file, validate each record with Pydantic.
    Raises ValueError if any record fails validation."""
    # Read file line-by-line, json.loads each, DatingPair.model_validate(record)
    # Return list of validated DatingPair objects

def load_metadata(path: Path) -> dict:
    """Load metadata JSON file. Returns raw dict."""

def pairs_to_texts(pairs: list[DatingPair]) -> tuple[list[str], list[str], list[int]]:
    """Extract parallel lists: text_1s, text_2s, labels (as int)."""

def get_categories(pairs: list[DatingPair]) -> list[str]:
    """Extract category list aligned with pairs."""

def get_pair_types(pairs: list[DatingPair]) -> list[str]:
    """Extract pair_type list aligned with pairs."""
```

### File: `tests/test_data_loader.py`
- `test_load_pairs_valid`: Load actual dating_pairs.jsonl, assert len == 1195
- `test_load_pairs_eval`: Load eval_pairs.jsonl, assert len == 295
- `test_dating_pair_valid_record`: Construct DatingPair with valid data, assert no error
- `test_dating_pair_invalid_gender`: text_1="cat: hello" → should raise ValidationError
- `test_dating_pair_no_colon`: text_1="no colon here" → should raise ValidationError
- `test_dating_pair_invalid_label`: label=2 → should raise ValidationError
- `test_pairs_to_texts`: Verify returns correct parallel lists
- `test_all_records_have_gender_prefix`: Load all training pairs, assert all have boy/girl prefix

---

## T1.3: SyntheticDataEvaluator

### File: `src/data_evaluator.py`

```python
class SyntheticDataEvaluator:
    def __init__(self, pairs: list[DatingPair]):
        self.pairs = pairs

    def evaluate(self) -> DataQualityReport:
        """Run all 4 dimensions, compute overall, return DataQualityReport."""

    # --- Dimension 1: Data Quality ---
    def _evaluate_data_quality(self) -> DimensionDetail:
        """4 sub-scores → average."""

    def _score_completeness(self) -> float:
        # % of records where all 6 fields are non-empty strings
        # Should be 100.0 for this dataset

    def _score_consistency(self) -> float:
        # % of records where label ∈ {0, 1}
        # Already enforced by Pydantic, but check raw data

    def _score_duplicates(self) -> float:
        # Count duplicate (text_1, text_2) tuples
        # Score = (1 - n_duplicates / n_total) * 100

    def _score_format(self) -> float:
        # % of texts matching r'^(boy|girl):\s+\S'
        # Check both text_1 and text_2

    # --- Dimension 2: Diversity ---
    def _evaluate_diversity(self) -> DimensionDetail:

    def _score_vocabulary_richness(self) -> float:
        # Collect all words (lowercased, split on whitespace) from all texts
        # Score = (unique_words / total_words) * 100, capped at 100

    def _score_category_entropy(self) -> float:
        # Shannon entropy of category distribution
        # H = -Σ p_i * log2(p_i)
        # Normalize: H / log2(n_categories) * 100
        # Use: from collections import Counter; math.log2

    def _score_label_balance(self) -> float:
        # min(n_compatible, n_incompatible) / max(...) * 100

    def _score_text_complexity(self) -> float:
        # avg_words = mean word count across all texts
        # if 5 <= avg_words <= 15: score = 100
        # elif avg_words < 3 or avg_words > 30: score = 50
        # else: linear interpolation

    # --- Dimension 3: Bias ---
    def _evaluate_bias(self) -> DimensionDetail:

    def _score_gender_bias(self) -> float:
        # Extract gender from text_1 (boy/girl)
        # Chi-squared test: gender × label independence
        # scipy.stats.chi2_contingency(contingency_table)
        # If p_value > 0.05: score = 100 (no bias detected)
        # Else: score = p_value / 0.05 * 100 (partial credit)

    def _score_category_label_correlation(self) -> float:
        # Chi-squared test: category × label independence
        # Same scoring as gender bias

    def _score_length_label_correlation(self) -> float:
        # Compute avg text length (words) per pair: (len(text_1) + len(text_2)) / 2
        # scipy.stats.pointbiserialr(labels, lengths)
        # |r| < 0.1 → 100; |r| > 0.3 → 0; linear between

    def _score_vocabulary_label_bias(self) -> float:
        # Collect word sets for compatible vs incompatible pairs
        # Jaccard similarity: |intersection| / |union| * 100

    # --- Dimension 4: Linguistic Quality ---
    def _evaluate_linguistic_quality(self) -> DimensionDetail:

    def _score_readability(self) -> float:
        # Flesch Reading Ease (simplified — count syllables by vowel groups)
        # FRE = 206.835 - 1.015*(total_words/total_sentences) - 84.6*(total_syllables/total_words)
        # Treat each text as 1 sentence
        # Map: FRE 60-80 → 100; FRE < 30 or > 100 → 60; linear between

    def _count_syllables(self, word: str) -> int:
        # Count vowel groups (a,e,i,o,u) as syllables
        # Minimum 1 syllable per word

    def _score_coherence(self) -> float:
        # For each pair: word overlap = |set(text_1_words) ∩ set(text_2_words)| / min(|set1|, |set2|)
        # Average across all pairs
        # Map: overlap 0.1-0.3 → 100 (some shared context is natural)
        # overlap > 0.5 → 70 (too similar); overlap 0 → 70 (no shared context)

    def _score_naturalness(self) -> float:
        # Collect all bigrams from all texts
        # unique_bigrams / total_bigrams * 100, capped at 100

    def _score_repetition(self) -> float:
        # most_common_bigram_frequency = count of most frequent bigram / total bigrams
        # Score = (1 - most_common_bigram_frequency) * 100
```

**Output generation (at end of evaluate()):**
- Save `DataQualityReport` to `data/evaluation/data_quality_report.json`
- Generate human-readable `data/evaluation/data_quality_summary.txt`
- Print Rich console table showing all scores with pass/fail (≥60 = pass)

### File: `tests/test_evaluator.py`
- `test_evaluate_returns_report`: Run evaluator on first 50 pairs of real data, check report structure
- `test_all_dimensions_in_range`: All scores between 0-100
- `test_overall_is_average`: overall == mean(data_quality, diversity, bias, linguistic_quality)
- `test_perfect_completeness`: All records complete → completeness = 100
- `test_syllable_counter`: Known words → known syllable counts ("hello" → 2, "a" → 1, "beautiful" → 3)

---

## T1.4: Generate Baseline Embeddings

### File: `src/baseline_analysis.py`

```python
import gc
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = Path("data/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16

def generate_and_save_embeddings(
    text_1s: list[str],
    text_2s: list[str],
    save_path: Path,
    model_name: str = MODEL_NAME,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Load model, encode both text lists, save to .npz, cleanup.

    Memory protocol:
    1. Load model (~90MB)
    2. Encode text_1s (batch_size=16)
    3. Encode text_2s (batch_size=16)
    4. np.savez(save_path, text1=emb1, text2=emb2)
    5. del model → gc.collect()
    """
    model = SentenceTransformer(model_name)
    emb1 = model.encode(text_1s, batch_size=batch_size, show_progress_bar=True)
    emb2 = model.encode(text_2s, batch_size=batch_size, show_progress_bar=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, text1=emb1, text2=emb2)
    del model
    gc.collect()

def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load saved embeddings. Returns (text1_emb, text2_emb)."""
    data = np.load(path)
    return data["text1"], data["text2"]
```

**Execution flow:**
1. Load train pairs → extract text_1s, text_2s
2. `generate_and_save_embeddings(text_1s, text_2s, EMBEDDINGS_DIR / "baseline_train.npz")`
3. Load eval pairs → extract text_1s, text_2s
4. `generate_and_save_embeddings(text_1s, text_2s, EMBEDDINGS_DIR / "baseline_eval.npz")`
   - NOTE: Model loaded TWICE (once per split) but never simultaneously. This is correct per memory protocol.
   - OPTIMIZATION: Could combine into single model load if memory allows. Sonnet should try combining first — load model once, encode train texts, encode eval texts, save both, del model. If OOM, split into two loads.

**Better approach (single model load):**
```python
def generate_all_baseline_embeddings(
    train_pairs: list[DatingPair],
    eval_pairs: list[DatingPair],
) -> None:
    """Generate and save embeddings for both train and eval splits in one model load."""
    model = SentenceTransformer(MODEL_NAME)

    # Train embeddings
    train_t1, train_t2, _ = pairs_to_texts(train_pairs)
    train_emb1 = model.encode(train_t1, batch_size=BATCH_SIZE, show_progress_bar=True)
    train_emb2 = model.encode(train_t2, batch_size=BATCH_SIZE, show_progress_bar=True)
    np.savez(EMBEDDINGS_DIR / "baseline_train.npz", text1=train_emb1, text2=train_emb2)

    # Eval embeddings
    eval_t1, eval_t2, _ = pairs_to_texts(eval_pairs)
    eval_emb1 = model.encode(eval_t1, batch_size=BATCH_SIZE, show_progress_bar=True)
    eval_emb2 = model.encode(eval_t2, batch_size=BATCH_SIZE, show_progress_bar=True)
    np.savez(EMBEDDINGS_DIR / "baseline_eval.npz", text1=eval_emb1, text2=eval_emb2)

    # WHY single cleanup: both splits encoded in one session, model released once
    del model
    gc.collect()
```

---

## T1.5: Baseline Cosine Similarity Analysis

### File: `src/metrics.py` — Reusable metric functions

```python
import numpy as np
from scipy import stats
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score, precision_score, recall_score
from src.models import CategoryMetrics

def compute_cosine_similarities(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between paired embeddings.
    Returns array of shape (N,) with cosine sim for each pair.

    WHY not sklearn.cosine_similarity: that computes NxN matrix.
    We only need the diagonal (paired similarities).
    """
    # Normalize rows to unit vectors
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    # Row-wise dot product = cosine similarity for unit vectors
    return np.sum(emb1_norm * emb2_norm, axis=1)

def compute_margin(
    similarities: np.ndarray, labels: list[int]
) -> tuple[float, float, float]:
    """Returns (compatible_mean, incompatible_mean, margin).
    Margin = compatible_mean - incompatible_mean.
    """
    labels_arr = np.array(labels)
    compat_sims = similarities[labels_arr == 1]
    incompat_sims = similarities[labels_arr == 0]
    compat_mean = float(np.mean(compat_sims))
    incompat_mean = float(np.mean(incompat_sims))
    return compat_mean, incompat_mean, compat_mean - incompat_mean

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two groups.
    d = (mean1 - mean2) / pooled_std
    pooled_std = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)

def compute_welch_ttest(
    group1: np.ndarray, group2: np.ndarray
) -> tuple[float, float]:
    """Welch's t-test (unequal variances). Returns (t_statistic, p_value).
    Use: scipy.stats.ttest_ind(group1, group2, equal_var=False)
    """
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    return float(t_stat), float(p_val)

def compute_spearman(
    similarities: np.ndarray, labels: list[int]
) -> float:
    """Spearman rank correlation between cosine similarities and labels.
    Use: scipy.stats.spearmanr(similarities, labels)
    Returns the correlation coefficient (not p-value).
    """
    corr, _ = stats.spearmanr(similarities, labels)
    return float(corr)

def compute_roc_metrics(
    similarities: np.ndarray, labels: list[int]
) -> tuple[float, float, float, float, float, float]:
    """Compute ROC-based metrics.
    Returns (auc_roc, best_threshold, best_f1, accuracy, precision, recall).

    Sweep thresholds to find best F1, then report other metrics at that threshold.
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    auc_roc = float(auc(fpr, tpr))

    # Sweep thresholds for best F1
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.0, 1.01, 0.01):
        preds = (similarities >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Metrics at best threshold
    preds = (similarities >= best_thresh).astype(int)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)

    return float(auc_roc), float(best_thresh), float(best_f1), float(acc), float(prec), float(rec)

def compute_false_positive_analysis(
    similarities: np.ndarray,
    labels: list[int],
    pair_types: list[str],
    threshold: float,
) -> dict[str, int]:
    """PRD Metric #4: False positive analysis.
    Finds incompatible pairs (label=0) with cosine sim >= threshold,
    groups them by pair_type, returns counts per pair_type.

    WHY this matters: identifies which pair_types the model confuses most —
    these are the "before" numbers for Day 3's false positive reduction chart.
    """
    labels_arr = np.array(labels)
    pair_types_arr = np.array(pair_types)
    # Mask: incompatible pairs that the model would classify as compatible
    fp_mask = (labels_arr == 0) & (similarities >= threshold)
    fp_pair_types = pair_types_arr[fp_mask]
    # Count per pair_type
    from collections import Counter
    counts = Counter(fp_pair_types)
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

def compute_category_metrics(
    similarities: np.ndarray,
    labels: list[int],
    groups: list[str],
) -> list[CategoryMetrics]:
    """Per-group breakdown (works for both category and pair_type).
    For each unique group: compute compatible/incompatible means, margin, Cohen's d.
    """
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)
    results = []
    for group in sorted(set(groups)):
        mask = groups_arr == group
        group_sims = similarities[mask]
        group_labels = labels_arr[mask]
        compat = group_sims[group_labels == 1]
        incompat = group_sims[group_labels == 0]
        compat_mean = float(np.mean(compat)) if len(compat) > 0 else 0.0
        incompat_mean = float(np.mean(incompat)) if len(incompat) > 0 else 0.0
        margin = compat_mean - incompat_mean
        d = compute_cohens_d(compat, incompat) if len(compat) >= 2 and len(incompat) >= 2 else None
        results.append(CategoryMetrics(
            category=group,
            compatible_mean=compat_mean,
            incompatible_mean=incompat_mean,
            margin=margin,
            count=int(mask.sum()),
            cohens_d=d,
        ))
    return results
```

### File: `tests/test_metrics.py`
- `test_cosine_identical_vectors`: Same vectors → similarity = 1.0
- `test_cosine_orthogonal_vectors`: Orthogonal → similarity = 0.0
- `test_cosine_opposite_vectors`: Negated → similarity = -1.0
- `test_margin_perfect_separation`: compatible all 0.9, incompatible all 0.1 → margin = 0.8
- `test_cohens_d_large_effect`: Two well-separated groups → d > 0.8
- `test_cohens_d_zero_effect`: Same distribution → d ≈ 0.0
- `test_spearman_perfect_correlation`: sims = [0.1, 0.2, 0.9, 1.0], labels = [0, 0, 1, 1] → high positive correlation
- `test_roc_random_classifier`: random sims → AUC ≈ 0.5
- `test_category_metrics_groups`: 2 categories with known values → verify per-group metrics
- `test_false_positive_analysis_basic`: sims=[0.8, 0.3, 0.7, 0.9], labels=[0, 0, 1, 1], pair_types=["dealbreaker", "subtle_mismatch", "compatible", "compatible"], threshold=0.5 → {"dealbreaker": 1} (only first pair is FP: label=0 and sim≥0.5)
- `test_false_positive_analysis_no_fps`: All incompatible pairs below threshold → empty dict
- `test_false_positive_analysis_multiple_types`: Multiple FPs across pair_types → verify counts and sorting

---

## T1.6: UMAP Visualization + HDBSCAN Clustering

### In `src/baseline_analysis.py` (continued)

```python
import umap
import hdbscan

def run_umap_projection(
    emb1: np.ndarray,
    emb2: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Project concatenated embeddings from 384d → 2d.

    Input: emb1 shape (N, 384), emb2 shape (N, 384)
    Concatenate to (2N, 384), project to (2N, 2).
    First N rows = text_1 projections, last N = text_2 projections.

    UMAP config: n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42
    """
    combined = np.vstack([emb1, emb2])
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    projections = reducer.fit_transform(combined)
    return projections  # shape (2N, 2)

def run_hdbscan_clustering(
    projections: np.ndarray,
    labels: list[int],
) -> tuple[float, int, float, np.ndarray]:
    """Cluster UMAP projections with HDBSCAN.

    Uses paired projections: for each pair i, average the two UMAP points
    (text_1[i] and text_2[i]) to get one point per pair. Then cluster.

    Returns (purity, n_clusters, noise_ratio, cluster_labels).

    Purity: for each non-noise cluster, fraction belonging to dominant label.
    Average across clusters, weighted by cluster size.
    """
    n = len(labels)
    # Average text_1 and text_2 projections per pair
    pair_projections = (projections[:n] + projections[n:]) / 2

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_method="eom",
    )
    cluster_labels = clusterer.fit_predict(pair_projections)

    # Compute purity
    labels_arr = np.array(labels)
    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)  # Remove noise cluster

    n_clusters = len(unique_clusters)
    noise_count = np.sum(cluster_labels == -1)
    noise_ratio = float(noise_count / len(cluster_labels))

    if n_clusters == 0:
        return 0.0, 0, noise_ratio, cluster_labels

    total_in_clusters = 0
    purity_sum = 0.0
    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_size = mask.sum()
        dominant_count = max(
            np.sum(labels_arr[mask] == 0),
            np.sum(labels_arr[mask] == 1),
        )
        purity_sum += dominant_count
        total_in_clusters += cluster_size

    purity = float(purity_sum / total_in_clusters) if total_in_clusters > 0 else 0.0

    return purity, n_clusters, noise_ratio, cluster_labels
```

---

## T1.7: Category-wise + Pair-type-wise Breakdown

Already handled by `compute_category_metrics()` in `src/metrics.py`. Call it twice:
1. `compute_category_metrics(sims, labels, categories)` → category breakdown
2. `compute_category_metrics(sims, labels, pair_types)` → pair_type breakdown

Store results in `BaselineMetrics.category_metrics` and `BaselineMetrics.pair_type_metrics`.

---

## T1.8: Save Metrics + Generate Charts + HTML Report

### Orchestrator function in `src/baseline_analysis.py`

```python
def run_full_baseline(
    train_pairs: list[DatingPair],
    eval_pairs: list[DatingPair],
) -> BaselineMetrics:
    """End-to-end baseline pipeline:
    1. Generate + save embeddings for BOTH train and eval (single model load)
    2. Load EVAL embeddings from disk (baseline_eval.npz)
       WHY eval not train: all metrics measure generalization on unseen data.
       Train embeddings are saved for Day 2 training but not used for baseline metrics.
    3. Compute cosine similarities on EVAL pairs
    4. Compute all metrics (margin, Cohen's d, t-test, Spearman, ROC, false positive analysis)
    5. Run UMAP + HDBSCAN on EVAL embeddings
    6. Generate all charts (7 charts: distributions, UMAP, HDBSCAN, ROC, category margins, pair_type margins, false positive breakdown)
    7. Save eval/baseline_metrics.json
    8. Generate eval/baseline_report.html
    """
```

### File: `src/visualizations.py` — All chart generation

Setup:
```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
BASELINE_VIZ_DIR = Path("eval/visualizations/baseline")
```

6 chart functions:

```python
def plot_cosine_distributions(
    similarities: np.ndarray,
    labels: list[int],
    save_path: Path | None = None,
) -> Path:
    """Overlapping histograms: compatible vs incompatible cosine similarity distributions.
    Use seaborn histplot with kde=True, alpha=0.5 for overlap.
    Annotate with vertical lines at each mean + margin text.
    """

def plot_umap_scatter(
    projections: np.ndarray,
    labels: list[int],
    categories: list[str],
    save_path: Path | None = None,
) -> Path:
    """Interactive plotly scatter plot of UMAP projections.
    Color by label (compatible=blue, incompatible=red).
    Hover shows category, pair_type.
    Save as HTML.
    """

def plot_hdbscan_clusters(
    projections: np.ndarray,
    cluster_labels: np.ndarray,
    save_path: Path | None = None,
) -> Path:
    """UMAP scatter colored by HDBSCAN cluster assignment.
    Noise points in gray. Use matplotlib scatter.
    Annotate with n_clusters and purity.
    """

def plot_roc_curve(
    similarities: np.ndarray,
    labels: list[int],
    save_path: Path | None = None,
) -> Path:
    """ROC curve with AUC annotation.
    Use sklearn.metrics.roc_curve. Plot diagonal reference line.
    """

def plot_category_margins(
    category_metrics: list[CategoryMetrics],
    title: str,
    save_path: Path | None = None,
) -> Path:
    """Horizontal bar chart of margin per category/pair_type.
    Sort by margin descending. Color gradient by margin value.
    """

def plot_false_positive_breakdown(
    false_positive_counts: dict[str, int],
    save_path: Path | None = None,
) -> Path:
    """PRD Metric #4 chart: Horizontal bar chart of false positive count per pair_type.
    Sort by count descending. This is the "before" chart for Day 3's
    "False Positive Reduction" comparison (#8 in PRD Section 4 Step 5).
    Use matplotlib horizontal barh. Annotate each bar with count.
    If no false positives, save a chart with "No false positives" text.
    """

def generate_baseline_report_html(
    metrics: BaselineMetrics,
    chart_paths: dict[str, Path],
    save_path: Path | None = None,
) -> Path:
    """Generate simple HTML report embedding all charts + metrics summary table.
    Use inline base64 for PNG images, iframe for HTML charts.
    Include summary table with all BaselineMetrics fields.
    """
```

### Save baseline_metrics.json
```python
# In run_full_baseline():
metrics_path = Path("eval/baseline_metrics.json")
metrics_path.parent.mkdir(parents=True, exist_ok=True)
metrics_path.write_text(metrics.model_dump_json(indent=2))
```

---

## Verification Plan

After all tasks complete, verify:

1. **Data loading:** `uv run python -c "from src.data_loader import load_pairs; pairs = load_pairs(Path('data/raw/dating_pairs.jsonl')); print(len(pairs))"`  → 1195
2. **Data quality:** `uv run python -c "..."` → DataQualityReport with overall ≥ 60
3. **Embeddings exist:** `ls data/embeddings/baseline_train.npz data/embeddings/baseline_eval.npz`
4. **Metrics saved:** `cat eval/baseline_metrics.json | python -m json.tool | head -20`
5. **Charts exist:** `ls eval/visualizations/baseline/` → 7 files (6 PNG + 1 HTML, includes false_positive_breakdown.png)
6. **HTML report:** `open eval/baseline_report.html` → viewable in browser
7. **Tests pass:** `uv run pytest tests/ -v`
8. **Lint clean:** `uv run ruff check src/ tests/`

---

## Git Strategy

- Branch: `feat/p3-day1-baseline` (create from `origin/main` before T1.1)
- **Commit after each completed task** for clean revert points (per CLAUDE.md: "After each file: run ruff check, run relevant tests, commit"):
  1. After D0 setup: `chore(p3): project setup — pyproject.toml, directory structure, uv sync`
  2. After T1.1: `feat(p3): project init — src/__init__.py, verify imports`
  3. After T1.2: `feat(p3): data loading — models.py, data_loader.py, tests`
  4. After T1.3: `feat(p3): data evaluator — SyntheticDataEvaluator, quality report`
  5. After T1.5: `feat(p3): metrics module — cosine, margin, cohen's d, ROC, false positive analysis`
  6. After T1.4+T1.6+T1.7+T1.8: `feat(p3): baseline analysis — embeddings, UMAP, HDBSCAN, charts, report`
- Before each commit: `uv run ruff check src/ tests/` + `uv run pytest tests/ -v`
- Push branch, create PR, merge to main

---

## Key Libraries/Functions Reference

| Task | Library | Function |
|------|---------|----------|
| Chi-squared test | `scipy.stats` | `chi2_contingency(table)` |
| Point-biserial | `scipy.stats` | `pointbiserialr(x, y)` |
| Welch's t-test | `scipy.stats` | `ttest_ind(a, b, equal_var=False)` |
| Spearman | `scipy.stats` | `spearmanr(a, b)` |
| ROC curve | `sklearn.metrics` | `roc_curve(y_true, y_score)` |
| AUC | `sklearn.metrics` | `auc(fpr, tpr)` |
| F1/Acc/Prec/Rec | `sklearn.metrics` | `f1_score`, `accuracy_score`, `precision_score`, `recall_score` |
| UMAP | `umap` | `UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)` |
| HDBSCAN | `hdbscan` | `HDBSCAN(min_cluster_size=10, min_samples=5, cluster_selection_method='eom')` |
| Cosine sim | `numpy` | Manual: normalize + dot product (NOT sklearn pairwise) |
| Cohen's d | Manual | `(mean1-mean2) / pooled_std` |
| Shannon entropy | `math` | `log2` for manual computation |
