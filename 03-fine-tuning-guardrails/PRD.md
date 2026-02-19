# PRD: P3 — Contrastive Embedding Fine-Tuning

> **This is the implementation contract.** Claude Code: read this + both CLAUDE.md files before starting.
> Do NOT re-debate architecture decisions. They are final. If something is ambiguous, ask the user.

**Project:** P3 — Contrastive Embedding Fine-Tuning (Dating Compatibility)
**Timeline:** Feb 18–20, 2026 (Tue–Thu, 3 sessions × 4h = 12h)
**Owner:** Developer (Java/TS background, completed P1 + P2)
**Source of Truth:** [Notion Requirements](https://www.notion.so/Mini_Project_3_Requirements-2ffdb630640a810cbfe5d23c33ee97c0)
**Concepts Primer:** `p3-concepts-primer.html` in project root — read for contrastive learning, LoRA, UMAP, HDBSCAN theory
**PRD Version:** v1

---

## 1. Objective

Build an end-to-end pipeline that **fine-tunes a sentence embedding model** using contrastive loss on dating compatibility data, then **systematically evaluates** whether fine-tuning reshapes the embedding space as intended.

The pipeline:

1. **Loads** provided synthetic dating pair data (1,195 train + 295 eval pairs)
2. **Validates** data quality using a comprehensive `SyntheticDataEvaluator` (5 dimensions, target ≥60% Overall Score)
3. **Establishes baseline** — generates embeddings with pre-trained `all-MiniLM-L6-v2`, measures cosine similarity separation, UMAP clustering, HDBSCAN purity
4. **Fine-tunes** with `CosineSimilarityLoss` (standard sentence-transformers training)
5. **Fine-tunes with LoRA** adapters via PEFT for parameter-efficient comparison
6. **Evaluates post-training** — re-runs all baseline metrics on the fine-tuned model
7. **Compares before/after** — side-by-side across 8 evaluation metrics
8. **Produces** publication-quality charts, interactive HTML report, and training curves

**The output is EVIDENCE, not a dating app.** The deliverable is the sentence: _"Fine-tuning improved compatibility margin from X to Y (Cohen's d from A to B), with category-wise analysis showing dealbreakers achieve near-perfect separation while subtle mismatches remain the hardest category."_ — backed by data.

**Success Criteria:**

- Spearman correlation on eval set: baseline ~0.76 → fine-tuned ≥0.86 (per requirements)
- Compatibility margin (compatible mean cosine - incompatible mean cosine): baseline ~0.04 → fine-tuned ≥0.20
- Cohen's d effect size: baseline <0.3 → fine-tuned >0.8

---

## 2. Architecture Decisions (FINAL — Do Not Re-Debate)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Data source** | Provided dataset (1,195 train + 295 eval pairs) | P1 already demonstrated synthetic data generation. Using provided data lets us focus on the core portfolio signal: fine-tuning + evaluation. We run the full `SyntheticDataEvaluator` to validate quality — this is the more realistic enterprise scenario (evaluating existing data, not generating from scratch). |
| **Base model** | `all-MiniLM-L6-v2` (22.7M params, 384 dims, ~90MB) | Fits in 8GB M2 RAM with room to spare. Small enough to fine-tune locally on CPU (slow but feasible). 384 dimensions are sufficient for dating compatibility signals. Same model used in P2 — provides portfolio continuity. |
| **Loss function** | `CosineSimilarityLoss` from `sentence-transformers` | Specified by requirements. Maps label=1 to cosine similarity 1.0 and label=0 to 0.0. The loss function pulls compatible pairs closer and pushes incompatible pairs apart in embedding space. |
| **Training approach** | Two-path comparison: Standard fine-tuning + LoRA via PEFT | Standard training updates all 22.7M parameters. LoRA adds low-rank adapters (~200K trainable params, <1% of total). Comparing both answers the interview question "When does LoRA help vs. hurt?" with empirical data. |
| **QLoRA / bitsandbytes** | **Skipped** — document in ADR as production technique | `bitsandbytes` NF4 quantization requires CUDA kernels. On Apple Silicon M2, it's experimental and unstable. For 22.7M params, quantization is unnecessary (model already fits in RAM). Time better spent on evaluation depth than debugging M2 compatibility. ADR documents when QLoRA matters (>1B param models). |
| **Instructional fine-tuning** | **Stretch goal** — Day 3 if time permits | Prepend instructions like "Family should outweigh hobbies" to pairs. Strong interview signal but incremental over base contrastive training. Core pipeline must be solid first. |
| **Cross-domain evaluation** | **Skipped** | Testing dating model on medical Q&A is conceptually interesting but low ROI. Before/after within dating domain is a complete evaluation story. |
| **Evaluation depth** | All 8 metrics from requirements — no shortcuts | This IS the portfolio differentiator. P2 proved that deep evaluation is where the value lives. Skipping data generation buys time for thorough evaluation. |
| **Visualization** | `matplotlib` + `seaborn` for static charts, `plotly` for interactive HTML report | Same stack as P2. Publication-quality before/after comparisons. Interactive UMAP with hover data. |
| **Demo** | Loom video + GitHub (not Streamlit) | Model weights are local artifacts. A Loom walkthrough showing the before/after transformation is more compelling than a web app for this project type. |

---

## 3. Provided Data — Schema & Validation

### 3a. Data Files

| File | Records | Purpose | Format |
|------|---------|---------|--------|
| `dating_pairs.jsonl` | 1,195 | Training data | JSONL |
| `eval_pairs.jsonl` | 295 | Evaluation (held out) | JSONL |
| `dating_pairs_metadata.json` | — | Generation stats | JSON |
| `eval_pairs_metadata.json` | — | Generation stats | JSON |
| `generate_dating_pairs.py` | — | Generation script (reference only) | Python |

### 3b. Record Schema

```jsonl
{
  "text_1": "boy: I love lazy Sundays",
  "text_2": "girl: I'm all about meeting new people",
  "label": 1,
  "category": "lifestyle",
  "subcategory": "relaxed_vs_extroverted",
  "pair_type": "subtle_mismatch"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text_1` | str | First person's statement, prefixed with `boy:` or `girl:` |
| `text_2` | str | Second person's statement, prefixed with `boy:` or `girl:` |
| `label` | int (0\|1) | 0 = incompatible, 1 = compatible |
| `category` | str | Preference domain (15 categories including compound like `lifestyle_and_values`) |
| `subcategory` | str | Specific preference within category |
| `pair_type` | str | One of 9 pair types (see distribution below) |

### 3c. Pre-Validated Data Statistics

**Field completeness:** 100% across all 6 fields — no nulls, no missing values.

**Label distribution (training):**
- Compatible (label=1): 628 (52.6%)
- Incompatible (label=0): 567 (47.4%)
- Balance ratio: 0.90 ✅ (no severe class imbalance)

**Pair type distribution (training):**

| Pair Type | Count | Actual % | Required % | Status |
|-----------|-------|----------|------------|--------|
| `compatible` | 240 | 20.1% | 20% | ✅ |
| `subtle_mismatch` | 230 | 19.2% | 19% | ✅ |
| `dealbreaker` | 180 | 15.1% | 15% | ✅ |
| `incompatible` | 180 | 15.1% | 15% | ✅ |
| `llm_judged_compatible` | 134 | 11.2% | 20% (combined) | ✅ |
| `llm_judged_incompatible` | 106 | 8.9% | ↑ | ✅ |
| `complex_compatible` | 63 | 5.3% | 10% (combined) | ✅ |
| `complex_incompatible` | 57 | 4.8% | ↑ | ✅ |
| `curated_realistic` | 5 | 0.4% | ~1% | ✅ |

**Category coverage:** 15 categories including compound categories (e.g., `lifestyle_and_values`). Core 5 domains: `lifestyle`, `interests`, `values`, `dealbreakers`, `realistic_example`.

**Quality signals already passing:**
- Dealbreaker sanity: All 180 dealbreaker pairs correctly labeled as incompatible (0 false positives) ✅
- Text format: All records follow `gender: statement` pattern ✅
- Text lengths: mean ~9 words (text_1), ~8 words (text_2) — typical for preference statements ✅

**Known characteristic (not a defect):** Subtle mismatch pairs skew toward compatible (190 vs 40). This is intentional — different hobbies don't necessarily mean incompatible. The model must learn this nuance.

---

## 4. Pipeline Steps — Detailed Specification

### Step 1: Data Loading & Quality Evaluation (Day 1, ~1.5h)

**Goal:** Load provided data, run comprehensive quality evaluation, document findings.

**Implementation — `SyntheticDataEvaluator`:**

Build evaluator with 5 scoring dimensions (0–100 each):

| Dimension | Weight | What It Measures | Key Checks |
|-----------|--------|------------------|------------|
| **Data Quality** | Equal | Completeness, consistency, duplicates, format | All fields present, labels ∈ {0,1}, no duplicate pairs, gender prefix format |
| **Diversity** | Equal | Vocabulary richness, category distribution entropy, label balance, text complexity | Unique words / total words ratio, Shannon entropy across categories, min/max label ratio |
| **Bias** | Equal | Gender bias, category-label correlation, length-label correlation, vocabulary-label bias | Chi-squared test for gender × label independence, point-biserial correlation for length × label |
| **Linguistic Quality** | Equal | Readability, coherence, naturalness, repetition | Flesch Reading Ease, word overlap between paired texts, n-gram repetition rate |
| **Overall** | Average | Simple average of 4 dimensions | Target: ≥60% |

**Output:**
- `data/evaluation/data_quality_report.json` — full scores with per-dimension breakdown
- `data/evaluation/data_quality_summary.txt` — human-readable findings
- Console output with Rich formatting showing pass/fail status

**Why run this on provided data:** Even though we pre-validated the schema, the evaluator checks *semantic* quality (vocabulary diversity, bias detection, linguistic naturalness) that schema validation can't catch. This is also the realistic enterprise scenario — you're handed a dataset and must assess whether it's fit for training.

```python
# Java parallel: This is like running a DataQualityValidator 
# against a pre-existing test dataset before training a model.
# You validate the data, not just the schema.

class SyntheticDataEvaluator:
    """Comprehensive quality evaluator for synthetic contrastive pairs.
    
    WHY this class: The requirements spec includes a full evaluation framework
    (scoring 0-100 across 5 dimensions). Running it on provided data proves
    you understand data quality assessment — the skill transfers to any
    ML project where you receive training data from another team.
    """
    
    def __init__(self, data_path: Path):
        self.records = self._load_jsonl(data_path)
    
    def evaluate(self) -> DataQualityReport:
        """Run all 5 evaluation dimensions, return overall score."""
        return DataQualityReport(
            data_quality=self._evaluate_data_quality(),      # Completeness, consistency
            diversity=self._evaluate_diversity(),              # Vocabulary, category entropy
            bias=self._evaluate_bias(),                        # Gender, label, length bias
            linguistic_quality=self._evaluate_linguistic(),    # Readability, coherence
        )
```

---

### Step 2: Baseline Analysis (Day 1, ~2.5h)

**Goal:** Establish pre-training metrics using the off-the-shelf `all-MiniLM-L6-v2` model. This becomes the "before" in every before/after comparison.

**Implementation:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

# WHY all-MiniLM-L6-v2: 22.7M params, 384 dims, ~90MB.
# Small enough to fine-tune on M2 CPU. Same model used in P2
# for retrieval — provides portfolio continuity.
model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Metrics to compute (all 8 from requirements):**

| # | Metric | Implementation | Expected Baseline |
|---|--------|---------------|-------------------|
| 1 | **Cosine Similarity Distributions** | Compute cosine sim for all pairs. Split by label. Plot overlapping histograms. | Compatible mean ~0.72, Incompatible mean ~0.68, Margin ~0.04 |
| 2 | **UMAP Visualization** | Project 384d → 2d with `umap-learn`. Color by label. Interactive plotly scatter. | Overlapping blobs — no clear separation |
| 3 | **HDBSCAN Clustering** | Cluster UMAP projections. Measure purity (% of clusters that are label-homogeneous). | Purity ~50% (random chance) |
| 4 | **False Positive Analysis** | Find pairs where label=0 but cosine sim > threshold. Categorize by pair_type. | Subtle mismatches and complex pairs will dominate false positives |
| 5 | **Statistical Separation** | Cohen's d effect size + Welch's t-test for compatible vs. incompatible distributions. | Cohen's d < 0.3 (negligible/small effect) |
| 6 | **Classification Metrics** | Sweep cosine threshold (0.0–1.0). Compute Accuracy, Precision, Recall, F1, AUC-ROC. | AUC-ROC ~0.55–0.65 (barely above random) |
| 7 | **Category-wise Performance** | Compute margin (compatible mean - incompatible mean) per category. | Dealbreakers likely have largest baseline margin; interests smallest |
| 8 | **Baseline Summary JSON** | Save all metrics to `baseline_metrics.json` for post-training comparison. | — |

**UMAP Configuration:**

```python
import umap

# WHY these UMAP parameters:
# n_neighbors=15 — balances local vs global structure. Lower values
#   preserve local clusters but may miss global patterns. 15 is default
#   and works well for ~1000 points.
# min_dist=0.1 — how tightly points can pack. Lower = tighter clusters.
#   0.1 lets clusters form without forcing everything into points.
# metric='cosine' — matches how we measure similarity in embedding space.
#   Using euclidean here would misrepresent the actual geometry.

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42  # WHY: Reproducibility. UMAP is stochastic.
)
```

**HDBSCAN Configuration:**

```python
import hdbscan

# WHY HDBSCAN over KMeans:
# KMeans forces you to specify k (number of clusters) — but we don't
# know how many natural groupings exist. HDBSCAN finds them automatically
# AND identifies noise points (outliers that don't belong to any cluster).
# 
# Java parallel: It's GROUP BY with auto-detected categories, including
# a null group for outliers.

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=10,       # WHY: Minimum meaningful cluster. Too small = noise.
    min_samples=5,             # WHY: Density threshold. Higher = more conservative.
    cluster_selection_method='eom'  # WHY: Excess of Mass — better for uneven clusters.
)
```

**Output:**
- `eval/visualizations/baseline/` — all charts (cosine histograms, UMAP scatter, HDBSCAN clusters, ROC curve, category breakdown)
- `eval/baseline_metrics.json` — all numeric metrics for post-training comparison
- `eval/baseline_report.html` — interactive HTML dashboard with plotly

---

### Step 3: Standard Contrastive Fine-Tuning (Day 2, ~2h for setup + ~1h training)

**Goal:** Fine-tune `all-MiniLM-L6-v2` using `CosineSimilarityLoss` to learn dating compatibility patterns.

**Implementation:**

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

# WHY CosineSimilarityLoss:
# The loss function maps label=1 to target cosine similarity of 1.0
# and label=0 to target of 0.0. During training, gradients flow to
# pull compatible pair embeddings closer and push incompatible ones apart.
#
# Alternative: ContrastiveLoss uses a margin-based approach (pairs within
# margin are penalized). CosineSimilarityLoss is simpler — directly
# optimizes the cosine similarity to match the label.
#
# Java parallel: Think of CosineSimilarityLoss as a compareTo() function
# that defines the "natural ordering" of pairs in embedding space.
# Training is the sort algorithm that rearranges the space accordingly.

model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert JSONL records to InputExample format
# WHY InputExample: sentence-transformers' internal format for pair training.
# texts=[text_1, text_2] provides the pair, label provides the target similarity.
train_examples = [
    InputExample(
        texts=[record['text_1'], record['text_2']],
        label=float(record['label'])  # WHY float: CosineSimilarityLoss expects float in [0, 1]
    )
    for record in train_records
]

train_dataloader = DataLoader(
    train_examples,
    shuffle=True,       # WHY: Prevents learning order-dependent patterns
    batch_size=16       # WHY 16: Fits in 8GB M2 RAM. 32 would be tighter.
)

train_loss = losses.CosineSimilarityLoss(model)

# WHY EmbeddingSimilarityEvaluator:
# Monitors Spearman rank correlation between predicted cosine similarities
# and ground truth labels during training. The best checkpoint is saved
# based on this metric. Spearman (not Pearson) because we care about
# rank ordering (is pair A more compatible than pair B?) not exact values.
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    eval_examples,
    name='dating-eval'
)
```

**Training Configuration:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 4 | Spec recommendation. Small dataset (1,195 pairs) — more epochs risk overfitting. |
| **Batch size** | 16 | Fits in 8GB M2 RAM. Larger batches would provide more stable gradients but risk OOM. |
| **Learning rate** | 2e-5 | Standard for sentence-transformers fine-tuning. Small enough to not destroy pre-trained knowledge. |
| **Warmup steps** | 100 | Prevents aggressive early updates that could destabilize pre-trained weights. ~1.3 epochs of warmup at 16 batch size. |
| **Evaluation steps** | 500 | Monitor Spearman correlation during training. Frequency balances insight vs. training speed. |
| **Optimizer** | AdamW | sentence-transformers default. Weight decay prevents overfitting. |
| **Scheduler** | WarmupLinear | Ramps up learning rate during warmup, then linearly decays. Prevents both early destabilization and late overfitting. |

**M2 CPU Training Time Estimate:** ~45–90 minutes for 4 epochs on 1,195 pairs at batch_size=16. Plan to start training early on Day 2 and work on LoRA setup while it runs.

**Output:**
- `training/model/standard_model/` — fine-tuned model weights
- `training/standard_training_info.json` — hyperparameters, training time, final Spearman
- Training logs showing Spearman correlation progression

---

### Step 4: LoRA Fine-Tuning Comparison (Day 2, ~1h setup + training)

**Goal:** Fine-tune the same base model with LoRA adapters (via PEFT) using <1% of parameters, then compare results against standard fine-tuning.

**First Principle — Why LoRA:**
Full fine-tuning updates all 22.7M parameters. LoRA decomposes weight updates into low-rank matrices (A × B, where rank << hidden dimension), reducing trainable parameters by ~96%. For a 22.7M model on M2, full fine-tuning already works — the comparison answers: **"Does LoRA sacrifice quality for efficiency, and by how much?"**

**Implementation:**

```python
from peft import LoraConfig, get_peft_model, TaskType

# WHY PEFT (Parameter-Efficient Fine-Tuning):
# HuggingFace library that wraps any model with LoRA adapters in 3 lines.
# Keeps the base model frozen, only trains the small adapter matrices.
#
# Java parallel: Think of LoRA as the Decorator pattern — you wrap the
# existing model with a thin adapter layer that modifies behavior without
# changing the original class. The adapter is serialized separately.

lora_config = LoraConfig(
    r=8,                        # WHY r=8: Rank of decomposition. Higher = more capacity
                                # but more parameters. 8 is good for small models.
                                # For MiniLM's 384-dim layers, r=8 gives 384×8 + 8×384 
                                # = 6,144 params per adapted layer.
    lora_alpha=16,              # WHY 16: Scaling factor = alpha/r = 2.0. Controls how
                                # much the adapter output is amplified. 2×r is standard.
    lora_dropout=0.1,           # WHY 0.1: Prevents overfitting on 1,195 examples.
                                # Without dropout, small datasets risk memorization.
    target_modules=["query", "value"],  # WHY query + value: These attention matrices
                                        # determine "what to attend to" and "what to extract."
                                        # Modifying them changes how the model weighs
                                        # different aspects of the input text.
                                        # Skipping "key" saves params with minimal quality loss.
    task_type=TaskType.FEATURE_EXTRACTION,  # WHY: We're producing embeddings, not classifying.
)
```

**LoRA Parameter Budget:**

| Component | Full Model | LoRA Adapters |
|-----------|-----------|---------------|
| Total parameters | 22.7M | ~200K |
| Trainable parameters | 22.7M (100%) | ~200K (<1%) |
| RAM for training | ~360MB (FP32) | ~90MB (base frozen) + ~0.8MB (adapters) |
| Training speed | Baseline | Potentially faster (fewer gradient computations) |

**Training uses the same configuration** as Step 3 (same loss, batch size, epochs, evaluator) to ensure the comparison is controlled — the *only* variable is LoRA vs. full fine-tuning.

**Output:**
- `training/model/lora_model/` — LoRA adapter weights (small, separate from base)
- `training/lora_training_info.json` — hyperparameters, trainable param count, final Spearman
- Comparison: standard vs. LoRA Spearman curves during training

**Potential M2 Compatibility Issue:** If PEFT + sentence-transformers integration has issues on M2 (some versions have MPS backend conflicts), the **fallback** is to skip LoRA and focus all evaluation depth on the standard fine-tuned model. The ADR documents LoRA as a production technique regardless.

---

### Step 5: Post-Training Evaluation — Full 8-Metric Comparison (Day 3, ~2.5h)

**Goal:** Re-run all baseline metrics on the fine-tuned model(s) and produce side-by-side before/after comparisons.

**This is the "money shot" of the entire project.** Every chart, every metric, every insight feeds the interview narrative.

**Implementation — Run identical evaluation pipeline on fine-tuned model:**

For each model (standard fine-tuned, LoRA fine-tuned):
1. Load fine-tuned model
2. Generate embeddings for all eval pairs (295 pairs)
3. Compute all 8 metrics (identical code to baseline)
4. Save to `finetuned_metrics.json` / `lora_metrics.json`
5. Generate comparison visualizations

**Before/After Comparison Charts (minimum 8):**

| # | Chart | What It Shows |
|---|-------|---------------|
| 1 | **Cosine Distribution Overlay** | Two overlapping histograms: baseline (transparent) vs fine-tuned (solid). Compatible and incompatible distributions should separate. |
| 2 | **UMAP Side-by-Side** | Two UMAP scatter plots: baseline (overlapping blobs) → fine-tuned (separated clusters). Most visually impactful chart. |
| 3 | **HDBSCAN Purity Comparison** | Bar chart: cluster purity baseline vs. fine-tuned. |
| 4 | **ROC Curve Overlay** | Baseline ROC vs fine-tuned ROC on same axes. AUC improvement visible. |
| 5 | **Category-wise Margin Heatmap** | Heatmap: categories × models (baseline, standard, LoRA). Cell value = compatibility margin. Shows which categories improved most. |
| 6 | **Cohen's d Effect Size Comparison** | Bar chart: baseline d vs fine-tuned d. Target: negligible → large. |
| 7 | **Training Curve** | Spearman correlation over training steps (for both standard and LoRA). Shows convergence. |
| 8 | **False Positive Reduction** | Count of false positives (high-similarity incompatible pairs) before vs. after, broken down by pair_type. |

**Expected Outcomes (from requirements and concept primer):**

| Metric | Baseline (Expected) | Fine-Tuned (Target) |
|--------|---------------------|---------------------|
| Compatibility margin | 0.02–0.05 | 0.20–0.40+ |
| Cohen's d | <0.3 (small) | >0.8 (large) |
| Spearman correlation | ~0.76 | ≥0.86 |
| UMAP visual separation | Overlapping blobs | Distinct clusters |
| Cluster purity | ~50% (random) | 70%+ |
| AUC-ROC | 0.55–0.65 | 0.80+ |

**Output:**
- `eval/visualizations/comparison/` — all comparison charts
- `eval/finetuned_metrics.json` — post-training metrics
- `eval/lora_metrics.json` — LoRA post-training metrics (if available)
- `eval/comparison_report.html` — interactive HTML dashboard with tabs for baseline/standard/LoRA

---

### Step 6: Documentation & Deliverables (Day 3, ~1.5h)

**Goal:** Produce all documentation artifacts, README, ADR, Loom script.

See Section 11 (Deliverables Checklist) for complete list.

---

## 5. Project Structure

```
03-fine-tuning-guardrails/
├── CLAUDE.md                          ← Project-specific Claude Code memory
├── PRD.md                             ← This file (implementation contract)
├── README.md                          ← Portfolio README with results
├── pyproject.toml                     ← uv project config
├── data/
│   ├── raw/
│   │   ├── dating_pairs.jsonl         ← 1,195 training pairs (provided)
│   │   ├── eval_pairs.jsonl           ← 295 evaluation pairs (provided)
│   │   ├── dating_pairs_metadata.json ← Generation metadata
│   │   └── eval_pairs_metadata.json   ← Eval metadata
│   └── evaluation/
│       ├── data_quality_report.json   ← SyntheticDataEvaluator output
│       └── data_quality_summary.txt   ← Human-readable findings
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 ← Load JSONL, Pydantic validation
│   ├── data_evaluator.py              ← SyntheticDataEvaluator (5 dimensions)
│   ├── baseline_analysis.py           ← Pre-training metrics + UMAP + HDBSCAN
│   ├── trainer.py                     ← Standard fine-tuning with CosineSimilarityLoss
│   ├── lora_trainer.py                ← LoRA fine-tuning via PEFT
│   ├── post_training_eval.py          ← Post-training metrics (same as baseline)
│   ├── comparison.py                  ← Before/after comparison + charts
│   └── models.py                      ← Pydantic models for all data types
├── training/
│   ├── model/
│   │   ├── standard_model/            ← Full fine-tuned weights
│   │   └── lora_model/                ← LoRA adapter weights
│   ├── standard_training_info.json
│   └── lora_training_info.json
├── eval/
│   ├── visualizations/
│   │   ├── baseline/                  ← Pre-training charts
│   │   └── comparison/                ← Before/after charts
│   ├── baseline_metrics.json
│   ├── finetuned_metrics.json
│   ├── lora_metrics.json
│   ├── baseline_report.html           ← Interactive baseline dashboard
│   └── comparison_report.html         ← Interactive comparison dashboard
├── docs/
│   └── adr-001-lora-vs-standard.md
└── tests/
    ├── test_data_loader.py
    ├── test_evaluator.py
    └── test_metrics.py
```

---

## 6. Pydantic Models

```python
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from enum import IntEnum

class CompatibilityLabel(IntEnum):
    """WHY IntEnum: Enforces label is 0 or 1 at the type level.
    IntEnum rather than Enum because labels are numeric (used in loss function)."""
    INCOMPATIBLE = 0
    COMPATIBLE = 1

class DatingPair(BaseModel):
    """One training/eval record from the dating pairs dataset.
    
    WHY Pydantic: Same pattern as P1 — validates schema at load time,
    catches data corruption before it reaches the training loop.
    """
    text_1: str = Field(..., description="First person's statement (boy: ...)")
    text_2: str = Field(..., description="Second person's statement (girl: ...)")
    label: CompatibilityLabel
    category: str
    subcategory: str
    pair_type: str

    @field_validator('text_1', 'text_2')
    @classmethod
    def validate_gender_prefix(cls, v: str) -> str:
        """WHY: Ensure every text follows 'gender: statement' format.
        The gender prefix is part of the training signal — the model
        sees 'boy:' and 'girl:' as context tokens."""
        if ':' not in v:
            raise ValueError(f"Text must follow 'gender: statement' format, got: {v}")
        gender = v.split(':', 1)[0].strip()
        if gender not in ('boy', 'girl'):
            raise ValueError(f"Gender prefix must be 'boy' or 'girl', got: {gender}")
        return v

class DataQualityScore(BaseModel):
    """Scores from SyntheticDataEvaluator — 0 to 100 per dimension."""
    data_quality: float = Field(..., ge=0, le=100)
    diversity: float = Field(..., ge=0, le=100)
    bias: float = Field(..., ge=0, le=100)
    linguistic_quality: float = Field(..., ge=0, le=100)
    overall: float = Field(..., ge=0, le=100, description="Average of 4 dimensions")

class BaselineMetrics(BaseModel):
    """All metrics from baseline (pre-training) analysis."""
    compatible_mean_cosine: float
    incompatible_mean_cosine: float
    compatibility_margin: float        # compatible_mean - incompatible_mean
    cohens_d: float                    # Effect size
    t_statistic: float                 # Welch's t-test
    p_value: float
    auc_roc: float
    best_threshold: float              # Cosine threshold maximizing F1
    best_f1: float
    accuracy_at_best_threshold: float
    precision_at_best_threshold: float
    recall_at_best_threshold: float
    cluster_purity: float              # HDBSCAN cluster purity
    n_clusters: int                    # HDBSCAN clusters found
    noise_ratio: float                 # % points classified as noise
    spearman_correlation: float        # Rank correlation

class ComparisonResult(BaseModel):
    """Side-by-side comparison of baseline vs fine-tuned metrics."""
    baseline: BaselineMetrics
    standard_finetuned: BaselineMetrics
    lora_finetuned: BaselineMetrics | None = None  # None if LoRA skipped
    margin_improvement: float          # Absolute improvement
    margin_improvement_pct: float      # Percentage improvement
    cohens_d_improvement: float
    spearman_improvement: float
```

---

## 7. Dependencies

```toml
[project]
name = "fine-tuning-guardrails"
requires-python = ">=3.12"

[project.dependencies]
# Core ML
sentence-transformers = ">=3.0"     # WHY: SentenceTransformer model + CosineSimilarityLoss + training loop. 
                                    # The central library for this project.
torch = ">=2.0"                     # WHY: sentence-transformers backend. CPU-only on M2 (no CUDA).
peft = ">=0.12"                     # WHY: LoRA adapter wrapping. HuggingFace's Parameter-Efficient Fine-Tuning library.
                                    # Enables LoRA without writing low-rank decomposition from scratch.

# Evaluation & Visualization
umap-learn = ">=0.5"                # WHY: Dimensionality reduction from 384d → 2d for visualization.
                                    # Preserves both local and global structure better than t-SNE.
hdbscan = ">=0.8"                   # WHY: Density-based clustering that auto-detects cluster count.
                                    # Identifies noise points — critical for measuring cluster purity.
scikit-learn = ">=1.4"              # WHY: cosine_similarity, classification_report, roc_curve, auc.
                                    # Standard ML utilities.
scipy = ">=1.12"                    # WHY: stats.ttest_ind (Welch's t-test), stats.spearmanr.
matplotlib = ">=3.8"                # WHY: Static publication-quality charts.
seaborn = ">=0.13"                  # WHY: Statistical visualizations (histograms, heatmaps).
plotly = ">=5.18"                   # WHY: Interactive HTML charts for reports.

# Data & Validation
pydantic = ">=2.5"                  # WHY: Data schema validation. Consistent with P1/P2 pattern.

# CLI & Formatting
click = ">=8.1"                     # WHY: CLI argument parsing. Consistent with P2 pattern.
rich = ">=13.0"                     # WHY: Pretty console output (tables, progress bars).

```

**Dependencies NOT included:**
- `bitsandbytes` — CUDA-only quantization, incompatible with M2 (see Decision table)
- `langchain` — not needed, no RAG pipeline
- `ragas` — not needed, no retrieval evaluation
- `openai` — not needed, no LLM API calls (all computation is local)
- `instructor` — not needed, no structured LLM output generation

---

## 8. Memory Management Protocol (8GB M2)

Training a 22.7M parameter model on M2 is feasible but requires care:

```
RULE 1: Never load two SentenceTransformer models simultaneously.
  - Load base model → train → save → del model → gc.collect()
  - Load fine-tuned model → evaluate → save metrics → del model → gc.collect()

RULE 2: Close all non-essential apps during training.
  - Chrome tabs are the biggest RAM competitor.
  - VS Code + terminal + one browser tab for monitoring.

RULE 3: batch_size=16, not 32.
  - 16 pairs × 384 dims × 2 texts × FP32 = ~50KB per batch (trivial).
  - But gradient computation + optimizer state is the real cost.
  - 16 is safe. Monitor with `htop` during first epoch.

RULE 4: UMAP + HDBSCAN on eval set (295 pairs), not train set (1,195).
  - UMAP on 1,195 points is fine (takes seconds).
  - But if memory is tight, eval set is sufficient for visualization.

RULE 5: For LoRA comparison, load base model + adapter separately.
  - PEFT's get_peft_model() wraps the base model in-place.
  - After training, save adapter only → del model → gc.collect()
  - For evaluation, load base + merge adapter → evaluate → cleanup.
```

---

## 9. Day-by-Day Execution Plan

### Day 1 (Tuesday Feb 18) — Data Evaluation + Baseline Analysis

**Time budget:** 4 hours

| Task | Time | Description | Output |
|------|------|-------------|--------|
| T1.1 | 20min | Project setup: directory structure, pyproject.toml, `uv sync` | Working environment |
| T1.2 | 15min | Data loading: copy provided files to `data/raw/`, Pydantic validation of all records | `data_loader.py`, all records validated |
| T1.3 | 45min | `SyntheticDataEvaluator`: implement 5-dimension scoring | `data_evaluator.py`, quality report |
| T1.4 | 30min | Generate baseline embeddings: load `all-MiniLM-L6-v2`, encode all train + eval texts | Embeddings cached to disk |
| T1.5 | 45min | Baseline cosine similarity analysis: distributions, margins, Cohen's d, t-test, ROC curve | Baseline metrics + charts |
| T1.6 | 30min | UMAP visualization + HDBSCAN clustering | Interactive UMAP plot, cluster purity |
| T1.7 | 20min | Category-wise + pair-type-wise baseline breakdown | Category margin heatmap |
| T1.8 | 15min | Save `baseline_metrics.json`, generate `baseline_report.html` | Complete baseline package |

**Git checkpoint after T1.8.** Commit message: `feat(p3): baseline analysis complete — margin=X.XX, cohen_d=X.XX`

**End of Day 1 state:** Full baseline established. Every metric has a "before" number. Ready to train.

---

### Day 2 (Wednesday Feb 19) — Fine-Tuning (Standard + LoRA)

**Time budget:** 4 hours

| Task | Time | Description | Output |
|------|------|-------------|--------|
| T2.1 | 30min | Implement `trainer.py`: DataLoader, CosineSimilarityLoss, EmbeddingSimilarityEvaluator setup | Training script ready |
| T2.2 | 60-90min | **Start standard training** (4 epochs). Monitor Spearman correlation. | `standard_model/` weights saved |
| T2.3 | 30min | While T2.2 trains: implement `lora_trainer.py` with PEFT LoraConfig | LoRA training script ready |
| T2.4 | 45-60min | **Start LoRA training** (same hyperparams as standard) | `lora_model/` adapter weights saved |
| T2.5 | 20min | Compare training curves: standard vs LoRA Spearman progression | Training comparison chart |
**Git checkpoint after T2.5.** Commit message: `feat(p3): fine-tuning complete — standard spearman=X.XX, lora spearman=X.XX`

**If LoRA has M2 compatibility issues:** Skip T2.3-T2.4. Use remaining time to start post-training evaluation on standard model. Document LoRA issue in ADR.

**End of Day 2 state:** One or two fine-tuned models saved. Training curves logged. Ready for the comparison.

---

### Day 3 (Thursday Feb 20) — Post-Training Evaluation + Documentation

**Time budget:** 4 hours

| Task | Time | Description | Output |
|------|------|-------------|--------|
| T3.1 | 60min | Post-training evaluation: re-run full 8-metric pipeline on fine-tuned model(s) | `finetuned_metrics.json`, `lora_metrics.json` |
| T3.2 | 45min | Before/after comparison charts (all 8 from Section 4, Step 5) | Publication-quality comparison visualizations |
| T3.3 | 30min | Generate `comparison_report.html` — interactive dashboard with tabs | Interactive HTML report |
| T3.4 | 15min | False positive deep dive: which pair_types/categories improved most? | Detailed breakdown chart |
| T3.5 | 30min | ADR: "LoRA vs Standard Fine-Tuning — When Parameter Efficiency Matters" | `docs/adr-001-lora-vs-standard.md` |
| T3.6 | 20min | README with Mermaid architecture diagram + key results + charts | Portfolio-ready README |
| T3.7 | 15min | Loom recording script + record 2-min demo | Loom link for README |
| T3.8 | 15min | Notion journal update via MCP | Journal entry with STAR talking points |

**Stretch (if time remains after T3.8):**
- T3.S1: Instructional fine-tuning variant — prepend instructions, train, compare

**Git checkpoint after T3.6.** Commit message: `feat(p3): evaluation complete — margin improvement X.XX→Y.YY, cohen_d A.AA→B.BB`

---

## 10. Session Management Protocol

Each session starts by telling Claude Code:

```
Read CLAUDE.md and PRD.md. Today is Day [N].
Here's where I left off: [checkpoint from yesterday]
Focus on tasks [#X through #Y] from the PRD.
```

Each session ends:

1. Git commit and push all work
2. Tell Claude Code to write a journal entry to Notion via MCP
3. Update CLAUDE.md "Current State" section with what was completed

---

## 11. ADRs to Write

| ADR | Title | When |
|-----|-------|------|
| 001 | LoRA vs Standard Fine-Tuning — When Parameter Efficiency Matters | Day 3 (after comparison results) |
| 002 | QLoRA Skipped — When to Use Quantized Training (Production Context) | Day 3 (documenting the decision with production reasoning) |
| 003 | CosineSimilarityLoss vs ContrastiveLoss — Which Contrastive Objective When | Day 2 (after training, document why this loss function) |

---

## 12. What NOT to Build

- **No synthetic data generation** — using provided dataset (P1 demonstrated generation skills)
- **No QLoRA / bitsandbytes** — CUDA dependency incompatible with M2 (documented in ADR)
- **No Streamlit app** — model weights are local artifacts; Loom demo is more appropriate
- **No cross-domain evaluation** — dating domain evaluation is a complete story
- **No RAG pipeline** — this project is about embedding quality, not retrieval
- **No LLM API calls** — all computation is local (sentence-transformers, PyTorch)
- **No database** — JSON files + model weights on disk
- **No FastAPI** — no serving layer (save for P5)

---

## 13. Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| M2 training too slow | 4-epoch training exceeds 2h, eating into eval time | Medium | Start training at beginning of Day 2. Work on LoRA setup while standard trains. If >2h, reduce to 3 epochs. |
| PEFT/LoRA M2 incompatibility | MPS backend errors, adapter wrapping fails | Medium | Fallback: skip LoRA entirely. Focus all evaluation depth on standard model. Document LoRA as production technique in ADR. The comparison is a bonus, not the core deliverable. |
| UMAP non-deterministic | Different runs produce different visualizations | Low | Set `random_state=42` everywhere. Save UMAP projections to disk for reproducibility. |
| HDBSCAN finds no clusters | All points classified as noise | Low | Tune `min_cluster_size` down (try 5 instead of 10). If still no clusters, report this as a finding (pre-trained model doesn't create dating-specific clusters). |
| Fine-tuning doesn't improve metrics | Spearman stays ~0.76, margin doesn't grow | Low | Even a null result is valid — document why (data too simple? Model already captures basic semantics?). Try: more epochs, lower learning rate, different loss function. |
---

## 14. Deliverables Checklist

| # | Deliverable | Priority | Portfolio Signal |
|---|-------------|----------|-----------------|
| D1 | Data quality evaluation report (5-dimension scoring) | **Core** | Data engineering discipline |
| D2 | Baseline analysis with all 8 metrics | **Core** | Scientific rigor |
| D3 | Standard contrastive fine-tuning with CosineSimilarityLoss | **Core** | "I can train ML models" |
| D4 | LoRA fine-tuning comparison | **Core** | "I know when LoRA helps vs hurts" |
| D5 | Before/after comparison (8 charts minimum) | **Core** | Data visualization + evidence |
| D6 | Interactive HTML comparison report | **Core** | Publication quality |
| D7 | Training curves (Spearman over steps) | **Core** | Training process understanding |
| D8 | Category-wise performance breakdown | **Core** | Domain-specific insights |
| D9 | ADRs (LoRA decision, QLoRA skip, loss function) | **Core** | Technical leadership |
| D10 | README with Mermaid arch diagram + results | **Core** | Communication |
| D11 | 2-min Loom walkthrough | **Core** | Presentation skills |
| D12 | Notion journal entry with STAR talking points | **Core** | Learning documentation |
| D13 | Instructional fine-tuning variant | **Stretch** | Instruction-following depth |

---

## 15. Expected Outcomes (Hypotheses to Validate)

| Metric | Baseline | Standard Fine-Tuned | LoRA Fine-Tuned | Notes |
|--------|----------|--------------------:|:---------------:|-------|
| Compatibility margin | 0.02–0.05 | 0.20–0.40+ | 0.18–0.35 | LoRA may be slightly lower due to fewer trainable params |
| Cohen's d | <0.3 | >0.8 (large) | >0.7 | — |
| Spearman ρ | ~0.76 | ≥0.86 | ≥0.83 | Requirements target for standard |
| AUC-ROC | 0.55–0.65 | 0.80–0.90 | 0.78–0.88 | — |
| Cluster purity | ~50% | 70%+ | 65%+ | — |
| UMAP visual | Overlapping | Separated | Separated | — |
| Dealbreaker margin | Moderate | Near-perfect separation | Near-perfect | Dealbreakers are the easiest category |
| Subtle mismatch margin | Near-zero | Moderate improvement | Moderate | Hardest category — expect smallest gains |

**If standard fine-tuning dramatically outperforms LoRA:** Document this as a finding — LoRA's reduced capacity matters more on small models (22.7M params) than large ones (7B+). This is a valuable insight.

**If LoRA matches or beats standard:** Also valuable — proves parameter-efficient training works even on small models, which means production deployments can serve multiple domain-specific adapters from a single base model.

---

## 16. Interview Talking Points

**"Tell me about a time you trained and evaluated an ML model"**
→ "I fine-tuned a sentence embedding model using contrastive learning to understand dating compatibility. The pre-trained model couldn't distinguish compatible from incompatible profiles — cosine similarity margin was only X. After fine-tuning with CosineSimilarityLoss, the margin jumped to Y, and Cohen's d went from negligible to large effect. I validated this with UMAP visualization showing clear cluster separation, HDBSCAN purity increasing from 50% to Z%, and category-wise analysis revealing dealbreakers achieved near-perfect separation while subtle lifestyle mismatches were the hardest category."

**"When would you fine-tune vs. prompt-engineer?"**
→ "When the task requires the model to understand domain-specific similarity that can't be expressed in prompts. In P3, I needed the model to understand that 'values alignment' outweighs 'hobby overlap' — that's geometric knowledge about the embedding space, not something a prompt can teach. I compared this to P2 where off-the-shelf embeddings were sufficient for document retrieval."

**"What's LoRA and when would you use it?"**
→ "LoRA decomposes weight updates into low-rank matrices, reducing trainable parameters by 96%. I compared LoRA vs full fine-tuning on a 22.7M parameter model and found [X]. For small models, [insight]. For production with 7B+ models, LoRA is essential — you can serve multiple domain-specific adapters from a single base model, switching between them without reloading weights."

**"How do you evaluate whether fine-tuning worked?"**
→ "I measure from three angles: numerical (Cohen's d effect size, AUC-ROC, compatibility margin), visual (UMAP projections showing cluster separation), and algorithmic (HDBSCAN cluster purity). All three must agree — if the numbers improve but the clusters don't separate visually, something is wrong. I also break down performance by category to find where the model struggles."

---

## 17. Key Analogies (For Concept Understanding)

**Embeddings → Coordinate System:** Pre-trained embeddings are like GPS coordinates optimized for "nearness by topic." Fine-tuning is like recalibrating the coordinate system so "nearness" means "compatibility" instead.

**Contrastive Loss → Sorting Algorithm:** CosineSimilarityLoss is like a `Comparator<float[]>` that defines the "natural ordering" of pairs in embedding space. Training is the sort algorithm that rearranges the space until compatible pairs are adjacent and incompatible pairs are distant.

**LoRA → Decorator Pattern:** LoRA wraps the existing model with thin adapter layers that modify behavior without changing the original weights. The adapter is serialized separately — like a plugin that can be swapped for different domains without rebuilding the base.

**Baseline vs Fine-Tuned → Factory Engine vs Custom Tune:** The pre-trained model is a factory engine — great for general driving but not optimized for dating compatibility. Fine-tuning takes it to a specialist who adjusts the engine using pairs of scenarios: "these two should be closer" (compatible) and "these two should be further apart" (incompatible).
