# ADR-003: CosineSimilarityLoss Selection

**Date**: 2026-02-20
**Status**: Accepted
**Project**: P3 — Fine-Tuning & Guardrails

## Context

We need to select a loss function for fine-tuning `all-MiniLM-L6-v2` embeddings on dating compatibility pairs. The baseline model produces inverted embeddings (Spearman -0.219), so the loss function must push compatible pairs closer and incompatible pairs farther apart in embedding space.

**Available options:**
1. **CosineSimilarityLoss** — Optimize cosine similarity directly
2. **ContrastiveLoss** — Margin-based loss for binary labels (compatible/incompatible)
3. **TripletLoss** — Anchor-positive-negative triplets with margin

All three are supported by Sentence-Transformers and proven for embedding fine-tuning.

## Decision

**Use CosineSimilarityLoss** for this project.

**Configuration:**
```python
from sentence_transformers import losses, InputExample

train_examples = [
    InputExample(texts=[profile1, profile2], label=1.0),  # compatible
    InputExample(texts=[profile3, profile4], label=0.0),  # incompatible
]

loss = losses.CosineSimilarityLoss(model)
```

**Why CosineSimilarityLoss:**
1. **Training-evaluation alignment** — We evaluate with cosine similarity (Spearman correlation, margin), so the loss should optimize the same metric
2. **Continuous labels** — CosineSimilarityLoss accepts floats in [0, 1], enabling future extensions to compatibility scores (not just binary)
3. **Simpler tuning** — No margin hyperparameter to tune (unlike ContrastiveLoss or TripletLoss)
4. **Proven performance** — Sentence-BERT paper uses CosineSimilarityLoss for STS (semantic textual similarity) tasks

## Alternatives Considered

| Loss Function | Input | Pros | Cons | Decision |
|---------------|-------|------|------|----------|
| **CosineSimilarityLoss** | Pairs + [0,1] labels | ✅ Direct cosine optimization<br>✅ No margin tuning<br>✅ Continuous labels<br>✅ Training-eval alignment | ❌ No explicit margin enforcement | ✅ Use |
| **ContrastiveLoss** | Pairs + binary labels + margin | ✅ Explicit margin enforcement<br>✅ Well-studied | ❌ Margin hyperparameter (0.5? 1.0?)<br>❌ Binary labels only<br>❌ Optimizes Euclidean distance (not cosine) | ❌ Skip |
| **TripletLoss** | (anchor, pos, neg) triplets + margin | ✅ Relative ranking<br>✅ Strong performance on retrieval | ❌ Requires triplet mining<br>❌ Margin hyperparameter<br>❌ 3x data prep complexity | ❌ Skip |
| **MultipleNegativesRankingLoss** | (anchor, pos) + in-batch negatives | ✅ No explicit negatives needed<br>✅ Scales well | ❌ Requires large batch sizes (≥64)<br>❌ Our batch size = 16 (memory constraint) | ❌ Skip |

## Results

Fine-tuning with CosineSimilarityLoss (4 epochs, batch size 16, LR 2e-5):

| Metric | Baseline | Fine-Tuned | Improvement |
|--------|----------|------------|-------------|
| **Spearman Correlation** | -0.219 | **0.853** | +1.072 |
| **Compatibility Margin** | -0.083 | **+0.940** | +1.023 |
| **Cohen's d** | -0.419 | **7.727** | +8.146 |
| **AUC-ROC** | 0.373 | **0.994** | +0.621 |

**Key finding:** CosineSimilarityLoss successfully reversed inverted embeddings and achieved near-perfect separation (AUC 0.994).

## Why Not ContrastiveLoss?

ContrastiveLoss optimizes Euclidean distance with a margin:
```
loss = (1 - y) * 0.5 * D^2 + y * 0.5 * max(0, margin - D)^2
```
where `D = ||emb1 - emb2||_2` (Euclidean distance).

**Problems:**
1. **Metric mismatch** — ContrastiveLoss optimizes Euclidean distance, but we evaluate with cosine similarity (different metric spaces)
2. **Margin tuning** — Must grid-search margin (common values: 0.5, 1.0, 2.0). CosineSimilarityLoss avoids this hyperparameter.
3. **Normalized embeddings** — Sentence-BERT normalizes embeddings to unit sphere. On the unit sphere, Euclidean distance and cosine similarity are monotonically related, but optimizing Euclidean explicitly is less direct.

**When to use ContrastiveLoss:**
- Face recognition (optimize Euclidean distance on raw embeddings, not unit sphere)
- Hard negative mining scenarios (margin helps push hard negatives farther)

## Why Not TripletLoss?

TripletLoss requires (anchor, positive, negative) triplets:
```
loss = max(0, ||anchor - positive||_2 - ||anchor - negative||_2 + margin)
```

**Problems:**
1. **Data complexity** — Must construct triplets from pairs:
   - For pair (A, B, label=1): need to find negative C
   - Requires triplet mining strategy (random, semi-hard, hard)
2. **3x data prep overhead** — Each training example needs 3 profiles instead of 2
3. **Margin tuning** — Same hyperparameter issue as ContrastiveLoss

**When to use TripletLoss:**
- Retrieval tasks (ranking matters more than absolute similarity)
- Large datasets where hard negative mining yields significant gains

## Consequences

### What becomes easier:
- **Training-evaluation alignment** — Loss function directly optimizes the metric we evaluate (cosine similarity)
- **Hyperparameter tuning** — One fewer hyperparameter (no margin)
- **Data pipeline** — Simple pair-label format, no triplet construction

### What becomes harder:
- **Hard negative mining** — CosineSimilarityLoss doesn't explicitly push hard negatives farther (but empirically performs well)
- **Future extensions** — If we later want to optimize for retrieval ranking, TripletLoss might outperform

## Java/TypeScript Parallel

Loss function selection is like **choosing a distance metric for K-Nearest Neighbors (KNN)**:

```java
interface DistanceMetric {
    double distance(Vector a, Vector b);
}

class EuclideanDistance implements DistanceMetric {
    // ContrastiveLoss optimizes this
    double distance(Vector a, Vector b) {
        return Math.sqrt(sumSquaredDifferences(a, b));
    }
}

class CosineDistance implements DistanceMetric {
    // CosineSimilarityLoss optimizes this
    double distance(Vector a, Vector b) {
        return 1 - dotProduct(a, b) / (norm(a) * norm(b));
    }
}

// Training-evaluation alignment principle:
// If you evaluate with cosineDistance(), train with CosineSimilarityLoss
// If you evaluate with euclideanDistance(), train with ContrastiveLoss
```

**Rule of thumb:**
- Use the same metric for training and evaluation
- For unit-normalized embeddings (Sentence-BERT), cosine similarity is the natural choice

## References

- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Sentence-Transformers Loss Functions Documentation](https://www.sbert.net/docs/package_reference/losses.html)
- [Improved Deep Metric Learning with Multi-class N-pair Loss Objective (Sohn, 2016)](https://papers.nips.cc/paper/2016/hash/6b180037abbebea991d8b1232f8a8ca9-Abstract.html)
- ADR-001: LoRA vs. Standard Fine-Tuning
- ADR-002: QLoRA Skipped for Small Models

## Update Log

- **2026-02-20**: Initial version — CosineSimilarityLoss selected for training-eval alignment
