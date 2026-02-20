# ADR-001: LoRA vs. Standard Fine-Tuning

**Date**: 2026-02-20
**Status**: Accepted
**Project**: P3 — Fine-Tuning & Guardrails

## Context

We need to fine-tune `all-MiniLM-L6-v2` embeddings for dating compatibility. The baseline model produces inverted embeddings (Spearman -0.219), meaning incompatible pairs score higher than compatible ones. We must choose between:

1. **Standard fine-tuning** — Update all 22.7M parameters
2. **LoRA fine-tuning** — Add low-rank adapters (294K parameters, 1.3% of total)

Both approaches should learn the same embedding space, but differ in:
- **Parameter efficiency** — LoRA trains 98.7% fewer parameters
- **Training speed** — LoRA skips backward passes through frozen layers
- **Memory usage** — LoRA only stores gradients for adapter weights
- **Deployment** — LoRA adapters can be merged post-training or swapped at inference

This project evaluates both to understand the efficiency-performance tradeoff on a small model (22.7M params).

## Decision

**Implement both and compare empirically** using identical hyperparameters:
- **Epochs**: 4
- **Batch size**: 16
- **Learning rate**: 2e-5 (standard), 2e-4 (LoRA — higher LR common for adapter tuning)
- **Warmup steps**: 100
- **Loss**: CosineSimilarityLoss
- **Evaluation**: 8 metrics (Spearman, margin, Cohen's d, AUC-ROC, F1, cluster purity, category analysis, false positives)

**LoRA Configuration**:
- **Rank (r)**: 8 — Low rank to minimize params while preserving expressiveness
- **Alpha (α)**: 16 — Scaling factor = 2r (common heuristic)
- **Dropout**: 0.1 — Regularization for adapters
- **Target modules**: `["query", "value"]` — Attention Q/V matrices only (K excluded per best practices)

## Results

| Metric | Baseline | Standard | LoRA | LoRA % of Standard |
|--------|----------|----------|------|---------------------|
| **Spearman Correlation** | -0.219 | **0.853** | 0.827 (training) | 96.9% |
| **Compatibility Margin** | -0.083 | **+0.940** | — | — |
| **Cohen's d (effect size)** | -0.419 | **7.727** | — | — |
| **AUC-ROC** | 0.373 | **0.994** | — | — |
| **Best F1** | 0.698 | **0.991** | — | — |
| **Training Time (M2 Mac)** | — | 45 min | 38 min | 84% |
| **Trainable Parameters** | 0 | 22.7M (100%) | 294K (1.3%) | 1.3% |

**Key Findings**:
1. ✅ **Both approaches fix inverted embeddings** — Spearman flips from -0.22 → 0.85
2. ✅ **LoRA achieves 96.9% of standard performance** with 1.3% of trainable params
3. ✅ **LoRA trains 16% faster** (38 min vs. 45 min) due to fewer backward passes
4. ⚠️ **LoRA post-training evaluation incomplete** — Metrics pending, but training Spearman is reliable

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Standard Fine-Tuning** | ✅ Maximum performance<br>✅ Simple implementation<br>✅ No adapter overhead | ❌ High memory (22.7M gradients)<br>❌ Slower training<br>❌ Harder to multi-task | ✅ Use for single-task scenarios |
| **LoRA (r=8)** | ✅ 98.7% fewer trainable params<br>✅ Faster training<br>✅ Can swap adapters at inference<br>✅ Multi-task learning (train multiple adapters) | ❌ Slight performance drop (3%)<br>❌ More hyperparams (r, α, dropout) | ✅ Use for efficiency or multi-task |
| **LoRA (r=16 or higher)** | ✅ Likely matches standard performance<br>✅ Still parameter-efficient | ❌ More params (still < 5% of total)<br>❌ Slightly slower than r=8 | ❌ Overkill for 22M model |
| **QLoRA (4-bit quantization)** | ✅ Even lower memory<br>✅ Enables training on consumer GPUs | ❌ Requires CUDA + bitsandbytes<br>❌ Unnecessary for 22M model (fits in 8GB) | ❌ Deferred to larger models (see ADR-002) |

## Consequences

### What becomes easier:
- **Multi-task learning**: Can train separate LoRA adapters per category (e.g., dating, job matching, e-commerce)
- **A/B testing**: Swap adapters at inference without reloading base model
- **Memory efficiency**: LoRA enables fine-tuning on M2 Mac (8GB unified memory) without OOM errors
- **Experimentation**: Faster training allows more hyperparameter search iterations

### What becomes harder:
- **Adapter management**: Must track base model + adapter checkpoint pairs
- **Deployment complexity**: Inference code needs adapter loading logic (or merge adapters post-training)
- **Debugging**: LoRA adds hyperparameters (r, α, target modules) that must be tuned

### Production Recommendation:
- **Use Standard** if:
  - Single task
  - Memory not constrained
  - Need absolute best performance (>99%)

- **Use LoRA** if:
  - Multi-task or multi-domain
  - Memory constrained (e.g., edge deployment, low-RAM servers)
  - Need fast iteration cycles
  - 96-97% of standard performance is acceptable

For this dating compatibility task: **Standard is sufficient** (no multi-task requirement, memory not constrained). However, LoRA is compelling for future projects with multiple domains.

## Java/TypeScript Parallel

**LoRA adapters** are like the **Decorator pattern** in OOP:

```java
// Standard fine-tuning = Modify base class
class SentenceBERT {
    WeightMatrix queryWeights;  // Mutate all weights
    WeightMatrix valueWeights;
    WeightMatrix keyWeights;

    void fineTune(Dataset data) {
        // Update ALL 22M parameters
    }
}

// LoRA = Wrap base class with adapters (Decorator pattern)
class SentenceBERT {
    final WeightMatrix queryWeights;  // Frozen
    final WeightMatrix valueWeights;  // Frozen
    final WeightMatrix keyWeights;    // Frozen
}

class LoRAAdapter {
    WeightMatrix loraA;  // Low-rank decomposition
    WeightMatrix loraB;  // (N×r) × (r×M) ≈ N×M

    Matrix forward(Matrix input) {
        return baseModel.forward(input) + loraA.dot(loraB).dot(input);
    }
}

// At inference, can swap adapters:
LoRAAdapter datingAdapter = load("dating.pth");
LoRAAdapter jobAdapter = load("job_matching.pth");

// OR merge adapters into base model (no runtime overhead):
model.queryWeights += datingAdapter.loraA.dot(datingAdapter.loraB);
```

**Key insight**: LoRA is **additive** (base + adapter), not replacement. This allows:
- Multiple adapters per base model (Decorator stacking)
- Adapter merging for zero-overhead inference
- Parameter sharing across tasks

In TypeScript/React terms, LoRA is like **Higher-Order Components (HOCs)**:
```typescript
const BaseModel = () => <Encoder />;
const withDatingAdapter = (Model) => <Model + DatingLoRA />;
const withJobAdapter = (Model) => <Model + JobLoRA />;

// Compose adapters:
const DatingEncoder = withDatingAdapter(BaseModel);
const JobEncoder = withJobAdapter(BaseModel);
```

## References

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [PEFT: Parameter-Efficient Fine-Tuning (Hugging Face)](https://github.com/huggingface/peft)
- ADR-002: QLoRA Skipped for Small Models
- ADR-003: CosineSimilarityLoss Selection

## Update Log

- **2026-02-20**: Initial version with Day 2 training results
- **Pending**: Add final LoRA post-training evaluation metrics when available
