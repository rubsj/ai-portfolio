# ADR-002: QLoRA Skipped for Small Models

**Date**: 2026-02-20
**Status**: Accepted
**Project**: P3 — Fine-Tuning & Guardrails

## Context

QLoRA (Quantized Low-Rank Adaptation) extends LoRA by quantizing the base model to 4-bit precision, enabling fine-tuning of large models (7B+ parameters) on consumer hardware. For our `all-MiniLM-L6-v2` fine-tuning project (22.7M parameters), we must decide whether to use QLoRA instead of standard LoRA.

**QLoRA Benefits:**
- 4-bit quantization reduces memory by ~75% (vs. fp16 baseline)
- Enables fine-tuning 7B–65B models on single 24GB GPU
- Minimal performance degradation (96-99% of full fine-tuning)

**Our Constraints:**
- Model size: 22.7M parameters (tiny compared to 7B+)
- Hardware: M2 Mac (8GB unified memory, no CUDA)
- Memory usage: Standard LoRA already fits comfortably in 8GB

## Decision

**Skip QLoRA for this project.** Use standard LoRA (fp32 weights + fp32 adapters) instead.

**Rationale:**
1. **CUDA dependency**: QLoRA requires `bitsandbytes` library, which depends on CUDA. M2 Mac uses Metal, not CUDA.
2. **Model too small**: 22.7M parameters fit easily in 8GB RAM. QLoRA's memory savings (75%) are unnecessary — we're not memory-constrained.
3. **Complexity cost**: QLoRA adds quantization logic, calibration data requirements, and potential numerical instability for minimal benefit.
4. **Performance risk**: 4-bit quantization could degrade 22M model more than 7B+ models (fewer parameters → less redundancy to absorb quantization errors).

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **QLoRA (4-bit + LoRA)** | ✅ Extreme memory efficiency<br>✅ Proven on 7B–65B models | ❌ Requires CUDA (M2 Mac incompatible)<br>❌ Overkill for 22M model<br>❌ Adds quantization complexity | ❌ Skip |
| **Standard LoRA (fp32)** | ✅ Works on M2 Mac (Metal)<br>✅ Simpler implementation<br>✅ No quantization overhead | ❌ Higher memory than QLoRA (but still fits in 8GB) | ✅ Use |
| **8-bit quantization** | ✅ Less extreme than 4-bit<br>✅ Better precision | ❌ Still requires bitsandbytes (CUDA)<br>❌ Unnecessary for 22M model | ❌ Skip |
| **Standard fine-tuning (fp32)** | ✅ Maximum performance<br>✅ No adapters | ❌ 77x more trainable params than LoRA<br>❌ Slower training | ❌ Use only for baseline comparison |

## When to Revisit QLoRA

QLoRA becomes compelling when:
1. **Model size ≥ 7B parameters** — Standard LoRA memory usage exceeds available RAM
2. **CUDA hardware available** — AWS/GCP GPU instances, NVIDIA consumer GPUs
3. **Multi-adapter serving** — Need to hot-swap adapters with minimal base model memory

**Example thresholds:**
- 7B model (fp16): ~14GB base model + ~2GB LoRA adapters → 16GB total (exceeds M2 Mac 8GB)
- 7B model (4-bit QLoRA): ~3.5GB base model + ~2GB adapters → 5.5GB total (fits in M2 Mac)

For P5 (Production RAG) or P6 (Digital Clone), if we fine-tune 7B+ models, revisit QLoRA with CUDA runtime (Docker container or cloud GPU).

## Consequences

### What becomes easier:
- **M2 Mac compatibility**: No CUDA dependency, native Metal support via PyTorch MPS backend
- **Simpler debugging**: No quantization numerical issues to troubleshoot
- **Faster iteration**: Skip calibration data preparation and quantization overhead

### What becomes harder:
- **Scaling to larger models**: If we later want to fine-tune 7B+ models locally, M2 Mac 8GB RAM will be insufficient
- **Cloud GPU costs**: Future projects requiring 7B+ fine-tuning must use cloud GPUs (AWS p3.2xlarge ~$3/hr)

## Java/TypeScript Parallel

QLoRA quantization is like **lossy compression** in image processing:

```java
// Standard LoRA = Store weights in full precision (like PNG)
class StandardLoRA {
    float[] weights;  // 32-bit floats (4 bytes per weight)
}

// QLoRA = Store weights in 4-bit quantized format (like JPEG)
class QLoRA {
    byte[] quantizedWeights;  // 4 bits per weight (8x compression)
    float[] scaleFactors;     // Dequantization metadata

    float[] dequantize() {
        // Lossy decompression: some precision lost, but 75% memory saved
        return applyScaleFactors(quantizedWeights);
    }
}
```

**Trade-off:**
- PNG: lossless, larger file, pixel-perfect
- JPEG: lossy, 75% smaller, visually similar (but not identical)

**When to use JPEG (QLoRA):**
- Large images (7B+ params) that don't fit in RAM
- Acceptable quality loss (96-99% of original)

**When to use PNG (standard LoRA):**
- Small images (22M params) that fit easily in RAM
- Need pixel-perfect quality (no quantization loss)

## References

- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [bitsandbytes: 8-bit & 4-bit Quantization](https://github.com/TimDettmers/bitsandbytes)
- ADR-001: LoRA vs. Standard Fine-Tuning
- ADR-003: CosineSimilarityLoss Selection

## Update Log

- **2026-02-20**: Initial version — QLoRA deferred until 7B+ models
