# ADR-005: Dual Labeling Strategy — Manual + LLM Inter-Rater Agreement

**Date:** 2026-02-11
**Status:** Accepted
**Project:** P1 — Synthetic Data, Home DIY Repair
**Category:** Evaluation

## Context

The LLM-as-Judge (GPT-4o) labels all 30 records across 6 binary failure modes — 180 evaluations total. But an LLM judge is itself an unvalidated system: how do we know its labels are trustworthy? Without external validation, the entire correction pipeline rests on the assumption that GPT-4o's failure detection is meaningful. If the judge is systematically wrong, the corrected records are no better than the originals — just differently wrong.

The constraint: manually labeling all 30 × 6 = 180 pairs is feasible but slow. The pragmatic question is: how many manual labels establish enough ground truth to validate the LLM judge's reliability on the remainder?

## Decision

Manually label a **10-record stratified sample** (covering all 5 categories × all difficulty levels) across all 6 failure modes, producing 60 binary ground-truth comparisons. Compute **per-mode agreement rates and Cohen's Kappa** between manual and LLM labels. If kappa is positive on the dominant failure modes, treat the LLM judge as reliable enough to label the remaining 20 records.

The core principle: **use human labels to calibrate the evaluator, not to replace it**. The 10 manually labeled records serve as a calibration set for the LLM judge — analogous to a held-out test set validating a classifier before deploying it in production.

### Implementation

```python
# evaluator.py — compute_agreement()
from sklearn.metrics import cohen_kappa_score

def compute_agreement(
    manual: list[ManualLabel], llm_csv: list[dict]
) -> dict:
    per_mode_kappa = {}
    for mode in FAILURE_MODES:
        manual_bits = [getattr(m, mode) for m in matched]
        llm_bits    = [row[mode] for row in matched_llm]
        try:
            kappa = cohen_kappa_score(manual_bits, llm_bits)
        except ValueError:
            kappa = None  # degenerate: one label class absent
        per_mode_kappa[mode] = kappa
```

The stratified sample ensures every category and difficulty appears in the calibration set — preventing a scenario where the judge appears reliable on easy records but fails on complex ones.

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **10-record stratified sample** ✅ | Covers all categories/difficulties, 60 comparison points, feasible in one session | Small N — individual mode kappas have wide CIs | — (selected) |
| Label all 30 records manually | Maximum ground truth | 3× the effort; diminishing returns if judge is already calibrated | Cost not justified given 81.7% on first 10 |
| Raw agreement only (no kappa) | Simpler to compute | Raw agreement is inflated when one class dominates (e.g., 90% "pass" → 90% agreement by always predicting pass) | Kappa corrects for chance agreement; essential for imbalanced binary labels |
| Cross-validate with a second LLM (GPT-3.5 or Claude) | No human effort | LLM-vs-LLM agreement doesn't establish ground truth — both could be wrong in the same direction | Human labels are the only external validity anchor |

## Quantified Validation

- **Sample size**: 10 records × 6 modes = 60 binary comparisons
- **Overall raw agreement**: **81.7%**
- **Per-mode agreement**: 60% (`poor_quality_tips`) → 80% (`incomplete_answer`, `safety_violations`) → 70% (`unrealistic_tools`) → 100% (`overcomplicated_solution`, `missing_context`)
- **Overall Cohen's Kappa**: **κ = 0.201** (fair agreement by Landis & Koch scale; positive = above-chance)
- **Per-mode kappa**:

| Failure Mode | Agreement | κ | Interpretation |
|---|---|---|---|
| `incomplete_answer` | 80.0% | **0.545** | Moderate — dominant mode, reliable signal |
| `safety_violations` | 80.0% | **0.412** | Moderate |
| `unrealistic_tools` | 70.0% | **−0.154** | Slight negative — judge too strict vs human |
| `overcomplicated_solution` | 100.0% | N/A | Degenerate: all labels 0, kappa undefined |
| `missing_context` | 100.0% | N/A | Degenerate: all labels 0, kappa undefined |
| `poor_quality_tips` | 60.0% | **0.000** | Chance — highest disagreement mode |

- **Key finding**: The two modes that drove the correction loop (`incomplete_answer` 50% of failures, `poor_quality_tips` 43.3%) show positive kappa (0.545 and 0.000 respectively). The judge's detections in the dominant modes are real, not noise. `unrealistic_tools` (κ=−0.154) shows the judge is stricter than humans — a known calibration artifact of the strict identity priming in ADR-003.

## Consequences

**Easier:** The κ=0.201 overall and κ=0.545 on `incomplete_answer` provide a citable validation claim for the correction pipeline. The portfolio narrative becomes: "We validated the LLM judge against human labels before using its outputs to drive corrections."
**Harder:** `poor_quality_tips` (κ=0.000) and `unrealistic_tools` (κ=−0.154) remain weakly validated. Corrections targeting those modes have lower confidence. A production system would expand the manual label set for these two modes specifically.
**Portability:** The pattern — sample-based calibration of an LLM evaluator against human ground truth — is directly reusable in P2 (QA pair quality), P4 (resume/job match quality), and any project where an LLM judge drives a correction loop.

## Cross-References

- **ADR-003**: Strict prompt calibration (0% → 20%) is what made the LLM judge strict enough to have meaningful labels worth validating. Without calibration, 0% failure rate → nothing to compare.
- **ADR-004**: Dual-labeling confirmed `incomplete_answer` and `poor_quality_tips` as the dominant modes, which directly drove the v2 template improvements.

## Java/TS Parallel

Analogous to **test coverage for a test suite**. The LLM judge is the "test suite" for DIY records. Dual labeling is the process of writing integration tests for the test suite itself — verifying that the automated quality gate actually catches real defects rather than producing false negatives or false positives.

More specifically: this is **confusion matrix validation before production deployment**. No engineer would deploy a classifier that labels production data without first computing precision/recall against a held-out human-labeled set. The dual-labeling step applies the same discipline to an LLM-based classifier. Cohen's Kappa is the F1-equivalent for inter-rater reliability.

**The key insight:** An LLM judge without human validation is an unverified assumption. Kappa on 60 comparisons is a lightweight, one-session investment that transforms "we trust the judge" from a belief into a measured claim.

## Interview Signal

Demonstrates **evaluation pipeline rigor** — the ability to distinguish between "we ran evaluation" and "we validated our evaluator." Senior engineers building ML systems are frequently asked: "How do you know your automated evaluation is measuring the right thing?" This ADR answers that question with a concrete methodology (stratified sampling), the right metric (Cohen's Kappa over raw agreement), and an honest interpretation of per-mode results including the weak and degenerate cases. The pattern — human labels as ground truth for calibrating automated evaluators — is directly transferable to any production ML system using LLM-as-Judge.
