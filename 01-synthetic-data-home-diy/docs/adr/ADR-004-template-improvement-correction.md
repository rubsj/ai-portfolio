# ADR-004: Template Improvement as Correction Strategy

**Date:** 2026-02-09
**Status:** Accepted
**Project:** P1 — Synthetic Data, Home DIY Repair
**Category:** Algorithm

## Context

After generating 30 v1 records, the LLM-as-Judge (ADR-003) identified a **20% failure rate** (36/180 evaluations). Two modes dominated: **incomplete_answer** (50% of records) and **poor_quality_tips** (43.3%), accounting for 28 of 36 total failures. The PRD specified two correction strategies: individual record correction and template improvement. The question: which delivers better failure rate reduction?

## Decision

Implement both strategies, measure independently, then combine:

- **Strategy A (Correction)**: Send failing records + judge feedback to GPT-4o-mini via Instructor for targeted fixes.
- **Strategy B (Template v2)**: Analyze failure distributions, then add explicit quality requirements to the generation prompt.

V2 templates append `--- QUALITY REQUIREMENTS ---` with mode-specific instructions (e.g., `incomplete_answer` now requires troubleshooting advice and what-if scenarios; `poor_quality_tips` explicitly bans generic advice).

**Why templates outperformed correction:** Templates prevent failures upstream. Individual correction asks the same model to fix its own output — it often repeats the same mistake class. The dominant failures were systemic (GPT-4o-mini's default "good enough" behavior), and template instructions override that default.

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Combined: v2 templates + correction** ✅ | 100% reduction, zero residual failures | Extra API cost for 6 records | — (selected) |
| Template improvement only | 77.8% reduction, prevents upstream | Misses edge cases, falls short of 80% target | Close but insufficient alone |
| Individual correction only | Simpler, no template changes | 66.7% reduction — treats symptoms, not cause | Misses systemic issue |
| Few-shot examples in templates | Potentially very effective | Significantly longer prompts, harder to maintain | Maintenance burden |

## Quantified Validation

| Metric | V1 Original | Strategy A (Correction) | Strategy B (V2 Templates) | Combined (B → A) |
|--------|:-----------:|:-----------------------:|:-------------------------:|:-----------------:|
| Total failures | 36 | 12 | 8 | **0** |
| Failure rate | 20.0% | 6.7% | 4.4% | **0.0%** |
| Reduction vs V1 | — | 66.7% | 77.8% | **100.0%** |
| `incomplete_answer` | 15 | 5 | 5 | **0** |
| `poor_quality_tips` | 13 | 6 | 2 | **0** |
| `safety_violations` | 3 | 0 | 0 | 0 |
| `unrealistic_tools` | 5 | 0 | 0 | 0 |

The combined pipeline achieved **100% failure elimination** (36 → 0), exceeding the PRD's >80% target. V2 templates reduced correction scope from 21/30 records to 6/30 — making correction targeted and precise rather than brute-force.

## Consequences

**Easier:** The failure analysis → template improvement → correction cycle is a reusable blueprint for LLM generation pipelines. All strategies are cached, so re-running is free.
**Harder:** V2 templates are more prescriptive, which may reduce output diversity. The combined pipeline requires two LLM calls for failing records (correction + re-evaluation), though this was only 6 of 30.
**Portability:** This upstream prevention pattern was the #1 lesson applied to P4's dual-channel enforcement strategy (schema validation + prompt requirements), achieving **100% validation on 550 records**.

## Cross-References

- **ADR-003**: Calibrated judge identified the failure patterns that drove template improvements. Without the 0% → 20% calibration, there was no signal to act on.
- **ADR-001**: Instructor powered both v2 re-generation and the correction loop, maintaining the self-healing retry pattern.
- **ADR-002**: Flat schema made DataFrame analysis tractable — `groupby('failure_mode').count()` directly identified the 50%/43.3% concentration that motivated v2 templates.

## Java/TS Parallel

Analogous to **shifting left in CI/CD**. Strategy A (correction) is catching bugs in QA and filing fix tickets. Strategy B (templates) is adding stricter TypeScript types and ESLint rules that prevent the bug class entirely. The combined approach is defense in depth: strict types catch most issues at compile time, and integration tests catch what slips through.

**The key insight:** Upstream prevention (better prompts) outperforms downstream correction (better retries) by 11 percentage points, just as compile-time checks outperform runtime patches.

## Interview Signal

Demonstrates **data-driven optimization** and the "shift left" philosophy applied to ML pipelines. The engineer didn't just build a correction loop — they analyzed failure distributions, identified systemic patterns, and applied upstream prevention that reduced the correction surface by 71% (21 → 6 records). This signals the ability to optimize at the system level rather than treating individual symptoms — a hallmark of senior engineering judgment.
