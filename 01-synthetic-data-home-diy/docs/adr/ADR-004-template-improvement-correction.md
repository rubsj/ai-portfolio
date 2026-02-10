# ADR-004: Template Improvement as Correction Strategy

**Date**: 2026-02-09
**Status**: Accepted
**Project**: P1 — Synthetic Data, Home DIY Repair

## Context

After generating 30 v1 records and evaluating them with the LLM-as-Judge
(GPT-4o), we observed a 20% overall failure rate (36 failures across 180
evaluations). Analysis revealed two dominant failure modes:

- **incomplete_answer** (50% of records): answers lacking troubleshooting
  advice, part sourcing info, and "what if this doesn't work" guidance
- **poor_quality_tips** (43.3%): generic platitudes like "be careful" instead
  of expert-level, repair-specific advice

These two modes accounted for 28 of 36 total failures (78%). The remaining
failures were `unrealistic_tools` (5) and `safety_violations` (3).
`overcomplicated_solution` and `missing_context` had zero failures.

The PRD (Section 8) specified two correction strategies: individual record
correction and template improvement. The question was: which delivers better
failure rate reduction?

## Decision

Implement both strategies and compare:

**Strategy A — Individual record correction**: For each record with ≥1 failure,
send the original record + judge feedback (flagged modes + reasoning) to
GPT-4o-mini via Instructor for targeted correction. The correction prompt
includes the original record JSON and specific failure reasons, asking the
model to fix only the flagged issues while preserving good content.

**Strategy B — Template v2 improvement**: Analyze which failure modes are most
common globally, then add explicit quality requirement instructions to the
system prompt. The v2 system prompt appends a `--- QUALITY REQUIREMENTS ---`
section with mode-specific instructions for all modes that had >0 failures.

### V2 Template Additions (Example)

For `incomplete_answer`:
```text
IMPORTANT: Your answer MUST include: (1) how to identify the specific
part/problem, (2) where to buy replacement parts if needed, (3) what to do
if the fix doesn't work, and (4) troubleshooting advice for common
complications. The answer should be at least 4-5 detailed sentences.
```

For `poor_quality_tips`:
```text
IMPORTANT: Tips MUST be specific and expert-level. Do NOT use generic advice
like 'be careful' or 'take your time'. Instead, provide concrete tips that
a professional would give — specific product recommendations, pro techniques,
common mistakes to avoid, and maintenance advice to prevent recurrence.
```

### Results

| Metric | V1 Original | Corrected (A) | V2 Generated (B) |
|--------|------------|---------------|-------------------|
| Total failures | 36 | 12 | 8 |
| Failure rate | 20.0% | 6.7% | 4.4% |
| Reduction vs V1 | — | 66.7% | 77.8% |
| incomplete_answer | 15 | 5 | 5 |
| poor_quality_tips | 13 | 6 | 2 |
| safety_violations | 3 | 0 | 0 |
| unrealistic_tools | 5 | 0 | 0 |

Strategy B (template v2) outperformed Strategy A (individual correction):
- V2 achieved 77.8% reduction vs 66.7% for correction
- V2 nearly eliminated `poor_quality_tips` (13 → 2)
- Both strategies fully eliminated `safety_violations` and `unrealistic_tools`
- Neither fully eliminated `incomplete_answer` (15 → 5 in both)

### Why Template Improvement Works Better

1. **Upstream fix vs downstream patch**: V2 templates prevent failures at
   generation time. Individual correction asks the same model to fix its own
   output — it often makes the same category of mistake again.

2. **Systemic patterns**: The dominant failures (`incomplete_answer`,
   `poor_quality_tips`) are systemic — they stem from GPT-4o-mini's default
   behavior of giving "good enough" answers. Template instructions override
   this default with specific requirements.

3. **`poor_quality_tips` almost eliminated**: V2's explicit "do NOT use generic
   advice" instruction was highly effective (13 → 2). Individual correction
   only achieved 13 → 6 because the model still defaults to generalities when
   "fixing" tips.

### Why 80% Target Was Not Quite Met (Single Strategy)

The PRD target was >80% reduction. V2 alone achieved 77.8% (close but not
met). The remaining 8 failures were mostly `incomplete_answer` (5) — this
mode is intrinsically harder to prevent because "completeness" is subjective
and the judge applies a strict standard (must include troubleshooting, part
sourcing, what-if scenarios).

### Combined Strategy: V2 Templates + Individual Correction

To close the gap, we applied individual correction (Strategy A) to the 6
remaining V2 records that had failures. This combined approach uses templates
to prevent systemic failures and correction to handle edge cases.

| Metric | V1 Original | V2 Generated | V2 + Correction |
|--------|------------|--------------|-----------------|
| Total failures | 36 | 8 | **0** |
| Failure rate | 20.0% | 4.4% | **0.0%** |
| Reduction vs V1 | — | 77.8% | **100.0%** |
| incomplete_answer | 15 | 5 | **0** |
| poor_quality_tips | 13 | 2 | **0** |
| safety_violations | 3 | 0 | 0 |
| unrealistic_tools | 5 | 0 | 0 |
| overcomplicated_solution | 0 | 1 | **0** |
| missing_context | 0 | 0 | 0 |

The combined pipeline achieved **100% failure reduction** (36 → 0),
far exceeding the >80% target. The correction pass was highly effective
on V2 records because:

1. **Fewer records to fix**: Only 6 of 30 V2 records had failures
   (vs 21 of 30 V1 records). This means correction was targeted and precise.
2. **Better starting point**: V2 records already had strong quality
   requirements baked in. The correction only needed to address residual
   gaps, not systemic defaults.
3. **Judge feedback is actionable**: The specific failure reasons from the
   judge (e.g., "lacks troubleshooting advice") give GPT-4o-mini clear
   instructions for what to add.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Individual correction only** (Strategy A) | Simpler, no template changes | 66.7% reduction — misses the systemic issue |
| **Template improvement only** (Strategy B, chosen as primary) | 77.8% reduction, prevents failures upstream | Requires analyzing failure patterns first |
| **Both strategies combined** (correct v2 records, chosen) | 100% reduction, zero failures | Extra API cost for 6 records |
| **Few-shot examples in templates** | Could be very effective | Significantly longer prompts, harder to maintain |

## Consequences

**Easier:**
- Template v2 improvement is reusable — the same pattern (analyze failures →
  add prevention instructions) applies to any LLM generation pipeline.
- The failure analysis → template improvement → correction cycle is a
  compelling portfolio artifact: "I analyzed quality metrics, improved prompt
  templates, and applied targeted correction to achieve 100% failure reduction."
- All strategies are cached, so re-running the full pipeline is free.
- The combined approach provides a blueprint for production LLM pipelines:
  templates prevent systemic failures, correction handles edge cases.

**Harder:**
- V2 templates are longer and more prescriptive, which may reduce output
  diversity. Future projects should monitor whether quality instructions
  lead to overly formulaic outputs.
- The combined pipeline requires two LLM calls for failing records
  (correction via GPT-4o-mini + re-evaluation via GPT-4o), increasing cost
  for those records. In practice this was only 6 of 30 records.

## Key Insight: Use BOTH Strategies — Templates Prevent, Correction Polishes

The most impactful single improvement came from adding explicit quality
requirements to the generation prompt (v2 templates). But the **combination**
of upstream prevention + downstream correction achieved total elimination
of failures. This maps to a general software principle: **defense in depth**.

Production pipelines should use BOTH strategies:
- **Templates** prevent systemic failure classes (the 77.8% reduction)
- **Individual correction** handles edge cases the templates miss (the
  remaining 22.2%)

In LLM terms: the generation prompt is the "design spec" — making it more
precise yields better first-pass output. The correction loop is the "code
review" — it catches the specific issues that slipped through.

## Java/TS Parallel

This is analogous to **shifting left in CI/CD**. Strategy A (individual
correction) is like catching bugs in QA and filing fix tickets. Strategy B
(template improvement) is like adding stricter TypeScript types, ESLint rules,
or Spring validation annotations that prevent the bug class entirely. Both
reduce defects, but the upstream fix is more sustainable.

Another parallel: **schema-first API design**. Defining an OpenAPI spec with
strict field requirements upfront (Strategy B) produces better API responses
than validating and patching responses after the fact (Strategy A).
