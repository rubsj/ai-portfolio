# ADR-003: Judge Prompt Calibration (0% → 20%)

**Date:** 2026-02-09
**Status:** Accepted
**Project:** P1 — Synthetic Data, Home DIY Repair
**Category:** Evaluation

## Context

The LLM-as-Judge uses GPT-4o to evaluate GPT-4o-mini records across 6 binary failure modes. Our first prompt produced **0% failure rate** — the judge passed every record. Manual inspection revealed genuine issues (electrical repairs without "turn off circuit breaker" as step 1, beginner tasks requiring specialty tools). The root cause: GPT-4o's RLHF training biases it toward positive assessments without explicit strictness cues.

## Decision

Rewrite the judge system prompt with **explicit strictness calibration** using three levers:

1. **Identity priming**: "You are a STRICT quality evaluator" + "Your job is to find deficiencies"
2. **Concrete criteria**: Replace vague definitions with checkable conditions, numeric thresholds, and named examples
3. **Distribution anchoring**: "Most guides are 3-4" and "most have at least 1-2 issues" to combat positivity bias

### What Changed (Key Calibration Moves)

| Element | Lenient (0%) | Strict (20%) | Why It Matters |
|---------|-------------|-------------|----------------|
| Identity | "quality evaluator" | "**STRICT** quality evaluator" | Primes critical stance |
| Expectation | *(none)* | "Most guides have 1-2 issues" | Normalizes finding failures |
| `safety_violations` | "Missing safety info" | Explicit checklist: power-off, PPE, professional referral, hazard callouts | Vague → pass/fail checklist |
| `unrealistic_tools` | "Tools a homeowner wouldn't have" | Named examples: multimeter, torque wrench + "fail if list suspiciously short" | Abstract → concrete |
| `overcomplicated` | "Too complex" | Hard limits: beginner ≤8 steps, intermediate ≤12 | Numeric thresholds eliminate subjectivity |
| Anti-rationalization | *(none)* | "Only give 0 when genuinely strong" | Directly combats positivity bias |

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Strict prompt with concrete criteria** ✅ | 20% failure rate matches manual review, reproducible, actionable reasons | Requires domain expertise to write criteria | — (selected) |
| Lenient prompt (original) | Simple, few failures | 0% failure rate is useless — defeats evaluation purpose | No diagnostic value |
| Multi-turn judge (ask, then challenge) | Could catch edge cases | 2× API cost, complex orchestration, diminishing returns on 30 records | Cost/complexity not justified |
| Temperature tuning (lower = stricter?) | No prompt rewrite | Temperature controls randomness, not strictness | Fundamental misunderstanding of the knob |

## Quantified Validation

- **Failure rate**: **0% → 20%** (0 → 36 failures across 180 evaluations)
- **Diagnostic yield**: Clear concentration in `incomplete_answer` (**50%**) and `poor_quality_tips` (**43.3%**) — actionable distribution that drove ADR-004
- **Manual vs LLM agreement**: **81.7%** raw agreement across 10 records × 6 modes (60 binary comparisons). Range: 60% (`poor_quality_tips`) to 100% (`overcomplicated_solution`, `missing_context`). Cohen's Kappa not computed — acknowledged gap, though 81.7% on strict binary labels indicates concordance well above chance
- **Score anchoring**: Without "most guides are 3-4", scores clustered at 4–5. With anchoring, realistic 3–4 distribution

## Consequences

**Easier:** The 20% failure rate gives the correction loop meaningful work. Failure reasons in `JudgeResult.labels[].reason` are specific enough to drive targeted template improvements (v2 prompts).
**Harder:** The "right" failure rate is subjective — 20% works for DIY repair but each new domain needs its own calibrated criteria, not a copy-paste of this prompt.
**Portability:** The three-lever calibration pattern (identity + specificity + anchoring) was reused in P2 evaluation judges and P4's quality scoring system.

## Cross-References

- **ADR-004**: Calibrated judge identified the failure patterns (`incomplete_answer` 50%, `poor_quality_tips` 43.3%) that directly drove v2 template improvements. Without 0% → 20% calibration, there was no signal to act on.
- **ADR-001**: Same Instructor pattern (`response_model=JudgeResult`) powers the judge, providing structured failure labels with typed reasons.

## Java/TS Parallel

Analogous to **configuring linter rule severity**. ESLint with all rules set to `"warn"` produces noise developers ignore. Changing to `"error"` with specific configs (`max-lines-per-function: 50`) forces compliance. The lenient prompt was ESLint with rules `"off"`; the strict prompt is a tuned `.eslintrc` catching real issues without false positives.

**The key insight:** LLM-as-Judge calibration is prompt engineering, not model selection. The 0% → 20% shift came entirely from prompt changes, not from switching models.

## Interview Signal

Demonstrates understanding of **RLHF bias in LLMs** and systematic prompt engineering. The engineer identified that helpfulness training creates measurable evaluation bias, then applied three calibration techniques with quantified before/after results. This signals experience designing evaluation pipelines — critical for production ML where "is my model good enough?" is the hardest question.
