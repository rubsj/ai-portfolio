# ADR-003: Two-Phase Validation — Structural (Instructor) vs Semantic (Labeler)

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 requires validating resumes at two distinct levels:

1. **Structural validity**: Is the JSON well-formed? Does it parse into a valid `Resume` Pydantic model? Are required fields present? Are enum values legal?
2. **Semantic validity**: Is the resume *content* reasonable? Does it hallucinate skills not mentioned in training prompts? Does it claim 15 years of experience for a Junior-level role? Does the writing style feel awkward?

These are fundamentally different failure modes. Structural failures are binary (valid/invalid Pydantic). Semantic failures are nuanced judgments that require comparing resume content against the paired job description.

Mixing both concerns into a single validation layer would create tight coupling: the labeler would need to parse JSON, the Instructor retry loop would need business-rule knowledge, and tests would become expensive (requiring real LLM calls to test label logic).

## Decision

Implement **two-phase validation** with strict separation of concerns:

**Phase 1 — Structural (Instructor / Pydantic)**:
- Runs at generation time in `run_generation.py`
- `instructor.patch(client)` enforces JSON → `Resume` Pydantic model, with up to 5 retries on validation failure
- Output: guaranteed valid `GeneratedResume` object (or exception propagated to caller)
- Cost: ~1–2 extra API calls on ~15% of resumes that fail first-pass validation

**Phase 2 — Semantic (Labeler)**:
- Runs post-generation in `run_labeling.py` via `label_pair(resume, job, pair_id, normalizer)`
- Computes: Jaccard skill overlap, experience year mismatch, seniority mismatch, missing core skills, hallucination flags, awkward language flags
- Pure Python computation (no LLM calls) — deterministic, fast (~1ms per pair), free
- Output: `FailureLabels` — 7 boolean flags + supporting numeric fields

**Optional Phase 3 — LLM Judge (GPT-4o)**:
- Runs selectively in `judge.py` when deeper semantic quality assessment is needed
- Structured extraction via Instructor: `has_hallucinations`, `has_awkward_language`, `overall_quality_score`, `fit_assessment`, `recommendations`, `red_flags`
- Most expensive (~$0.002/pair × 250 = ~$0.50); skippable via `--skip-judge` flag

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| Single-phase: validate everything in Instructor | One system to learn | Semantic rules in retry prompts are fragile; costs $0.002 per pair just for labeling |
| Single-phase: validate everything in labeler | Pure Python, fast | Labeler can't enforce JSON schema; would need manual parse + fallback logic |
| LLM for all validation (including structural) | Maximum flexibility | 10–100× more expensive; non-deterministic; latency ~1s per pair |
| **Two-phase** (chosen) | Each tool used for what it's best at; independent testability | Slight pipeline complexity (two separate stages) |

## Consequences

**Easier**:
- `test_labeler.py` has zero LLM dependencies — fully unit-testable with constructed fixtures
- Labeler logic can be iterated rapidly without API calls or cost
- Structural failures are caught early (at generation) — downstream stages receive clean data
- Phase separation allows skipping expensive LLM judge via flag

**Harder**:
- Two separate pipeline stages means two config points (generation flags + judge flags)
- Edge cases that fall between structural and semantic (e.g., date range that parses correctly but is logically impossible) require explicit handling in labeler

## Java/TS Parallel

This is the exact split between **`@Valid` (Bean Validation)** and **business rule validation** in Spring:
- `@Valid` on a DTO = Instructor structural phase: enforces type, required fields, enum values
- `@Service` with custom logic = labeler semantic phase: enforces business invariants (experience years, seniority match)
- In Java, you'd never put `"experience must be >= 3 years for senior role"` into a `@NotNull` constraint — that's service-layer logic. Same principle applies here.
