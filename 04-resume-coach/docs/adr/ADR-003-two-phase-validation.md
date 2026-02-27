# ADR-003: Two-Phase Validation — Structural (Instructor) vs Semantic (Labeler)

**Date**: 2026-02-26
**Project**: P4 — Resume Coach

## Status

Accepted

## Context

P4 validates resumes at two fundamentally different levels:

1. **Structural**: Is the JSON well-formed? Does it parse into a valid `Resume` Pydantic model with all 30+ required fields, correct enum values (`ProficiencyLevel`, `ExperienceLevel`), and passing field validators (ISO dates, GPA 0.0–4.0, phone >= 10 digits)?
2. **Semantic**: Is the *content* reasonable? Does a resume claim 15 years of experience for a Junior-level role? Does it list skills that never appeared in the job description? Does the writing feel like awkward AI-generated prose?

These are different failure modes with different detection costs. Structural failures are binary — Pydantic either accepts or rejects the object. Semantic failures require comparing resume content against the paired job description using domain-specific rules (Jaccard similarity, experience year thresholds, seniority level mapping).

Mixing both into one layer creates three problems:
- **Cost**: Putting semantic rules into Instructor retry prompts means every labeling check costs $0.002/pair (GPT-4o-mini round-trip). For 250 pairs, that's $0.50 just for deterministic comparisons that Python can compute in 250ms total.
- **Testability**: If the labeler requires an LLM to run, `test_labeler.py` needs API mocking or real API calls. P4 has 504 tests — making any of them depend on LLM responses introduces flakiness.
- **Coupling**: The generator would need to know about business rules (experience thresholds, seniority mappings) to inject them as retry prompts, and the labeler would need to parse raw JSON. Each module would do two jobs instead of one.

## Decision

Implement two-phase validation with strict separation:

**Phase 1 — Structural (Instructor/Pydantic, at generation time)**:
- Runs in `generator.py` via `instructor.from_openai(client, mode=Mode.JSON)` with `max_retries=5`
- Validates against 35 Pydantic models with ~30 validation points per `Resume`
- On failure: Instructor injects the exact `ValidationError` back as a correction prompt and retries
- Output: guaranteed valid `GeneratedResume` and `GeneratedJob` objects. Downstream stages never see malformed data.
- Cost: ~1–2 extra API calls on the ~15% of resumes that fail first-pass validation

**Phase 2 — Semantic (Labeler, post-generation)**:
- Runs in `labeler.py` via `label_pair(resume: Resume, job: JobDescription, pair_id: str, normalizer: SkillNormalizer)`
- Computes 5 boolean failure flags: `experience_mismatch`, `seniority_mismatch`, `missing_core_skills`, `has_hallucinations`, `has_awkward_language`
- Plus supporting numeric fields: `skills_overlap` (Jaccard 0.0–1.0), `experience_years_resume`, `seniority_level_resume` (0–4), `missing_skills: list[str]`
- Pure Python — deterministic, ~1ms per pair, zero LLM calls, zero cost
- Output: `FailureLabels` with 18 fields total

**Optional Phase 3 — LLM Judge (GPT-4o, on demand)**:
- Runs in `judge.py` when deeper quality assessment is needed
- Structured output via Instructor: `has_hallucinations`, `has_awkward_language`, `overall_quality_score` (0–1), `fit_assessment`, `recommendations`, `red_flags`
- Cost: ~$0.002/pair × 250 = ~$0.50. Skippable via `--skip-judge` flag
- Provides a second opinion on the rule-based labeler's output — used for agreement analysis (`judge_vs_rules_agreement.png`)

## Alternatives Considered

**Single-phase in Instructor (validate everything at generation time)**: This would embed semantic rules like "experience must be >= 3 years for senior role" into the LLM retry loop. The model would get correction prompts for content quality issues, not just structural errors. Problem: semantic rules are fuzzy and context-dependent. Instructor's retry mechanism works well for binary constraints (field present/absent, enum valid/invalid) but struggles with judgment calls like "is this experience mismatch bad enough to flag?" Mixing both types of validation into one retry loop means the model could fix a semantic issue by introducing a structural one. It also means labeling costs $0.002/pair in API calls for checks that Python can do in 1ms.

**Single-phase in labeler (validate everything post-generation)**: The labeler would handle both JSON parsing and content analysis. But the labeler receives deserialized Pydantic objects — it never sees raw JSON. To also handle structural validation, it would need to accept raw strings, wrap `model_validate_json()` with fallback logic, and return partially-parsed objects. This turns a clean analytical function into a messy parser with two responsibilities. The labeler's `label_pair()` signature takes typed `Resume` and `JobDescription` objects precisely because Phase 1 guarantees they're valid.

**LLM for all validation**: Maximum flexibility — the model can catch anything. But at 10–100× the cost of rule-based checks, with non-deterministic outputs that change between runs, and ~1s latency per pair instead of ~1ms. P4's failure mode analysis requires reproducible numbers. Running the labeler twice should produce identical results. An LLM-only approach makes that impossible.

## Consequences

### What This Enabled

Because structural validation runs at generation time via Instructor, every downstream module — labeler, judge, corrector, analyzer, multi_hop — receives guaranteed-valid typed objects. No defensive parsing, no `try/except` around `model_validate()`, no "what if this field is None?" guards in 5 separate modules. The labeler processes all 250 pairs in ~250ms at zero cost because its 5 failure flags and 13 supporting fields are pure Python comparisons against clean data — making `test_labeler.py` fully deterministic with constructed fixtures and no LLM mocking. The phase separation also enabled the judge-vs-labeler agreement analysis (`judge_vs_rules_agreement.png`): comparing Phase 2's rule-based labels against Phase 3's GPT-4o assessments is only meaningful because the two phases produce genuinely independent evaluations of the same data.

### Accepted Trade-offs

- Two separate pipeline stages means two invocation points: `run_generation.py` (Phase 1) and `run_labeling.py` (Phase 2), orchestrated by `pipeline.py`. Adding a new validation concern requires deciding which phase it belongs in
- Edge cases that are structurally valid but semantically absurd (e.g., a date range `"2020-01"` to `"2019-06"` that passes ISO validation but is logically backwards) fall between phases. Currently handled by explicit checks in the labeler, but these are easy to miss
- The `--skip-judge` flag means some pipeline runs have Phase 3 data and some don't. Downstream code (analyzer, API) must handle the missing-judge-data case

## Cross-Project Context

P1 (Synthetic Data) mixed structural and semantic validation in a single `evaluator.py`, which worked at small scale (30 records, 6 failure modes) but made testing expensive — every evaluator test needed either a real LLM call or complex mocking. The lesson from P1: separate what the machine validates (schema) from what the human cares about (content quality). P4's two-phase architecture is the direct application of that lesson at 8× the scale.

## Java/TS Parallel

This maps directly to Java's separation of `@Valid` (Bean Validation / JSR-380 at deserialization) from `@Service` business rules. Phase 1 is `@Valid` on a `@RequestBody` DTO: the framework enforces `@NotNull`, `@Size`, `@Pattern` before your code sees the object. Phase 2 is a service method that receives the validated DTO and applies domain logic: "if experience < 3 years and seniority == SENIOR, flag as mismatch." The insight is identical: structural validation fails fast at the boundary, business rules run on guaranteed-clean data downstream.

## Validation

The two-phase split delivered on all three design goals. **Cost**: Phase 2 labeling runs in 250ms for 250 pairs at $0.00; Phase 3 judge is $0.50 and skippable. **Testability**: 532 tests pass without API keys; `test_labeler.py` alone covers all 5 failure modes with pure Python fixtures. **Decoupling**: the corrector (8/8 records fixed, 100% rate) operates on Phase 2 labels — it never needs to re-validate structure because Phase 1 already guaranteed it.

## Reversibility

**Low** — Merging the phases would require rewriting the labeler to handle raw JSON input, embedding semantic rules into Instructor retry prompts, and refactoring all downstream modules that currently assume typed inputs. The phase boundary is load-bearing.
