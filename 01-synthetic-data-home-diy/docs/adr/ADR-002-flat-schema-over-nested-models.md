# ADR-002: Flat Schema over Nested Models

**Date:** 2026-02-09
**Status:** Accepted
**Project:** P1 — Synthetic Data, Home DIY Repair
**Category:** Data Model

## Context

The `DIYRepairRecord` Pydantic model had two viable designs: a **flat** schema (7 top-level fields, lists as `list[str]`) matching the PRD exactly, or **nested** sub-models (`Tool(name, category, is_common)`, `RepairStep(order, instruction, estimated_time)`) for richer typing. The question was whether to follow the spec literally or "improve" it with nested models for stronger type safety.

## Decision

Use the **flat schema** exactly as specified in the PRD. `tools_required` is `list[str]` with per-item `min_length=2` via `Annotated`, `steps` is `list[str]` with per-item `min_length=10`. All fields use plain strings with length constraints.

```python
tools_required: list[Annotated[str, Field(min_length=2)]] = Field(
    min_length=1, description="List of tools needed for this repair",
)
steps: list[Annotated[str, Field(min_length=10)]] = Field(
    min_length=2, description="Ordered repair steps the homeowner should follow",
)
```

The key trade-off: we sacrifice domain-rich typing for LLM generation reliability. `Annotated[str, Field(min_length=N)]` on list items gives per-element validation without full sub-models — the Pydantic v2 sweet spot.

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Flat schema** ✅ | Matches PRD, simpler JSON for LLM, fewer validation errors | Less type-safe (tool is just a string) | — (selected) |
| Nested models (`Tool`, `RepairStep`) | Richer domain modeling, stronger validation | Deeper JSON schema → more LLM errors → more retries → higher cost | Over-engineers a project where data flows to CSV/DataFrame |
| Hybrid (nest steps, flat tools) | Partial richness | Inconsistent — "why nest steps but not tools?" | Arbitrary boundary |

## Quantified Validation

- **First-attempt success**: **30/30 records** (100%) with zero structural parse errors — the LLM never produced malformed JSON
- **Expected nested overhead**: Our flat schema achieved **0 retries across 30 records**. Nested schemas increase the JSON structure the LLM must produce correctly in a single pass — each additional nesting level adds `properties`, `required`, and `type` blocks that compound error surface. Even a conservative 5% retry rate on nested schemas would have added 3+ unnecessary API calls and ~$0.01 in wasted cost per retry cycle
- **Token efficiency**: Flat schema injects ~200 tokens of JSON schema into the prompt. Nested models with `Tool`, `RepairStep`, and `SafetyPrecaution` sub-models would inject ~500+ tokens — a **2.5× increase** in schema overhead per call
- **Downstream simplicity**: Flat records map directly to DataFrame rows with zero flattening logic, enabling immediate analysis in `analysis.py`

## Consequences

**Easier:** Schema maps 1:1 to the PRD — any reviewer can verify correctness by comparing the two. Downstream consumers (evaluator, analysis) work with simple dicts/DataFrames without flattening nested structures.
**Harder:** If we later need structured tool metadata (e.g., "is this a specialty tool?" for `unrealistic_tools` detection), we'd need to parse strings or restructure. Currently the LLM judge handles this semantic analysis.
**Portability:** P4 (Resume Coach) adopted the same flat-first principle — `Job` and `Resume` schemas use `list[str]` for skills rather than nested `Skill(name, category, years)`, achieving **100% validation on 550 records**.

## Cross-References

- **ADR-001**: Instructor handles schema injection regardless of complexity, but flat schemas maximize the benefit — simpler schema = fewer tokens = higher success rate.
- **ADR-004**: Flat schema made DataFrame analysis trivial for failure pattern identification. The `groupby('failure_mode').count()` operations that drove template improvements would have required flattening logic with nested models.

## Java/TS Parallel

Flat schema is a **single-level DTO with `@NotBlank` and `@Size` annotations** rather than nested `@Valid` entity graphs. In Spring: `RepairRecordDTO` with `List<String> tools` annotated `@Size(min=1)` instead of `List<ToolEntity> tools` with `@Valid` cascading into `ToolEntity.name`, `ToolEntity.category`. The flat DTO serializes cleanly to CSV or DataFrame; the entity graph requires flattening.

**The key insight:** Schema design for LLM consumers follows different constraints than schema design for API consumers. In traditional APIs, deeper types buy safety; in LLM generation, deeper types buy retry costs. Simplicity is a first-class requirement, not a compromise — just as flat DTOs optimize for serialization correctness over domain richness.

## Interview Signal

Demonstrates **pragmatic schema design** over theoretical purity. The engineer recognized that LLMs have different constraints than traditional APIs — deeper schemas don't buy safety, they buy retry costs. Choosing spec compliance over "clever" nesting shows understanding of LLM-specific failure modes and the discipline to solve the actual problem rather than an imagined one.
