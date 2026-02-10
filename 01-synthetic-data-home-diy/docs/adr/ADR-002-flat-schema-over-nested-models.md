# ADR-002: Why Flat Schema Matching Spec over Nested Models

**Date**: 2026-02-09
**Status**: Accepted
**Project**: P1 — Synthetic Data, Home DIY Repair

## Context

When designing the `DIYRepairRecord` Pydantic model, we had a choice between:

- **Flat**: 7 top-level fields (`question`, `answer`, `equipment_problem`,
  `tools_required`, `steps`, `safety_info`, `tips`) with `tools_required` and
  `steps` as simple `list[str]`.
- **Nested**: Dedicated sub-models like `Tool(name, category, is_common)`,
  `RepairStep(order, instruction, estimated_time)`,
  `SafetyPrecaution(warning, severity)` — richer typing but more complex JSON
  for the LLM to produce.

The PRD specifies the flat structure. The question was whether to follow the spec
literally or "improve" it with nested models for stronger type safety.

## Decision

Use the **flat schema** exactly as specified in the PRD. `tools_required` is
`list[str]` (with per-item `min_length=2` via `Annotated`), `steps` is
`list[str]` (with per-item `min_length=10`), and all other fields are plain
strings with length constraints.

```python
tools_required: list[Annotated[str, Field(min_length=2)]] = Field(
    min_length=1,
    description="List of tools needed for this repair",
)
steps: list[Annotated[str, Field(min_length=10)]] = Field(
    min_length=2,
    description="Ordered repair steps the homeowner should follow",
)
```

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Flat schema** (chosen) | Matches PRD spec exactly, simpler JSON schema for the LLM, fewer nested validation errors, higher generation success rate | Less type-safe (a tool is just a string), no structured metadata on steps |
| **Nested models** (`Tool`, `RepairStep`, `SafetyPrecaution`) | Richer domain modeling, stronger validation (e.g., `Tool.is_common: bool`), better for downstream analysis | Deeper JSON schema → more LLM errors → more retries → higher cost. Over-engineers a first project where the data flows into a CSV for analysis |
| **Hybrid** (some nesting, e.g., `RepairStep` but flat tools) | Partial richness | Inconsistent design — "why nest steps but not tools?" |

### Why flat wins for LLM generation

The JSON schema that Instructor injects into the prompt grows with nesting
depth. Each nested model adds `properties`, `required`, and `type` blocks.
Empirically, GPT-4o-mini handles flat schemas with near-100% first-attempt
success. Nested schemas increase:

1. **Token count** in the prompt (more schema = more input tokens = more cost).
2. **Structural errors** — the LLM might produce `{"tool": "wrench"}` instead
   of `{"name": "wrench", "category": "hand_tool", "is_common": true}`.
3. **Retry rate** — each structural error triggers an Instructor retry, burning
   another API call.

For a 30-record batch at $0.15/1M input tokens, this matters less for cost and
more for reliability. We achieved **100% generation success (30/30)** with the
flat schema.

## Consequences

**Easier:**
- LLM produces valid JSON on first attempt in almost all cases — 30/30 success.
- Schema is easy to read and maps 1:1 to the PRD, so any reviewer can verify
  correctness by comparing the two.
- Downstream consumers (evaluator, analysis) work with simple dicts/DataFrames
  without needing to flatten nested structures.
- `Annotated[str, Field(min_length=N)]` on list items gives us per-element
  validation without full sub-models — a Pydantic v2 pattern that hits the
  sweet spot.

**Harder:**
- If we later need structured tool metadata (e.g., "is this a specialty tool?"
  for the `unrealistic_tools` failure mode), we'd need to either parse the
  string or restructure the schema. Currently, the LLM judge handles this
  semantic analysis.
- No type distinction between a step like "Turn off power" (safety-critical)
  and "Tighten the screw" (mechanical) — both are plain strings.

## Java/TS Parallel

This is the classic **DTO vs. rich domain model** trade-off. In a Spring app,
you might model tools as a `List<ToolDTO>` with typed fields. But when the
"API" is an LLM that returns free-form JSON, simpler DTOs reduce parse failures
— like using `Map<String, Object>` for a flaky external API instead of a rigid
`@JsonDeserialize` class. In TypeScript terms: `tools: string[]` instead of
`tools: { name: string; category: ToolCategory; isCommon: boolean }[]`. You can
always parse the strings later; you can't un-fail a generation call.
