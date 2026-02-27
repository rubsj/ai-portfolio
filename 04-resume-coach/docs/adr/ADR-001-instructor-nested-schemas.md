# ADR-001: Use Instructor with max_retries=5 for Nested Schema Generation

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 requires generating deeply nested JSON objects: a `Resume` contains `ContactInfo`, a list of `Skill` (each with `name`, `proficiency_level`, `years`), a list of `Experience` (each with `responsibilities: list[str]`), a list of `Education`, and an optional `Summary`. Across 250 resumes × 50 jobs, the generator must produce valid Pydantic models every time without manual parsing.

Raw OpenAI responses fail Pydantic validation ~15–30% of the time on complex nested schemas due to:
- Missing required fields in nested objects
- Wrong enum values (e.g., `"advanced"` instead of `ProficiencyLevel.ADVANCED`)
- Malformed dates (`"2020"` instead of `"2020-01"`)
- Arrays that are `null` instead of `[]`

A retry-from-scratch strategy would waste API calls and add latency. We need a library that understands *why* validation failed and patches the response intelligently.

## Decision

Use **Instructor** (`instructor` library) with `max_retries=5` for all structured LLM generation in P4.

Instructor wraps the OpenAI client and:
1. Validates the LLM response against the Pydantic schema
2. On failure, injects the validation error message back into the conversation as a correction prompt
3. Retries up to `max_retries` times, each time giving the model precise feedback on what to fix

Use `instructor.patch(client, mode=instructor.Mode.JSON)` to enforce JSON mode at the API level — prevents the model from prefacing output with prose like `"Here is the JSON:"`.

Strategic schema design to reduce retry frequency:
- Mark rarely-provided fields `Optional[str] = None` (e.g., `linkedin`, `portfolio`) so missing fields don't fail validation
- Keep `responsibilities: list[str]` flexible in length — no `min_length` enforcement
- Use `@field_validator` for date format normalization rather than strict regex patterns

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| Raw OpenAI + manual JSON parse | No extra dependency | Brittle: manual `try/except json.loads()`, no retry intelligence |
| LangChain structured output | Familiar, broad ecosystem | Heavy dependency; output parsers are less precise than Instructor's validation feedback |
| Pydantic `model_validate_json()` with manual retry | Pure stdlib | Retry prompt engineering is all on us; no automatic error injection |
| **Instructor** (chosen) | Validation errors → automatic correction prompts; `max_retries` covers transient model failures | One extra dependency; slight coupling to OpenAI client |

## Consequences

**Easier**:
- 100% validation rate achieved on 250 resumes in P4 (vs expected ~70–85% with raw parse)
- Schema changes are automatically propagated to retry prompts — no prompt updates needed
- `Mode.JSON` eliminates prose-wrapping failures without manual prompt engineering

**Harder**:
- `max_retries=5` means worst-case 5× API cost per record on validation failures (rare in practice: ~2 retries avg)
- Instructor occasionally has breaking changes on major OpenAI client version bumps — pin versions in CI

## Java/TS Parallel

Instructor is the equivalent of a **Bean Validation framework** (JSR-380 / Hibernate Validator) combined with an **automatic correction loop**. In Java, `@Valid` on a DTO tells the framework to validate on deserialization, but it throws immediately — you'd have to write retry logic manually. Instructor automates the "validation failure → rephrase → retry" cycle that you'd otherwise build yourself with `@Retryable` from Spring Retry + custom error message construction.
