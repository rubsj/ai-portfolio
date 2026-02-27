# ADR-001: Instructor over Raw OpenAI API

**Date:** 2026-02-09
**Status:** Accepted
**Project:** P1 — Synthetic Data, Home DIY Repair
**Category:** Tool Choice

## Context

We needed to generate structured JSON from GPT-4o-mini conforming to a 7-field Pydantic schema (`DIYRepairRecord`). The raw OpenAI API returns strings requiring manual parsing, validation, and retry logic. As the first of 9 portfolio projects, the LLM → structured output → validation pattern chosen here would propagate across every subsequent project.

## Decision

Use **Instructor** (`instructor.from_openai(OpenAI())`) as the interface between our code and the OpenAI API. Instructor wraps the client and provides three capabilities we'd otherwise build by hand:

1. **Schema injection** — calls `model_json_schema()` on the Pydantic model and appends it to the system prompt automatically.
2. **Response parsing** — calls `model_validate_json()` on the LLM response, returning a typed Pydantic object.
3. **Auto-retry with error feedback** — catches `ValidationError`, formats it as a follow-up message, and retries up to `max_retries=3`. The LLM sees its own mistake and self-corrects.

```python
record = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=DIYRepairRecord,
    messages=messages,
    max_retries=3,
)
```

The same pattern powers the evaluator (`response_model=JudgeResult`).

## Alternatives Considered

| Option | Pros | Cons | Why Not |
|--------|------|------|---------|
| **Instructor** ✅ | One-line call, auto-retry with error feedback, Pydantic-native | Extra dependency, hides retry internals | — (selected) |
| Raw OpenAI + `json_object` | No dependency, full control | ~60 lines boilerplate per call site: schema injection, JSON parsing, validation, retry loop | Build-vs-buy — not worth it across 9 projects |
| LangChain `StructuredOutputParser` | Part of LangChain ecosystem | Heavy dependency for P1, weaker Pydantic integration | Premature — LangChain introduced in P5 |
| OpenAI function calling | Built into API | Manual validation, no auto-retry, API being superseded | Missing the self-healing loop |

## Quantified Validation

- **Generation success**: **30/30 records** (100%), zero manual retries needed
- **LOC saved**: ~60 lines of boilerplate eliminated per call site (measured by counting the manual implementation: `model_json_schema()` serialization into the system prompt, `json.loads()` + `try/except JSONDecodeError`, `model_validate()` + `try/except ValidationError`, retry loop with error formatting into natural language, conversation history management across retries, and retry exhaustion handling). With 2 call sites (generator + evaluator), **~120 lines** of error-prone code avoided
- **Retry handling**: Instructor's auto-retry feeds `ValidationError` details back to the LLM. A manual retry loop requires formatting Pydantic errors into natural language, managing conversation history, and handling retry exhaustion — a non-trivial state machine
- **Reuse**: Same `response_model=` pattern used in `generator.py` and `evaluator.py` with zero code duplication

## Consequences

**Easier:** Adding new Pydantic models works automatically — just pass `response_model=NewModel`. Validation errors become self-healing, which is why we hit **100% generation success**.
**Harder:** Debugging failed retries requires Instructor logging hooks — intermediate LLM responses aren't visible by default. We accept a dependency on Instructor's API stability (mitigated: actively maintained, widely adopted).
**Portability:** Pattern reused in P2 (QA pair generation), P4 (resume/job generation — 550 records at 100% success), and planned for P5–P9.

## Cross-References

- **ADR-002**: Flat schema works because Instructor handles schema injection regardless of complexity — simplicity pays dividends at both schema and library level.
- **ADR-003**: Same Instructor pattern powers the GPT-4o judge (`response_model=JudgeResult`).
- **ADR-004**: Correction loop uses Instructor for re-generation, maintaining the self-healing retry pattern throughout the pipeline.

## Java/TS Parallel

Instructor is analogous to **Jackson + Bean Validation** with automatic retry — like `@Valid` on a `@RequestBody` that, instead of returning a 400, re-sends the request with validation errors appended so the client self-corrects. `response_model` is the `Class<T>` passed to `objectMapper.readValue()`, and `max_retries` maps to `@Retryable(maxAttempts=3)`.

**The key insight:** The highest-leverage library decisions eliminate entire categories of bugs, not just lines of code. Instructor didn't just save 120 LOC — it eliminated the retry state machine as a failure surface, just as Spring Boot's `@Valid` eliminates manual deserialization bugs rather than merely shortening the code.

## Interview Signal

Demonstrates **build-vs-buy reasoning** and library evaluation methodology. Rather than building a custom retry loop (the "not invented here" trap), the engineer evaluated four options, chose a thin wrapper that provides maximum leverage, and validated the decision with quantified success metrics. This signals mature judgment about where to invest engineering effort vs. adopting existing solutions — a key skill for engineering managers building production ML systems.
