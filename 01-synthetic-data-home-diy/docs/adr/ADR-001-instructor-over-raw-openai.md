# ADR-001: Why Instructor over Raw OpenAI API

**Date**: 2026-02-09
**Status**: Accepted
**Project**: P1 — Synthetic Data, Home DIY Repair

## Context

We need to generate structured JSON records from GPT-4o-mini that conform to a
7-field Pydantic schema (`DIYRepairRecord`). The raw OpenAI API offers
`response_format={"type": "json_object"}` for JSON mode, but it returns raw
strings that we'd need to parse and validate ourselves. When validation fails
(missing fields, wrong types, constraint violations), we'd need to build our own
retry loop that feeds the error back to the LLM.

This is the first project in a 9-project portfolio. The pattern chosen here
(LLM → structured output → validation) will be reused across all subsequent
projects.

## Decision

Use the **Instructor** library (`instructor.from_openai(OpenAI())`) as the
interface between our code and the OpenAI API.

Instructor wraps the OpenAI client and provides three things we'd otherwise
build by hand:

1. **Schema injection** — calls `model_json_schema()` on our Pydantic model and
   appends it to the system prompt automatically. We never manually serialize
   the schema or write "return JSON matching this format" instructions.

2. **Response parsing** — calls `model_validate_json()` on the LLM's raw
   response, turning it into a typed Pydantic object. No `json.loads()` +
   manual construction in our code.

3. **Auto-retry with error feedback** — when Pydantic validation fails (e.g.,
   `question` doesn't end with `?`, `tools_required` is empty), Instructor
   catches the `ValidationError`, formats it into a follow-up message, and
   retries the LLM call (up to `max_retries=3`). The LLM sees its own mistake
   and self-corrects.

The call site in `generator.py` is a single expression:

```python
record = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=DIYRepairRecord,
    messages=messages,
    temperature=0.7,
    max_retries=3,
)
```

The same pattern is used in `evaluator.py` for the judge (`response_model=JudgeResult`).

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **Instructor** (chosen) | One-line call, auto-retry with error feedback, Pydantic-native, reusable across projects | Extra dependency, hides retry logic (harder to log intermediate attempts) |
| **Raw OpenAI + json_object mode** | No dependency, full control over retry logic | Must manually: inject schema into prompt, parse JSON, validate, build retry loop, format errors. ~50-80 lines of boilerplate per call site |
| **LangChain StructuredOutputParser** | Part of LangChain (used later in P5) | Heavy dependency for P1's needs, less direct Pydantic integration, would require LangChain in a project that doesn't need it yet |
| **OpenAI function calling** | Built into the API, no extra library | Pydantic validation still manual, no auto-retry, function_call API is being superseded by structured outputs |

## Consequences

**Easier:**
- Adding new Pydantic models automatically works with the LLM — just pass
  `response_model=NewModel` and Instructor handles the rest.
- Validation errors become self-healing: the LLM gets its own errors and fixes
  them, which is why we hit 100% generation success rate (30/30 records).
- Same pattern ports directly to the evaluator (GPT-4o judge) and will port to
  P2–P9.

**Harder:**
- Debugging failed retries requires checking Instructor internals or adding
  logging hooks — we can't see intermediate LLM responses by default.
- We take a dependency on Instructor's API staying compatible with OpenAI's SDK
  updates (mitigated: Instructor is actively maintained and widely adopted).

## Java/TS Parallel

Instructor is analogous to using **Jackson + Bean Validation annotations** in
Java, but with automatic retry — like a request validator middleware that
re-prompts the client on 400 errors. In Spring terms: imagine `@Valid` on a
`@RequestBody` that, instead of returning a 400 to the caller, automatically
re-sends the request with the validation errors appended so the client can
self-correct. The `response_model` parameter is the `Class<T>` you'd pass to
`objectMapper.readValue()`, and `max_retries` is `@Retryable(maxAttempts=3)`.
