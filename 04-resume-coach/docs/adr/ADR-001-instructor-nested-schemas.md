# ADR-001: Instructor with max_retries=5 for Nested Schema Generation

**Date**: 2026-02-26
**Project**: P4 — Resume Coach

## Status

Accepted

## Context

P4 generates 250 resumes as deeply nested Pydantic models. A single `Resume` has 4 levels of nesting — `ContactInfo`, `list[Education]` (each with optional `gpa` constrained to 0.0–4.0 and ISO-format `graduation_date`), `list[Experience]` (each with `responsibilities: list[str]` requiring min 1 item and dates validated in sequence), and `list[Skill]` (each with a `ProficiencyLevel` enum and `years` constrained to 0–30). Across the full schema there are roughly 30 individual validation points where raw GPT-4o-mini output can fail Pydantic.

Raw OpenAI completions fail validation 15–30% of the time on schemas this complex. Common breakages: `null` instead of `[]` for lists, `"advanced"` instead of the `ProficiencyLevel.ADVANCED` enum literal, `"2020"` instead of ISO `"2020-01"` for dates, missing required fields in nested objects. A naive retry-from-scratch strategy wastes API calls because the model gets no feedback on *what* went wrong — it just tries again blind.

P1 had ~7 validation points per record and could get away with simpler prompting. P4's 4× increase in schema complexity made an intelligent retry mechanism a hard requirement.

## Decision

Use **Instructor** (`instructor.from_openai(client, mode=instructor.Mode.JSON)`) with `max_retries=5` for all structured LLM generation.

The two strongest reasons:

1. **Automatic error injection**: When Pydantic validation fails, Instructor extracts the `ValidationError` message and injects it back into the conversation as a correction prompt. The model sees exactly which field failed and why — `"1 validation error for Resume: skills.0.years — Value error, years must be 0–30"` — not just "try again." This turns a blind retry into a targeted fix.

2. **Zero retry logic in application code**: The entire retry loop is `max_retries=5` on the `create()` call. No `try/except`, no backoff, no prompt rewriting. The generator functions (`generate_job`, `generate_resume`) are 40-line functions, not 100-line retry state machines.

Supporting design choice: `Mode.JSON` forces the API to return valid JSON at the protocol level, eliminating the class of failures where the model prefixes output with prose like `"Here is the JSON:"`.

Schema-side mitigation: fields that LLMs rarely produce correctly (`linkedin`, `portfolio`, `coursework`) are marked `Optional[str] = None` so their absence doesn't trigger retries. `responsibilities: list[str]` has no `max_length` — constraining list length causes more retries than it prevents bad data.

## Alternatives Considered

**Raw OpenAI + manual `json.loads()`**: The caller receives a raw string from `chat.completions.create()`, calls `json.loads()` to parse it, then `Resume.model_validate(data)` to validate. When validation fails — and it fails 15–30% of the time on 30-field nested schemas — the caller must catch `JSONDecodeError` and `ValidationError` separately, extract the error message, format it into a correction prompt, append it to the conversation history, and re-call the API. Each retry needs conversation management: the original system prompt, the failed response, and the correction must all be tracked. In P1 this was viable because `RepairRecord` had 7 fields and ~2 validation points — the retry logic was a 10-line `for` loop. With P4's 35 Pydantic models, 30 validation points per resume, and 4 nesting levels, the manual retry logic would be the single largest source of complexity in the generator — easily 100+ lines of error handling per generation function, repeated across `generator.py`, `judge.py`, and `corrector.py`.

**LangChain structured output (`PydanticOutputParser`)**: LangChain's `PydanticOutputParser` validates LLM output against a Pydantic model and can be paired with `OutputFixingParser` or `RetryWithErrorOutputParser` for automatic retries. The critical difference from Instructor is *what gets sent back on failure*. `OutputFixingParser` re-sends the entire malformed output with a generic instruction: "Fix the following output to match the schema." It does not include the specific `ValidationError` — the model sees the broken JSON but not *which field* broke or *why*. For a schema with 30 validation points, this means the model often fixes the obvious structural issue (e.g., missing `contact_info`) while introducing a new one (e.g., `skills[0].years = -1`). Instructor's approach is fundamentally different: it extracts the exact `ValidationError` message — `"skills.0.years: Value error, must be 0–30, got -1"` — and injects it as a user message in the conversation, giving the model a field-level correction target. This is the difference between "your JSON is wrong, try again" and "field X at path Y failed constraint Z with value W."

**Pydantic `model_validate_json()` with hand-rolled retry**: Pure stdlib approach — call `model_validate_json(response.content)` in a `try/except ValidationError` loop, no third-party dependency beyond Pydantic itself. The problem is that the *entire value* of Instructor is the correction prompt construction: parsing `ValidationError.errors()` into a list of `{"loc": ("skills", 0, "years"), "msg": "must be 0–30", "input": -1}` dicts, formatting each into natural language the model can act on, appending it to the OpenAI conversation as a user message, and tracking the multi-turn history across retries. Building this manually means writing a `format_validation_errors()` function, a conversation history manager, and a retry controller with configurable max attempts — essentially reimplementing Instructor's core loop. The result would be 80–100 lines of infrastructure code that's identical across `generator.py`, `judge.py`, and `corrector.py`, versus Instructor's single `max_retries=5` parameter.

## Consequences

### What This Enabled

The 100% validation rate across 250 resumes — each with 30+ validation points across 35 Pydantic model types — required zero manual intervention. Roughly 15% of resumes needed 1–2 retries to pass, but none exhausted all 5 attempts, because Instructor's error injection gives the model a targeted correction path rather than a blind retry. This kept generator functions under 50 lines each: no try/except blocks, no backoff logic, no prompt rewriting — just `max_retries=5` on the `create()` call. The same Instructor pattern carried across three pipeline stages — `generator.py` for resume/job creation, `judge.py` for GPT-4o structured evaluation, and `corrector.py` for the correction loop — proving the approach scales to any Pydantic schema, not just the generation use case.

### Accepted Trade-offs

- Worst-case 5× API cost per record on persistent validation failures. In practice this added ~$0.02 across the full 250-resume run — negligible against the ~$0.50 total generation cost
- Instructor pins to a specific OpenAI client version. Major `openai` SDK bumps (v0 → v1 was painful for the community) require waiting for an Instructor update. Mitigated by pinning both in `pyproject.toml`
- Instructor's `Mode.JSON` doesn't support streaming — acceptable since P4 generation is batch, not interactive

## Cross-Project Context

P1 (Synthetic Data) used raw `model_validate_json()` with manual retry for its simpler 7-field `RepairRecord` schema — that worked at P1's scale but revealed that retry prompt engineering, not parsing, was the real bottleneck. When P4's schema grew to 35 models with 30+ validation points, the manual approach would have dominated development time. Instructor was the direct response to P1's lesson: let the library handle retry intelligence so the developer handles domain logic.

## Java/TS Parallel

Instructor is the Python equivalent of combining Jackson's `@JsonCreator` with Spring Retry's `@Retryable` — automatic deserialization with intelligent retry on validation failure. The key difference: Instructor feeds the specific `ValidationError` back to the LLM as correction context, which has no direct Java equivalent since Jackson deserialization errors aren't self-correcting. In Java you'd need a custom `RetryCallback` that reads `ConstraintViolationException.getConstraintViolations()`, formats them into natural language, and feeds them back into the request — Instructor gives you that entire flow for free with one parameter.

## Validation

The decision delivered exactly what was needed. 100% validation rate across 250 resumes, 50 jobs, and 250 pairs. The generator ran end-to-end without human intervention. The same Instructor pattern powered the GPT-4o judge (250 evaluations, avg quality score 0.541) and the corrector (8/8 records fixed in 1.0 attempts avg). Total pipeline cost: ~$0.50 — the retry overhead was undetectable in the budget.

## Reversibility

**Medium** — Removing Instructor would require adding ~100 lines of retry + error-injection logic per generation function (generator, judge, corrector). The schema definitions stay the same, but the glue code between OpenAI and Pydantic would need to be rebuilt.
