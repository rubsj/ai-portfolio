# ADR-004: FastAPI over Flask for the Resume Coach REST API

**Date**: 2026-02-26
**Project**: P4 — Resume Coach

## Status

Accepted

## Context

P4 needs a REST API exposing 9 endpoints across 4 domains: resume review (live labeling + optional GPT-4o judge), pipeline analysis (failure rates, A/B template comparison), vector search (ChromaDB semantic similarity), and feedback (human ratings). The API serves two audiences: it's a functional backend for the Streamlit demo, and it's a portfolio artifact that demonstrates production-grade engineering to recruiters via interactive Swagger documentation at `/docs`.

The hard constraint: P4 already has 14 Pydantic request/response schemas defined in `schemas.py` — `ReviewRequest`, `ReviewResponse`, `FailureRateResponse`, `TemplateComparisonResponse`, `MultiHopRequest`, `MultiHopResponse`, `SimilarCandidatesResponse`, `FeedbackRequest`, `FeedbackResponse`, `JobListResponse`, `PairDetailResponse`, plus supporting models like `TemplateStats`, `MultiHopQuestion`, `SimilarCandidate`, `JobSummary`. These schemas define the API contract. The framework must consume them directly — writing adapter DTOs or serialization boilerplate for 14 schemas would double the code without adding value.

Secondary constraint: P4's data lives in JSONL files on disk, loaded into memory at startup by `DataStore`. There is no database, no ORM, no migrations. The framework must work with a stateless, file-backed architecture.

## Decision

Use **FastAPI** for the Resume Coach REST API.

The two strongest reasons:

1. **14 Pydantic schemas plug in with zero adapter code**: `ReviewRequest` is both the Pydantic validation model AND the FastAPI request body parameter. `response_model=ReviewResponse` on the decorator is both the serialization format AND the OpenAPI schema. In Flask, each of these 14 schemas would need a manual `request.get_json()` → `model_validate()` round-trip on input and a `model_dump()` call on output — that's 28 lines of boilerplate across 9 endpoints. FastAPI eliminates all of it.

2. **Auto-generated Swagger UI at `/docs` with zero configuration**: Every endpoint, every query parameter, every request/response schema is rendered in interactive documentation the moment the server starts. Recruiters can send real requests to `/review-resume` or `/search/similar-candidates` from the browser. With Flask, achieving the same requires installing `flask-restx` or `flask-smorest`, writing `@api.doc()` decorators, and maintaining schema descriptions separately from the Pydantic models.

Supporting factor: `Annotated[int, Query(ge=1, le=50)]` on query parameters (e.g., `page_size` on `/jobs`) expresses validation constraints inline with the type. FastAPI auto-generates the constraint documentation in Swagger and returns structured 422 errors on violation. Flask equivalent: `page_size = request.args.get('page_size', 20); page_size = int(page_size); if page_size < 1 or page_size > 50: abort(422)` — three lines of manual validation per parameter.

## Alternatives Considered

**Flask**: Simpler, lighter, and the most familiar Python web framework. But Flask's design assumes you'll build request parsing yourself. Without native Pydantic support, every endpoint needs: (1) `request.get_json()`, (2) `try/except` around `model_validate()`, (3) manual `model_dump()` for the response, (4) manual error formatting for validation failures. For 9 endpoints with 14 schemas, this adds ~120 lines of boilerplate that FastAPI handles automatically. Flask also lacks built-in OpenAPI generation — `flask-restx` adds it but requires duplicate schema definitions in marshmallow format, defeating the purpose of having Pydantic models in `schemas.py`.

**Django REST Framework**: The most full-featured Python API framework, with authentication, ORM integration, admin panels, and browsable API. But P4 has no database — data comes from JSONL files loaded into `DataStore` at startup. Django's ORM, migration system, and model serializers would be unused dead weight. The `django.contrib.*` dependencies alone are larger than P4's entire codebase. DRF is the right choice for a CRUD app with PostgreSQL; it's the wrong choice for a stateless analytics API backed by flat files.

**Litestar**: The fastest Python web framework by benchmark, with auto-generated OpenAPI and native support for both dataclasses and Pydantic. A legitimate contender on technical merits. Rejected because Litestar's design favors a dataclass-first approach — its dependency injection, DTOs, and middleware are built around `@dataclass` and `msgspec.Struct` types, with Pydantic support added as a plugin. P4 already has 14 Pydantic v2 schemas in `schemas.py` that serve triple duty: LLM structured output (Instructor), API contracts (request/response models), and data validation (pipeline stages). Adapting these to Litestar's DTO layer would require either wrapping each Pydantic model in a Litestar-specific DTO class or configuring the Pydantic plugin for each endpoint — friction that doesn't exist in FastAPI where Pydantic is the native type system, not an adapter. Litestar's performance advantage (microseconds per request) is also irrelevant when endpoints call GPT-4o at 1s+ latency.

## Consequences

### What This Enabled

The 346-line `api.py` serves 9 endpoints using all 14 Pydantic schemas from `schemas.py` with zero adapter code — every schema is both the FastAPI parameter type and the OpenAPI documentation source. Swagger UI at `/docs` renders all endpoints interactively the moment the server starts, letting recruiters send real requests to `/review-resume` or `/search/similar-candidates` from the browser with no additional setup. Because FastAPI's TestClient wraps httpx, all 26 endpoint tests run without starting a server process — same pattern as Spring's MockMvc but with zero configuration — contributing to the 504-test suite that runs in ~2 seconds. Adding the `/feedback` endpoint (POST with `FeedbackRequest` body, returns `FeedbackResponse`) took 15 lines of code including file I/O, because the framework handled parsing, validation, serialization, and documentation automatically. Structured 422 errors on invalid input — wrong types, constraint violations like `page_size > 50` — come free with no custom error handling code.

### Accepted Trade-offs

- FastAPI's dependency injection system (`Depends()`) has a learning curve for shared state patterns beyond module-level singletons. P4 uses three module-level singletons (`_store: DataStore`, `_normalizer: SkillNormalizer`, `_collection: chromadb.Collection`), which is simple but won't scale to multi-tenant or request-scoped state. Acceptable for P4's single-user portfolio context
- `TestClient` requires `httpx` as a dev dependency — already included in `pyproject.toml`, but it's one more package in the dev group
- FastAPI's startup is heavier than Flask (~200ms vs ~50ms) due to OpenAPI schema generation. Irrelevant for a long-lived server process, but noticeable in test suites that create the app per test class

## Cross-Project Context

P1 and P2 were batch pipelines with no API layer — P2 used Click CLI for its evaluation interface. P4 is the first project that serves results over HTTP, making framework choice consequential. The decision to use Pydantic v2 for all schemas (made in P1) paid dividends here: the same `Resume`, `FailureLabels`, and `JudgeResult` models created for batch processing became API contracts with zero refactoring.

## Java/TS Parallel

FastAPI with Pydantic is what Spring Boot would be if Jackson, Bean Validation, and springdoc-openapi were a single annotation. In Spring you need `@RestController` for routing, `@RequestBody @Valid` for deserialization + validation, separate DTO classes for Jackson, and springdoc config for Swagger — four concerns wired together with boilerplate. FastAPI collapses all four into `response_model=MyPydanticClass` on the route decorator. The `Annotated[int, Query(ge=1, le=50)]` pattern maps to `@RequestParam @Min(1) @Max(50)`, but FastAPI auto-generates the constraint documentation in Swagger where Spring requires explicit `@Parameter` annotations.

## Validation

9 endpoints shipped, all documented in Swagger, all tested. The `/review-resume` endpoint demonstrates live labeling: POST a resume + job description, get back `FailureLabels` with Jaccard score, failure flags, and optional GPT-4o judge assessment — all powered by the same Pydantic schemas used in the batch pipeline. The `/analysis/template-comparison` endpoint serves the A/B test results (χ²=32.74, p=1.35e-06) directly from `pipeline_summary.json` through the `TemplateComparisonResponse` schema. Zero schema duplication between the pipeline and the API.

## Reversibility

**Medium** — Switching to Flask would require adding ~120 lines of request parsing and response serialization boilerplate across 9 endpoints, plus installing and configuring `flask-restx` for OpenAPI support. The Pydantic schemas stay the same; only the glue code between HTTP and Pydantic changes. The 26 API tests would need rewriting from `TestClient` to Flask's `test_client`.
