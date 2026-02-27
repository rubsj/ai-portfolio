# ADR-004: FastAPI over Flask for the Resume Coach REST API

**Date**: 2026-02-26
**Status**: Accepted
**Project**: P4 — Resume Coach

## Context

P4 requires a REST API exposing 9 endpoints across 4 domains: resume review, pipeline analysis, vector search, and feedback. The API serves as both a portfolio artifact (demonstrates production engineering) and a functional backend for the Streamlit demo.

Key requirements:
- Request/response schemas are already defined as Pydantic v2 models in `schemas.py`
- Auto-generated API documentation for recruiter demos (`/docs` Swagger UI)
- Type-safe request parsing and response serialization — no manual `request.json.get("field")` boilerplate
- Async support for future extensions (streaming LLM responses in P5)

The two realistic Python web framework options are Flask and FastAPI.

## Decision

Use **FastAPI** for the Resume Coach REST API.

Key factors:
1. **Native Pydantic integration**: FastAPI validates incoming request bodies against Pydantic models automatically. `ReviewRequest`, `FeedbackRequest`, `MultiHopRequest` are used directly as endpoint parameters — zero boilerplate parsing code.
2. **Auto OpenAPI at `/docs`**: FastAPI generates Swagger UI and ReDoc automatically from endpoint type annotations. Recruiters can explore all 9 endpoints interactively without additional tooling.
3. **Response model enforcement**: `response_model=ReviewResponse` on the decorator ensures the return value is validated and serialized — no accidental field leakage or missing fields in responses.
4. **`Annotated` query parameters**: `Annotated[int, Query(ge=1, le=50)]` expresses validation constraints inline with the parameter type. Flask requires manual `int(request.args.get('page_size', 20))` + validation checks.
5. **Async-ready**: FastAPI supports both sync and async handlers; the endpoints are sync today but migration to async (for streaming P5 features) requires only adding `async def`.

## Alternatives Considered

| Option | Pros | Cons |
|--------|------|------|
| **FastAPI** (chosen) | Pydantic-native; auto OpenAPI; typed query params; async-ready | Slightly steeper learning curve than Flask; heavier startup |
| Flask | Simpler, lighter; huge ecosystem; familiar to Python community | No native Pydantic integration; manual request parsing; no auto OpenAPI without flask-restx; no async |
| Django REST Framework | Full-featured; auth, ORM, admin built-in | Far too heavy for a stateless analytics API; ORM not needed (data from JSONL files) |
| Litestar | Fastest Python web framework | Less mature ecosystem; fewer tutorials; unnecessary optimization for this scale |

## Consequences

**Easier**:
- Adding a new endpoint = add a function with type-annotated parameters — Swagger UI auto-updates
- Client SDK generation is possible from the OpenAPI spec (useful for P5 frontend)
- Request validation errors return structured 422 responses automatically — no custom error handling needed
- All 14 Pydantic schemas reused directly as API contracts with zero additional code

**Harder**:
- FastAPI's dependency injection system (`Depends()`) adds a learning curve if endpoints need shared state beyond module-level singletons (not an issue for P4's pattern)
- `TestClient` requires `httpx` as a dev dependency (already included in `pyproject.toml`)

## Java/TS Parallel

FastAPI is the Python equivalent of **Spring Boot with Jackson + Bean Validation**:
- `@RestController` → `@app.get()`/`@app.post()`
- Jackson JSON deserialization + `@Valid DTO` → FastAPI Pydantic model parameter
- Swagger via `springdoc-openapi` → FastAPI's built-in `/docs` (zero config)
- `@RequestParam` with constraints → `Annotated[int, Query(ge=1)]`
- Spring `ResponseEntity<T>` → `response_model=T` on the decorator

The philosophical alignment is strong: both frameworks prefer declarative type-driven APIs over imperative request parsing.
