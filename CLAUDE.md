# CLAUDE.md — AI Portfolio Monorepo

> This file is Claude Code's persistent memory. Read it at the start of every session.

## Who I'm Working With

- **Background**: Senior Java/TypeScript developer (10+ years), learning Python for AI
- **Goal**: Building 9 AI projects in 8 weeks for Tier-1 Engineering Manager roles
- **Timeline**: Feb 6 – Apr 2, 2026
- **Working hours**: Weeknights 9PM–1AM (4h), Sundays 6-8h deep work
- **Hardware**: MacBook Air M2 (M4 Max arriving ~Mar 13)

## Communication Style

- Explain Python patterns when they differ from Java/TS — map concepts to Java/TS equivalents
- Comment code with "WHY" not "what" — every non-obvious line gets a comment explaining the reasoning
- Treat as expert in software engineering, novice in Python ecosystem and AI/ML concepts
- Be direct. No filler. If something is wrong, say so.
- When introducing a library or function, explain: what it does, why this one over alternatives, key parameters

## Tech Stack

- **Python**: 3.12+ with type hints everywhere
- **Package Manager**: uv (NOT pip, NOT poetry)
- **Validation**: Pydantic v2 (model_validate_json, Field, field_validator with @classmethod)
- **LLMs**: OpenAI API (GPT-4o-mini for generation, GPT-4o for evaluation)
- **Embeddings**: Sentence-Transformers (local) + OpenAI text-embedding-3-small (comparison)
- **Vector Store**: ChromaDB
- **RAG Framework**: LangChain (use specific components, don't over-abstract)
- **Multi-Agent**: CrewAI (P7, P8, P9)
- **Evaluation**: RAGAS, Braintrust
- **Visualization**: matplotlib, seaborn
- **CLI**: Click
- **API**: FastAPI
- **Demo**: Streamlit
- **Testing**: pytest
- **Linting**: ruff

## Claude Model Strategy

- **Planning mode**: Use Opus 4.6 (maximum capability for architecture decisions)
  - Specify `model: "opus"` when launching Plan agents
- **Implementation**: Use Sonnet 4.5 (best balance of quality and cost for coding)
  - Most code tasks run with Sonnet by default
- **Fast mode**: Use `/fast` to switch to Opus with faster output when needed
- When entering plan mode via EnterPlanMode, specify `model: "opus"` in the call

## Monorepo Structure

```
ai-portfolio/
├── CLAUDE.md                     ← THIS FILE (root-level Claude Code memory)
├── STATUS.md                     ← Auto-generated project status (YAML frontmatter)
├── README.md                     ← Portfolio overview + links to demos
├── docs/
│   ├── adr/                      ← Cross-project ADRs
│   ├── learnings/                ← Monthly learning notes (YYYY-MM.md)
│   └── progress/                 ← Per-project progress snapshots
├── shared/                       ← Shared utilities across projects
│   ├── llm_client.py
│   ├── cache.py
│   └── eval_helpers.py
├── 01-synthetic-data-home-diy/   ← P1 (has its own CLAUDE.md)
│   ├── CLAUDE.md                 ← Project-specific Claude Code memory
│   ├── src/
│   ├── tests/
│   ├── data/
│   ├── results/
│   ├── docs/adr/
│   └── pyproject.toml
├── 02-rag-evaluation/            ← P2
│   └── ...
└── ...through 09-devops-assistant/
```

## Coding Standards

### Python Style
- Use type hints on ALL function signatures and return types
- Use `from __future__ import annotations` for forward references
- Prefer `list[str]` over `List[str]` (Python 3.12 native generics)
- Use `str | None` over `Optional[str]`
- Use `@classmethod` with `@field_validator` (Pydantic v2 pattern)
- Use `model_validate_json()` not `parse_raw()` (deprecated in v2)
- Use `model_json_schema()` to generate schemas for LLM prompts
- Prefer dataclasses for simple data containers, Pydantic when validation needed
- Use pathlib.Path over os.path

### Naming
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private: `_leading_underscore`

### Error Handling
- Use specific exception types, never bare `except:`
- Log errors with context (what operation, what input caused it)
- For LLM calls: always wrap in try/except, implement retry with backoff

### Testing
- pytest, not unittest
- Test file mirrors source: `src/schemas.py` → `tests/test_schemas.py`
- Test names: `test_<what>_<when>_<expected>` (e.g., `test_tool_name_when_empty_raises_validation_error`)
- Include both happy path and failure cases
- Use parametrize for multiple inputs

## Documentation Protocol

### ADRs (Architecture Decision Records)
Write an ADR for every significant decision. Store in project's `docs/adr/` folder.

Template:
```markdown
# ADR-NNNN: [Decision Title]

**Date**: YYYY-MM-DD
**Status**: Accepted | Superseded | Deprecated
**Project**: P[N] — [Project Name]

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing and/or doing?

## Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| ... | ... | ... |

## Consequences
What becomes easier or harder because of this change?

## Java/TS Parallel
How does this map to a pattern the developer already knows?
```

### Learning Notes
After every session, append to `docs/learnings/YYYY-MM.md`:
```markdown
### [Date] — [Topic]
- **What**: What was learned
- **Why it matters**: How it connects to the project or AI concepts
- **Python pattern**: Any Python-specific idiom discovered
- **Java/TS equivalent**: What this replaces from the Java/TS world
```

### STATUS.md
Auto-generate this file after significant milestones. Use YAML frontmatter for machine parsing:
```yaml
---
last_updated: "2026-02-07"
current_project: P1
overall_progress: "1/9 complete"
---
```

## LLM Cost Management

- Cache ALL LLM responses during development
- Cache key: `hashlib.md5(prompt.encode()).hexdigest()`
- Cache location: `data/cache/` as JSON files
- Before any LLM call, check cache first
- Log estimated cost per call to console
- GPT-4o-mini: ~$0.15/1M input, ~$0.60/1M output
- GPT-4o: ~$2.50/1M input, ~$10.00/1M output

## Git Conventions

- **NEVER commit directly to `main`**. Always work on a feature branch and merge via PR.
- Branch naming: `type/scope-short-description` (e.g., `feat/p1-pydantic-schemas`, `fix/p1-validator-edge-case`)
- Commit messages: `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
  - Scope: project folder name (e.g., `p1`, `p2`, `shared`)
  - Example: `feat(p1): add Pydantic schemas for DIY repair data`
- Commit after each logical unit of work (not at end of night)
- Workflow: create branch → commit → push → create PR → merge to main

## Session Workflow

1. At session start: read this CLAUDE.md + project-specific CLAUDE.md
2. Check STATUS.md for current state
3. Build / test / evaluate
4. Before session end:
   - Git commit + push
   - Update STATUS.md if milestone reached
   - Append to learning notes if something non-obvious was learned
   - Generate a handoff summary (compact block for next chat session)

## Current Sprint Context

- **Active Project**: P2 — RAG Evaluation Benchmarking Framework
- **Week**: 2 of 8
- **Phase**: Day 5 deliverables complete (CLI, Streamlit, README, charts)
- **Status**: All core tasks done, ready for PR merge. Polish tasks deferred to Week 8.

## Project Schedule Reference

| Project | Week | Dates | Status |
|---------|------|-------|--------|
| P1: Synthetic Data | W1 | Feb 8–11 | Complete |
| P2: RAG Evaluation | W1–2 | Feb 12–16 | Ready for Merge |
| P3: Fine-Tuning & Guardrails | W2 | Feb 17–19 | Backlog |
| P4: Resume Coach | W3 | Feb 20–25 | Backlog |
| P5: Production RAG | W4 | Feb 27–Mar 2 | Backlog |
| P6: Digital Clone | W4–5 | Mar 3–8 | Backlog |
| P7: Feedback Intelligence | W5 | Mar 8–12 | Backlog |
| P8: Jira AI Agent | W6 | Mar 13–18 | Backlog |
| P9: DevOps Assistant | W7 | Mar 20–26 | Backlog |
| Portfolio Polish | W8 | Mar 27–Apr 2 | Backlog |
