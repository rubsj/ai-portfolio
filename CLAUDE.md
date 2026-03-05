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

## Writing Rules (applies to all docs — ADRs, journal entries, comments)

- Write as a practitioner documenting real decisions, not a consultant producing a deliverable
- First person is allowed and preferred where natural ("I picked X because", "this burned us")
- Never narrate the document's own importance — if it mattered, just state what happened
- No section whose only purpose is to make the author look good (e.g., "Interview Signal")
- Analogies go inline as parentheticals — never in their own dedicated section
- Bold emotional category labels ("Easier:", "Harder:") are banned — write plain prose or plain bullets
- Numbers and benchmarks stay where they're contextually relevant — never aggregate into a "Validation" section
- Section headers are plain nouns — not action phrases, not corporate labels
- If a sentence could have been written without knowing anything specific about this project, delete it
- Code comments explain WHY, never what — if the code is readable, no comment needed
- No hedging openers in comments: ban "Note that", "This ensures", "It's worth mentioning"
- Docstrings: one sentence what + one sentence non-obvious how/why — no parameter narration
- Inline comments for short context, block comments only for genuinely non-obvious decisions
- Comment like you're explaining to a teammate at 11pm — direct, no filler

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

## Monorepo Structure

```
ai-portfolio/
├── CLAUDE.md                     ← THIS FILE (root-level Claude Code memory)
├── README.md                     ← Portfolio overview + links to demos
├── docs/
│   ├── adr/                      ← Cross-project ADRs
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

Write an ADR for every significant architectural, tool, or algorithmic decision.
**Destination**: Create as a new page in the Notion ADR Log database (via Claude Chat + MCP), AND save a markdown copy in the project's `docs/adr/` folder.

**Notion properties to set on each ADR entry:**
- `Decision`: `ADR-NNN: [Title]`
- `Project`: e.g., `P1: Synthetic Data`
- `Category`: one of `Data Model | Architecture | Tool Choice | Algorithm | Deployment | Evaluation`
- `Status`: `Accepted | Superseded | Proposed`
- `Date`: YYYY-MM-DD

**Page body template** (what goes inside the ADR page):

```markdown
## Context
What situation forced this decision? Be specific about the constraints — scale, failure modes,
prior project lessons. Reference real incidents (e.g., "this burned us in P4 because...").
Write in first person where it's natural. Skip background the team already knows.

## Decision
One sentence stating what was chosen. Then explain how it actually works — the mechanism,
not the marketing. Include the exact code/config used, because that's what future-me needs.

```python
# concrete usage pattern here — not pseudocode
```

## Alternatives Considered

| Option | Trade-off | Why rejected |
|--------|-----------|--------------|
| **Chosen** ✅ | what you give up | — selected |
| Alternative A | what it offers vs. costs | specific reason, ideally from experience |
| Alternative B | what it offers vs. costs | specific reason, ideally from experience |

## What the numbers said
Any benchmarks, error rates, latency measurements, or cost figures that informed the call.
State them inline as facts, not as a proof section. If you didn't measure it, don't invent precision.

## What this changes
Plain bullets on what gets easier, what gets harder, and which future projects (P2–P9) can reuse this.
No bold category labels — just write it out.

## Cross-References
- Links to other ADRs this depends on or affects
```

---

### Learning Journal Entries

After every session, create a new page in the Notion Learning Journal database (via Claude Chat + MCP).

**Notion properties to set on each entry:**
- `Entry`: `P[N] Day [X] — [Short descriptive title of what was built/learned]`
- `Project`: e.g., `P4: Resume Coach`
- `Phase`: one of `Foundation | Build | Evaluate | Document | Polish | Implementation | Testing`
- `Session Type`: one of `Weeknight | Sunday Deep Work | Saturday`
- `Hours`: numeric
- `Date`: YYYY-MM-DD (or range)
- `Python Pattern Learned`: 2–5 bullet summary of Python patterns discovered (inline text field, not the page body)
- `Blocked By`: brief description of blockers, or blank

**Page body template** (what goes inside the journal entry page):

```markdown
## What I shipped
For each component built this session: what it does, what file it lives in, and why I designed
it the way I did. Focus on the non-obvious calls — anyone can read the code for the obvious parts.
Include specific implementation details that would trip me up if I came back cold in two weeks.

## Numbers
Whatever I actually measured: pass rates, error counts, latency, API cost, test coverage.
No invented precision — if I didn't measure it, I don't include it.
Before/after if something changed. One line per metric is fine.

## What I actually learned
### [Name the concept]
Write this as if explaining to past-me from two months ago. Full paragraphs, not bullets.
Cover: what it is, why it matters beyond this specific project, and where it maps to
something I already know from Java/TS. One subsection per real insight — not one per topic touched.

## What blocked me
### [Name the blocker] — RESOLVED / ONGOING
- What broke and how I found it
- The actual root cause (not just the symptom)
- What let it get this far without being caught
- What I did to fix it
- The principle I'm taking forward

## Python pattern of the day
### [Name it]
The one Python or library pattern worth a deep dive. Show the real code with WHY comments.
Explain what's happening under the hood, why this over the obvious alternative, and the
Java/TS equivalent side-by-side. Cover the parameters that actually matter and why I set them
the way I did.

## Next session
Concrete tasks, not intentions. Reference PRD task IDs where applicable.
```

---

## LLM Cost Management

- Cache ALL LLM responses during development
- Cache key: `hashlib.md5(f"{model}\n{system_prompt}\n---\n{user_prompt}".encode()).hexdigest()`
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
2. Check Notion Command Center for current project state
3. Build / test / evaluate
4. Before session end:
   - Git commit + push
   - Create Learning Journal entry in Notion (using the template above)
   - Create any new ADR entries in Notion for decisions made this session
   - Generate a handoff summary (compact block for next chat session)


