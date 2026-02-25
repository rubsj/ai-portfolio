# CLAUDE.md — P4: AI-Powered Resume Coach

> **Read this file + PRD.md at the start of EVERY session.**
> This is your persistent memory across sessions. Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P4 — AI-Powered Resume Coach: Synthetic Data Pipeline
- **Location:** `04-resume-coach/` within `ai-portfolio` monorepo
- **Timeline:** Feb 24–26, 2026 (3 sessions, no hard time cap)
- **PRD:** `PRD.md` in this directory — the implementation contract
- **Concepts Primer:** `p4-concepts-primer.html` in project root — read for Jaccard, skill normalization, FastAPI, correction loops

---

## Model Routing Protocol (CRITICAL)

This project uses the **Opus-plans, Sonnet-executes** workflow:

### When to use Opus (Planning)
- **Start of each day:** Read PRD tasks for the day, create a detailed implementation plan
- **Architecture decisions:** If something in the PRD is ambiguous, reason through it
- **Debugging:** When stuck on a non-trivial bug (not typos — conceptual issues)
- **Evaluation interpretation:** Analyzing results, deciding chart generation strategy
- **Instruction:** Plan the exact files, classes, functions, and interfaces before writing code

### When to use Sonnet (Execution)
- **All code writing** — implement what the Opus plan specified
- **File creation** — pyproject.toml, __init__.py, tests
- **Running commands** — uv sync, pytest, python scripts, uvicorn
- **Routine edits** — fixing imports, adjusting parameters, formatting
- **Chart generation** — matplotlib/seaborn code (follow the plan)

### Session Workflow
```
1. Switch to Opus
2. "Read CLAUDE.md and PRD.md. Today is Day [N]. Plan tasks T[X.Y] through T[X.Z]."
3. Opus produces: file-by-file plan, function signatures, key logic, validation criteria
4. Switch to Sonnet
5. "Execute the plan. Start with [first file]."
6. Sonnet implements, tests, commits
7. If blocked → switch to Opus for debugging/replanning
8. At session end → use Sonnet for git commit, journal entry, CLAUDE.md update
```

---

## Developer Context

- **Background:** Java/TypeScript developer learning Python. Completed P1 (Synthetic Data) + P2 (RAG Evaluation) + P3 (Contrastive Embeddings).
- **P1 patterns to reuse:** Pydantic validation, JSON caching, Instructor integration, correction loops, matplotlib/seaborn charts, Rich console output
- **P2 patterns to reuse:** ChromaDB/FAISS knowledge, SentenceTransformer model loading/unloading, `gc.collect()` memory management, Click CLI, comparison heatmaps, Braintrust experiment tracking
- **P3 patterns to reuse:** Embedding model lifecycle management, before/after comparison methodology, publication-quality charts
- **Python comfort level:** Intermediate-to-comfortable after P1-P3. Strong with: Pydantic v2, type hints, list comprehensions, ThreadPoolExecutor, generators, numpy, matplotlib/seaborn, Instructor. **New for P4:** FastAPI, uvicorn, httpx TestClient, ChromaDB, scipy chi-squared test
- **IDE:** VS Code + Claude Code terminal
- **Hardware:** MacBook Air M2, 8GB RAM — constraint for ChromaDB + SentenceTransformers

---

## Architecture Rules (Do NOT Re-Debate)

These decisions are FINAL. Refer to PRD Section 2 for full rationale.

1. **OpenAI GPT-4o-mini** for generation + correction. **GPT-4o** for LLM-as-Judge.
2. **Instructor for ALL structured LLM output** — `max_retries=5`, `mode=instructor.Mode.JSON`. No raw OpenAI SDK.
3. **Deeply nested Pydantic schemas** — Resume → Education[], Experience[], Skills[]. Use strategic Optional fields to separate structural from semantic validation.
4. **Two-phase validation:** Instructor handles structural parsing (JSON → Pydantic). Labeling pipeline handles semantic quality (Jaccard, hallucination, etc.).
5. **5 prompt templates** for A/B testing — formal, casual, technical, achievement, career-changer.
6. **6 rule-based failure modes** — skills overlap, experience mismatch, seniority mismatch, missing core skills, hallucinations, awkward language.
7. **LLM-as-Judge is CORE**, not stretch — GPT-4o evaluates all pairs.
8. **Correction loop is CORE** — max 3 retries per invalid record, target >50% success rate.
9. **FastAPI** with 9 endpoints — POST /review-resume, GET /health, GET /analysis/failure-rates, GET /analysis/template-comparison, POST /evaluate/multi-hop, GET /search/similar-candidates, POST /feedback, GET /jobs, GET /pairs/{pair_id}.
10. **ChromaDB** for vector store — semantic resume search, persistent storage, metadata filtering. New tool (FAISS used in P2 for benchmarking — different tool for different job).
11. **Prompt A/B testing** — Core. Chi-squared significance test on template failure rates.
12. **Multi-hop evaluation questions** — Core. Cross-section reasoning questions.
13. **Feedback mechanism** — Core. POST /feedback logs thumbs up/down to JSON (Braintrust if time permits).

---

## Memory Management Protocol (8GB M2)

ChromaDB + SentenceTransformers on M2:

```
RULE 1: Load SentenceTransformer ONCE for embedding, then del + gc.collect().
  - Embed all resumes → save to ChromaDB → del model → gc.collect()
  - ChromaDB handles persistence — no need to keep model in memory.

RULE 2: FastAPI runs AFTER pipeline, not during.
  - Pipeline generates/analyzes data → saves to disk
  - API reads from disk/ChromaDB — no LLM calls except /review-resume

RULE 3: LLM judge runs with ThreadPoolExecutor (max 4 workers).
  - 250 pairs × 5s ÷ 4 threads ≈ 5 minutes
  - API calls are I/O-bound — threads are safe

RULE 4: Close non-essential apps during generation batch.
  - Generation of 250 pairs takes ~20 minutes with caching
```

---

## Notion Integration

Claude Code can write to Notion via MCP. Use these IDs:

| Resource | ID / URL |
|----------|----------|
| Command Center | `https://www.notion.so/2ffdb630640a81f58df5f5802aa51550` |
| Project Tracker (data source) | `collection://4eb4a0f8-83c5-4a78-af3a-10491ba75327` |
| P4 Tracker Card | *(create on Day 1 — update this field with the page ID)* |
| Learning Journal (data source) | `collection://c707fafc-4c0e-4746-a3bc-6fc4cd962ce5` |
| ADR Log (data source) | `collection://629d4644-ca7a-494f-af7c-d17386e1189b` |
| Chat Index | `303db630640a81ccb026f767597b023f` |

### Journal Entry Template

At the end of each session, create a journal entry in the Learning Journal:

```
Properties:
  - Title: "P4 Day [N] — [summary]"
  - Project: P4
  - Date: [today]
  - Hours: [session hours]

Content:
  ## What I Built
  [files created/modified, key functionality]

  ## What I Learned
  [concepts understood, Python patterns, surprises]

  ## What Blocked Me
  [issues, workarounds, things deferred]

  ## Python Pattern of the Day
  [one specific Python pattern with Java/TS comparison]

  ## Tomorrow's Plan
  [next session tasks from PRD]
```

---

## Code Conventions

### From P1/P2/P3 (continue these):
- **Comment with "WHY" not "what"** — `# WHY: max_retries=5 because nested schemas have ~30 failure points vs P1's ~7`
- **Type hints everywhere** — `def calculate_jaccard(resume_skills: list[str], job_skills: list[str]) -> tuple[float, int, int]:`
- **Pydantic for all data models** — no raw dicts
- **f-strings for everything** — prompts, log messages, file paths
- **pathlib.Path** over os.path

### New for P4:
- **FastAPI route decorators** — `@app.post("/review-resume")` — like Spring's `@PostMapping`
- **Query parameters** — `use_judge: bool = Query(False)` — like Spring's `@RequestParam`
- **Pydantic in routes** — FastAPI auto-validates request/response. Like `@RequestBody @Valid MyDTO`
- **httpx TestClient** — `client = TestClient(app)` — like MockMvc in Spring Boot
- **ChromaDB collection** — `collection.add()` / `collection.query()` — like a simplified Elasticsearch
- **scipy.stats.chi2_contingency** — Chi-squared test for A/B testing significance

---

## Prompts for Claude Code

### Starting a session (Opus):
```
Read CLAUDE.md and PRD.md. Today is Day [N].
Plan tasks T[X.Y] through T[X.Z] from PRD Section 15.
For each task: specify files to create/modify, function signatures,
key logic, and validation criteria (how do we know it works?).
```

### Switching to execution (Sonnet):
```
Execute the plan from Opus. Start with [first file].
After each file: run ruff check, run relevant tests, commit.
Do NOT re-debate architecture — follow the plan.
```

### Ending a session (Sonnet):
1. **Git commit and push** all work
2. **Update CLAUDE.md** "Current State" section below
3. **Write journal entry** to Notion Learning Journal via MCP
4. **Produce handoff summary** in this format:

```
## P4 Handoff — Session End [Date]

### Branch / Commit
- Branch: `feat/p4-[description]`
- Working tree: [clean/dirty]

### What's Done
[list of completed PRD tasks with task numbers]

### Key Files Created/Modified
[file list with brief description]

### Key Metrics (if any)
[validation rate, Jaccard averages, correction rate — whatever was measured today]

### What's Next
[next session's tasks from PRD Section 15]

### Blockers / Open Questions
[anything unresolved — flag for Opus planning chat]
```

---

## Current State

> **Claude Code: UPDATE this section at the end of every session.**

### Day 0 (Pre-start)
- [ ] Project directory created
- [ ] Dependencies installed (`uv sync` passes)
- [ ] .env with OPENAI_API_KEY configured
- [ ] P4 card created in Notion Project Tracker

### Day 1 — Schemas + Generation + Validation (Mon Feb 24)
- [ ] T1.1: Project setup (directory, pyproject.toml, uv sync)
- [ ] T1.2: schemas.py — ALL Pydantic models (Resume, Job, metadata, failure labels, judge)
- [ ] T1.3: test_schemas.py — Valid/invalid data tests for every model
- [ ] T1.4: normalizer.py + test_normalizer.py — Skill normalization
- [ ] T1.5: templates.py — 5 prompt templates for A/B testing
- [ ] T1.6: generator.py — Instructor-based generation + caching
- [ ] T1.7: validator.py — Validation tracking + error categorization
- [ ] T1.8: Run generation pipeline — 50 jobs + 250 resumes
- [ ] T1.9: Sanity check — spot-check 5 pairs
- [ ] **Checkpoint:** 250+ pairs generated and validated. Ready for analysis.

### Day 2 — Labeling + Judge + Correction + Analysis (Tue Feb 25)
- [ ] T2.1: labeler.py — All 6 failure modes
- [ ] T2.2: test_labeler.py — Unit tests for each mode
- [ ] T2.3: Run labeling pipeline on all pairs
- [ ] T2.4: judge.py — LLM-as-Judge (GPT-4o) on all pairs
- [ ] T2.5: corrector.py — Correction loop implementation
- [ ] T2.6: test_corrector.py — Correction tests
- [ ] T2.7: Run correction pipeline on invalid records
- [ ] T2.8: analyzer.py — DataFrame + 9 visualizations + A/B chi-squared test
- [ ] T2.9: pipeline_summary.json generation
- [ ] T2.10: Multi-hop evaluation questions (bonus)
- [ ] **Checkpoint:** Complete analysis pipeline. All metrics calculated.

### Day 3 — API + Vector Store + Demo + Documentation (Wed Feb 26)
- [ ] T3.1: api.py — FastAPI with 9 endpoints (all features exposed)
- [ ] T3.2: test_api.py — TestClient tests for all endpoints
- [ ] T3.3: vector_store.py — ChromaDB integration (embed, persist, search with metadata filtering)
- [ ] T3.4: Wire vector store + feedback into API endpoints
- [ ] T3.5: pipeline.py — End-to-end orchestrator
- [ ] T3.6: streamlit_app.py — Full demo (browse jobs → analyze → search → feedback)
- [ ] T3.7: ADRs (ADR-001 through ADR-005)
- [ ] T3.8: README.md with Mermaid diagram + API endpoint table + results
- [ ] T3.9: Loom recording
- [ ] T3.10: Final git push + Notion update
- [ ] **P4 COMPLETE**

---

## P1/P2/P3 Patterns to Reuse

| Pattern | Source | P4 Usage |
|---------|--------|----------|
| Pydantic models with validators | P1 `src/schemas.py` | Deeply nested: Resume → Experience[] → responsibilities[] |
| Instructor + auto-retry | P1 `src/generator.py` | max_retries=5, JSON mode for nested schemas |
| JSON file cache | P1/P2 `data/cache/` | Cache around Instructor calls, same MD5 key pattern |
| LLM-as-Judge | P1 `src/evaluator.py` | GPT-4o evaluates hallucinations, awkward language, holistic fit |
| Correction loop | P1 `src/corrector.py` | Extended with structured error context for nested schemas |
| matplotlib/seaborn charts | P1/P2 analysis modules | 9 publication-ready charts |
| Rich progress bars | P2 CLI | Batch generation progress tracking |
| SentenceTransformer lifecycle | P2/P3 embedder | Load → encode → del → gc.collect() for ChromaDB |
| Before/after comparison | P3 evaluation | Correction loop before/after, template A/B testing |
| ADR template | P1/P2/P3 `docs/adr/` | Same structure — Context, Decision, Alternatives, Consequences |

---

## Key Concepts Quick Reference

(For deep explanation, read `p4-concepts-primer.html`)

- **Jaccard similarity:** |A ∩ B| / |A ∪ B| — set-based overlap. 0=no overlap, 1=identical sets. Use for skill matching.
- **Skill normalization:** lowercase → version removal → suffix stripping → alias mapping. Essential for accurate Jaccard.
- **Controlled fit levels:** Generate resumes at 5 quality tiers (excellent→mismatch). Tests whether labeling correctly identifies quality.
- **Two-phase validation:** Instructor (structural) → Labeling pipeline (semantic). Separation of concerns.
- **FastAPI:** Python web framework. Auto-generates OpenAPI docs from Pydantic models. Like Spring Boot + Swagger combined.
- **ChromaDB:** Vector database with persistence. Insert embeddings → query by similarity. Like simplified Elasticsearch for embeddings.
- **Chi-squared test:** Statistical test for whether template failure rates differ significantly. `scipy.stats.chi2_contingency()`.
- **A/B testing:** Compare 5 templates on validation rate + failure modes. Chi-squared determines if differences are real or noise.

---

## Troubleshooting Guide

### "Instructor fails >30% of the time on Resume schema"
- Check which fields fail most often → make them Optional
- Increase max_retries to 7 (temporary, find root cause)
- Simplify the most complex sub-model (e.g., make achievements a single string instead of list)
- If still failing: add field descriptions to schema (`Field(description="...")`) — Instructor passes these to the LLM

### "Jaccard similarity doesn't correlate with fit level"
- Check normalization: are "Python 3.10" and "python" being merged?
- Spot-check: print normalized skill sets for 5 excellent-fit pairs
- The skill alias map may be missing common aliases for the target domain

### "FastAPI returns 422 on valid-looking requests"
- 422 = Pydantic validation failed. Check the response body for details.
- Common cause: date format. Ensure ISO format in request JSON.
- Test with the auto-generated /docs page (Swagger UI) — it shows expected schema.

### "ChromaDB OOM on M2"
- Reduce batch size for embedding: embed 50 resumes at a time, not all 250
- Del SentenceTransformer model before starting ChromaDB queries
- ChromaDB's persistent mode uses disk, not RAM, for stored vectors — the OOM is during embedding, not querying

### "LLM judge takes too long"
- Use ThreadPoolExecutor with max_workers=4
- Cache judge results — same MD5 caching pattern
- If still slow: run judge on a sample (50 pairs) instead of all 250, document sampling strategy
