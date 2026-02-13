# CLAUDE.md — P2: RAG Evaluation Benchmarking Framework

> **Read this file + PRD.md at the start of EVERY session.**
> This is your persistent memory across sessions. Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P2 — Evaluating RAG for Any PDF
- **Location:** `02-rag-evaluation/` within `ai-portfolio` monorepo
- **Timeline:** Feb 12–17, 2026 (5 sessions, ~22-24h total)
- **PRD:** `PRD.md` in this directory — the implementation contract
- **Concepts Primer:** `p2-concepts-primer.html` in project root — read for theory background

---

## Developer Context

- **Background:** Java/TypeScript developer learning Python. Completed P1 (Synthetic Data) in ~13h.
- **P1 patterns to reuse:** Instructor for structured LLM output, JSON file caching (MD5 hash), Pydantic for validation, matplotlib/seaborn for charts
- **Python comfort level:** Intermediate after P1 — comfortable with Pydantic, type hints, f-strings, list comprehensions. New concepts for P2: FAISS, embeddings, ThreadPoolExecutor, generators for large data
- **IDE:** VS Code + Claude Code terminal
- **Hardware:** MacBook Air M2, 8GB RAM — this is a HARD constraint

---

## Architecture Rules (Do NOT Re-Debate)

These decisions are FINAL. Refer to PRD Section 2 for full rationale.

1. **FAISS IndexFlatIP** for vector stores — brute force is correct for <1000 vectors. Normalized embeddings + inner product = cosine similarity.
2. **5 chunk configs (A-E):** A(128/32), B(256/64), C(512/128), D(256/128), E(semantic headers). Config E splits on `##`/`###` Markdown headers.
3. **3 embedding models:** MiniLM-L6-v2 (384d), mpnet-base-v2 (768d), text-embedding-3-small (1536d via LiteLLM)
4. **Local embeddings SEQUENTIAL, API embeddings PARALLEL** — 8GB RAM constraint. Never load two SentenceTransformer models simultaneously. Use ThreadPoolExecutor for API calls only.
5. **Instructor** for synthetic QA generation — auto-retry on Pydantic validation. Same pattern as P1.
6. **`judges` library** (quotient-ai/judges) for LLM-as-Judge — NOT `judgy`. Fallback: manual openai structured outputs if dep conflicts.
7. **Braintrust** for experiment tracking. One project, one experiment per config.
8. **Logfire** is a STRETCH GOAL — implement only if core is done.
9. **`judgy`** is a STRETCH GOAL — implement only if core is done.
10. **No full LangChain** — only `langchain-text-splitters` for `RecursiveCharacterTextSplitter`.

---

## Memory Management Protocol (CRITICAL — 8GB M2)

Claude Code MUST follow this when writing embedding/indexing code:

```
For EACH local embedding model:
  1. Load model
  2. Embed ALL 5 chunk configs with this model
  3. Save all 5 FAISS indices to disk
  4. Delete model from memory
  5. Call gc.collect()
  6. Proceed to next model

For API embeddings (OpenAI via LiteLLM):
  - Use ThreadPoolExecutor(max_workers=8)
  - No local model loaded — safe for parallel I/O
  
NEVER:
  - Load two SentenceTransformer models at once
  - Keep FAISS indices in memory while embedding
  - Run local embeddings in parallel
```

---

## Notion Integration

Claude Code can write to Notion via MCP. Use these IDs:

| Resource | ID / URL |
|----------|----------|
| Command Center | `https://www.notion.so/2ffdb630640a81f58df5f5802aa51550` |
| Project Tracker (data source) | `collection://4eb4a0f8-83c5-4a78-af3a-10491ba75327` |
| P2 Tracker Card | *(create on Day 1 — update this field with the page ID)* |
| Learning Journal (data source) | `collection://c707fafc-4c0e-4746-a3bc-6fc4cd962ce5` |
| ADR Log (data source) | `collection://629d4644-ca7a-494f-af7c-d17386e1189b` |
| Chat Index | `303db630640a81ccb026f767597b023f` |

### Journal Entry Template

At the end of each session, create a journal entry in the Learning Journal:

```
Properties:
  - Title: "P2 Day [N] — [summary]"
  - Project: P2
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

### From P1 (continue these):
- **Comment with "WHY" not "what"** — `# WHY: Pydantic catches LLM hallucinations that type hints miss`
- **Type hints everywhere** — `def embed(text: str) -> list[float]:`
- **Pydantic for all data models** — no raw dicts, no dataclasses for validated data
- **Cache ALL LLM calls** — JSON file cache keyed on MD5(model + prompt). Check before calling, save after.
- **f-strings for prompts** — multi-line with `f"""..."""`

### New for P2:
- **Factory pattern for parsers** — PDF vs Markdown, return same interface
- **Factory pattern for embedders** — Local (SentenceTransformers) vs API (LiteLLM), same `.embed()` interface
- **ABC (Abstract Base Class)** for embedder interface — like Java interfaces
- **Generators (`yield`)** for chunking — process chunks lazily without holding all in memory
- **`ThreadPoolExecutor`** for API embedding parallelism — like Java's `ExecutorService`
- **numpy arrays** for embedding storage — not Python lists (FAISS requires ndarray)
- **Normalize embeddings** before adding to FAISS — `faiss.normalize_L2(vectors)` then `IndexFlatIP`

### Test conventions:
- `pytest` for all tests
- Test file per module: `test_models.py`, `test_chunker.py`, `test_embedder.py`, etc.
- Mock API calls in tests — never hit real APIs in CI
- Pydantic validation tests: always test both valid and invalid inputs

---

## File Structure

```
02-rag-evaluation/
├── CLAUDE.md                          # THIS FILE
├── PRD.md                             # Implementation contract
├── p2-concepts-primer.html            # Theory background (read-only reference)
├── pyproject.toml                     # Dependencies
├── .env                               # API keys (gitignored)
├── src/
│   ├── __init__.py
│   ├── config.py                      # Constants, 5 chunk configs, 3 embedding models, paths
│   ├── models.py                      # ALL Pydantic schemas from PRD Section 4
│   ├── parser.py                      # Factory: PyMuPDF (PDF) + Markdown parser
│   ├── chunker.py                     # Fixed-size (RecursiveCharacterTextSplitter) + semantic (Config E)
│   ├── embedder.py                    # Factory: SentenceTransformers (sequential) + LiteLLM (ThreadPool)
│   ├── vector_store.py                # FAISS IndexFlatIP: create, add, search, save, load
│   ├── bm25_baseline.py               # BM25Okapi from rank-bm25
│   ├── synthetic_qa.py                # 5 QA strategies via Instructor + quality report
│   ├── retrieval_evaluator.py         # Recall@K, Precision@K, MRR@K
│   ├── reranker.py                    # Cohere Rerank API wrapper
│   ├── generation_evaluator.py        # RAGAS evaluation wrapper
│   ├── judge.py                       # judges library + custom BloomTaxonomyClassifier
│   ├── braintrust_logger.py           # Experiment tracking + feedback classification
│   ├── observability.py               # Logfire instrumentation (STRETCH)
│   ├── grid_search.py                 # Orchestrator — runs full evaluation matrix
│   ├── cache.py                       # LLM response cache (MD5 hash → JSON file)
│   ├── visualization.py               # 12 charts from PRD Section 7
│   └── cli.py                         # Click + Rich CLI
├── tests/
│   ├── test_models.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_retrieval_evaluator.py
│   └── test_synthetic_qa.py
├── data/
│   ├── input/                         # Source documents (Kaggle MD or PDFs)
│   ├── cache/                         # LLM response cache
│   └── output/                        # Chunks, QA pairs, FAISS indices
├── results/
│   ├── charts/                        # 12 PNG visualizations
│   ├── metrics/                       # Per-config JSON metrics
│   └── reports/                       # GridSearchReport + QADatasetReport
├── docs/
│   └── adr/                           # 5 Architecture Decision Records
├── streamlit_app.py                   # Demo app
└── README.md                          # Portfolio README
```

---

## Session Protocol

### Starting a session:
```
Read CLAUDE.md and PRD.md. Today is Day [N].
Here's where I left off: [paste handoff from previous session]
Focus on tasks [#X through #Y] from PRD Section 10.
Stop after each task and wait for my approval before moving to the next one.
```

### Ending a session:
1. **Git commit and push** all work
2. **Update CLAUDE.md** "Current State" section below
3. **Write journal entry** to Notion Learning Journal via MCP
4. **Produce handoff summary** in this format:

```
## P2 Handoff — Session End [Date]

### Branch / Commit
- Branch: `main` at `[hash]`
- Working tree: [clean/dirty]

### What's Done
[list of completed PRD tasks with task numbers]

### Key Files Created/Modified
[file list with brief description]

### Key Results / Decisions
[any metrics, surprises, deviations from PRD]

### What's Next
[next session's tasks from PRD Section 10]

### Blockers / Open Questions
[anything unresolved for the planning chat to address]
```

---

## Environment Setup (Day 1 Task)

```bash
# From monorepo root
cd 02-rag-evaluation

# Initialize project
uv init
uv add PyMuPDF tiktoken langchain-text-splitters
uv add sentence-transformers litellm faiss-cpu rank-bm25 numpy
uv add cohere
uv add openai instructor
uv add ragas judges braintrust
uv add pydantic python-dotenv
uv add click rich matplotlib seaborn plotly
uv add streamlit
uv add pytest ruff --dev

# Optional (stretch goals)
uv add logfire

# Create .env
echo "OPENAI_API_KEY=sk-..." > .env
echo "COHERE_API_KEY=..." >> .env
echo "BRAINTRUST_API_KEY=..." >> .env
```

---

## Current State

> **Claude Code: UPDATE this section at the end of every session.**

### Day 0 (Pre-start)
- [ ] Project directory created
- [ ] Dependencies installed (`uv sync` passes)
- [ ] .env configured with API keys
- [ ] P2 card created in Notion Project Tracker

### Day 1 — Foundation: Parse, Chunk, Embed (Thu Feb 12)
- [ ] Task 1: Dependencies installed
- [ ] Task 2: All Pydantic models in `models.py`
- [ ] Task 3: Config constants in `config.py`
- [ ] Task 4: Document parser (PDF + Markdown) in `parser.py`
- [ ] Task 5: Chunker (fixed-size A-D + semantic E) in `chunker.py`
- [ ] Task 6: Tests for models + chunker
- [ ] Checkpoint: Parse Kaggle MD file, print chunk counts per config

### Day 2 — Embeddings, FAISS, BM25 (Fri Feb 13)
- [ ] Task 7: Embedder factory in `embedder.py`
- [ ] Task 8: FAISS vector store in `vector_store.py`
- [ ] Task 9: BM25 baseline in `bm25_baseline.py`
- [ ] Task 10: LLM cache in `cache.py`
- [ ] Task 11: Build all 15 FAISS indices + BM25
- [ ] Task 12: Embedder tests
- [ ] Checkpoint: 16 searchable retrieval backends on disk

### Day 3 — Synthetic QA + Full Grid Search (Sun Feb 15)
- [ ] Task 13: Synthetic QA generation (5 strategies, ≥50 questions)
- [ ] Task 14: QA quality report (`QADatasetReport`)
- [ ] Task 15: Retrieval evaluator (R/P/MRR @1,3,5)
- [ ] Task 16: Grid search orchestrator
- [ ] Task 17: Visualization (at least 4 core charts)
- [ ] Task 18: Run full grid search
- [ ] Task 19: QA + evaluator tests
- [ ] Checkpoint: First heatmap, best config identified, BM25 comparison

### Day 4 — Reranking + RAGAS + Judges + Braintrust (Mon Feb 16)
- [ ] Task 20: Reranker (Cohere)
- [ ] Task 21: Generation evaluator (RAGAS)
- [ ] Task 22: LLM-as-Judge (`judges` library + Bloom)
- [ ] Task 23: Braintrust experiment logging
- [ ] Task 24: Run reranking + all evaluations
- [ ] Checkpoint: Complete evaluation pipeline, Braintrust dashboard

### Day 5 — CLI, Streamlit, Docs, Deploy (Tue Feb 17)
- [ ] Task 25: CLI with Rich output
- [ ] Task 26: Streamlit app
- [ ] Task 27: All 12 charts generated
- [ ] Task 28: GridSearchReport compiled
- [ ] Task 29: README with Mermaid diagram
- [ ] Task 30: 5 ADRs written
- [ ] Deployed to Streamlit Cloud
- [ ] Loom recorded
- [ ] P2 marked Done in Notion

---

## P1 Patterns to Reuse

Reference these from `01-synthetic-data-home-diy/` when implementing P2:

| Pattern | P1 File | P2 Usage |
|---------|---------|----------|
| Instructor client setup | `src/generator.py` | `src/synthetic_qa.py` — same `instructor.from_openai()` pattern |
| JSON file cache | `src/generator.py` | `src/cache.py` — extract into shared utility |
| Pydantic model with validators | `src/schemas.py` | `src/models.py` — same Field() + @field_validator patterns |
| LLM-as-Judge prompt structure | `src/evaluator.py` | `src/judge.py` — same structured evaluation approach |
| Chart generation | `src/analysis.py` | `src/visualization.py` — same matplotlib/seaborn patterns |
| Streamlit app structure | `streamlit_app.py` | `streamlit_app.py` — same sidebar navigation pattern |

---

## Key Concepts Quick Reference

(For deep explanation, read `p2-concepts-primer.html`)

- **RAG:** Retrieve relevant context from your docs → inject into LLM prompt → generate grounded answer
- **Embeddings:** Vectors (arrays of floats) that represent text meaning. Similar text → similar vectors.
- **Cosine similarity:** Measures angle between vectors. Range -1 to +1. >0.7 = semantically similar.
- **FAISS IndexFlatIP:** Brute-force inner product search. For normalized vectors, IP = cosine similarity.
- **Chunking:** Splitting documents into pieces. Chunk size and overlap are hyperparameters to tune.
- **BM25:** Classical word-matching retrieval (like grep). The "floor" your vector search must beat.
- **Reranking:** Two-stage retrieval — fast vector search (top-20) → accurate cross-encoder scoring (top-5).
- **RAGAS:** Framework measuring faithfulness (no hallucination), relevancy, context recall/precision.
- **Recall@K:** Did the gold chunk appear in top-K? (binary per question, averaged across all)
- **Precision@K:** What fraction of top-K were relevant?
- **MRR@K:** Reciprocal rank of first relevant result. High MRR = right answer near top.
