# CLAUDE.md — P5: ShopTalk Knowledge Management Agent

> **Read this file + PRD.md at the start of EVERY session.**
> This is your persistent memory across sessions. Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P5 — ShopTalk Knowledge Management Agent: Production RAG System
- **Location:** `05-shoptalk-knowledge-agent/` within `ai-portfolio` monorepo
- **Timeline:** 5 sessions (~22h total), flexible pacing prioritizing learning depth
- **PRD:** `PRD.md` in this directory — the product requirements contract (v3, all decisions finalized)
- **Concepts Primer:** `p5-concepts-primer.html` in project root — deep reference for chunking, hybrid retrieval, score fusion, NDCG, cross-encoders, ABC pattern, citation extraction

---

## Model Routing Protocol (CRITICAL)

**Opus plans. Sonnet executes.**

### Opus (Planning & Architecture)
- Start of each day: read PRD tasks, create detailed implementation plan with file-by-file approach
- Design Pydantic schemas, ABC interfaces, function signatures
- Debug non-trivial issues (conceptual, not typos)
- Analyze experiment results and decide what findings matter
- Any ambiguity in the PRD

### Sonnet (Implementation)
- All code writing — implement what Opus planned
- File creation, dependency setup, test writing
- Running commands (uv sync, pytest, experiment runs)
- Routine fixes (imports, parameters, formatting)
- Chart generation, documentation

### Session Workflow
```
1. Opus: "Read CLAUDE.md and PRD.md. Today is Day [N]. Plan implementation."
2. Opus produces: file-by-file plan, function signatures, key logic, validation criteria
3. Sonnet: "Execute the plan. Start with [first file]."
4. Sonnet implements, tests, commits
5. If blocked → Opus for debugging
6. Session end → Sonnet for git commit, journal entry, CLAUDE.md update
```

---

## Developer Context

- **Background:** Java/TypeScript developer learning Python. Completed P1–P4.
- **Learning Priority:** Learning depth over speed. Rabbit holes encouraged if they produce genuine insight.
- **Hardware:** MacBook Air M2, 8GB RAM — hard constraint for model loading.
- **IDE:** VS Code + Claude Code terminal

### Patterns Proven in P1–P4 (Reuse These)
| Pattern | Source | P5 Application |
|---------|--------|----------------|
| Pydantic models + validators | P1/P4 | Document, Chunk, RetrievalResult, ExperimentConfig |
| Instructor + auto-retry | P1/P4 | Ground truth generation, LLM-as-Judge |
| JSON file cache (MD5 key) | P1/P2/P4 | Cache LLM responses + embeddings |
| FAISS + SentenceTransformers | P2 | Vector indexing with explicit persistence |
| Model lifecycle: load → use → del → gc.collect() | P2/P3 | Memory management for 3 embedding models |
| Retrieval metrics (Precision, Recall, MRR) | P2 | Plus NDCG@K (new) |
| matplotlib/seaborn charts | P1–P4 | 10+ experiment comparison charts |
| Click CLI | P2 | 3 commands: ingest, serve, evaluate |
| Rich progress bars | P2 | Batch processing progress |
| ADR template | P1–P4 | 5 ADRs (distributed across Days 1-3) |

### New for P5 (Learn These)
- **PyMuPDF** (`import fitz`) — PDF text extraction. `fitz.open()` = Java's `PDDocument.load()`.
- **rank-bm25** — `BM25Okapi(tokenized_corpus)`. Corpus = list of token lists, NOT embeddings.
- **LiteLLM** — `from litellm import completion`. Wraps OpenAI/Anthropic/Cohere behind one API. Like Java's adapter pattern.
- **NDCG** — Implement from scratch. `DCG@K = Σ(rel_i / log₂(i+1))`. `NDCG = DCG / ideal_DCG`.
- **Score fusion** — ALWAYS normalize BM25 [0,∞) to [0,1] via min-max before combining with cosine [0,1].
- **CrossEncoder** — `from sentence_transformers import CrossEncoder`. Takes (query, doc) pairs. Slower but more accurate than bi-encoder.
- **Embedding-semantic chunking** — Cosine similarity between consecutive sentence embeddings. Split where similarity drops. ALWAYS use MiniLM for boundary detection (PRD Decision 1).
- **tiktoken** — `enc = tiktoken.get_encoding("cl100k_base")` for accurate token counting. Like Java's `StringTokenizer` but for LLM tokens.
- **YAML + Pydantic** — `yaml.safe_load()` → Pydantic model. NEVER `yaml.load()` (security risk like Java's `ObjectInputStream` deserialization).
- **ABC pattern** — `from abc import ABC, abstractmethod`. Python's interface equivalent. `@abstractmethod` = Java `abstract`. Enforced at runtime, not compile time.

---

## Architecture Rules (FINAL — Do Not Re-Debate)

These come from PRD Sections 3 and 5. All design decisions are finalized.

1. **FAISS** — not ChromaDB. Explicit persistence (index + metadata files).
2. **No LangChain** — build from ABCs. First-principles RAG.
3. **LiteLLM** — multi-provider wrapper behind BaseLLM. Default to OpenAI models.
4. **Instructor** — structured output for ground truth + judge. `max_retries=3`, `mode=instructor.Mode.JSON`.
5. **5 chunking strategies** — Fixed, Recursive, Sliding Window, Heading-Semantic, Embedding-Semantic.
6. **3 embedding models** — MiniLM (384d), mpnet (768d), OpenAI text-embedding-3-small (1536d, Day 3–4).
7. **3 retrieval methods** — Dense, BM25, Hybrid (α-weighted, min-max normalized).
8. **2 rerankers** — Cohere API + local CrossEncoder. Both behind BaseReranker.
9. **5-axis LLM-as-Judge** — Relevance, Accuracy, Completeness, Conciseness, Citation Quality. NOT RAGAS.
10. **NDCG from scratch** — understand the math. 4-level grading: 3=gold, 2=same section, 1=same doc, 0=irrelevant.
11. **YAML + Pydantic** for experiment configs.
12. **Click CLI** (ingest, serve, evaluate) + **Streamlit** web UI. NO FastAPI.
13. **Hybrid ground truth** — LLM generates 30, developer curates 15. 1-3 gold chunks per query.
14. **35 experiment configs** in core grid (5 chunkers × 3 embedders × 2 retrieval methods + 5 BM25 baselines).
15. **Index-reference citations** — [N] markers parsed to chunk objects. Parse-only validation for grid; judge handles semantic validation.
16. **System performance metrics** — track ingestion time, query latency, index size, peak memory.
17. **Braintrust** — optional stretch. JSON baseline first.
18. **Embedding-semantic chunker** — ALWAYS uses MiniLM for boundary detection, regardless of indexing embedder.
19. **HybridRetriever** — internal composition. Owns both DenseRetriever and BM25Retriever.
20. **Experiment runner** — pools experiments by embedding model. MiniLM first, mpnet second, BM25, then OpenAI.
21. **Test coverage** — ≥95% on all core modules. pytest --cov enforced.

---

## Memory Management Protocol (8GB M2)

```
RULE 1: ONE SentenceTransformer at a time.
  Embed all chunks with model A → save index → del model → gc.collect()
  → load model B → repeat. NEVER two models simultaneously.

RULE 2: CrossEncoder loads ONLY during reranking.
  Reranking = top-20 candidates only. Load → rerank → del → gc.collect().

RULE 3: FAISS indices stay loaded during query session.
  IndexFlatIP for <10K vectors ≈ 30MB at 768d. Fine alongside Streamlit.

RULE 4: Embedding-semantic chunker uses MiniLM (smallest).
  del model after chunking, before indexing phase begins.

RULE 5: OpenAI embeddings = API calls = zero local RAM.
  Batch 100 texts per call. Respect rate limits.

RULE 6: Pool experiments by embedder.
  Run ALL MiniLM configs, then unload, load mpnet, run ALL mpnet configs.
  Avoids loading same model 10+ times.

MONITORING: import psutil; psutil.virtual_memory().percent
```

---

## Notion Integration

| Resource | ID / URL |
|----------|----------|
| Command Center | `https://www.notion.so/2ffdb630640a81f58df5f5802aa51550` |
| Requirements Doc | `https://www.notion.so/Mini_Project_5_requirements-2ffdb630640a81668269f59ead053417` |
| Requirements Traceability | `https://www.notion.so/Requirements-Traceability-312db630640a816fa2e0cee9dc7ba829` |
| Project Tracker (data source) | `collection://4eb4a0f8-83c5-4a78-af3a-10491ba75327` |
| P5 Tracker Card | *(create on Day 1 — update this field)* |
| Learning Journal (data source) | `collection://c707fafc-4c0e-4746-a3bc-6fc4cd962ce5` |
| ADR Log (data source) | `collection://629d4644-ca7a-494f-af7c-d17386e1189b` |
| Chat Index | `303db630640a81ccb026f767597b023f` |

### Journal Entry Template

```
Properties:
  - Title: "P5 Day [N] — [summary]"
  - Project: P5
  - Phase: [Foundation | Build | Evaluate | Document | Polish]
  - Session Type: [Weeknight | Sunday Deep Work | Saturday]
  - Date: [today]
  - Hours: [session hours]
  - Python Pattern Learned: [2-5 bullet summary]
  - Blocked By: [brief description or blank]

Content:
  ## What I Built
  ## Key Metrics
  ## What I Learned
  ## What Blocked Me
  ## Python Pattern of the Day (with Java/TS comparison)
  ## Tomorrow's Plan
```

---

## Code Conventions

- **Comment with "WHY" not "what"** — `# WHY: min-max normalize because BM25 ranges [0,∞) vs cosine [0,1]`
- **Type hints everywhere** — all function signatures, all variables where non-obvious
- **Pydantic for all data** — no raw dicts crossing module boundaries
- **f-strings** — prompts, logs, paths
- **pathlib.Path** — over os.path
- **Factory pattern** — config string → class instance via registry dict
- **yaml.safe_load()** — NEVER yaml.load()
- **One model at a time** — load/use/del/gc.collect()
- **≥95% test coverage** — pytest --cov enforced on every module

---

## Session End Protocol

At the end of every session, Sonnet must:

1. **Git commit and push** all work
2. **Update Current State** below (check off completed tasks)
3. **Write journal entry** to Notion Learning Journal
4. **Produce handoff summary:**

```
## P5 Handoff — Session End [Date]

### Branch / Commit
### What's Done (PRD task references)
### Key Files Created/Modified
### Key Metrics (if measured)
### What's Next
### Key Learnings (the interview story)
### Blockers / Open Questions
```

---

## Current State

> **UPDATE at the end of every session.**

### Day 0 (Pre-start)
- [ ] Project directory created
- [ ] pyproject.toml with all dependencies
- [ ] `uv sync` passes
- [ ] .env configured (OPENAI_API_KEY, COHERE_API_KEY)
- [ ] P5 card created in Notion Project Tracker
- [ ] Test PDFs downloaded to `data/pdfs/`

### Day 1 — Foundation: Extraction + Chunking + Schemas
- [ ] All Pydantic schemas (matching requirements data models + PRD additions)
- [ ] All 6 ABCs (BaseChunker, BaseEmbedder, BaseVectorStore, BaseRetriever, BaseReranker, BaseLLM)
- [ ] PDF extraction with PyMuPDF + text cleaning
- [ ] 5 chunking strategies (Fixed, Recursive, Sliding Window, Heading-Semantic, Embedding-Semantic)
- [ ] Tests for schemas + all chunkers (≥95% coverage)
- [ ] **ADR-001: FAISS over ChromaDB** written and committed
- [ ] **ADR-002: No LangChain — first-principles RAG** written and committed
- [ ] **Checkpoint:** All chunkers produce valid Chunks from test PDFs.

### Day 2 — Retrieval Pipeline (Sunday Deep Work)
- [ ] 2 local embedders (MiniLM, mpnet)
- [ ] FAISSVectorStore with save/load persistence
- [ ] Dense retriever
- [ ] BM25 retriever
- [ ] Hybrid retriever with min-max normalization + α weighting
- [ ] Cohere reranker + CrossEncoder reranker
- [ ] LiteLLM generator with citation extraction
- [ ] Factory pattern: config string → class
- [ ] End-to-end smoke test: PDF → chunk → embed → index → retrieve → generate → cited answer
- [ ] Tests for all new components (≥95% coverage)
- [ ] **ADR-003: Hybrid retrieval with min-max score fusion** written and committed
- [ ] **ADR-004: LiteLLM over raw OpenAI SDK** written and committed
- [ ] **Checkpoint:** Full query pipeline works end-to-end.

### Day 3 — Evaluation + Experiment Setup
- [ ] Retrieval metrics from scratch (Precision@K, Recall@K, MRR, NDCG@K)
- [ ] Ground truth: LLM generates 30, developer curates 15
- [ ] 5-axis LLM-as-Judge via Instructor
- [ ] OpenAI embedder (3rd model)
- [ ] 30+ YAML experiment configs
- [ ] Experiment runner (grid orchestrator pooled by embedder)
- [ ] System performance metric collection (ingestion time, latency, index size, memory)
- [ ] Tests for metrics + judge (≥95% coverage)
- [ ] **ADR-005: YAML + Pydantic for experiment configs** written and committed
- [ ] **Checkpoint:** Evaluation framework complete. Ready for big run.

### Day 4 — Run Experiments + Analysis
- [ ] Full experiment grid run (30+ configs)
- [ ] 10+ visualization charts
- [ ] Comparison report answering 4 required questions (Q1-Q4)
- [ ] Extended experiments (α sweep, reranking comparison)
- [ ] Judge calibration (manual vs LLM, 5 answers)
- [ ] Pipeline orchestrator (ties all phases together)
- [ ] **Checkpoint:** All experiments complete. Best config meets retrieval targets.

### Day 5 — CLI + Streamlit + Documentation + Polish
- [ ] Click CLI (ingest, serve, evaluate)
- [ ] Streamlit app (5 panels: upload, config, question, answer+citations, source viewer)
- [ ] README.md (Mermaid diagram, results, findings, setup)
- [ ] Loom recording (2-min walkthrough)
- [ ] Self-evaluation questions answered (5 from requirements)
- [ ] Final ≥95% coverage verified across all modules
- [ ] Final git push + Notion update
- [ ] **P5 COMPLETE**

---

## Troubleshooting Guide

### "BM25 scores dominate hybrid retrieval"
BM25 scores [0,∞) vs cosine [0,1]. Apply min-max normalization to BM25 BEFORE combining with α. Verify: both arrays in [0,1] after normalization. Handle edge case: all BM25 scores identical → division by zero → set all to 0.0.

### "FAISS IndexFlatIP returns low similarity scores"
IndexFlatIP = inner product. Equals cosine similarity ONLY if vectors are L2-normalized. Call `faiss.normalize_L2(embeddings)` before add() AND before search(). SentenceTransformers normalizes by default — verify with `np.linalg.norm(emb)` ≈ 1.0.

### "Embedding-semantic chunker is slow"
It embeds every sentence during chunking. Uses MiniLM (smallest, per PRD Decision 1). If still too slow, fall back to heading-semantic for that run.

### "OOM during batch embedding"
Reduce batch size to 50. `del model` + `gc.collect()` between models. Pool experiments by embedder. Monitor: `psutil.virtual_memory().percent`.

### "CrossEncoder too slow for full corpus"
CrossEncoder is O(n) per candidate. ONLY use for reranking top-20, NOT full retrieval. Pipeline: bi-encoder → top-20 → cross-encoder → top-5.

### "LLM judge gives all 5s"
Leniency bias. Manual calibration: score 5 answers yourself, compare. Add negative examples to judge prompt.

### "YAML config validation error at runtime"
This is why Pydantic validates at load time. Common: missing hybrid_alpha when method=hybrid, missing reranker when use_reranking=True.

### "No config meets Recall@5 > 0.80 target"
Iterate: increase chunk overlap, try reranking, try larger embedder. The debugging story is valuable in interviews regardless. Document what you tried.

### "PyMuPDF two-column PDF text interleaved"
Use `page.get_text("blocks")` for coordinate-based extraction, sort by column position. Or: choose single-column technical docs.

### "LiteLLM import error or routing issue"
LiteLLM model strings: `"gpt-4o-mini"` (not `"openai/gpt-4o-mini"` unless explicitly routing). Check `litellm.model_list` for supported models. Fallback: raw OpenAI SDK behind BaseLLM if LiteLLM is problematic.
