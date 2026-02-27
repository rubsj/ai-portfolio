# P4 Day 3 Plan — API + Vector Store + Demo + Documentation

## Context

Day 1+2 are complete and merged. On disk: 50 jobs, 250 resumes, 250 pairs, failure_labels, judge_results, correction_results, 9 charts, pipeline_summary.json. Working tree is clean on `main`.

Day 3 turns the pipeline from a batch script into a **production-grade service**: REST API, vector search, interactive demo, and full documentation. Tasks T3.1–T3.10 from PRD Section 15 (skipping T3.9 Loom — batched to Week 8).

---

## Build Order & Dependencies

```
pyproject.toml (uncomment deps)
       ↓
  src/data_paths.py (new — centralized file discovery + loaders)
       ↓
  src/vector_store.py (new — ChromaDB, depends on data_paths)
       ↓
  src/api.py (new — 9 endpoints, depends on data_paths + vector_store + labeler + judge + multi_hop)
       ↓
  tests/test_api.py (new — depends on api.py)
       ↓
  src/pipeline.py (new — orchestrator, depends on all modules)
       ↓
  streamlit_app.py (new — standalone, reads data from disk)
       ↓
  docs/adr/ (5 files) + README.md
       ↓
  git branch → commits → push → PR
```

---

## Step 0: Setup (~5 min)

- **Branch**: `feat/p4-day3-api-vectorstore-demo` off `main`
- **pyproject.toml**: Uncomment `chromadb` and `sentence-transformers`
- Run `uv sync` to install new deps
- Commit: `chore(p4): uncomment chromadb + sentence-transformers deps`

---

## Step 1: `src/data_paths.py` — Centralized Data Loading (~15 min) | T3.1 prereq

**Problem**: 4 modules (analyzer.py, multi_hop.py, run_labeling.py, judge.py) each hardcode `jobs_20260225_141615.jsonl` etc. and duplicate JSONL loading logic. API needs file discovery too.

**Solution**: Single module with:
- `find_latest(directory, prefix) -> Path | None` — glob for `{prefix}_*.jsonl`, return largest file (reuses existing pattern from run_generation.py)
- `load_jobs(path) -> dict[str, GeneratedJob]` — keyed by trace_id
- `load_resumes(path) -> dict[str, GeneratedResume]` — keyed by trace_id
- `load_pairs(path) -> list[ResumeJobPair]`
- `load_failure_labels(path) -> dict[str, FailureLabels]` — keyed by pair_id
- `load_judge_results(path) -> dict[str, JudgeResult]` — keyed by pair_id
- `load_correction_results(path) -> dict[str, list[CorrectionResult]]` — keyed by pair_id
- `load_feedback(path) -> dict[str, list[FeedbackRequest]]` — keyed by pair_id
- `load_pipeline_summary() -> dict` — reads `results/pipeline_summary.json`
- `DataStore` class — loads everything into memory at init. API imports this as a singleton. ~1MB total, O(1) lookups per request (like a Spring `@Bean`).

Path constants: `DATA_DIR`, `GENERATED_DIR`, `LABELED_DIR`, `CORRECTED_DIR`, `FEEDBACK_DIR`, `ANALYSIS_DIR`, `CHROMADB_DIR`, `RESULTS_DIR`

Commit: `feat(p4): add data_paths.py — centralized file discovery and loaders`

---

## Step 2: `src/vector_store.py` — ChromaDB Integration (~30 min) | T3.3

Four functions + one helper class:

1. **`_SentenceTransformerEF`** — ChromaDB `EmbeddingFunction` wrapper around `all-MiniLM-L6-v2`. Loaded once at query time via `get_collection()`, stays in memory for fast repeated searches (no 2s model-load per query).

2. **`resume_to_text(resume: Resume) -> str`** — Flatten resume (summary + skills + experience titles/responsibilities + education) into searchable text string.

3. **`build_resume_index(resumes?, persist_dir?) -> chromadb.Collection`**
   - Load `all-MiniLM-L6-v2` SentenceTransformer **manually** (batch path)
   - Encode 250 resumes, store in `PersistentClient` at `data/chromadb/`
   - Metadata per resume: `fit_level`, `writing_style`, `name`, `skills` (comma-separated)
   - `del model; gc.collect()` after encoding (8GB M2 constraint, same P2/P3 pattern)
   - Collection: `hnsw:space = cosine`
   - Delete + recreate collection to avoid duplicates on re-index

4. **`get_collection(persist_dir?) -> chromadb.Collection`** — Read-only access to existing collection. Passes `_SentenceTransformerEF` as the collection's `embedding_function` so ChromaDB handles query encoding automatically. Used by API at startup.

5. **`search_similar(collection, query_text, top_k=5, fit_level=None) -> list[dict]`**
   - Call `collection.query(query_texts=[query_text], ...)` — ChromaDB uses the built-in EmbeddingFunction to encode the query (no manual model load needed)
   - Optional `where={"fit_level": ...}` filter
   - Return `[{trace_id, score (1-distance), metadata}, ...]`
   - Fast queries (~50ms) since model is already loaded in the EmbeddingFunction

After creating: manually run `build_resume_index()` to populate `data/chromadb/`.

Commit: `feat(p4): add vector_store.py — ChromaDB semantic search integration`

---

## Step 3: `src/api.py` — FastAPI with 9 Endpoints (~75 min) | T3.1 + T3.4

**Startup**: Module-level `DataStore()` singleton + `get_collection()` (try/except for missing index).

### 9 Endpoints

| # | Route | Method | Logic | Response Model |
|---|-------|--------|-------|----------------|
| 1 | `/health` | GET | Return status + version + pair/job counts from DataStore | `dict` |
| 2 | `/review-resume` | POST | Call `labeler.label_pair()` (fast). If `use_judge=True` query param, also call `judge.judge_pair()` via `_create_judge_client()` | `ReviewResponse` |
| 3 | `/analysis/failure-rates` | GET | Read from `pipeline_summary.json` via DataStore | `FailureRateResponse` |
| 4 | `/analysis/template-comparison` | GET | Read A/B stats from `pipeline_summary.json`, build `TemplateStats` per template | `TemplateComparisonResponse` |
| 5 | `/evaluate/multi-hop` | POST | Call `labeler.label_pair()` then `multi_hop.generate_multi_hop_questions()` | `MultiHopResponse` |
| 6 | `/search/similar-candidates` | GET | Call `vector_store.search_similar()`, enrich with resume details from DataStore | `SimilarCandidatesResponse` |
| 7 | `/feedback` | POST | Append JSON to `data/feedback/feedback.jsonl`, update in-memory store | `FeedbackResponse` |
| 8 | `/jobs` | GET | Paginate + filter from DataStore.jobs. Query: `page`, `page_size` (1-50), `industry`, `is_niche` | `JobListResponse` |
| 9 | `/pairs/{pair_id}` | GET | Lookup pair + resume + job + labels + judge + corrections + feedback from DataStore. 404 if not found | `PairDetailResponse` |

**Key reused functions** (no refactoring needed — signatures already accept ad-hoc inputs):
- `labeler.label_pair(resume, job, pair_id, normalizer)` → labeler.py:336
- `judge.judge_pair(client, job, resume, pair_id, use_cache=True)` → judge.py:181
- `judge._create_judge_client()` → judge.py:95
- `multi_hop.generate_multi_hop_questions(resume, job, labels, pair_id)` → multi_hop.py:264
- `labeler.calculate_total_experience(experiences)` → labeler.py:131 (for SimilarCandidate.experience_years)
- `vector_store.search_similar(collection, text, top_k, fit_level)` → new

**All 14 API request/response schemas verified present** in schemas.py (lines 364-482): `ReviewRequest`, `ReviewResponse`, `FailureRateResponse`, `TemplateStats`, `TemplateComparisonResponse`, `MultiHopQuestion`, `MultiHopRequest`, `MultiHopResponse`, `SimilarCandidate`, `SimilarCandidatesResponse`, `FeedbackRequest`, `FeedbackResponse`, `JobSummary`, `JobListResponse`, `PairDetailResponse`. No Step 1.5 needed.

Commit: `feat(p4): add api.py — FastAPI with 9 endpoints`

---

## Step 4: `tests/test_api.py` — API Tests (~45 min) | T3.2

FastAPI TestClient (httpx-based). Mock `DataStore`, ChromaDB collection, and judge client.

**Test coverage per endpoint:**

| Endpoint | Happy Path | Edge Cases |
|----------|-----------|------------|
| `/health` | 200 + status/version/counts | — |
| `/review-resume` | Labels returned (no judge) | With judge; invalid resume (422); empty skills (422) |
| `/analysis/failure-rates` | Stats from summary | No pipeline run (503) |
| `/analysis/template-comparison` | A/B results | No data (503) |
| `/evaluate/multi-hop` | 4 questions returned | Invalid request (422) |
| `/search/similar-candidates` | Results with scores | With fit_level filter; no vector store (503) |
| `/feedback` | Logs entry, returns UUID | Missing pair_id (422) |
| `/jobs` | Paginated results | Page 2; filter industry; filter niche; page_size>50 (422) |
| `/pairs/{pair_id}` | Full detail response | Invalid ID (404) |

**Mocking strategy**: Patch `src.api._store`, `src.api._collection`, `src.api._normalizer` at module level. Build test fixtures for Resume, Job, Pair, FailureLabels using helper functions.

Commit: `test(p4): add test_api.py — TestClient tests for all endpoints`

---

## Step 5: `src/pipeline.py` — Orchestrator (~15 min) | T3.5

Thin wrapper calling existing module entry points in sequence:

1. `run_generation.main()` → generate 50 jobs + 250 resumes
2. `run_labeling.run()` → label all pairs
3. `judge.judge_batch()` → GPT-4o evaluation
4. `corrector` → correction loop
5. `analyzer` → charts + pipeline_summary
6. `multi_hop.run()` → evaluation questions
7. `vector_store.build_resume_index()` → ChromaDB index

Skip flags (`skip_generation`, `skip_judge`, `skip_vector_store`) for iterative dev.

Commit: `feat(p4): add pipeline.py — end-to-end orchestrator`

---

## Step 6: `streamlit_app.py` — Interactive Demo (~60 min) | T3.6

**Standalone mode** — reads data from disk via `data_paths.py`. No API server required.

5 sidebar pages:

1. **Browse Jobs** — Table of 50 jobs, filter by industry/niche, click for details
2. **Review Resume** — Select a pre-generated pair, display failure labels + judge results side by side, show Jaccard/seniority metrics
3. **Analysis Dashboard** — Display 9 charts from `results/charts/` + key metrics from pipeline_summary (st.metric widgets)
4. **Search Similar** — Text input + top_k slider + fit_level dropdown → query ChromaDB → display ranked results with similarity scores
5. **Feedback** — Thumbs up/down on displayed pair → append to feedback.jsonl

**Patterns**: `@st.cache_data` for data loading, `st.sidebar.radio` for navigation, `st.image()` for charts, `st.metric()` for KPIs.

ChromaDB import wrapped in try/except — search page shows "not available" if missing.

Commit: `feat(p4): add streamlit_app.py — interactive demo`

---

## Step 7: ADRs (5 files in `docs/adr/`) (~25 min) | T3.7

| File | Decision | Key Points |
|------|----------|------------|
| `ADR-001-instructor-nested-schemas.md` | Instructor with max_retries=5 | ~30 failure points in nested Resume; strategic Optional fields reduce retry frequency; Mode.JSON for API-level enforcement |
| `ADR-002-skill-normalization.md` | Custom normalizer over library | lowercase→version→suffix→aliases pipeline; deterministic+testable; Jaccard accuracy depends on it |
| `ADR-003-two-phase-validation.md` | Structural vs semantic separation | Instructor = JSON→Pydantic; labeler = content quality. Java parallel: `@Valid` vs business rules |
| `ADR-004-fastapi-over-flask.md` | FastAPI over Flask | Native Pydantic integration; auto OpenAPI/Swagger at `/docs`; async support; dependency injection. Spring Boot parallel |
| `ADR-005-chromadb-over-faiss.md` | ChromaDB for P4 (FAISS in P2) | Persistence across restarts; native metadata filtering; different tools for different jobs |

Commit: `docs(p4): add ADR-001 through ADR-005`

---

## Step 8: `README.md` (~25 min) | T3.8

Sections:
1. Problem statement (1 paragraph)
2. Architecture diagram (Mermaid: Generator→Validator→Labeler→Judge→Corrector→Analyzer→VectorStore→API→Demo)
3. Key results table (Jaccard gradient, χ²=32.74 p<0.001, correction rate 100%, judge avg 0.541)
4. API endpoint table (9 endpoints × method/path/description)
5. Setup instructions (`uv sync`, `.env`, `python -m src.pipeline`, `uvicorn`, `streamlit run`)
6. File structure (abbreviated)
7. Technical highlights (Instructor, two-phase validation, normalization, ChromaDB)

Commit: `docs(p4): add README.md with architecture diagram and results`

---

## Step 9: Git Push + PR (~10 min) | T3.10

- Push branch to origin
- Create PR: `feat(p4): API (9 endpoints), ChromaDB vector store, Streamlit demo, documentation`
- Merge to main

---

## Verification Plan

1. **Dependencies**: `uv sync` succeeds with chromadb + sentence-transformers
2. **Vector store**: Run `python -c "from src.vector_store import build_resume_index; build_resume_index()"` — verify `data/chromadb/` populated
3. **API smoke test**: `uvicorn src.api:app --reload` then:
   - `curl localhost:8000/health` → 200 with pair/job counts
   - `curl localhost:8000/jobs?page=1&page_size=3` → 3 jobs
   - `curl localhost:8000/analysis/failure-rates` → metrics
   - Open `localhost:8000/docs` → Swagger UI renders all 9 endpoints
4. **Tests**: `uv run pytest tests/test_api.py -v` — all pass
5. **Full test suite**: `uv run pytest tests/ -v` — no regressions
6. **Streamlit**: `uv run streamlit run streamlit_app.py` — all 5 pages load
7. **Linting**: `uv run ruff check src/ tests/ streamlit_app.py`

---

## Time Estimates

| Step | Task | Est. |
|------|------|------|
| 0 | Setup (branch + deps) | 5 min |
| 1 | data_paths.py | 15 min |
| 2 | vector_store.py + manual test | 35 min |
| 3 | api.py (9 endpoints) | 75 min |
| 4 | test_api.py | 45 min |
| 5 | pipeline.py | 15 min |
| 6 | streamlit_app.py | 60 min |
| 7 | ADRs (5 docs) | 25 min |
| 8 | README.md | 25 min |
| 9 | Git push + PR | 10 min |
| — | **Total** | **~5.2 hrs** |

If time is tight, cut in this order: Streamlit polish > individual ADR depth > pipeline.py
