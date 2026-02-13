# Day 2 Execution Plan — Tasks 7-12: Embeddings, FAISS, BM25

## Context

Day 1 is complete: 3 PRs merged, 204 tests at 99% coverage. We have Pydantic models (`src/models.py`), config (`src/config.py`), parser (`src/parser.py`), and chunker (`src/chunker.py`) fully built and tested. 3 Kaggle Markdown files are parsed and chunkable. Today we build the **retrieval layer** — the engine that turns chunks into searchable embeddings.

**Document scope:** All 3 Kaggle Markdown files combined (financial_services.md + healthcare.md + technology.md, ~523KB total).

---

## Task-by-Task Acceptance Criteria

### Task 10 — LLM Cache (`src/cache.py`)
- Reuse P1 pattern: MD5 hash of `f"{model}\n---\n{prompt}"` as key
- `compute_cache_key(model: str, prompt: str) -> str`
- `load_cached(cache_key: str) -> dict | None` — returns raw dict or None
- `save_cached(cache_key: str, response: dict, metadata: dict | None = None) -> None`
- Cache dir: `data/cache/` (from `config.CACHE_DIR`)
- Graceful degradation: stale/corrupt cache returns None (logged), never crashes
- Not used by embedder (FAISS indices ARE the embedding cache), but needed for Day 3 QA generation

### Task 7 — Embedder Factory (`src/embedder.py`)
- `BaseEmbedder` ABC with `.embed(texts: list[str]) -> np.ndarray` and `.dimensions` property
- `SentenceTransformerEmbedder`: loads model on init, `batch_size=32`, sequential processing
- `LiteLLMEmbedder`: `ThreadPoolExecutor(max_workers=8)`, batches texts in groups of 20-50 per API call, parallelizes the _batches_ (not individual texts) to minimize API call overhead
- Factory function `create_embedder(model: EmbeddingModel) -> BaseEmbedder`
- **All embeddings L2-normalized** (unit vectors) so `IndexFlatIP` = cosine similarity
- Return type: `np.ndarray` with shape `(n_texts, dimensions)`, dtype `float32`

### Task 8 — FAISS Vector Store (`src/vector_store.py`)
- `FAISSVectorStore` class wrapping `faiss.IndexFlatIP`
- `__init__(dimension: int)` creates empty index + chunk ID mapping (`list[str]`)
- `add(embeddings: np.ndarray, chunk_ids: list[str])` — adds vectors, stores ID mapping
- `search(query: np.ndarray, k: int) -> list[tuple[str, float]]` — returns `(chunk_id, score)` pairs (see Note on Return Types below)
- `save(path: Path)` / `load(path: Path)` class method — saves to `data/output/indices/{model}_{config}.faiss` + `.json` ID mapping
- Validation: embeddings dimension must match index dimension; chunk_ids length must match embeddings count

### Task 9 — BM25 Baseline (`src/bm25_baseline.py`)
- `BM25Retriever` class wrapping `BM25Okapi`
- `__init__(chunks: list[Chunk])` — tokenizes chunk texts with `.lower().split()`
- `search(query: str, k: int) -> list[tuple[str, float]]` — same return signature as FAISS (see Note on Return Types below)
- `save(path: Path)` / `load(path: Path)` — pickle serialization for BM25 object + chunk IDs
- Saves to `data/output/indices/bm25_B.pkl` + `.json`
- Built from Config B chunks only (PRD: "BM25 uses Config B chunks")

### Note on Return Types for `search()`

`RetrievalResult` from `models.py` has 15 fields including evaluation metrics (recall/precision/MRR @1,3,5) and question metadata (query_id, question, question_type, gold_chunk_ids). The search function doesn't have access to any of these — it only knows the query and returns ranked chunks.

**Proposed approach:** `search()` returns `list[tuple[str, float]]` (chunk_id, score) as the raw retrieval primitive. Then the `retrieval_evaluator.py` (Day 3, Task 15) wraps this: takes raw search results + question metadata + gold chunk IDs → computes metrics → builds full `RetrievalResult` objects.

This keeps the vector store and BM25 as clean retrieval primitives (like a Java DAO) and puts evaluation logic in the evaluator (like a service layer). The evaluator is the only place that builds `RetrievalResult`.

### Task 12 — Tests (`tests/test_embedder.py`, `test_vector_store.py`, `test_bm25_baseline.py`, `test_cache.py`)
- All external dependencies mocked (SentenceTransformers, LiteLLM). FAISS not mocked (CPU-only, fast).
- ~35 new tests covering shape, normalization, roundtrips, edge cases
- Full test list in Test Strategy section below

### Task 11 — Build All 15 FAISS Indices + BM25
- Orchestration function in a new `src/index_builder.py` module
- **Input:** Re-parses and re-chunks all 3 Markdown files at runtime (not loading Day 1 outputs). Uses `parser.parse_document()` → `chunker.chunk_document()` for each file, then concatenates chunk lists per config. This ensures consistency — same chunker code, fresh chunks.
- Flow:
  ```
  Parse all 3 Markdown files → get 3 ParseResults
  For each chunk config (A-E):
    Chunk all 3 docs → concatenate into one chunk list

  For each LOCAL model (MiniLM, mpnet):
    1. Load SentenceTransformer model
    2. For each chunk config (A, B, C, D, E):
       - Embed all chunks (batch_size=32, sequential)
       - Create FAISS index, add vectors
       - Save index to data/output/indices/{model}_{config}.faiss + .json
    3. del model; gc.collect()

  For OpenAI (API model):
    - For each chunk config (A, B, C, D, E):
       - Batch texts (20-50 per API call), parallelize batches via ThreadPoolExecutor
       - Create FAISS index, save to data/output/indices/openai_{config}.faiss + .json

  BM25:
    - Build from Config B chunks (already chunked above)
    - Save to data/output/indices/bm25_B.pkl + .json
  ```
- **Index file naming:** `{model}_{config}.faiss` so files for the same model sort together
  - Examples: `minilm_A.faiss`, `minilm_B.faiss`, ..., `mpnet_A.faiss`, ..., `openai_A.faiss`, ...
  - Each `.faiss` file has a matching `.json` file with chunk ID mapping
- **Chunk file output:** Also save chunk lists as JSON to `data/output/chunks_{config}.json` for later use by QA generation and the checkpoint verification
- Total output: 15 FAISS indices + 1 BM25 = **16 retrieval backends** in `data/output/indices/`

---

## Implementation Order

| Step | Task | Why This Order |
|------|------|----------------|
| 1 | Task 10 (cache) | Zero dependencies on other Day 2 code. Foundation for Day 3. Quick win. |
| 2 | Task 7 (embedder) | Core building block. Everything else depends on embeddings. |
| 3 | Task 8 (vector_store) | Consumes embedder output. Needed before Task 11. |
| 4 | Task 9 (bm25) | Independent of FAISS. Simple module. |
| 5 | Task 12 (tests) | Test all modules before the big orchestration run. Catch bugs early. |
| 6 | Task 11 (build indices) | Final step — runs the full pipeline. Only safe to run after tests pass. |

**Rationale**: Bottom-up — build + test each component, then orchestrate. This avoids debugging 3 modules at once during the RAM-critical Task 11.

---

## Key Risk: 8GB RAM During Task 11

### RAM Budget Estimate
| Component | RAM Usage |
|-----------|-----------|
| Python runtime + imports | ~200MB |
| MiniLM model (all-MiniLM-L6-v2) | ~80MB |
| mpnet model (all-mpnet-base-v2) | ~420MB |
| Chunks in memory (3 docs, all configs) | ~10MB |
| FAISS index (max: 1000 vectors × 1536d × 4B) | ~6MB |
| **Peak (mpnet loaded)** | **~720MB** |
| Available (8GB - macOS overhead ~3GB) | **~5GB** |

### Verdict: Comfortable, but discipline matters

The real risk isn't a single model — it's **accidental accumulation** (not freeing the previous model). Mitigations:

1. **Explicit `del model` + `gc.collect()`** between local model loads
2. **Never load two SentenceTransformer models simultaneously**
3. **`batch_size=32`** for SentenceTransformers (controls per-batch tensor RAM)
4. **Don't hold all FAISS indices in memory** — save to disk immediately after creation
5. **Combined 3 docs = ~523KB** text — well within RAM even with all chunk configs
6. **Log memory usage** at each stage if `psutil` is available

### Fallback if RAM is tight
- Reduce `batch_size` to 16 or 8
- Process chunk configs one at a time per model (instead of all 5)
- Close Chrome / VS Code during embedding runs

---

## Test Strategy (Task 12)

### `tests/test_cache.py`

| Test | What It Verifies |
|------|-----------------|
| `test_cache_miss_returns_none` | Non-existent key returns None |
| `test_save_and_load_roundtrip` | Save a response, load it back, data matches |
| `test_cache_key_deterministic` | Same inputs produce same hash |
| `test_cache_key_differs_for_different_inputs` | Different prompts produce different hashes |
| `test_corrupt_cache_returns_none` | Malformed JSON returns None, no crash |
| `test_cache_dir_created_on_save` | Directory created lazily on first write |

### `tests/test_embedder.py`

**Mocking approach**: Mock `sentence_transformers.SentenceTransformer` and `litellm.embedding` to avoid loading real models or making API calls in tests.

| Test | What It Verifies |
|------|-----------------|
| `test_sentence_transformer_embed_shape` | Output shape is `(n, dimensions)` — 384 for MiniLM, 768 for mpnet |
| `test_sentence_transformer_embed_normalized` | All vectors have unit L2 norm (within 1e-6 tolerance) |
| `test_sentence_transformer_batch_processing` | Large input is processed in batch_size=32 chunks |
| `test_litellm_embed_shape` | Output shape is `(n, 1536)` for OpenAI |
| `test_litellm_embed_normalized` | Unit L2 norm after normalization |
| `test_litellm_thread_pool_executor` | ThreadPoolExecutor is used (mock verify concurrent calls) |
| `test_litellm_order_preserved` | Parallel execution returns embeddings in input order |
| `test_litellm_batching` | Texts are grouped into batches of 20-50 before API calls |
| `test_factory_creates_correct_type` | `create_embedder(MINILM)` returns `SentenceTransformerEmbedder`, etc. |
| `test_factory_all_models` | Parametrized: all 3 enum values produce a valid embedder |
| `test_embed_empty_list` | Edge case: empty input returns empty array with correct shape |
| `test_embed_single_text` | Single text produces `(1, dimensions)` array |

### `tests/test_vector_store.py` (no mocks — FAISS is CPU-only and fast)

| Test | What It Verifies |
|------|-----------------|
| `test_create_empty_index` | New store has 0 vectors |
| `test_add_and_search` | Add known vectors, search returns correct chunk IDs |
| `test_search_k_greater_than_total` | k > stored vectors returns all, no crash |
| `test_search_scores_descending` | Scores are in descending order |
| `test_save_and_load_roundtrip` | Save to `data/output/indices/`, load back, same results |
| `test_dimension_mismatch_raises` | Adding vectors with wrong dimension raises error |
| `test_chunk_id_mapping` | Returned IDs match what was added |

### `tests/test_bm25_baseline.py`

| Test | What It Verifies |
|------|-----------------|
| `test_bm25_search_returns_results` | Search returns up to k results |
| `test_bm25_exact_match_ranks_first` | Query matching a chunk's text ranks that chunk highest |
| `test_bm25_scores_nonnegative` | All BM25 scores >= 0 |
| `test_bm25_save_load_roundtrip` | Pickle save/load produces same search results |
| `test_bm25_tokenization` | Tokenizes with `.lower().split()` as specified |

---

## Files to Create

| File | Action | Lines (est.) |
|------|--------|-------------|
| `src/cache.py` | Create | ~60 |
| `src/embedder.py` | Create | ~160 |
| `src/vector_store.py` | Create | ~120 |
| `src/bm25_baseline.py` | Create | ~80 |
| `src/index_builder.py` | Create | ~150 |
| `tests/test_cache.py` | Create | ~60 |
| `tests/test_embedder.py` | Create | ~160 |
| `tests/test_vector_store.py` | Create | ~100 |
| `tests/test_bm25_baseline.py` | Create | ~70 |

**Existing files read but not modified**: `src/models.py`, `src/config.py`, `src/chunker.py`, `src/parser.py`

---

## Verification Plan

1. **Unit tests pass**: `uv run pytest tests/ -v` — all existing 204 + new ~35 tests green
2. **Coverage**: `uv run pytest --cov=src --cov-report=term-missing` — maintain >=95%
3. **Manual smoke test (Task 11 checkpoint)**:
   - Parse all 3 Kaggle Markdown files
   - Chunk into 5 configs
   - Embed with all 3 models → 15 FAISS indices saved to `data/output/indices/`
   - Build BM25 from Config B chunks → saved to `data/output/indices/`
   - **Manual query test**: search "total revenue" across a few indices → print **chunk text snippets + scores** (not just IDs) so relevance can be visually verified. Load chunk texts from saved `chunks_{config}.json` files and display first 200 chars of each result alongside its score.
   - Compare Config E retrieval vs Config B on same query — show both chunk texts side by side
4. **RAM monitoring**: Watch Activity Monitor during Task 11, confirm peak under 1GB

---

## Git Strategy

- Branch: `feat/p2-day2-embed-search`
- Commit after each task completes (6 commits)
- Push branch, create PR, merge to main
