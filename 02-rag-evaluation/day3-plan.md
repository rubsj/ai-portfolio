# P2 Day 3 Execution Plan — Synthetic QA + Full Grid Search

## Context

Day 3 is a Sunday deep work session (6-8h). Days 1-2 built the full retrieval infrastructure: 10 modules, 268 tests, 16 searchable indices (15 FAISS + 1 BM25), 5 chunk data files. Today we generate synthetic QA, build the evaluation pipeline, and run the first full grid search to produce the config × metric heatmap.

**Goal**: Produce the sentence: _"Config X achieved Y Recall@5, outperforming BM25 and all other configurations."_

**Critical path**: Task 13 → 15 → 16 → 18 (must complete)
**Secondary path**: Task 14 → 17 → 19 (complete if time allows)

**Branch**: `feat/p2-day3-qa-grid-search`

---

## Key Design Decisions

### 1. Config B as QA reference config

Generate all QA pairs from Config B chunks (256/64 tokens). Rationale:
- Industry baseline chunk size
- BM25 uses the same chunks — no gold mapping needed for BM25 evaluation
- Moderate count (~589 chunks) — not too many (A=~1222) or few (C=~284)

### 2. Cross-config gold chunk mapping

**Verified**: `Chunk` model has `start_char` (line 131) and `end_char` (line 135) fields in `models.py`. All chunk JSON files include these fields. No schema changes needed.

QA pairs have `gold_chunk_ids` in Config B namespace (e.g., `B_0_42`). When evaluating retrieval on Config A/C/D/E, we map gold chunks using **character position overlap**:

```
For each gold B chunk, find target config chunks where:
  overlap = max(0, min(gold.end_char, target.end_char) - max(gold.start_char, target.start_char))
  overlap_ratio = overlap / (gold.end_char - gold.start_char)
  if overlap_ratio >= 0.5 → target chunk is "gold"
```

Only compare chunks from the same document (extract `doc_idx` from chunk ID: `B_{doc_idx}_{chunk_idx}`).

### 3. Strategy 2 embeddings from FAISS

Extract pre-computed embeddings from `minilm_B.faiss` using `index.reconstruct_n(0, n)`. IndexFlatIP stores full vectors, so reconstruction is exact. No need to load any embedding model.

### 4. Memory-efficient grid search

One embedding model at a time: load → embed all questions → search all 5 chunk configs → unload → `gc.collect()`. Same pattern as `index_builder.py`.

### 5. `INDICES_DIR` path constant

Currently defined only in `index_builder.py`. Add `INDICES_DIR = OUTPUT_DIR / "indices"` to `config.py` so grid_search.py and synthetic_qa.py can import it cleanly.

### 6. `model_key()` moved to config.py

`_model_key()` is currently a private function in `index_builder.py:358`. Move to `config.py` as a **public** function `model_key(model: EmbeddingModel) -> str`. Update `index_builder.py` to import from `config.py` instead. Don't import private functions across modules.

### 7. `RETRIEVAL_TOP_N = 10`

Change `RETRIEVAL_TOP_N` from 20 to **10** in `config.py`. Grid search retrieves top-10 for evaluation (no reranking stage in Day 3). `RERANK_TOP_N` stays at 5 for Day 4.

### 8. pandas dependency

Add `pandas` via `uv add pandas` — needed for clean seaborn heatmap DataFrames.

---

## Task 13: Synthetic QA Generation

**File**: `src/synthetic_qa.py` (~350 lines)

**Public API**:
- `generate_synthetic_qa(chunks: list[Chunk], *, use_cache: bool = True) -> list[SyntheticQAPair]`
- `compute_qa_quality(questions: list[SyntheticQAPair], total_chunks: int) -> QADatasetReport`
- `save_qa_pairs(pairs: list[SyntheticQAPair], path: Path | None = None) -> Path`
- `load_qa_pairs(path: Path | None = None) -> list[SyntheticQAPair]`

**Instructor setup**: `instructor.from_openai(OpenAI())` — same P1 pattern from `01-synthetic-data-home-diy/src/generator.py`.

**Internal response models** (Pydantic models for Instructor, NOT in models.py):
- `QuestionChainResponse` — 3 question/answer pairs for Strategy 1
- `MultiChunkQuestionResponse` — single Q/A for Strategy 2
- `SingleQuestionResponse` — reusable for Strategies 3, 4, 5

**Caching wrapper**: `_cached_instructor_call(client, prompt, response_model, use_cache)` — checks `cache.py` before calling LLM, saves after. Key = `compute_cache_key("gpt-4o-mini", prompt)`.

**`source_chunk_text` requirement** (PRD Section 4 — needed by Day 4 RAGAS faithfulness + Day 5 Streamlit display):
Every strategy MUST populate `source_chunk_text` on `SyntheticQAPair`:
- **Single-chunk strategies (1, 3, 5)**: `source_chunk_text = chunk.text`
- **Multi-chunk strategies (2, 4)**: `source_chunk_text = "\n---\n".join(c.text for c in relevant_chunks)`

### Strategy 1 — Per-Chunk Question Chains (~24 questions)

1. `_sample_diverse_chunks(chunks, n=8)` — spread across 3 documents (3 + 3 + 2), sampling at beginning/middle/end positions
2. For each chunk, prompt asks for 3 progressive questions: factual → analytical → connective
3. Instructor validates `QuestionChainResponse`, auto-retries on validation failure
4. Convert to 3 `SyntheticQAPair` objects:
   - Factual: `question_type=FACTUAL`, `hierarchy=PARAGRAPH`
   - Analytical: `question_type=ANALYTICAL`, `hierarchy=PARAGRAPH`
   - Connective: `question_type=MULTI_HOP`, `hierarchy=SECTION`
5. `gold_chunk_ids=[chunk.id]`, `generation_strategy="per_chunk_chain"`, `source_chunk_text=chunk.text`

### Strategy 2 — Multi-Chunk Questions (~10 questions)

1. `_load_precomputed_embeddings()` — load from `minilm_B.faiss`:
   ```python
   import faiss
   index = faiss.read_index(str(INDICES_DIR / "minilm_B.faiss"))
   embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
   index.reconstruct_n(0, index.ntotal, embeddings)
   chunk_ids = json.loads((INDICES_DIR / "minilm_B.json").read_text())
   ```
2. `_find_semantically_similar_chunks(source_idx, embeddings, chunk_ids, top_k=3)` — dot product similarity, exclude self, return top-k indices
3. Sample 10 source chunks, find 3 similar chunks each
4. Prompt: "Generate a question requiring info from ALL these chunks"
5. `gold_chunk_ids = [source.id, *similar_ids]`, `hierarchy=SECTION`, `generation_strategy="multi_chunk"`, `source_chunk_text="\n---\n".join(texts of source + similar chunks)`

### Strategy 3 — Overlap Region Questions (~8 questions)

1. `_find_overlap_pairs(chunks)` — find consecutive chunk pairs (same doc) where `c1.end_char > c2.start_char`
2. Sample 8 pairs with largest overlap zones
3. Extract overlap text from the shared character range
4. Prompt: generate question about the overlap content
5. `gold_chunk_ids=[c1.id, c2.id]`, `is_overlap_region=True`, `hierarchy=PARAGRAPH`, `generation_strategy="overlap_region"`, `source_chunk_text=overlap_text` (the shared character range extracted in step 3)

### Strategy 4 — Hierarchical Questions (~8 questions)

1. **Paragraph** (3 Qs): 1 chunk, `hierarchy=PARAGRAPH`
2. **Section** (3 Qs): 2-4 consecutive chunks, `hierarchy=SECTION`
3. **Page** (2 Qs): 5+ consecutive chunks, `hierarchy=PAGE`, `question_type=SUMMARIZATION`
4. `generation_strategy="hierarchical"`, `source_chunk_text="\n---\n".join(texts of all chunks in the group)`

### Strategy 5 — Academic Pattern Questions (~6 questions)

1. Define `ACADEMIC_PATTERNS: dict[QuestionType, list[str]]` with template strings per PRD Section 5d
2. Sample 6 chunks, one per template pattern
3. Prompt: LLM fills `{concept}`, `{topic}` slots from chunk content and provides answer
4. `generation_strategy="academic_pattern"`, `source_chunk_text=chunk.text`

**Output**: Save to `data/output/qa_pairs.json`. Target: 24+10+8+8+6 = **56 questions** (≥50 threshold).

**Commit**: `feat(p2): add synthetic QA generation with 5 strategies`

---

## Sanity Check (after Task 13, before full grid search)

**Purpose**: Catch bad QA pairs early before burning 3-5 min on full grid search.

**Steps**:
1. Pick 3 QA pairs from the generated set (one per-chunk-chain, one multi-chunk, one overlap)
2. Load `minilm_B.faiss` + `minilm_B.json` via `FAISSVectorStore.load()`
3. Embed the 3 questions using MiniLM embedder
4. Search Config B index with `k=10`
5. Check: for each question, is at least one `gold_chunk_id` in the top-10 results?
6. **Pass gate**: Recall@10 > 0 for at least 2 out of 3 questions
7. **If all zeros**: STOP. Debug QA quality or gold chunk assignment before proceeding.

This runs inside `grid_search.py` as a `sanity_check(qa_pairs, n=3)` function, called from `__main__` before `run_grid_search()`.

---

## Task 15: Retrieval Evaluator

**File**: `src/retrieval_evaluator.py` (~120 lines)

**Pure computation — no LLM calls, no I/O, no external deps.**

**Functions**:
- `compute_recall_at_k(gold_ids, retrieved_ids, k)` — `|gold ∩ top_k| / |gold|`
- `compute_precision_at_k(gold_ids, retrieved_ids, k)` — `|gold ∩ top_k| / k`
- `compute_mrr_at_k(gold_ids, retrieved_ids, k)` — `1/rank` of first gold in top-k, else 0
- `evaluate_single_question(qa_pair, retrieved, gold_chunk_ids) -> RetrievalResult` — computes all 9 metrics (R/P/MRR @ 1,3,5)
- `evaluate_config(qa_pairs, retrieval_results, gold_ids_per_q, config_id, ...) -> ConfigEvaluation` — averages across questions + breakdown by `QuestionType`

**Key**: `retrieved` comes as `list[tuple[str, float]]` from search — extract just the chunk IDs for metric computation.

**Commit**: `feat(p2): add retrieval evaluator (R/P/MRR @1,3,5)`

---

## Task 16: Grid Search Orchestrator

**File**: `src/grid_search.py` (~250 lines)

**Functions**:
- `map_gold_chunks(gold_b_ids, b_chunks_lookup, target_chunks, overlap_threshold=0.5) -> list[str]` — character position overlap mapping
- `run_grid_search(qa_pairs, *, b_chunks=None) -> list[ConfigEvaluation]` — full 16-config evaluation
- `save_grid_results(evaluations) -> Path` — save to `results/metrics/grid_search_results.json`
- `_load_all_chunks() -> dict[str, list[Chunk]]` — load chunks_{A-E}.json

**Execution loop** (memory-efficient, BM25 separate):
```
Load all chunk files into memory (dict of config→chunks)
Build b_chunks_lookup = {chunk.id: chunk for chunk in chunks["B"]}

# ── Phase 1: Vector search (15 configs) ──
For each embedding model in [MINILM, MPNET, OPENAI]:
  embedder = create_embedder(model)
  query_embeddings = embedder.embed([qa.question for qa in qa_pairs])  # embed all once

  For each chunk config in [A, B, C, D, E]:
    store = FAISSVectorStore.load(INDICES_DIR / f"{model_key}_{config}")

    all_results = [store.search(query_embeddings[i], k=RETRIEVAL_TOP_N) for i, qa in enumerate(qa_pairs)]

    if config == "B":
      mapped_golds = [qa.gold_chunk_ids for qa in qa_pairs]
    else:
      mapped_golds = [map_gold_chunks(qa.gold_chunk_ids, b_chunks_lookup, chunks[config]) for qa in qa_pairs]

    config_eval = evaluate_config(...)
    evaluations.append(config_eval)

  del embedder; gc.collect()

# ── Phase 2: BM25 (1 config, runs exactly once, OUTSIDE embedding loop) ──
bm25 = BM25Retriever.load(INDICES_DIR / "bm25_B")
all_results = [bm25.search(qa.question, k=RETRIEVAL_TOP_N) for qa in qa_pairs]
mapped_golds = [qa.gold_chunk_ids for qa in qa_pairs]  # Config B — no mapping needed
config_eval = evaluate_config(..., config_id="bm25")
evaluations.append(config_eval)
```

**Config ID naming**: `{config}-{model_key}` (e.g., `B-minilm`, `E-openai`, `bm25`)

**Imports from config.py**: `model_key()` (public function, moved from `index_builder.py`).

**Commit**: `feat(p2): add grid search orchestrator with cross-config gold mapping`

---

## Task 18: Run Full Grid Search

Add `if __name__ == "__main__"` block to `grid_search.py`:

1. Load QA pairs from `data/output/qa_pairs.json`
2. Run sanity check (3 QA pairs vs Config B + MiniLM)
3. Run `run_grid_search(qa_pairs)` — produces 16 `ConfigEvaluation` objects
4. Save results to `results/metrics/grid_search_results.json`
5. Print summary (see format below)
6. Generate charts (if Task 17 done)

**Print summary format**:

```
=== Top-3 Configs by Recall@5 ===
#1  B-openai:  Recall@5=0.XX, P@5=0.XX, MRR@5=0.XX
#2  D-mpnet:   Recall@5=0.XX, P@5=0.XX, MRR@5=0.XX
#3  B-minilm:  Recall@5=0.XX, P@5=0.XX, MRR@5=0.XX

=== BM25 Baseline ===
bm25:  Recall@5=0.XX (rank #N/16)
Best vector config beats BM25 by X.X%

=== Config E (Semantic) Analysis ===
E-minilm: Recall@5=0.XX (rank #N/16)
E-mpnet:  Recall@5=0.XX (rank #N/16)
E-openai: Recall@5=0.XX (rank #N/16)
Best fixed-size comparator (B-{model}): Recall@5=0.XX
Delta: Config E [beats/loses to] Config B by X.X%
```

The Config E section ALWAYS prints, regardless of where E ranks. This is the core portfolio talking point — "semantic chunking did/didn't beat fixed-size, by how much, on which metrics."

**Expected runtime**: ~3-5 min (dominated by embedding 56 questions × 3 models).

**Checkpoint verification**:
- 16 ConfigEvaluations produced
- At least one vector config beats BM25 on Recall@5
- Results saved to disk
- Top-3 configs identified
- Config E analysis printed with delta vs Config B

**Commit**: `feat(p2): run full grid search, generate charts and results`

---

## Task 14: QA Quality Report (secondary)

Part of `synthetic_qa.py` — the `compute_qa_quality()` function.

Compute from generated QA pairs:
- `total_questions`: count
- `questions_per_strategy`: Counter by `generation_strategy`
- `questions_per_type`: Counter by `question_type`
- `questions_per_hierarchy`: Counter by `hierarchy`
- `chunk_coverage_percent`: `len(unique gold chunk IDs) / total_chunks * 100`
- `overlap_question_count`: count where `is_overlap_region=True`
- `avg_questions_per_chunk`: `total_questions / total_chunks`

Save to `results/reports/qa_dataset_report.json`.

---

## Task 17: Visualization (secondary)

**File**: `src/visualization.py` (~200 lines)

**Dependency**: add `pandas` via `uv add pandas`

4 charts for Day 3 checkpoint:

| Chart | Description | File |
|-------|-------------|------|
| Config × Metric Heatmap | 16 rows × 7 columns (R@1,3,5 P@1,3,5 MRR@5), seaborn heatmap | `results/charts/config_heatmap.png` |
| Metric Bar Chart | Grouped bars — all 16 configs on R@5, P@5, MRR@5 | `results/charts/metric_comparison.png` |
| BM25 vs Vector | Side-by-side bars — BM25 vs best vector config | `results/charts/bm25_comparison.png` |
| Semantic vs Fixed-Size | Config E vs Config B per embedding model | `results/charts/semantic_vs_fixed.png` |

Style: `matplotlib` + `seaborn`, `dpi=150`, `bbox_inches="tight"`.

---

## Task 19: Tests (secondary)

### `tests/test_retrieval_evaluator.py` (~120 lines)

Pure computation tests, no mocking needed:
- `TestRecallAtK`: perfect recall, zero recall, partial recall multi-gold, parametrize K=[1,3,5]
- `TestPrecisionAtK`: perfect, decreasing with K, zero
- `TestMRRAtK`: gold at rank 1/2/3, no gold → 0.0, parametrize
- `TestEvaluateConfig`: averages correct, per-question-type breakdown correct

### `tests/test_synthetic_qa.py` (~180 lines)

Mock all LLM calls (Instructor + OpenAI):
- `TestFindSemanticallySimilarChunks`: pure numpy test — correct top-k, excludes self
- `TestFindOverlapPairs`: consecutive chunks with positive overlap found, doc boundaries skipped
- `TestComputeQAQuality`: total count, coverage %, strategy distribution, overlap count
- `TestStrategyPerChunkChain`: produces 3 Qs per chunk, correct types, gold matches source
- Strategy tests: mock Instructor client, verify output structure

---

## Config.py Additions

Three changes to `config.py`:

```python
# 1. Centralize indices path (was only in index_builder.py)
INDICES_DIR = OUTPUT_DIR / "indices"

# 2. Public model_key function (was _model_key in index_builder.py)
def model_key(model: EmbeddingModel) -> str:
    """Convert EmbeddingModel enum to short key for file naming."""
    return {
        EmbeddingModel.MINILM: "minilm",
        EmbeddingModel.MPNET: "mpnet",
        EmbeddingModel.OPENAI: "openai",
    }[model]

# 3. Change RETRIEVAL_TOP_N from 20 → 10
RETRIEVAL_TOP_N = 10
```

Also update `index_builder.py` to import `model_key` from `config.py` and delete its local `_model_key()` function.

---

## ADR Plan

Write after grid search results are available (Task 18):

| ADR | Title | When |
|-----|-------|------|
| ADR-002 | Chunk size selection + overlap experiment + semantic chunking | After grid search results |
| ADR-003 | Embedding model comparison — local vs API, cost vs quality | After grid search results |
| ADR-004 | Synthetic QA generation — 5 strategies, Instructor validation | After QA generation (Task 13) |

Format: follow existing ADR-001 template with Context → Decision → Alternatives → Consequences → Java/TS Parallel.

---

## Git Plan

**Branch**: `feat/p2-day3-qa-grid-search`

Commit sequence:
1. `chore(p2): add INDICES_DIR, model_key(), RETRIEVAL_TOP_N=10 to config, add pandas`
2. `feat(p2): add synthetic QA generation with 5 strategies`
3. `feat(p2): add retrieval evaluator (R/P/MRR @1,3,5)`
4. `feat(p2): add grid search orchestrator with cross-config gold mapping`
5. `feat(p2): run full grid search, generate charts and results`
6. `test(p2): add QA generation and retrieval evaluator tests`
7. `docs(p2): add ADR-002, ADR-003, ADR-004`

Push → PR to main.

---

## Verification Checklist

- [ ] ≥50 QA pairs generated with all 5 strategies
- [ ] All 5 QuestionTypes and all 3 QuestionHierarchy levels present
- [ ] Sanity check passes: ≥2/3 QA pairs have Recall@10 > 0 on Config B + MiniLM
- [ ] 16 ConfigEvaluations in `results/metrics/grid_search_results.json`
- [ ] At least one vector config beats BM25 on Recall@5
- [ ] Config heatmap at `results/charts/config_heatmap.png`
- [ ] Top-3 configs identified by Recall@5
- [ ] `uv run pytest` — all tests pass (existing 268 + new)
- [ ] QA quality report at `results/reports/qa_dataset_report.json`
