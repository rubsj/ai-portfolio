# P5 Day 1 Plan — Foundation: Schemas, Interfaces, PDF Extraction, and Chunking

## Context

P5 (ShopTalk Knowledge Agent) is a production RAG system built from first principles — no LangChain. Day 1 builds the foundation layer: Pydantic data models, abstract interfaces (ABCs), PDF extraction via PyMuPDF, and 5 chunking strategies. All files exist as docstring-only stubs from Day 0 scaffolding.

**Branch:** `feat/p5-day1-foundation` (from `origin/main`)

**Exit criteria:**
- All 5 chunkers produce valid `Chunk` objects from test PDFs
- `pytest` passes with ≥95% coverage on schemas + chunkers
- ADR-001 and ADR-002 written and committed

---

## Implementation Order

```
 1. src/schemas.py              ← no deps, everything imports from here
 2. src/interfaces.py           ← imports schemas
 3. tests/test_schemas.py       ← validate schemas before building on them
 4. src/extraction.py           ← imports schemas
 5. tests/test_extraction.py    ← validate extraction
 6. src/chunkers/_utils.py      ← NEW shared helper (find_page_number)
 7. src/chunkers/fixed.py       ← imports schemas + interfaces + _utils
 8. src/chunkers/recursive.py
 9. src/chunkers/sliding_window.py
10. src/chunkers/heading_semantic.py
11. src/chunkers/embedding_semantic.py
12. src/chunkers/__init__.py    ← re-exports all 5 chunkers
13. tests/test_chunkers.py      ← all 5 chunkers tested
14. docs/adr/ADR-001.md         ← FAISS over ChromaDB
15. docs/adr/ADR-002.md         ← No LangChain — first-principles RAG
16. Delete tests/test_placeholder.py
```

---

## File 1: `src/schemas.py` — 13 Pydantic Models

All models use: `from __future__ import annotations`, Pydantic v2, `Field()` with descriptions, `ConfigDict(arbitrary_types_allowed=True)` where numpy is used.

### Models (in dependency order)

| Model | Key Fields | Validators |
|-------|-----------|------------|
| `PageInfo` | `page_number: int (ge=0)`, `text: str`, `char_count: int (ge=0)` | — |
| `DocumentMetadata` | `source: str (min_length=1)`, `title: str`, `author: str`, `page_count: int (ge=1)` | — |
| `Document` | `id: str (uuid4 default)`, `content: str (min_length=1)`, `metadata: DocumentMetadata`, `pages: list[PageInfo] (min_length=1)` | — |
| `ChunkMetadata` | `document_id`, `source`, `page_number`, `start_char`, `end_char`, `chunk_index` (all ge=0) | `@model_validator`: end_char >= start_char |
| `Chunk` | `id: str (uuid4)`, `content: str (min_length=1)`, `metadata: ChunkMetadata`, `embedding: np.ndarray \| None` | `ConfigDict(arbitrary_types_allowed=True)` |
| `RetrievalResult` | `chunk: Chunk`, `score: float`, `retriever_type: RetrieverType`, `rank: int (ge=1)` | — |
| `Citation` | `chunk_id`, `source`, `page_number`, `text_snippet`, `relevance_score: float (ge=0, le=1)` | — |
| `QAResponse` | `query`, `answer`, `citations: list[Citation]`, `chunks_used: list[Chunk]`, `confidence: float \| None`, `latency: float (ge=0)` | — |
| `ExperimentConfig` | `chunking_strategy: ChunkingStrategy`, `chunk_size`, `chunk_overlap`, `embedding_model`, `retriever_type`, `hybrid_alpha`, `use_reranking`, `reranker_type`, `top_k`, `window_size_tokens`, `step_size_tokens`, `breakpoint_threshold`, `min_chunk_size` | 4 `@model_validator`s: hybrid_alpha required iff hybrid; reranker_type required iff use_reranking; embedding_model required for dense/hybrid, None for bm25; sliding_window requires window/step. 1 `@field_validator`: overlap < size |
| `RetrievalMetrics` | `recall_at_5`, `precision_at_5`, `mrr`, `ndcg_at_5` (all ge=0, le=1) | — |
| `JudgeScores` | 5 axes + `overall_average` (all ge=1, le=5) | — |
| `PerformanceMetrics` | `ingestion_time_seconds`, `avg_query_latency_ms`, `index_size_bytes`, `peak_memory_mb` | — |
| `ExperimentResult` | `experiment_id`, `config: ExperimentConfig`, `metrics: RetrievalMetrics`, `judge_scores: JudgeScores \| None`, `performance: PerformanceMetrics` | — |

**Type aliases:** `RetrieverType = Literal["dense", "bm25", "hybrid"]`, `ChunkingStrategy = Literal[...]`, `EmbeddingModel = Literal["minilm", "mpnet", "openai"]`, `RerankerType = Literal["cohere", "cross_encoder"]`

---

## File 2: `src/interfaces.py` — 6 ABCs

All use `from abc import ABC, abstractmethod`. Each has Java parallel in docstring.

| ABC | Methods |
|-----|---------|
| `BaseChunker` | `chunk(document: Document) -> list[Chunk]` |
| `BaseEmbedder` | `embed(texts: list[str]) -> np.ndarray`, `embed_query(query: str) -> np.ndarray`, `@property dimensions -> int` |
| `BaseVectorStore` | `add(chunks, embeddings)`, `search(query_embedding, top_k) -> list[tuple[Chunk, float]]`, `save(path)`, `load(path)` |
| `BaseRetriever` | `retrieve(query: str, top_k: int) -> list[RetrievalResult]` |
| `BaseReranker` | `rerank(query, results, top_k) -> list[RetrievalResult]` |
| `BaseLLM` | `generate(prompt, system_prompt="", temperature=0.0) -> str` |

---

## File 3: `src/extraction.py` — PDF Extraction

### Functions

1. **`clean_text(text: str) -> str`** — Ligature replacement (ﬁ→fi, ﬂ→fl, ﬃ→ffi, ﬄ→ffl, ﬀ→ff), hyphenation rejoin (`re-\njoin` → `rejoin`), collapse excessive whitespace. All regexes pre-compiled at module level.

2. **`_is_header_or_footer(line, page_number, total_pages) -> bool`** — Heuristics: standalone digits, "Page N", "N of M", "arXiv:", short ALL-CAPS lines.

3. **`remove_headers_footers(text, page_number, total_pages) -> str`** — Check first 3 and last 3 lines only. Skip pages with ≤6 lines.

4. **`extract_pdf(pdf_path: str | Path) -> Document`** — `fitz.open()`, page-by-page extraction, clean, build `Document`. Read metadata before `doc.close()`.

5. **`extract_all_pdfs(pdf_dir: str | Path) -> list[Document]`** — Glob `*.pdf`, extract each.

---

## File 4: `src/chunkers/_utils.py` — NEW Shared Helper

```python
def find_page_number(document: Document, char_offset: int) -> int:
    """Map char offset to page number. Accounts for \\n\\n separators between pages."""
```

All 5 chunkers import this instead of reimplementing.

---

## Files 5-9: 5 Chunking Strategies

### FixedSizeChunker (`fixed.py`)
- `__init__(chunk_size=512, chunk_overlap=50)` — validate overlap < size
- `chunk()` — character-based sliding window, step = size - overlap
- Uses `find_page_number()` for metadata

### RecursiveChunker (`recursive.py`)
- `__init__(chunk_size=512, chunk_overlap=50, separators=None)`
- Default separators: `["\n\n", "\n", ". ", " ", ""]`
- `_split_text(text, separators)` — try largest separator, recurse with smaller if segment too large
- `_merge_with_overlap(chunks)` — prepend last N chars from previous chunk

### SlidingWindowChunker (`sliding_window.py`)
- `__init__(window_size=200, step_size=150, encoding_name="cl100k_base")`
- Uses `tiktoken` for token counting — encode → window → decode
- Token-to-char offset mapping for metadata

### HeadingSemanticChunker (`heading_semantic.py`)
- `__init__(heading_patterns=None, min_chunk_size=50, max_chunk_size=3000)`
- 4 pre-compiled regex patterns: markdown `#`, numbered sections, ALL-CAPS, academic section names
- `_find_heading_boundaries()` → `_split_at_boundaries()` → `_split_oversized()` for large sections
- Heading line included at start of its chunk

### EmbeddingSemanticChunker (`embedding_semantic.py`)
- `__init__(breakpoint_threshold=0.85, min_chunk_size=100)`
- **ALWAYS uses MiniLM** (`all-MiniLM-L6-v2`) for boundary detection
- Algorithm: sentence-split → embed → consecutive cosine similarities → split where < threshold → merge small chunks
- `del model; gc.collect()` after embedding (8GB constraint)
- `_consecutive_cosine_similarities()` — normalize + dot product
- `_group_sentences()` — split at breakpoint indices
- `_merge_small_chunks()` — merge fragments below min_chunk_size

### `src/chunkers/__init__.py`
Re-export all 5 chunker classes via `__all__`.

---

## Tests

### `tests/test_schemas.py` (~50-60 tests)
- Inline `_make_*()` builders for each model (no conftest)
- Happy path creation for all 13 models
- Validation error cases: empty content, end_char < start_char, overlap >= size, hybrid without alpha, reranker without use_reranking, bm25 with embedding_model, etc.
- `@pytest.mark.parametrize` for retriever_type, chunking_strategy Literal values

### `tests/test_extraction.py` (~20-25 tests)
- **Unit tests** (no PDFs): `TestCleanText` (ligatures, hyphenation, whitespace), `TestIsHeaderOrFooter` (page numbers, arXiv, normal text), `TestRemoveHeadersFooters`
- **Integration tests** (require PDFs in `data/pdfs/`): `TestExtractPdf` — extract each paper, verify page count, content keywords, Document structure. Mark with `@pytest.mark.skipif` if PDFs not present.

### `tests/test_chunkers.py` (~50 tests)
- Shared `_make_test_document()` builder with configurable content/pages
- **TestFixedSizeChunker**: overlap verification, metadata offsets, sequential indices, edge cases
- **TestRecursiveChunker**: paragraph splits, fallback separators, content preservation
- **TestSlidingWindowChunker**: token-based windows, char offset mapping
- **TestHeadingSemanticChunker**: markdown/numbered/CAPS/academic heading detection, oversized splitting
- **TestEmbeddingSemanticChunker**: **Mock `SentenceTransformer`** — craft embeddings to control where boundaries fall. Verify model loaded with correct name, verify `del` called. Test `_consecutive_cosine_similarities`, `_group_sentences`, `_merge_small_chunks` as unit methods.

---

## ADRs

### ADR-001: FAISS over ChromaDB
- Context: P4 used ChromaDB for REST API. P5 needs 35+ experiment configs with different embedding dimensions, explicit index management.
- Decision: FAISS (faiss-cpu) with IndexFlatIP. Separate .faiss + .json persistence.
- Why not ChromaDB: abstracts too much, global singleton issues (P4 lesson), wrong granularity for experiments.
- Interview signal: different tools for different jobs — production API vs experiment platform.

### ADR-002: No LangChain — First-Principles RAG
- Context: LangChain provides pre-built splitters/chains. P5's purpose is demonstrating first-principles understanding.
- Decision: Build all components behind ABCs. 6 interfaces, 5 chunkers, 3 retrievers.
- Why not LangChain: hides internals, portfolio needs to show understanding not configuration skill.
- Interview signal: "I can use LangChain, but I chose first-principles to prove I understand the internals."

---

## Verification Checklist

1. `uv run pytest tests/test_schemas.py tests/test_extraction.py tests/test_chunkers.py -v`
2. `uv run pytest --cov=src/schemas --cov=src/interfaces --cov=src/extraction --cov=src/chunkers --cov-report=term-missing` → ≥95%
3. Sanity REPL: extract a PDF, run all 5 chunkers, print chunk counts
4. ADR-001.md and ADR-002.md exist in `docs/adr/`
5. All chunk.content is non-empty, chunk.metadata.document_id matches document.id, chunk_index is sequential
6. `git diff --stat` shows only files in the implementation order above
