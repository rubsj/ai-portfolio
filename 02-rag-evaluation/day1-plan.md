# P2 Day 1 Plan — Tasks #1-6: Foundation (Parse, Chunk, Embed Setup)

## Context

P2 is a RAG evaluation benchmarking framework. Day 1 builds the foundation: project init, data models, configuration, document parsing, and chunking. No code exists yet — only CLAUDE.md, PRD.md, and 3 Kaggle Markdown input files in `data/input/`.

Each task will be implemented one-at-a-time, with approval requested between tasks.

---

## Task 1: Install Dependencies + Project Init

**What**: Initialize the project with `uv`, install all dependencies, create directory structure.

**Key decision — Python 3.12 (not 3.14)**: `sentence-transformers` doesn't officially support Python 3.14 yet. P1 used `>=3.14` but had no ML deps. We'll pin P2 to 3.12 for compatibility with `sentence-transformers`, `faiss-cpu`, and `torch`.

**Steps**:
1. `uv init --python 3.12 --no-readme --bare` in `02-rag-evaluation/`
2. Add all deps from PRD Section 9 in groups (core, ML, eval, CLI, dev)
3. Add `[tool.pytest.ini_options] pythonpath = ["."]` to pyproject.toml (matches P1)
4. Create directories: `src/`, `tests/`, `data/cache/`, `data/output/`, `results/charts/`, `results/metrics/`, `results/reports/`, `docs/adr/`
5. Create `src/__init__.py` (empty)
6. Create `.env` with placeholder API keys
7. Verify: `uv sync` succeeds, key imports work

**Risk**: Dependency conflicts between `ragas`, `judges`, `instructor`. Mitigation: install in groups, isolate conflicting package if needed.

---

## Task 2: Pydantic Models (`src/models.py`)

**What**: All data models from PRD Section 4.

**Key decisions**:
- **`StrEnum`** for all enums (not `Literal` like P1, not plain `Enum`). StrEnum serializes to clean JSON strings while still being a proper Enum. Java parallel: TypeScript string enums.
- **Model ordering**: enums first, then leaf models, then composite models (bottom-up)
- **Cross-field validation**: `@model_validator(mode="after")` for ChunkConfig (overlap < chunk_size) and Chunk (end_char > start_char)

**Models to implement** (15 total):
1. Enums: `EmbeddingModel`, `RetrievalMethod`, `QuestionType`, `QuestionHierarchy`, `BloomLevel`
2. Config: `ChunkConfig` (with overlap < chunk_size validator)
3. Document: `Chunk` (with whitespace-only text validator, end > start validator)
4. QA: `SyntheticQAPair`, `QADatasetReport`
5. Retrieval: `RetrievalResult`, `ConfigEvaluation`, `RerankingComparison`
6. Generation: `RAGASResult`, `JudgeResult`
7. Report: `GridSearchReport`

**Pattern**: Follow P1's `01-synthetic-data-home-diy/src/schemas.py` — `from __future__ import annotations`, `Field()` with descriptions, `@field_validator` with `@classmethod`, WHY comments.

---

## Task 3: Configuration (`src/config.py`)

**What**: Centralize all constants — 5 ChunkConfigs, embedding model dimensions, file paths, env vars.

**Contents**:
- File paths via `pathlib.Path` (PROJECT_ROOT, DATA_DIR, INPUT_DIR, CACHE_DIR, OUTPUT_DIR, etc.)
- `load_dotenv()` for API keys
- 5 `ChunkConfig` instances (A-E) with `chunking_goal` annotations from PRD Section 3a
- Config E: `chunk_size=512` (subdivision threshold), `overlap=0`, `is_semantic=True`
- `EMBEDDING_DIMENSIONS` dict mapping `EmbeddingModel` to dimension count
- `ALL_CHUNK_CONFIGS`, `FIXED_CHUNK_CONFIGS`, `ALL_EMBEDDING_MODELS` lists
- LLM model constants

---

## Task 4: Document Parser (`src/parser.py`)

**What**: Factory pattern — parse PDF (PyMuPDF) or Markdown, return same `ParseResult` structure.

**Key data structures** (dataclasses, not Pydantic — internal pipeline objects):
- `HeaderInfo`: level, text, char_offset, page_number
- `ParseResult`: full_text, page_map (list of tuples), headers, source_path, num_pages

**Kaggle Markdown format handling**:
- Page separators: `{N}------------------------------------------------` (regex: `r'^\{(\d+)\}-{48}$'`)
- Strip HTML tags (`<span>`, `<br>`)
- Strip image refs (`![](_page_N_...)`)
- Extract all header levels (#, ##, ###, ####) with char offsets
- Build `page_map`: list of `(start_char, end_char, page_num)` in cleaned text

**Functions**:
- `parse_document(file_path) -> ParseResult` — factory dispatcher
- `_parse_markdown(file_path) -> ParseResult` — Kaggle format
- `_parse_pdf(file_path) -> ParseResult` — PyMuPDF
- `get_page_for_offset(page_map, char_offset) -> int` — binary search helper

---

## Task 5: Chunker (`src/chunker.py`)

**What**: Two strategies — fixed-size (A-D) and semantic (E) — both returning `list[Chunk]`.

**Fixed-size strategy (Configs A-D)**:
- `RecursiveCharacterTextSplitter` with `tiktoken` `length_function` (cl100k_base encoding)
- Char offset recovery: scan forward in full_text to find each chunk's position
- Page number assignment via `get_page_for_offset()`

**Semantic strategy (Config E)**:
- Split on `##` and `###` headers only (not `#` or `####`)
- Merge sections <32 tokens with next section
- Subdivide sections >512 tokens using Config B params (256/64)
- Fall back to Config B if no `##`/`###` headers found
- Populate `section_header` field

**Functions**:
- `chunk_document(parse_result, config) -> list[Chunk]` — dispatcher
- `_chunk_fixed_size(parse_result, config) -> list[Chunk]`
- `_chunk_semantic(parse_result, config) -> list[Chunk]`
- `_count_tokens(text) -> int` — tiktoken wrapper
- `_assign_page_numbers(start_char, end_char, page_map) -> list[int]`
- `_find_char_offset(chunk_text, full_text, search_start) -> int`

---

## Task 6: Tests (`test_models.py`, `test_chunker.py`)

**test_models.py**:
- Valid/invalid data for `ChunkConfig`, `Chunk`, `SyntheticQAPair`, enums
- Cross-field validators: overlap >= chunk_size, end_char <= start_char, whitespace-only text
- JSON roundtrip for key models
- Parametrize edge cases

**test_chunker.py**:
- Small synthetic markdown fixture (not real 1344-line file — fast, deterministic)
- Config A produces more chunks than Config B (smaller size = more chunks)
- Config E chunks align with headers, `section_header` populated
- Config E produces fewer chunks than Config A
- Chunk IDs follow `{config}_{index}` convention
- All chunks have positive token_count and valid page_numbers

---

## Verification

After all 6 tasks:
```bash
cd 02-rag-evaluation
uv run pytest tests/ -v         # All tests pass
uv run python -c "
from src.parser import parse_document
from src.chunker import chunk_document
from src.config import ALL_CHUNK_CONFIGS
from pathlib import Path

result = parse_document(Path('data/input/financial_services.md'))
for cfg in ALL_CHUNK_CONFIGS:
    chunks = chunk_document(result, cfg)
    print(f'Config {cfg.name}: {len(chunks)} chunks')
"
# Expected: A most, C fewest, E variable (section-based)
```

Then: git commit + push on a feature branch, create PR.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `02-rag-evaluation/pyproject.toml` | Create (uv init + deps) |
| `02-rag-evaluation/.env` | Create (placeholder API keys) |
| `02-rag-evaluation/src/__init__.py` | Create (empty) |
| `02-rag-evaluation/src/models.py` | Create (15 Pydantic models) |
| `02-rag-evaluation/src/config.py` | Create (constants, configs, paths) |
| `02-rag-evaluation/src/parser.py` | Create (factory parser) |
| `02-rag-evaluation/src/chunker.py` | Create (2 strategies) |
| `02-rag-evaluation/tests/test_models.py` | Create |
| `02-rag-evaluation/tests/test_chunker.py` | Create |

## Reusable Patterns from P1

| Pattern | Source | Usage in P2 |
|---------|--------|-------------|
| Pydantic model structure | `01-synthetic-data-home-diy/src/schemas.py` | models.py — Field(), validators, docstrings |
| pytest config | `01-synthetic-data-home-diy/pyproject.toml` | pyproject.toml — `pythonpath = ["."]` |
