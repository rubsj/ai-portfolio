# Plan: P5 Day 0 — Scaffold ShopTalk Knowledge Agent

## Context

P5 is the Production RAG project. The directory currently exists as `05-production-rag/` with CLAUDE.md, PRD.md, concepts-primer.html, and `data/pdf/README.md`. We need to:
- Rename to `05-shoptalk-knowledge-agent/` (matches PRD Section 9)
- Rename `data/pdf/` → `data/pdfs/` (matches PRD download instructions)
- Scaffold the full directory structure, pyproject.toml, empty source files, and test infrastructure
- Download the 4 test PDFs from arXiv
- Verify `uv sync` and `pytest` pass

## Steps

### 1. Create branch `feat/p5-day0-scaffold` from `origin/main`

### 2. Rename directory
- `git mv 05-production-rag 05-shoptalk-knowledge-agent`
- `git mv data/pdf data/pdfs` inside the new directory

### 3. Create directory structure (PRD Section 9)
```
05-shoptalk-knowledge-agent/
├── CLAUDE.md                 (existing — keep)
├── PRD.md                    (existing — keep)
├── concepts-primer.html      (existing — keep)
├── README.md                 (empty placeholder)
├── pyproject.toml            (new — full deps)
├── .env.example              (new — template)
├── .gitignore                (new — project-specific)
├── src/
│   ├── __init__.py
│   ├── schemas.py
│   ├── interfaces.py
│   ├── extraction.py
│   ├── vector_store.py
│   ├── generator.py
│   ├── pipeline.py
│   ├── experiment_runner.py
│   ├── visualization.py
│   ├── factories.py
│   ├── cache.py
│   ├── streamlit_app.py
│   ├── chunkers/
│   │   ├── __init__.py
│   │   ├── fixed.py
│   │   ├── recursive.py
│   │   ├── sliding_window.py
│   │   ├── heading_semantic.py
│   │   └── embedding_semantic.py
│   ├── embedders/
│   │   ├── __init__.py
│   │   ├── minilm.py
│   │   ├── mpnet.py
│   │   └── openai_embedder.py
│   ├── retrievers/
│   │   ├── __init__.py
│   │   ├── dense.py
│   │   ├── bm25.py
│   │   └── hybrid.py
│   ├── rerankers/
│   │   ├── __init__.py
│   │   ├── cohere_reranker.py
│   │   └── cross_encoder.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       ├── judge.py
│       └── ground_truth.py
├── scripts/
│   ├── ingest.py
│   ├── serve.py
│   └── evaluate.py
├── experiments/
│   └── configs/              (empty — YAML configs go here Day 3)
├── data/
│   ├── pdfs/
│   │   └── README.md         (existing — keep)
│   ├── indices/              (empty — FAISS indices go here)
│   ├── ground_truth.json     (empty placeholder)
│   └── cache/                (empty — LLM cache)
├── results/
│   ├── experiments/
│   ├── comparison/
│   └── charts/
├── tests/
│   ├── __init__.py
│   └── test_placeholder.py   (passes so pytest works)
└── docs/
    └── adr/                  (empty — ADRs start Day 1)
```

### 4. pyproject.toml

Dependencies from PRD Section 3a + CLAUDE.md patterns:

**Runtime:**
- `pymupdf` (PyMuPDF, `import fitz`)
- `sentence-transformers`
- `faiss-cpu` (Apple Silicon compatible)
- `rank-bm25`
- `litellm`
- `pydantic>=2.0`
- `streamlit`
- `cohere` (reranker API)
- `click` (CLI)
- `python-dotenv`
- `matplotlib`
- `seaborn`
- `numpy`
- `psutil` (memory monitoring)
- `pyyaml` (YAML config loading)
- `instructor` (structured LLM output)
- `tiktoken` (token counting)
- `rich` (progress bars)

**Dev:**
- `pytest`
- `pytest-cov>=7.0.0`
- `ruff`

Config:
- `requires-python = ">=3.12"`
- `[tool.pytest.ini_options] pythonpath = ["."]`
- `[tool.ruff] line-length = 100`

### 5. Source files — docstring-only stubs

Each `.py` file gets a module docstring describing its purpose (from PRD Section 9 comments). No implementation code. Example:

```python
"""Pydantic data models for the ShopTalk Knowledge Agent.

Models: Document, Chunk, RetrievalResult, QAResponse, Citation,
ExperimentConfig, ExperimentResult.
"""
```

Sub-package `__init__.py` files get a brief docstring. `src/__init__.py` is empty.

### 6. .env.example
```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...
```

### 7. .gitignore (project-specific)
```
data/pdfs/*.pdf
data/indices/
data/cache/
results/
.env
__pycache__/
*.pyc
.venv/
.coverage
coverage.json
```

### 8. Download test PDFs — SKIPPED (manual)
> Network is sandboxed. Developer will download the 4 PDFs manually using
> the curl commands in `data/pdfs/README.md`.

### 9. Verify
- `uv sync` passes (all deps resolve)
- `uv run pytest` passes (placeholder test)

### 10. Commit and push
- Stage all new/moved files
- Commit: `feat(p5): scaffold project structure, deps, and empty source stubs`
- Push branch, create PR

## Files to modify
- Rename: `05-production-rag/` → `05-shoptalk-knowledge-agent/`
- Rename: `data/pdf/` → `data/pdfs/`
- New: ~40 files (pyproject.toml, source stubs, sub-package stubs, test, .gitignore, .env.example)

## Verification
1. `uv sync` — all dependencies install without errors
2. `uv run pytest` — 1 test passes
3. `git status` — clean working tree after commit
4. PDFs downloaded manually later per `data/pdfs/README.md`
