"""Centralized configuration for the RAG evaluation benchmarking framework.

All constants, file paths, chunk configs, embedding dimensions, and env vars
live here — single source of truth. Import from this module, never hardcode.

Java/TS parallel: like a Spring @Configuration class or a Next.js config file
that centralizes environment variables, feature flags, and resource paths.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.models import ChunkConfig, EmbeddingModel

# ===========================================================================
# File Paths — pathlib.Path for cross-platform safety
# Java/TS parallel: like `path.resolve(__dirname, '..')` in Node.js
# WHY pathlib over os.path: cleaner API, operator overloading (`/`), type-safe.
# ===========================================================================

# WHY .resolve(): ensures absolute path regardless of where the script is run
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = DATA_DIR / "output"

# WHY separate indices dir: keeps 16+ index files (FAISS + BM25) separate
# from other output files (chunk JSONs, QA pairs). Clean directory structure.
INDICES_DIR = OUTPUT_DIR / "indices"

RESULTS_DIR = PROJECT_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"

DOCS_DIR = PROJECT_ROOT / "docs"
ADR_DIR = DOCS_DIR / "adr"


# ===========================================================================
# Environment Variables — loaded from .env via python-dotenv
# WHY load_dotenv() at module level: ensures env vars are available as soon
# as config.py is imported, before any API client is created.
# Java/TS parallel: like dotenv.config() in Node.js or Spring's @Value.
# ===========================================================================

load_dotenv(PROJECT_ROOT / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
BRAINTRUST_API_KEY = os.getenv("BRAINTRUST_API_KEY", "")


# ===========================================================================
# Chunk Configurations (A-E) — PRD Section 3a
# Each config tests a different chunking hypothesis. chunking_goal maps
# directly to the Notion requirements spec.
# ===========================================================================

CONFIG_A = ChunkConfig(
    name="A",
    chunk_size=128,
    overlap=32,
    chunking_goal="Preserve semantic units (100-300 tokens) — max retrieval granularity",
    is_semantic=False,
)

CONFIG_B = ChunkConfig(
    name="B",
    chunk_size=256,
    overlap=64,
    chunking_goal="Enable dense search (256-512 tokens) — industry baseline",
    is_semantic=False,
)

CONFIG_C = ChunkConfig(
    name="C",
    chunk_size=512,
    overlap=128,
    chunking_goal="Support long reasoning (512-1024 tokens) — analytical questions",
    is_semantic=False,
)

# WHY D has same chunk_size as B but double overlap: isolates the impact
# of overlap on retrieval quality. B vs D is a controlled experiment.
CONFIG_D = ChunkConfig(
    name="D",
    chunk_size=256,
    overlap=128,
    chunking_goal="Maximize retrievability (50% overlap) — overlap impact control experiment vs B",
    is_semantic=False,
)

# WHY is_semantic=True: Config E uses Markdown header splitting, not fixed-size.
# chunk_size=512 is only used as the subdivision threshold — sections larger
# than 512 tokens get split using Config B params (256/64).
CONFIG_E = ChunkConfig(
    name="E",
    chunk_size=512,
    overlap=0,
    chunking_goal="Layout-aware chunking (split by text structure) — semantic vs fixed-size",
    is_semantic=True,
)

# Ordered lists for iteration — ALL includes semantic, FIXED excludes it.
ALL_CHUNK_CONFIGS: list[ChunkConfig] = [CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D, CONFIG_E]
FIXED_CHUNK_CONFIGS: list[ChunkConfig] = [CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D]

# Config B params used as fallback for Config E subdivision (PRD Section 3a)
SEMANTIC_SUBDIVISION_CONFIG = CONFIG_B


# ===========================================================================
# Embedding Model Configuration — PRD Section 3b
# Dimensions are needed for FAISS IndexFlatIP initialization.
# Java/TS parallel: like a Map<EmbeddingModel, Integer> for vector dimensions.
# ===========================================================================

EMBEDDING_DIMENSIONS: dict[EmbeddingModel, int] = {
    EmbeddingModel.MINILM: 384,
    EmbeddingModel.MPNET: 768,
    EmbeddingModel.OPENAI: 1536,
}

ALL_EMBEDDING_MODELS: list[EmbeddingModel] = list(EmbeddingModel)

# WHY separate local vs API: different parallelization strategies (8GB RAM constraint).
# Local models: sequential (RAM-bound). API: ThreadPoolExecutor (I/O-bound).
LOCAL_EMBEDDING_MODELS: list[EmbeddingModel] = [EmbeddingModel.MINILM, EmbeddingModel.MPNET]
API_EMBEDDING_MODELS: list[EmbeddingModel] = [EmbeddingModel.OPENAI]


# ===========================================================================
# LLM Model Constants — PRD Section 2
# WHY separate generation vs judge models: generation uses cheaper 4o-mini,
# evaluation uses higher-quality 4o for more reliable judgments.
# ===========================================================================

GENERATION_MODEL = "gpt-4o-mini"     # Synthetic QA + RAG answer generation
JUDGE_MODEL = "gpt-4o"               # RAGAS evaluation + LLM-as-Judge


# ===========================================================================
# Retrieval Constants
# ===========================================================================

# K values for Recall@K, Precision@K, MRR@K (PRD Section 6a)
RETRIEVAL_K_VALUES: list[int] = [1, 3, 5]

# WHY 10: Day 3 grid search retrieves top-10 for evaluation (no reranking).
# Day 4 reranking will use RERANK_TOP_N=5 to narrow down.
RETRIEVAL_TOP_N = 10
RERANK_TOP_N = 5

# WHY 20: wider candidate pool for Cohere cross-encoder reranking.
# FAISS retrieves top-20, Cohere narrows to top-5 (RERANK_TOP_N).
RERANK_RETRIEVAL_TOP_N = 20
COHERE_RERANK_MODEL = "rerank-v3.5"

# BM25 uses Config B chunks (PRD Section 3c: "Config 16: BM25 baseline using Config B chunks")
BM25_CHUNK_CONFIG = CONFIG_B

# Minimum synthetic QA questions (PRD Section 1: "≥50 questions")
MIN_QA_QUESTIONS = 50


# ===========================================================================
# API Parallelization — PRD Section 3b
# WHY 8 workers: balances throughput vs rate limits for OpenAI embedding API.
# ===========================================================================

API_EMBEDDING_MAX_WORKERS = 8


# ===========================================================================
# Caching — PRD Section 2, root CLAUDE.md
# ===========================================================================

CACHE_ENABLED = True


# ===========================================================================
# Config E Constants — PRD Section 3a
# ===========================================================================

# Sections below this token count get merged with the next section
SEMANTIC_MIN_TOKENS = 32

# Sections above this token count get subdivided using SEMANTIC_SUBDIVISION_CONFIG
SEMANTIC_MAX_TOKENS = 512


# ===========================================================================
# Model Key Mapping — used by index_builder, grid_search, synthetic_qa
# ===========================================================================

def model_key(model: EmbeddingModel) -> str:
    """Convert EmbeddingModel enum to a short key for file naming.

    WHY short keys: file names like 'minilm_A.faiss' are cleaner than
    'all-MiniLM-L6-v2_A.faiss'. Also avoids special characters in paths.

    WHY in config.py (not index_builder.py): multiple modules need this
    mapping (grid_search, synthetic_qa). Central location avoids importing
    private functions across modules.
    """
    return {
        EmbeddingModel.MINILM: "minilm",
        EmbeddingModel.MPNET: "mpnet",
        EmbeddingModel.OPENAI: "openai",
    }[model]
