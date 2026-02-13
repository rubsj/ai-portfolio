"""Tests for config.py — all constants, configs, paths, and dimensions.

Verifies the centralized configuration matches PRD Section 3a/3b requirements.
Config values are the foundation of the grid search — wrong values here would
produce invalid experimental results.
"""

from __future__ import annotations

from src.config import (
    ADR_DIR,
    ALL_CHUNK_CONFIGS,
    ALL_EMBEDDING_MODELS,
    API_EMBEDDING_MAX_WORKERS,
    API_EMBEDDING_MODELS,
    BM25_CHUNK_CONFIG,
    CACHE_DIR,
    CACHE_ENABLED,
    CHARTS_DIR,
    CONFIG_A,
    CONFIG_B,
    CONFIG_C,
    CONFIG_D,
    CONFIG_E,
    DATA_DIR,
    DOCS_DIR,
    EMBEDDING_DIMENSIONS,
    FIXED_CHUNK_CONFIGS,
    GENERATION_MODEL,
    INPUT_DIR,
    JUDGE_MODEL,
    LOCAL_EMBEDDING_MODELS,
    METRICS_DIR,
    MIN_QA_QUESTIONS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    RERANK_TOP_N,
    REPORTS_DIR,
    RESULTS_DIR,
    RETRIEVAL_K_VALUES,
    RETRIEVAL_TOP_N,
    SEMANTIC_MAX_TOKENS,
    SEMANTIC_MIN_TOKENS,
    SEMANTIC_SUBDIVISION_CONFIG,
)
from src.models import EmbeddingModel


# ===========================================================================
# Path Constants
# ===========================================================================

class TestPaths:
    """Verify file path constants resolve correctly."""

    def test_project_root_is_absolute(self) -> None:
        assert PROJECT_ROOT.is_absolute()

    def test_data_dir_under_project_root(self) -> None:
        assert DATA_DIR == PROJECT_ROOT / "data"

    def test_input_dir_under_data(self) -> None:
        assert INPUT_DIR == DATA_DIR / "input"

    def test_cache_dir_under_data(self) -> None:
        assert CACHE_DIR == DATA_DIR / "cache"

    def test_output_dir_under_data(self) -> None:
        assert OUTPUT_DIR == DATA_DIR / "output"

    def test_results_subtree(self) -> None:
        assert CHARTS_DIR == RESULTS_DIR / "charts"
        assert METRICS_DIR == RESULTS_DIR / "metrics"
        assert REPORTS_DIR == RESULTS_DIR / "reports"

    def test_docs_subtree(self) -> None:
        assert DOCS_DIR == PROJECT_ROOT / "docs"
        assert ADR_DIR == DOCS_DIR / "adr"


# ===========================================================================
# Chunk Configurations (A-E) — PRD Section 3a
# ===========================================================================

class TestChunkConfigs:
    """Verify the 5 chunk configs match PRD Section 3a."""

    def test_config_a(self) -> None:
        assert CONFIG_A.name == "A"
        assert CONFIG_A.chunk_size == 128
        assert CONFIG_A.overlap == 32
        assert CONFIG_A.is_semantic is False

    def test_config_b(self) -> None:
        assert CONFIG_B.name == "B"
        assert CONFIG_B.chunk_size == 256
        assert CONFIG_B.overlap == 64
        assert CONFIG_B.is_semantic is False

    def test_config_c(self) -> None:
        assert CONFIG_C.name == "C"
        assert CONFIG_C.chunk_size == 512
        assert CONFIG_C.overlap == 128
        assert CONFIG_C.is_semantic is False

    def test_config_d_same_size_as_b_but_more_overlap(self) -> None:
        """D vs B is a controlled experiment — same chunk_size, double overlap."""
        assert CONFIG_D.name == "D"
        assert CONFIG_D.chunk_size == CONFIG_B.chunk_size  # Same size
        assert CONFIG_D.overlap == 128  # Double B's overlap (64)
        assert CONFIG_D.is_semantic is False

    def test_config_e_semantic(self) -> None:
        assert CONFIG_E.name == "E"
        assert CONFIG_E.chunk_size == 512
        assert CONFIG_E.overlap == 0
        assert CONFIG_E.is_semantic is True

    def test_all_chunk_configs_has_five(self) -> None:
        assert len(ALL_CHUNK_CONFIGS) == 5
        names = [c.name for c in ALL_CHUNK_CONFIGS]
        assert names == ["A", "B", "C", "D", "E"]

    def test_fixed_chunk_configs_excludes_semantic(self) -> None:
        assert len(FIXED_CHUNK_CONFIGS) == 4
        assert all(not c.is_semantic for c in FIXED_CHUNK_CONFIGS)
        assert CONFIG_E not in FIXED_CHUNK_CONFIGS

    def test_semantic_subdivision_config_is_config_b(self) -> None:
        assert SEMANTIC_SUBDIVISION_CONFIG is CONFIG_B


# ===========================================================================
# Embedding Configuration — PRD Section 3b
# ===========================================================================

class TestEmbeddingConfig:
    """Verify embedding model dimensions and groupings."""

    def test_embedding_dimensions_minilm(self) -> None:
        assert EMBEDDING_DIMENSIONS[EmbeddingModel.MINILM] == 384

    def test_embedding_dimensions_mpnet(self) -> None:
        assert EMBEDDING_DIMENSIONS[EmbeddingModel.MPNET] == 768

    def test_embedding_dimensions_openai(self) -> None:
        assert EMBEDDING_DIMENSIONS[EmbeddingModel.OPENAI] == 1536

    def test_all_embedding_models_has_three(self) -> None:
        assert len(ALL_EMBEDDING_MODELS) == 3

    def test_local_models_count(self) -> None:
        assert len(LOCAL_EMBEDDING_MODELS) == 2
        assert EmbeddingModel.MINILM in LOCAL_EMBEDDING_MODELS
        assert EmbeddingModel.MPNET in LOCAL_EMBEDDING_MODELS

    def test_api_models_count(self) -> None:
        assert len(API_EMBEDDING_MODELS) == 1
        assert EmbeddingModel.OPENAI in API_EMBEDDING_MODELS

    def test_no_overlap_between_local_and_api(self) -> None:
        local_set = set(LOCAL_EMBEDDING_MODELS)
        api_set = set(API_EMBEDDING_MODELS)
        assert local_set.isdisjoint(api_set)


# ===========================================================================
# LLM + Retrieval + Semantic Constants
# ===========================================================================

class TestConstants:
    """Verify LLM, retrieval, and semantic constants."""

    def test_generation_model(self) -> None:
        assert GENERATION_MODEL == "gpt-4o-mini"

    def test_judge_model(self) -> None:
        assert JUDGE_MODEL == "gpt-4o"

    def test_retrieval_k_values(self) -> None:
        assert RETRIEVAL_K_VALUES == [1, 3, 5]

    def test_retrieval_top_n(self) -> None:
        assert RETRIEVAL_TOP_N == 20

    def test_rerank_top_n(self) -> None:
        assert RERANK_TOP_N == 5

    def test_bm25_uses_config_b(self) -> None:
        assert BM25_CHUNK_CONFIG is CONFIG_B

    def test_min_qa_questions(self) -> None:
        assert MIN_QA_QUESTIONS == 50

    def test_api_embedding_max_workers(self) -> None:
        assert API_EMBEDDING_MAX_WORKERS == 8

    def test_cache_enabled(self) -> None:
        assert CACHE_ENABLED is True

    def test_semantic_min_tokens(self) -> None:
        assert SEMANTIC_MIN_TOKENS == 32

    def test_semantic_max_tokens(self) -> None:
        assert SEMANTIC_MAX_TOKENS == 512
