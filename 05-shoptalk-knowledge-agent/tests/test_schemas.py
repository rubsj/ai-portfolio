"""Tests for src/schemas.py — all 13 Pydantic models.

Strategy:
  - Inline _make_*() builders so each test is self-contained (no conftest dependency)
  - Happy path: verify fields are stored correctly
  - Validation error cases: confirm Pydantic raises ValidationError with the right message
  - Parametrize for Literal type aliases to ensure every valid value is accepted
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.schemas import (
    Chunk,
    ChunkMetadata,
    ChunkingStrategy,
    Citation,
    Document,
    DocumentMetadata,
    EmbeddingModel,
    ExperimentConfig,
    ExperimentResult,
    JudgeScores,
    PageInfo,
    PerformanceMetrics,
    QAResponse,
    RerankerType,
    RetrievalMetrics,
    RetrievalResult,
    RetrieverType,
)

# ---------------------------------------------------------------------------
# Builders — return minimal valid instances
# ---------------------------------------------------------------------------


def _make_page_info(page_number: int = 0, text: str = "hello world") -> PageInfo:
    return PageInfo(page_number=page_number, text=text, char_count=len(text))


def _make_document_metadata(source: str = "paper.pdf") -> DocumentMetadata:
    return DocumentMetadata(source=source, title="Test Paper", author="A. Author", page_count=5)


def _make_document() -> Document:
    pages = [_make_page_info(i, f"page {i} content") for i in range(3)]
    content = "\n\n".join(p.text for p in pages)
    return Document(content=content, metadata=_make_document_metadata(), pages=pages)


def _make_chunk_metadata(document_id: str = "doc-1", chunk_index: int = 0) -> ChunkMetadata:
    return ChunkMetadata(
        document_id=document_id,
        source="paper.pdf",
        page_number=0,
        start_char=0,
        end_char=20,
        chunk_index=chunk_index,
    )


def _make_chunk(document_id: str = "doc-1", chunk_index: int = 0) -> Chunk:
    return Chunk(
        content="This is a test chunk.",
        metadata=_make_chunk_metadata(document_id=document_id, chunk_index=chunk_index),
    )


def _make_experiment_config(**overrides) -> ExperimentConfig:
    defaults = dict(
        chunking_strategy="fixed",
        chunk_size=512,
        chunk_overlap=50,
        embedding_model="minilm",
        retriever_type="dense",
        top_k=5,
    )
    defaults.update(overrides)
    return ExperimentConfig(**defaults)


def _make_retrieval_metrics() -> RetrievalMetrics:
    return RetrievalMetrics(recall_at_5=0.8, precision_at_5=0.7, mrr=0.6, ndcg_at_5=0.75)


def _make_judge_scores() -> JudgeScores:
    return JudgeScores(
        avg_relevance=4.0,
        avg_accuracy=4.0,
        avg_completeness=3.5,
        avg_conciseness=4.0,
        avg_citation_quality=3.0,
        overall_average=3.7,
    )


def _make_performance_metrics() -> PerformanceMetrics:
    return PerformanceMetrics(
        ingestion_time_seconds=5.2,
        avg_query_latency_ms=120.0,
        index_size_bytes=1024 * 1024,
        peak_memory_mb=512.0,
    )


# ---------------------------------------------------------------------------
# PageInfo
# ---------------------------------------------------------------------------


class TestPageInfo:
    def test_happy_path(self):
        p = _make_page_info(page_number=2, text="hello")
        assert p.page_number == 2
        assert p.text == "hello"
        assert p.char_count == 5

    def test_page_number_zero_allowed(self):
        p = _make_page_info(page_number=0)
        assert p.page_number == 0

    def test_negative_page_number_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PageInfo(page_number=-1, text="x", char_count=1)

    def test_negative_char_count_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PageInfo(page_number=0, text="x", char_count=-1)


# ---------------------------------------------------------------------------
# DocumentMetadata
# ---------------------------------------------------------------------------


class TestDocumentMetadata:
    def test_happy_path(self):
        m = _make_document_metadata("my.pdf")
        assert m.source == "my.pdf"
        assert m.page_count == 5

    def test_empty_source_raises(self):
        with pytest.raises(ValidationError, match="at least 1 character"):
            DocumentMetadata(source="", title="t", author="a", page_count=1)

    def test_zero_page_count_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            DocumentMetadata(source="x.pdf", title="t", author="a", page_count=0)

    def test_defaults_for_optional_fields(self):
        m = DocumentMetadata(source="x.pdf", page_count=1)
        assert m.title == ""
        assert m.author == ""


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_happy_path(self):
        doc = _make_document()
        assert doc.content
        assert len(doc.pages) == 3
        assert doc.id  # uuid4 auto-generated

    def test_id_auto_generated(self):
        doc1 = _make_document()
        doc2 = _make_document()
        assert doc1.id != doc2.id

    def test_empty_content_raises(self):
        pages = [_make_page_info()]
        with pytest.raises(ValidationError, match="at least 1 character"):
            Document(content="", metadata=_make_document_metadata(), pages=pages)

    def test_empty_pages_raises(self):
        with pytest.raises(ValidationError):
            Document(content="text", metadata=_make_document_metadata(), pages=[])


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    def test_happy_path(self):
        m = _make_chunk_metadata()
        assert m.start_char == 0
        assert m.end_char == 20

    def test_equal_start_end_allowed(self):
        # Zero-length chunk is degenerate but not invalid
        m = ChunkMetadata(
            document_id="d", source="s.pdf", page_number=0,
            start_char=5, end_char=5, chunk_index=0,
        )
        assert m.end_char == m.start_char

    def test_end_less_than_start_raises(self):
        with pytest.raises(ValidationError, match="end_char"):
            ChunkMetadata(
                document_id="d", source="s.pdf", page_number=0,
                start_char=10, end_char=5, chunk_index=0,
            )

    def test_negative_chunk_index_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ChunkMetadata(
                document_id="d", source="s.pdf", page_number=0,
                start_char=0, end_char=5, chunk_index=-1,
            )


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_happy_path(self):
        c = _make_chunk()
        assert c.content == "This is a test chunk."
        assert c.embedding is None

    def test_id_auto_generated(self):
        c1 = _make_chunk()
        c2 = _make_chunk()
        assert c1.id != c2.id

    def test_empty_content_raises(self):
        with pytest.raises(ValidationError, match="at least 1 character"):
            Chunk(content="", metadata=_make_chunk_metadata())

    def test_numpy_embedding_stored(self):
        import numpy as np
        emb = np.ones(384, dtype=np.float32)
        c = Chunk(content="text", metadata=_make_chunk_metadata(), embedding=emb)
        assert c.embedding is not None
        assert c.embedding.shape == (384,)


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------


class TestRetrievalResult:
    def test_happy_path(self):
        rr = RetrievalResult(
            chunk=_make_chunk(), score=0.87, retriever_type="dense", rank=1
        )
        assert rr.rank == 1
        assert rr.score == 0.87

    def test_rank_zero_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            RetrievalResult(
                chunk=_make_chunk(), score=0.5, retriever_type="bm25", rank=0
            )

    @pytest.mark.parametrize("rt", ["dense", "bm25", "hybrid"])
    def test_all_retriever_types_accepted(self, rt: RetrieverType):
        rr = RetrievalResult(chunk=_make_chunk(), score=0.5, retriever_type=rt, rank=1)
        assert rr.retriever_type == rt


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------


class TestCitation:
    def test_happy_path(self):
        c = Citation(
            chunk_id="c1", source="paper.pdf", page_number=3,
            text_snippet="relevant passage", relevance_score=0.9,
        )
        assert c.relevance_score == 0.9

    def test_relevance_score_above_1_raises(self):
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Citation(
                chunk_id="c1", source="paper.pdf", page_number=0,
                text_snippet="text", relevance_score=1.1,
            )

    def test_relevance_score_below_0_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Citation(
                chunk_id="c1", source="paper.pdf", page_number=0,
                text_snippet="text", relevance_score=-0.1,
            )


# ---------------------------------------------------------------------------
# QAResponse
# ---------------------------------------------------------------------------


class TestQAResponse:
    def test_happy_path(self):
        qa = QAResponse(
            query="What is attention?",
            answer="Attention is a mechanism [1].",
            chunks_used=[_make_chunk()],
            latency=0.35,
        )
        assert qa.citations == []
        assert qa.confidence is None

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError, match="at least 1 character"):
            QAResponse(query="", answer="ans", chunks_used=[_make_chunk()], latency=0.1)

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            QAResponse(query="q", answer="a", chunks_used=[_make_chunk()], latency=-1.0)


# ---------------------------------------------------------------------------
# ExperimentConfig — the most complex model
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_happy_path_dense(self):
        cfg = _make_experiment_config()
        assert cfg.retriever_type == "dense"
        assert cfg.embedding_model == "minilm"
        assert cfg.hybrid_alpha is None

    def test_happy_path_hybrid(self):
        cfg = _make_experiment_config(retriever_type="hybrid", hybrid_alpha=0.7)
        assert cfg.hybrid_alpha == 0.7

    def test_happy_path_bm25(self):
        cfg = _make_experiment_config(retriever_type="bm25", embedding_model=None)
        assert cfg.embedding_model is None

    def test_hybrid_without_alpha_raises(self):
        with pytest.raises(ValidationError, match="hybrid_alpha is required"):
            _make_experiment_config(retriever_type="hybrid")

    def test_alpha_without_hybrid_raises(self):
        with pytest.raises(ValidationError, match="hybrid_alpha must be None"):
            _make_experiment_config(retriever_type="dense", hybrid_alpha=0.5)

    def test_reranking_without_type_raises(self):
        with pytest.raises(ValidationError, match="reranker_type is required"):
            _make_experiment_config(use_reranking=True)

    def test_reranker_type_without_reranking_raises(self):
        with pytest.raises(ValidationError, match="reranker_type must be None"):
            _make_experiment_config(use_reranking=False, reranker_type="cohere")

    def test_dense_without_embedding_model_raises(self):
        with pytest.raises(ValidationError, match="embedding_model is required"):
            _make_experiment_config(retriever_type="dense", embedding_model=None)

    def test_bm25_with_embedding_model_raises(self):
        with pytest.raises(ValidationError, match="embedding_model must be None"):
            _make_experiment_config(retriever_type="bm25", embedding_model="minilm")

    def test_overlap_gte_size_raises(self):
        with pytest.raises(ValidationError, match="less than chunk_size"):
            _make_experiment_config(chunk_size=100, chunk_overlap=100)

    def test_sliding_window_without_params_raises(self):
        with pytest.raises(ValidationError, match="window_size_tokens"):
            _make_experiment_config(chunking_strategy="sliding_window")

    def test_sliding_window_with_params_ok(self):
        cfg = _make_experiment_config(
            chunking_strategy="sliding_window",
            window_size_tokens=200,
            step_size_tokens=100,
        )
        assert cfg.window_size_tokens == 200

    def test_reranking_config_ok(self):
        cfg = _make_experiment_config(use_reranking=True, reranker_type="cross_encoder")
        assert cfg.reranker_type == "cross_encoder"

    @pytest.mark.parametrize("strategy", ["fixed", "recursive", "sliding_window", "heading_semantic", "embedding_semantic"])
    def test_all_chunking_strategies_accepted(self, strategy: ChunkingStrategy):
        extra = {}
        if strategy == "sliding_window":
            extra = {"window_size_tokens": 200, "step_size_tokens": 100}
        cfg = _make_experiment_config(chunking_strategy=strategy, **extra)
        assert cfg.chunking_strategy == strategy

    @pytest.mark.parametrize("model", ["minilm", "mpnet", "openai"])
    def test_all_embedding_models_accepted(self, model: EmbeddingModel):
        cfg = _make_experiment_config(embedding_model=model)
        assert cfg.embedding_model == model

    @pytest.mark.parametrize("rt", ["cohere", "cross_encoder"])
    def test_all_reranker_types_accepted(self, rt: RerankerType):
        cfg = _make_experiment_config(use_reranking=True, reranker_type=rt)
        assert cfg.reranker_type == rt


# ---------------------------------------------------------------------------
# RetrievalMetrics
# ---------------------------------------------------------------------------


class TestRetrievalMetrics:
    def test_happy_path(self):
        m = _make_retrieval_metrics()
        assert 0.0 <= m.recall_at_5 <= 1.0

    def test_value_above_1_raises(self):
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            RetrievalMetrics(recall_at_5=1.1, precision_at_5=0.5, mrr=0.5, ndcg_at_5=0.5)

    def test_negative_value_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            RetrievalMetrics(recall_at_5=-0.1, precision_at_5=0.5, mrr=0.5, ndcg_at_5=0.5)


# ---------------------------------------------------------------------------
# JudgeScores
# ---------------------------------------------------------------------------


class TestJudgeScores:
    def test_happy_path(self):
        j = _make_judge_scores()
        assert j.overall_average == 3.7

    def test_score_below_1_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            JudgeScores(
                avg_relevance=0.9, avg_accuracy=4.0, avg_completeness=4.0,
                avg_conciseness=4.0, avg_citation_quality=4.0, overall_average=4.0,
            )

    def test_score_above_5_raises(self):
        with pytest.raises(ValidationError, match="less than or equal to 5"):
            JudgeScores(
                avg_relevance=5.1, avg_accuracy=4.0, avg_completeness=4.0,
                avg_conciseness=4.0, avg_citation_quality=4.0, overall_average=4.0,
            )


# ---------------------------------------------------------------------------
# PerformanceMetrics
# ---------------------------------------------------------------------------


class TestPerformanceMetrics:
    def test_happy_path(self):
        p = _make_performance_metrics()
        assert p.ingestion_time_seconds >= 0
        assert p.index_size_bytes >= 0

    def test_negative_latency_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PerformanceMetrics(
                ingestion_time_seconds=-1.0,
                avg_query_latency_ms=100.0,
                index_size_bytes=1024,
                peak_memory_mb=256.0,
            )


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class TestExperimentResult:
    def test_happy_path(self):
        result = ExperimentResult(
            experiment_id="exp-001",
            config=_make_experiment_config(),
            metrics=_make_retrieval_metrics(),
            performance=_make_performance_metrics(),
        )
        assert result.judge_scores is None
        assert result.query_results == []

    def test_with_judge_scores(self):
        result = ExperimentResult(
            experiment_id="exp-002",
            config=_make_experiment_config(),
            metrics=_make_retrieval_metrics(),
            judge_scores=_make_judge_scores(),
            performance=_make_performance_metrics(),
        )
        assert result.judge_scores is not None
        assert result.judge_scores.overall_average == 3.7

    def test_empty_experiment_id_raises(self):
        with pytest.raises(ValidationError, match="at least 1 character"):
            ExperimentResult(
                experiment_id="",
                config=_make_experiment_config(),
                metrics=_make_retrieval_metrics(),
                performance=_make_performance_metrics(),
            )
