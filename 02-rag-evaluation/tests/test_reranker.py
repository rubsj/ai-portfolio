"""Tests for reranker.py — Cohere reranking with before/after metric evaluation.

Mocks Cohere API and FAISS store. Verifies correct reranking order,
before/after metric computation, improvement percentages, and edge cases.

Java/TS parallel: like integration tests with @MockBean for external
API clients. We test the logic, not the network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import (
    Chunk,
    QuestionHierarchy,
    QuestionType,
    RerankingComparison,
    SyntheticQAPair,
)
from src.reranker import rerank_chunks, rerank_config, run_reranking


# ===========================================================================
# Helpers
# ===========================================================================

def _make_chunk(
    chunk_id: str,
    start_char: int = 0,
    end_char: int = 100,
    config_name: str = "B",
    text: str = "Some meaningful text content for testing purposes",
) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text=text,
        token_count=10,
        start_char=start_char,
        end_char=end_char,
        page_numbers=[1],
        config_name=config_name,
    )


def _make_qa_pair(
    qa_id: str = "q1",
    question: str = "What is the best way to fix a leaky faucet?",
    gold_ids: list[str] | None = None,
) -> SyntheticQAPair:
    """Create a minimal SyntheticQAPair for testing."""
    return SyntheticQAPair(
        id=qa_id,
        question=question,
        question_type=QuestionType.FACTUAL,
        hierarchy=QuestionHierarchy.PARAGRAPH,
        gold_chunk_ids=gold_ids or ["B_0_1"],
        expected_answer="Turn off the water supply and replace the washer.",
        source_chunk_text="To fix a leaky faucet, turn off the water supply.",
        generation_strategy="per_chunk_chain",
    )


# ===========================================================================
# rerank_chunks Tests
# ===========================================================================

class TestRerankChunks:
    """Tests for the core rerank_chunks function."""

    @patch("src.reranker.load_cached", return_value=None)
    @patch("src.reranker.save_cached")
    @patch("cohere.ClientV2")
    def test_rerank_returns_sorted_results(
        self, mock_client_cls, mock_save, mock_load,
    ) -> None:
        """Cohere results are mapped back to chunk IDs and sorted by score."""
        # Mock Cohere client and response
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        # WHY index=2 first: Cohere returns results sorted by relevance,
        # and the index refers to the position in the input documents list.
        mock_result_0 = MagicMock()
        mock_result_0.index = 2
        mock_result_0.relevance_score = 0.95

        mock_result_1 = MagicMock()
        mock_result_1.index = 0
        mock_result_1.relevance_score = 0.80

        mock_response = MagicMock()
        mock_response.results = [mock_result_0, mock_result_1]
        mock_client.rerank.return_value = mock_response

        chunk_ids = ["c1", "c2", "c3"]
        chunk_texts = ["text1", "text2", "text3"]

        results = rerank_chunks("query", chunk_ids, chunk_texts, top_n=2)

        assert len(results) == 2
        # Index 2 → "c3" with 0.95, Index 0 → "c1" with 0.80
        assert results[0] == ("c3", 0.95)
        assert results[1] == ("c1", 0.80)

    @patch("src.reranker.load_cached", return_value=None)
    @patch("src.reranker.save_cached")
    @patch("cohere.ClientV2")
    def test_rerank_passes_top_n_to_cohere(
        self, mock_client_cls, mock_save, mock_load,
    ) -> None:
        """top_n parameter is forwarded to Cohere API."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.results = []
        mock_client.rerank.return_value = mock_response

        rerank_chunks("query", ["c1"], ["text1"], top_n=3)

        mock_client.rerank.assert_called_once()
        call_kwargs = mock_client.rerank.call_args[1]
        assert call_kwargs["top_n"] == 3

    def test_rerank_empty_input_returns_empty(self) -> None:
        """Empty chunk list returns empty results without calling Cohere."""
        results = rerank_chunks("query", [], [])
        assert results == []

    @patch("src.reranker.load_cached")
    def test_rerank_uses_cache_on_hit(self, mock_load) -> None:
        """Cached response is returned without calling Cohere."""
        mock_load.return_value = [
            {"chunk_id": "c2", "relevance_score": 0.9},
            {"chunk_id": "c1", "relevance_score": 0.7},
        ]

        results = rerank_chunks("query", ["c1", "c2"], ["t1", "t2"])

        assert len(results) == 2
        assert results[0] == ("c2", 0.9)
        assert results[1] == ("c1", 0.7)

    @patch("src.reranker.load_cached", return_value=None)
    @patch("src.reranker.save_cached")
    @patch("cohere.ClientV2")
    def test_rerank_caches_response(
        self, mock_client_cls, mock_save, mock_load,
    ) -> None:
        """Response is saved to cache after successful API call."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.relevance_score = 0.85
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        mock_client.rerank.return_value = mock_response

        rerank_chunks("query", ["c1"], ["text1"])

        mock_save.assert_called_once()


# ===========================================================================
# rerank_config Tests
# ===========================================================================

class TestRerankConfig:
    """Tests for the per-config reranking evaluation."""

    @patch("src.reranker.rerank_chunks")
    def test_before_after_metrics_computed(self, mock_rerank) -> None:
        """Before-metrics on FAISS top-5, after-metrics on Cohere top-5."""
        qa = _make_qa_pair(gold_ids=["c1", "c2"])

        # Mock FAISS store: returns 20 results, c1 at position 0, c2 at position 6
        mock_store = MagicMock()
        faiss_results = [(f"c{i}", 0.9 - i * 0.04) for i in range(20)]
        mock_store.search.return_value = faiss_results

        # Chunk lookup — all 20 chunks
        chunk_lookup = {f"c{i}": _make_chunk(f"c{i}") for i in range(20)}

        # Cohere rerank moves c2 into top-5 (it was at position 6 in FAISS)
        mock_rerank.return_value = [
            ("c0", 0.98), ("c2", 0.95), ("c1", 0.90), ("c3", 0.85), ("c4", 0.80),
        ]

        query_embeddings = np.random.randn(1, 1536).astype(np.float32)

        result = rerank_config(
            config_id="E-openai",
            qa_pairs=[qa],
            store=mock_store,
            chunk_lookup=chunk_lookup,
            gold_ids_per_question=[["c1", "c2"]],
            query_embeddings=query_embeddings,
        )

        assert isinstance(result, RerankingComparison)
        assert result.config_id == "E-openai"

        # Before: FAISS top-5 = [c0, c1, c2, c3, c4]. Gold = {c1, c2}.
        # c1 at pos 1 (index 1), c2 at pos 6 — only c1 in top-5.
        # Wait, our mock returns c0-c19, so top-5 is c0,c1,c2,c3,c4
        # Both c1 and c2 are in top-5 → recall = 2/2 = 1.0
        # But wait, the FAISS results are [(c0, 0.9), (c1, 0.86), (c2, 0.82), ...]
        # So before top-5 contains both c1 and c2 → recall_before = 1.0
        assert result.recall_at_5_before == 1.0
        # After also has both → recall_after = 1.0
        assert result.recall_at_5_after == 1.0

    @patch("src.reranker.rerank_chunks")
    def test_rerank_improves_recall(self, mock_rerank) -> None:
        """Reranking can improve recall by surfacing gold chunks into top-5."""
        qa = _make_qa_pair(gold_ids=["c10"])

        # FAISS returns c0-c19; c10 is at position 10 (outside top-5)
        mock_store = MagicMock()
        faiss_results = [(f"c{i}", 0.9 - i * 0.04) for i in range(20)]
        mock_store.search.return_value = faiss_results

        chunk_lookup = {f"c{i}": _make_chunk(f"c{i}") for i in range(20)}

        # Cohere moves c10 into position 2
        mock_rerank.return_value = [
            ("c0", 0.99), ("c10", 0.97), ("c1", 0.90), ("c2", 0.85), ("c3", 0.80),
        ]

        query_embeddings = np.random.randn(1, 1536).astype(np.float32)

        result = rerank_config(
            config_id="B-openai",
            qa_pairs=[qa],
            store=mock_store,
            chunk_lookup=chunk_lookup,
            gold_ids_per_question=[["c10"]],
            query_embeddings=query_embeddings,
        )

        # Before: c10 not in FAISS top-5 → recall = 0
        assert result.recall_at_5_before == 0.0
        # After: c10 in Cohere top-5 → recall = 1
        assert result.recall_at_5_after == 1.0

    @patch("src.reranker.rerank_chunks")
    def test_improvement_percentage_division_by_zero(self, mock_rerank) -> None:
        """When before-metric is 0, improvement percentage should be 0 (not crash)."""
        qa = _make_qa_pair(gold_ids=["c99"])  # Gold chunk not in any results

        mock_store = MagicMock()
        mock_store.search.return_value = [(f"c{i}", 0.5) for i in range(20)]

        chunk_lookup = {f"c{i}": _make_chunk(f"c{i}") for i in range(20)}

        # Reranking also doesn't find c99
        mock_rerank.return_value = [
            ("c0", 0.9), ("c1", 0.8), ("c2", 0.7), ("c3", 0.6), ("c4", 0.5),
        ]

        query_embeddings = np.random.randn(1, 1536).astype(np.float32)

        result = rerank_config(
            config_id="D-openai",
            qa_pairs=[qa],
            store=mock_store,
            chunk_lookup=chunk_lookup,
            gold_ids_per_question=[["c99"]],
            query_embeddings=query_embeddings,
        )

        # Both before and after are 0 — improvement should be 0, not crash
        assert result.recall_at_5_before == 0.0
        assert result.recall_at_5_after == 0.0
        assert result.recall_improvement_pct == 0.0

    @patch("src.reranker.rerank_chunks")
    def test_multiple_questions_averaged(self, mock_rerank) -> None:
        """Metrics are averaged across multiple questions."""
        qa1 = _make_qa_pair(qa_id="q1", gold_ids=["c0"])
        qa2 = _make_qa_pair(qa_id="q2", gold_ids=["c10"])

        mock_store = MagicMock()
        faiss_results = [(f"c{i}", 0.9 - i * 0.04) for i in range(20)]
        mock_store.search.return_value = faiss_results

        chunk_lookup = {f"c{i}": _make_chunk(f"c{i}") for i in range(20)}

        # Rerank keeps c0 in top-5, doesn't move c10 in
        mock_rerank.return_value = [
            ("c0", 0.99), ("c1", 0.90), ("c2", 0.85), ("c3", 0.80), ("c4", 0.75),
        ]

        query_embeddings = np.random.randn(2, 1536).astype(np.float32)

        result = rerank_config(
            config_id="E-openai",
            qa_pairs=[qa1, qa2],
            store=mock_store,
            chunk_lookup=chunk_lookup,
            gold_ids_per_question=[["c0"], ["c10"]],
            query_embeddings=query_embeddings,
        )

        # q1: c0 in top-5 → recall=1.0; q2: c10 not in top-5 → recall=0.0
        # Average = 0.5
        assert result.recall_at_5_before == pytest.approx(0.5)
        # After reranking, same scenario: c0 in, c10 out
        assert result.recall_at_5_after == pytest.approx(0.5)


# ===========================================================================
# run_reranking Tests
# ===========================================================================

class TestRunReranking:
    """Tests for the top-level reranking orchestrator."""

    @patch("src.reranker.rerank_config")
    @patch("src.reranker.create_embedder")
    @patch("src.reranker.FAISSVectorStore")
    @patch("src.grid_search.map_gold_chunks")
    def test_reranks_all_configs(
        self, mock_map_gold, mock_store_cls, mock_create_embedder, mock_rerank_config,
    ) -> None:
        """run_reranking processes all provided config IDs."""
        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(1, 1536).astype(np.float32)
        mock_create_embedder.return_value = mock_embedder

        # Mock FAISS store loading
        mock_store = MagicMock()
        mock_store_cls.load.return_value = mock_store

        # Mock gold chunk mapping
        mock_map_gold.return_value = ["E_0_1"]

        # Mock rerank_config return
        mock_rerank_config.return_value = RerankingComparison(
            config_id="E-openai",
            recall_at_5_before=0.6, recall_at_5_after=0.7,
            precision_at_5_before=0.3, precision_at_5_after=0.4,
            mrr_at_5_before=0.5, mrr_at_5_after=0.6,
            recall_improvement_pct=16.7,
            precision_improvement_pct=33.3,
            mrr_improvement_pct=20.0,
        )

        qa = _make_qa_pair()
        chunks_b = [_make_chunk("B_0_1", config_name="B")]
        chunks_e = [_make_chunk("E_0_1", config_name="E")]

        comparisons = run_reranking(
            top_config_ids=["E-openai", "B-openai"],
            qa_pairs=[qa],
            chunks_by_config={"E": chunks_e, "B": chunks_b},
            b_chunks_lookup={"B_0_1": chunks_b[0]},
        )

        # Both configs should be processed
        assert len(comparisons) == 2
        assert mock_rerank_config.call_count == 2

    @patch("src.reranker.rerank_config")
    @patch("src.reranker.create_embedder")
    @patch("src.reranker.FAISSVectorStore")
    def test_embeds_queries_once(
        self, mock_store_cls, mock_create_embedder, mock_rerank_config,
    ) -> None:
        """Query embeddings are computed once and reused across configs."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(1, 1536).astype(np.float32)
        mock_create_embedder.return_value = mock_embedder

        mock_store_cls.load.return_value = MagicMock()
        mock_rerank_config.return_value = RerankingComparison(
            config_id="test",
            recall_at_5_before=0.5, recall_at_5_after=0.6,
            precision_at_5_before=0.3, precision_at_5_after=0.4,
            mrr_at_5_before=0.4, mrr_at_5_after=0.5,
            recall_improvement_pct=20.0,
            precision_improvement_pct=33.3,
            mrr_improvement_pct=25.0,
        )

        qa = _make_qa_pair()

        run_reranking(
            top_config_ids=["B-openai", "D-openai"],
            qa_pairs=[qa],
            chunks_by_config={"B": [_make_chunk("B_0_1")], "D": [_make_chunk("D_0_1", config_name="D")]},
            b_chunks_lookup={"B_0_1": _make_chunk("B_0_1")},
        )

        # Embedder.embed() called exactly once — queries embedded once for all configs
        mock_embedder.embed.assert_called_once()

    @patch("src.reranker.rerank_config")
    @patch("src.reranker.create_embedder")
    @patch("src.reranker.FAISSVectorStore")
    def test_config_b_skips_gold_mapping(
        self, mock_store_cls, mock_create_embedder, mock_rerank_config,
    ) -> None:
        """Config B uses gold IDs directly without mapping."""
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(1, 1536).astype(np.float32)
        mock_create_embedder.return_value = mock_embedder

        mock_store_cls.load.return_value = MagicMock()
        mock_rerank_config.return_value = RerankingComparison(
            config_id="B-openai",
            recall_at_5_before=0.5, recall_at_5_after=0.6,
            precision_at_5_before=0.3, precision_at_5_after=0.4,
            mrr_at_5_before=0.4, mrr_at_5_after=0.5,
            recall_improvement_pct=20.0,
            precision_improvement_pct=33.3,
            mrr_improvement_pct=25.0,
        )

        qa = _make_qa_pair(gold_ids=["B_0_1"])

        run_reranking(
            top_config_ids=["B-openai"],
            qa_pairs=[qa],
            chunks_by_config={"B": [_make_chunk("B_0_1")]},
            b_chunks_lookup={"B_0_1": _make_chunk("B_0_1")},
        )

        # Verify rerank_config was called with the original gold IDs (no mapping)
        call_kwargs = mock_rerank_config.call_args[1]
        assert call_kwargs["gold_ids_per_question"] == [["B_0_1"]]
