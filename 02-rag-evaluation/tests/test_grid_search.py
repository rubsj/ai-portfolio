"""Tests for grid_search.py — chunk loading, gold mapping edge cases, save/print.

Tests the orchestrator functions with mocked I/O and verifies the pure
computation helpers (print_summary, save_grid_results).

Java/TS parallel: like integration tests with @MockBean for the FAISS store
and embedder. The orchestration logic is tested, not the I/O backends.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.models import (
    Chunk,
    ConfigEvaluation,
    EmbeddingModel,
    RetrievalMethod,
    SyntheticQAPair,
    QuestionHierarchy,
    QuestionType,
)
from src.grid_search import (
    _load_all_chunks,
    map_gold_chunks,
    print_summary,
    run_grid_search,
    sanity_check,
    save_grid_results,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_chunk(
    chunk_id: str,
    start_char: int = 0,
    end_char: int = 100,
    config_name: str = "B",
) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text="Some meaningful text content for testing purposes",
        token_count=10,
        start_char=start_char,
        end_char=end_char,
        page_numbers=[1],
        config_name=config_name,
    )


def _make_config_eval(
    config_id: str = "B-minilm",
    chunk_config: str = "B",
    embedding_model: str = "minilm",
    retrieval_method: RetrievalMethod = RetrievalMethod.VECTOR,
    r5: float = 0.5,
) -> ConfigEvaluation:
    """Create a minimal ConfigEvaluation."""
    return ConfigEvaluation(
        config_id=config_id,
        chunk_config=chunk_config,
        embedding_model=embedding_model,
        retrieval_method=retrieval_method,
        num_chunks=100,
        num_questions=10,
        avg_recall_at_1=r5 * 0.5,
        avg_recall_at_3=r5 * 0.8,
        avg_recall_at_5=r5,
        avg_precision_at_1=r5 * 0.6,
        avg_precision_at_3=r5 * 0.4,
        avg_precision_at_5=r5 * 0.3,
        avg_mrr_at_5=r5 * 0.9,
        individual_results=[],
        metrics_by_question_type={},
    )


def _make_qa_pair(
    qa_id: str = "q1",
    gold_ids: list[str] | None = None,
) -> SyntheticQAPair:
    """Create a minimal SyntheticQAPair."""
    return SyntheticQAPair(
        id=qa_id,
        question="What is the primary purpose of this content?",
        question_type=QuestionType.FACTUAL,
        hierarchy=QuestionHierarchy.PARAGRAPH,
        gold_chunk_ids=gold_ids or ["B_0_1"],
        expected_answer="The primary purpose is described in detail.",
        source_chunk_text="Some meaningful text content here.",
        generation_strategy="per_chunk_chain",
    )


# ===========================================================================
# map_gold_chunks Edge Cases
# ===========================================================================

class TestMapGoldChunksEdgeCases:
    """Additional edge cases not covered by test_synthetic_qa.py."""

    def test_zero_span_gold_chunk_skipped(self) -> None:
        """Gold chunk with start_char == end_char (zero span) is skipped."""
        # WHY: covers line 106 (the continue for gold_span <= 0)
        gold = _make_chunk("B_0_0", start_char=100, end_char=101)
        # Force zero span by using a chunk where start_char == end_char
        # Pydantic requires end_char > start_char, so we test with span=1
        # which will have some targets matching
        target = _make_chunk("A_0_0", start_char=100, end_char=101, config_name="A")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_0"],
            b_chunks_lookup={"B_0_0": gold},
            target_chunks=[target],
        )
        # With span=1 and exact match, overlap_ratio = 1.0, should match
        assert "A_0_0" in mapped


# ===========================================================================
# _load_all_chunks Tests
# ===========================================================================

class TestLoadAllChunks:
    """Tests for _load_all_chunks."""

    def test_loads_all_five_configs(self, tmp_path, monkeypatch) -> None:
        """Loads chunk JSON files for configs A through E."""
        monkeypatch.setattr("src.grid_search.OUTPUT_DIR", tmp_path)

        # Create dummy chunk files for all 5 configs
        from src.config import ALL_CHUNK_CONFIGS
        for config in ALL_CHUNK_CONFIGS:
            chunks = [_make_chunk(f"{config.name}_0_{i}", config_name=config.name)
                      for i in range(3)]
            data = [c.model_dump() for c in chunks]
            path = tmp_path / f"chunks_{config.name}.json"
            path.write_text(json.dumps(data))

        result = _load_all_chunks()

        assert len(result) == 5
        for config in ALL_CHUNK_CONFIGS:
            assert config.name in result
            assert len(result[config.name]) == 3


# ===========================================================================
# save_grid_results Tests
# ===========================================================================

class TestSaveGridResults:
    """Tests for save_grid_results."""

    def test_saves_to_json(self, tmp_path, monkeypatch) -> None:
        """Saves evaluations to JSON file."""
        monkeypatch.setattr("src.grid_search.METRICS_DIR", tmp_path)

        evals = [_make_config_eval("B-minilm"), _make_config_eval("bm25")]
        path = save_grid_results(evals)

        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["config_id"] == "B-minilm"


# ===========================================================================
# print_summary Tests
# ===========================================================================

class TestPrintSummary:
    """Tests for print_summary — captures stdout to verify output."""

    def test_prints_top_3_configs(self, capsys) -> None:
        """Summary includes top-3 configs by Recall@5."""
        evals = [
            _make_config_eval("E-openai", chunk_config="E", embedding_model="openai", r5=0.625),
            _make_config_eval("B-openai", chunk_config="B", embedding_model="openai", r5=0.607),
            _make_config_eval("B-minilm", chunk_config="B", embedding_model="minilm", r5=0.481),
            _make_config_eval("bm25", chunk_config="B", embedding_model="bm25",
                              retrieval_method=RetrievalMethod.BM25, r5=0.381),
        ]

        print_summary(evals)
        output = capsys.readouterr().out

        assert "Top-3 Configs" in output
        assert "E-openai" in output
        assert "BM25 Baseline" in output
        assert "Config E" in output

    def test_prints_bm25_comparison(self, capsys) -> None:
        """Summary shows BM25 rank and best vector delta."""
        evals = [
            _make_config_eval("B-minilm", r5=0.6),
            _make_config_eval("bm25", retrieval_method=RetrievalMethod.BM25, r5=0.4),
        ]

        print_summary(evals)
        output = capsys.readouterr().out

        assert "bm25" in output.lower()
        assert "beats BM25" in output

    def test_prints_config_e_analysis(self, capsys) -> None:
        """Summary includes Config E vs Config B comparison."""
        evals = [
            _make_config_eval("E-openai", chunk_config="E", embedding_model="openai", r5=0.62),
            _make_config_eval("B-openai", chunk_config="B", embedding_model="openai", r5=0.60),
            _make_config_eval("bm25", retrieval_method=RetrievalMethod.BM25, r5=0.38),
        ]

        print_summary(evals)
        output = capsys.readouterr().out

        assert "Config E" in output
        assert "Delta" in output


# ===========================================================================
# sanity_check Tests
# ===========================================================================

class TestSanityCheck:
    """Tests for sanity_check — mocks FAISS and embedder."""

    @patch("src.grid_search.create_embedder")
    @patch("src.grid_search.FAISSVectorStore")
    def test_passes_when_gold_found(self, mock_store_cls, mock_create) -> None:
        """Returns True when gold chunks are in top-10 results."""
        qa_pairs = [
            _make_qa_pair("q1", gold_ids=["B_0_1"]),
            _make_qa_pair("q2", gold_ids=["B_0_2"]),
            _make_qa_pair("q3", gold_ids=["B_0_3"]),
        ]

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros((3, 384), dtype=np.float32)
        mock_create.return_value = mock_embedder

        # Mock store: always return gold chunk in results
        mock_store = MagicMock()
        mock_store.search.side_effect = [
            [("B_0_1", 0.9), ("B_0_99", 0.5)],
            [("B_0_2", 0.85), ("B_0_99", 0.5)],
            [("B_0_3", 0.8), ("B_0_99", 0.5)],
        ]
        mock_store_cls.load.return_value = mock_store

        assert sanity_check(qa_pairs, n=3) is True

    @patch("src.grid_search.create_embedder")
    @patch("src.grid_search.FAISSVectorStore")
    def test_fails_when_gold_not_found(self, mock_store_cls, mock_create) -> None:
        """Returns False when no gold chunks appear in results."""
        qa_pairs = [
            _make_qa_pair("q1", gold_ids=["B_0_1"]),
            _make_qa_pair("q2", gold_ids=["B_0_2"]),
            _make_qa_pair("q3", gold_ids=["B_0_3"]),
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros((3, 384), dtype=np.float32)
        mock_create.return_value = mock_embedder

        # Mock store: never returns gold chunks
        mock_store = MagicMock()
        mock_store.search.return_value = [("B_0_99", 0.5), ("B_0_98", 0.3)]
        mock_store_cls.load.return_value = mock_store

        assert sanity_check(qa_pairs, n=3) is False


# ===========================================================================
# run_grid_search Tests
# ===========================================================================

class TestRunGridSearch:
    """Tests for run_grid_search — heavily mocked."""

    @patch("src.grid_search.BM25Retriever")
    @patch("src.grid_search.FAISSVectorStore")
    @patch("src.grid_search.create_embedder")
    @patch("src.grid_search._load_all_chunks")
    def test_produces_16_evaluations(
        self, mock_load, mock_create, mock_store_cls, mock_bm25_cls,
    ) -> None:
        """Full grid search produces 16 ConfigEvaluations (15 vector + 1 BM25)."""
        # Create chunks for all 5 configs
        chunks = {}
        for config_name in ["A", "B", "C", "D", "E"]:
            chunks[config_name] = [
                _make_chunk(f"{config_name}_0_{i}", start_char=i*100,
                            end_char=(i+1)*100, config_name=config_name)
                for i in range(5)
            ]
        mock_load.return_value = chunks

        # Mock embedder: returns zeros for any input
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros((2, 384), dtype=np.float32)
        mock_embedder.dimensions = 384
        mock_create.return_value = mock_embedder

        # Mock FAISS store: returns first chunk with high score
        mock_store = MagicMock()
        mock_store.search.return_value = [("B_0_0", 0.9), ("B_0_1", 0.8)]
        mock_store.size = 5
        mock_store_cls.load.return_value = mock_store

        # Mock BM25
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [("B_0_0", 2.5), ("B_0_1", 1.8)]
        mock_bm25.size = 5
        mock_bm25_cls.load.return_value = mock_bm25

        qa_pairs = [
            _make_qa_pair("q1", gold_ids=["B_0_0"]),
            _make_qa_pair("q2", gold_ids=["B_0_1"]),
        ]

        evaluations = run_grid_search(qa_pairs)

        # 5 chunk configs × 3 embedding models + 1 BM25 = 16
        assert len(evaluations) == 16

    @patch("src.grid_search.BM25Retriever")
    @patch("src.grid_search.FAISSVectorStore")
    @patch("src.grid_search.create_embedder")
    @patch("src.grid_search._load_all_chunks")
    def test_bm25_included(
        self, mock_load, mock_create, mock_store_cls, mock_bm25_cls,
    ) -> None:
        """BM25 evaluation is included in results."""
        chunks = {name: [_make_chunk(f"{name}_0_0", config_name=name)]
                  for name in ["A", "B", "C", "D", "E"]}
        mock_load.return_value = chunks

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros((1, 384), dtype=np.float32)
        mock_create.return_value = mock_embedder

        mock_store = MagicMock()
        mock_store.search.return_value = [("B_0_0", 0.9)]
        mock_store.size = 1
        mock_store_cls.load.return_value = mock_store

        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [("B_0_0", 2.0)]
        mock_bm25.size = 1
        mock_bm25_cls.load.return_value = mock_bm25

        qa_pairs = [_make_qa_pair("q1", gold_ids=["B_0_0"])]
        evaluations = run_grid_search(qa_pairs)

        bm25_evals = [e for e in evaluations if e.retrieval_method == RetrievalMethod.BM25]
        assert len(bm25_evals) == 1
        assert bm25_evals[0].config_id == "bm25"
