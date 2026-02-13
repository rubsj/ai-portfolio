"""Tests for bm25_baseline.py — BM25Okapi lexical retrieval.

Uses real BM25Okapi (no mocks) — it's a pure Python library with no
external dependencies. Tests verify search ranking, scoring, tokenization,
and save/load roundtrip.
"""

from __future__ import annotations

import pytest

from src.bm25_baseline import BM25Retriever, _tokenize
from src.models import Chunk


# ===========================================================================
# Helpers
# ===========================================================================

def _make_chunk(chunk_id: str, text: str, config_name: str = "B") -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text=text,
        token_count=len(text.split()),
        start_char=0,
        end_char=len(text),
        page_numbers=[1],
        config_name=config_name,
    )


def _sample_chunks() -> list[Chunk]:
    """Create a small corpus of chunks for BM25 testing."""
    return [
        _make_chunk("B_0", "The total revenue for the financial quarter was impressive"),
        _make_chunk("B_1", "Healthcare spending increased by twenty percent this year"),
        _make_chunk("B_2", "Machine learning models require large datasets for training"),
        _make_chunk("B_3", "The revenue growth was driven by technology investments"),
        _make_chunk("B_4", "Patient care quality improved across all hospital departments"),
    ]


# ===========================================================================
# Tokenization Tests
# ===========================================================================

class TestTokenization:
    """Verify BM25 tokenization behavior."""

    def test_tokenize_lowercases(self) -> None:
        """Tokenizer lowercases all text."""
        tokens = _tokenize("Hello World FOO")
        assert tokens == ["hello", "world", "foo"]

    def test_tokenize_splits_on_whitespace(self) -> None:
        """Tokenizer splits on whitespace."""
        tokens = _tokenize("one  two\tthree\nfour")
        assert tokens == ["one", "two", "three", "four"]

    def test_tokenize_empty_string(self) -> None:
        """Empty string produces empty token list."""
        tokens = _tokenize("")
        assert tokens == []


# ===========================================================================
# Search Tests
# ===========================================================================

class TestBM25Search:
    """Verify BM25 search ranking and scoring."""

    def test_search_returns_results(self) -> None:
        """Search returns up to k results."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("revenue growth", k=3)
        assert len(results) <= 3
        assert len(results) > 0

    def test_exact_match_ranks_first(self) -> None:
        """Query matching a chunk's text ranks that chunk highest."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)
        # "total revenue financial quarter" should match B_0 most strongly
        results = retriever.search("total revenue financial quarter", k=3)
        top_id = results[0][0]
        assert top_id == "B_0"

    def test_scores_nonnegative(self) -> None:
        """All BM25 scores >= 0."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("revenue", k=5)
        for _, score in results:
            assert score >= 0.0

    def test_scores_descending(self) -> None:
        """Results are sorted by descending score."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("healthcare spending", k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_k_greater_than_corpus(self) -> None:
        """k > corpus size returns all documents, no crash."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)
        results = retriever.search("test", k=100)
        assert len(results) == len(chunks)

    def test_search_empty_corpus(self) -> None:
        """Searching an empty corpus returns empty list."""
        retriever = BM25Retriever([])
        results = retriever.search("hello", k=5)
        assert results == []

    def test_size_property(self) -> None:
        """size property reflects number of indexed documents."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)
        assert retriever.size == 5


# ===========================================================================
# Save / Load Roundtrip Tests
# ===========================================================================

class TestSaveAndLoad:
    """Verify persistence and loading."""

    def test_save_load_roundtrip(self, tmp_path) -> None:
        """Pickle save/load produces same search results."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)

        # Search before save
        results_before = retriever.search("revenue growth", k=3)

        # Save and load
        path = tmp_path / "bm25_B"
        retriever.save(path)
        loaded = BM25Retriever.load(path)

        # Search after load
        results_after = loaded.search("revenue growth", k=3)

        assert len(results_before) == len(results_after)
        for (id_b, score_b), (id_a, score_a) in zip(results_before, results_after):
            assert id_b == id_a
            assert score_b == pytest.approx(score_a, abs=1e-6)

    def test_save_creates_two_files(self, tmp_path) -> None:
        """save() creates .pkl and .json files."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)

        path = tmp_path / "bm25_test"
        retriever.save(path)

        assert (tmp_path / "bm25_test.pkl").exists()
        assert (tmp_path / "bm25_test.json").exists()

    def test_loaded_size_matches(self, tmp_path) -> None:
        """Loaded retriever has same size as original."""
        chunks = _sample_chunks()
        retriever = BM25Retriever(chunks)

        path = tmp_path / "bm25_size"
        retriever.save(path)
        loaded = BM25Retriever.load(path)

        assert loaded.size == retriever.size
