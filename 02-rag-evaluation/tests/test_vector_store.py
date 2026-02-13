"""Tests for vector_store.py — FAISS IndexFlatIP vector store.

No mocks needed — FAISS is CPU-only and fast. Tests use small random vectors
to verify add/search/save/load/validation behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vector_store import FAISSVectorStore


# ===========================================================================
# Helpers
# ===========================================================================

def _normalized_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """Generate random L2-normalized float32 vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


# ===========================================================================
# Creation Tests
# ===========================================================================

class TestCreateEmptyIndex:
    """Verify empty index initialization."""

    def test_empty_index_has_zero_size(self) -> None:
        store = FAISSVectorStore(dimension=384)
        assert store.size == 0

    def test_dimension_property(self) -> None:
        store = FAISSVectorStore(dimension=768)
        assert store.dimension == 768


# ===========================================================================
# Add and Search Tests
# ===========================================================================

class TestAddAndSearch:
    """Verify adding vectors and searching the index."""

    def test_add_increases_size(self) -> None:
        store = FAISSVectorStore(dimension=8)
        vectors = _normalized_vectors(5, 8)
        store.add(vectors, [f"chunk_{i}" for i in range(5)])
        assert store.size == 5

    def test_add_multiple_batches(self) -> None:
        """Multiple add() calls accumulate vectors."""
        store = FAISSVectorStore(dimension=8)
        store.add(_normalized_vectors(3, 8, seed=1), ["a", "b", "c"])
        store.add(_normalized_vectors(2, 8, seed=2), ["d", "e"])
        assert store.size == 5

    def test_search_returns_correct_chunk_ids(self) -> None:
        """Search returns chunk IDs that were added."""
        store = FAISSVectorStore(dimension=8)
        vectors = _normalized_vectors(5, 8)
        ids = ["c0", "c1", "c2", "c3", "c4"]
        store.add(vectors, ids)

        # Query with one of the stored vectors — should find itself as top result
        results = store.search(vectors[2], k=3)
        returned_ids = [r[0] for r in results]
        assert "c2" in returned_ids

    def test_search_self_has_score_near_one(self) -> None:
        """Searching with a stored vector should return score ~1.0 for itself."""
        store = FAISSVectorStore(dimension=8)
        vectors = _normalized_vectors(5, 8)
        store.add(vectors, ["c0", "c1", "c2", "c3", "c4"])

        results = store.search(vectors[0], k=1)
        chunk_id, score = results[0]
        assert chunk_id == "c0"
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_search_scores_descending(self) -> None:
        """Scores are returned in descending order."""
        store = FAISSVectorStore(dimension=8)
        vectors = _normalized_vectors(10, 8)
        store.add(vectors, [f"c{i}" for i in range(10)])

        results = store.search(vectors[0], k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_k_greater_than_total(self) -> None:
        """k > stored vectors returns all stored vectors, no crash."""
        store = FAISSVectorStore(dimension=8)
        vectors = _normalized_vectors(3, 8)
        store.add(vectors, ["a", "b", "c"])

        results = store.search(vectors[0], k=100)
        assert len(results) == 3

    def test_search_empty_index_returns_empty(self) -> None:
        """Searching an empty index returns empty list."""
        store = FAISSVectorStore(dimension=8)
        query = _normalized_vectors(1, 8)[0]
        results = store.search(query, k=5)
        assert results == []

    def test_chunk_id_mapping_preserved(self) -> None:
        """Returned IDs match what was added in the correct order."""
        store = FAISSVectorStore(dimension=4)
        # Use distinct, easily identifiable vectors
        v1 = np.array([[1, 0, 0, 0]], dtype=np.float32)
        v2 = np.array([[0, 1, 0, 0]], dtype=np.float32)
        store.add(v1, ["first"])
        store.add(v2, ["second"])

        # Query v1 — should return "first" with highest score
        results = store.search(v1[0], k=2)
        assert results[0][0] == "first"
        assert results[1][0] == "second"


# ===========================================================================
# Validation Tests
# ===========================================================================

class TestValidation:
    """Verify input validation in add()."""

    def test_dimension_mismatch_raises(self) -> None:
        """Adding vectors with wrong dimension raises ValueError."""
        store = FAISSVectorStore(dimension=384)
        wrong_dim = _normalized_vectors(3, 768)  # 768 != 384

        with pytest.raises(ValueError, match="dimension"):
            store.add(wrong_dim, ["a", "b", "c"])

    def test_count_mismatch_raises(self) -> None:
        """Mismatched embedding count and chunk ID count raises ValueError."""
        store = FAISSVectorStore(dimension=8)
        vectors = _normalized_vectors(3, 8)

        with pytest.raises(ValueError, match="chunk IDs"):
            store.add(vectors, ["a", "b"])  # 3 vectors, 2 IDs

    def test_1d_array_raises(self) -> None:
        """1D array raises ValueError (must be 2D)."""
        store = FAISSVectorStore(dimension=8)
        vector_1d = np.ones(8, dtype=np.float32)

        with pytest.raises(ValueError, match="2D"):
            store.add(vector_1d, ["a"])


# ===========================================================================
# Save / Load Roundtrip Tests
# ===========================================================================

class TestSaveAndLoad:
    """Verify persistence to disk and loading back."""

    def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """Save to disk, load back, search produces same results."""
        store = FAISSVectorStore(dimension=8)
        vectors = _normalized_vectors(5, 8)
        ids = ["c0", "c1", "c2", "c3", "c4"]
        store.add(vectors, ids)

        # Search before save
        results_before = store.search(vectors[0], k=3)

        # Save and load
        path = tmp_path / "test_index"
        store.save(path)
        loaded_store = FAISSVectorStore.load(path)

        # Search after load should produce identical results
        results_after = loaded_store.search(vectors[0], k=3)

        assert len(results_before) == len(results_after)
        for (id_b, score_b), (id_a, score_a) in zip(results_before, results_after):
            assert id_b == id_a
            assert score_b == pytest.approx(score_a, abs=1e-6)

    def test_loaded_store_has_correct_properties(self, tmp_path) -> None:
        """Loaded store preserves dimension and size."""
        store = FAISSVectorStore(dimension=384)
        vectors = _normalized_vectors(10, 384)
        store.add(vectors, [f"c{i}" for i in range(10)])

        path = tmp_path / "props_test"
        store.save(path)
        loaded = FAISSVectorStore.load(path)

        assert loaded.dimension == 384
        assert loaded.size == 10

    def test_save_creates_two_files(self, tmp_path) -> None:
        """save() creates .faiss and .json sidecar files."""
        store = FAISSVectorStore(dimension=8)
        store.add(_normalized_vectors(2, 8), ["a", "b"])

        path = tmp_path / "idx"
        store.save(path)

        assert (tmp_path / "idx.faiss").exists()
        assert (tmp_path / "idx.json").exists()

    def test_save_creates_parent_dirs(self, tmp_path) -> None:
        """save() creates parent directories if they don't exist."""
        store = FAISSVectorStore(dimension=8)
        store.add(_normalized_vectors(1, 8), ["a"])

        path = tmp_path / "nested" / "dir" / "idx"
        store.save(path)

        assert (tmp_path / "nested" / "dir" / "idx.faiss").exists()
