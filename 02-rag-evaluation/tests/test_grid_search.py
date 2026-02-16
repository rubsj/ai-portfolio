"""Tests for grid_search.py — Gold chunk mapping and evaluation orchestration.

WHY focused on map_gold_chunks: This is a pure function with complex logic
that's critical for cross-config evaluation. Testing the full orchestration
(main()) requires extensive mocking of file I/O, vector stores, and LLM calls,
so we defer that to integration testing.
"""

from __future__ import annotations

from src.grid_search import map_gold_chunks
from src.models import Chunk


# ===========================================================================
# Helpers
# ===========================================================================

def _make_chunk(
    chunk_id: str,
    start_char: int,
    end_char: int,
    config_name: str = "B",
) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text=f"Text content for {chunk_id}",
        token_count=50,
        start_char=start_char,
        end_char=end_char,
        page_numbers=[1],
        config_name=config_name,
    )


# ===========================================================================
# map_gold_chunks Tests
# ===========================================================================

class TestMapGoldChunks:
    """Tests for gold chunk ID mapping across configs."""

    def test_exact_overlap_maps_correctly(self) -> None:
        """Chunk with 100% overlap maps to target config."""
        # Gold B chunk: chars 0-1000
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=1000, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target config has exact same span
        target_chunks = [
            _make_chunk("C_0_1", start_char=0, end_char=1000, config_name="C"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks)

        assert result == ["C_0_1"]

    def test_partial_overlap_above_threshold_maps(self) -> None:
        """Chunk with ≥50% overlap maps to target."""
        # Gold B chunk: chars 0-1000
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=1000, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target chunk overlaps 600 chars (60% of gold span)
        # Gold: [0, 1000], Target: [400, 1200] → overlap: [400, 1000] = 600 chars
        target_chunks = [
            _make_chunk("D_0_1", start_char=400, end_char=1200, config_name="D"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks, overlap_threshold=0.5)

        assert result == ["D_0_1"]

    def test_partial_overlap_below_threshold_skips(self) -> None:
        """Chunk with <50% overlap is not mapped."""
        # Gold B chunk: chars 0-1000
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=1000, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target chunk overlaps 400 chars (40% of gold span) — below 50% threshold
        # Gold: [0, 1000], Target: [600, 1500] → overlap: [600, 1000] = 400 chars
        target_chunks = [
            _make_chunk("A_0_1", start_char=600, end_char=1500, config_name="A"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks, overlap_threshold=0.5)

        assert result == []

    def test_multiple_gold_chunks_map_to_multiple_targets(self) -> None:
        """Multiple gold chunks can map to multiple target chunks."""
        # Two gold B chunks in different positions
        gold_b1 = _make_chunk("B_0_1", start_char=0, end_char=500, config_name="B")
        gold_b2 = _make_chunk("B_0_2", start_char=1000, end_char=1500, config_name="B")
        b_lookup = {"B_0_1": gold_b1, "B_0_2": gold_b2}

        # Two target chunks that overlap each gold chunk
        target_chunks = [
            _make_chunk("E_0_1", start_char=0, end_char=600, config_name="E"),
            _make_chunk("E_0_2", start_char=900, end_char=1600, config_name="E"),
        ]

        result = map_gold_chunks(["B_0_1", "B_0_2"], b_lookup, target_chunks, overlap_threshold=0.5)

        # Both targets should be mapped
        assert set(result) == {"E_0_1", "E_0_2"}

    def test_multiple_gold_chunks_map_to_same_target_deduplicated(self) -> None:
        """Multiple gold chunks can map to the same target (deduplicated)."""
        # Two small gold B chunks
        gold_b1 = _make_chunk("B_0_1", start_char=0, end_char=300, config_name="B")
        gold_b2 = _make_chunk("B_0_2", start_char=200, end_char=500, config_name="B")
        b_lookup = {"B_0_1": gold_b1, "B_0_2": gold_b2}

        # One large target chunk that covers both
        target_chunks = [
            _make_chunk("C_0_1", start_char=0, end_char=1000, config_name="C"),
        ]

        result = map_gold_chunks(["B_0_1", "B_0_2"], b_lookup, target_chunks, overlap_threshold=0.5)

        # WHY only one result: both gold chunks map to C_0_1, deduplicated
        assert result == ["C_0_1"]

    def test_different_document_chunks_dont_match(self) -> None:
        """Chunks from different documents never match (even if chars overlap)."""
        # Gold B chunk from doc 0
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=1000, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target chunk from doc 1 (different document index)
        target_chunks = [
            _make_chunk("A_1_1", start_char=0, end_char=1000, config_name="A"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks)

        # No match because doc indices differ (0 vs 1)
        assert result == []

    def test_same_document_different_spans_no_overlap(self) -> None:
        """Chunks from same document with no overlap don't match."""
        # Gold B chunk: chars 0-500
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=500, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target chunk: chars 1000-1500 (no overlap)
        target_chunks = [
            _make_chunk("D_0_1", start_char=1000, end_char=1500, config_name="D"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks)

        assert result == []

    def test_gold_chunk_not_in_lookup_skips_with_warning(self) -> None:
        """Missing gold chunk ID logs warning and is skipped."""
        b_lookup = {}  # Empty lookup — gold chunk doesn't exist

        target_chunks = [
            _make_chunk("C_0_1", start_char=0, end_char=1000, config_name="C"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks)

        # No crash, returns empty list
        assert result == []

    def test_zero_span_gold_chunk_cannot_be_created(self) -> None:
        """Pydantic validator prevents zero-span chunks at creation time.

        WHY removed from map_gold_chunks test: The guard clause `if gold_span <= 0`
        in map_gold_chunks() is defensive programming, but in practice Pydantic's
        @field_validator on Chunk prevents zero/negative spans at construction.
        Testing the validator directly belongs in test_models.py, not here.
        """
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            _make_chunk("B_0_1", start_char=500, end_char=500, config_name="B")

        assert "end_char" in str(exc_info.value)
        assert "must be greater than start_char" in str(exc_info.value)

    def test_custom_overlap_threshold(self) -> None:
        """Custom overlap threshold (e.g., 0.7) is respected."""
        # Gold B chunk: chars 0-1000
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=1000, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target chunk overlaps 600 chars (60% of gold span)
        target_chunks = [
            _make_chunk("D_0_1", start_char=400, end_char=1200, config_name="D"),
        ]

        # With 0.7 threshold, 60% overlap is not enough
        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks, overlap_threshold=0.7)
        assert result == []

        # With 0.5 threshold, 60% overlap is sufficient
        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks, overlap_threshold=0.5)
        assert result == ["D_0_1"]

    def test_edge_touching_spans_no_overlap(self) -> None:
        """Adjacent chunks that touch but don't overlap should not match."""
        # Gold B chunk: chars 0-500
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=500, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target chunk: chars 500-1000 (edge touch, no overlap)
        target_chunks = [
            _make_chunk("A_0_1", start_char=500, end_char=1000, config_name="A"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks)

        # No overlap → no match
        assert result == []

    def test_one_char_overlap_not_enough(self) -> None:
        """Single character overlap (below threshold) doesn't match."""
        # Gold B chunk: chars 0-1000
        gold_b = _make_chunk("B_0_1", start_char=0, end_char=1000, config_name="B")
        b_lookup = {"B_0_1": gold_b}

        # Target chunk overlaps by just 1 char (0.1% of 1000)
        # Gold: [0, 1000], Target: [999, 1500] → overlap: [999, 1000] = 1 char
        target_chunks = [
            _make_chunk("C_0_1", start_char=999, end_char=1500, config_name="C"),
        ]

        result = map_gold_chunks(["B_0_1"], b_lookup, target_chunks, overlap_threshold=0.5)

        assert result == []


# ===========================================================================
# sanity_check Tests
# ===========================================================================

class TestSanityCheck:
    """Tests for QA dataset sanity check before grid search."""

    def test_passes_when_gold_chunks_retrievable(self, monkeypatch) -> None:
        """Returns True when ≥2/3 QA pairs retrieve their gold chunks."""
        from unittest.mock import MagicMock, patch
        import numpy as np
        from src.grid_search import sanity_check
        from src.models import QuestionType, QuestionHierarchy, SyntheticQAPair

        # Create 3 QA pairs
        qa_pairs = [
            SyntheticQAPair(
                id="q1",
                question="Question 1?",
                question_type=QuestionType.FACTUAL,
                hierarchy=QuestionHierarchy.PARAGRAPH,
                gold_chunk_ids=["B_0_1"],
                expected_answer="Answer 1",
                source_chunk_text="Source 1",
                generation_strategy="per_chunk_chain",
            ),
            SyntheticQAPair(
                id="q2",
                question="Question 2?",
                question_type=QuestionType.COMPARATIVE,
                hierarchy=QuestionHierarchy.SECTION,
                gold_chunk_ids=["B_0_2"],
                expected_answer="Answer 2",
                source_chunk_text="Source 2",
                generation_strategy="multi_chunk",
            ),
            SyntheticQAPair(
                id="q3",
                question="Question 3?",
                question_type=QuestionType.ANALYTICAL,
                hierarchy=QuestionHierarchy.PAGE,
                gold_chunk_ids=["B_0_3"],
                expected_answer="Answer 3",
                source_chunk_text="Source 3",
                generation_strategy="overlap_region",
            ),
        ]

        # Mock FAISS store
        mock_store = MagicMock()
        # First 2 queries retrieve gold chunks, 3rd doesn't
        mock_store.search.side_effect = [
            [("B_0_1", 0.9), ("B_0_5", 0.8)],  # Hit: B_0_1 in results
            [("B_0_2", 0.95), ("B_0_6", 0.7)],  # Hit: B_0_2 in results
            [("B_0_7", 0.85), ("B_0_8", 0.75)],  # Miss: B_0_3 not in results
        ]

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(3, 384).astype(np.float32)

        with patch("src.grid_search.FAISSVectorStore") as mock_store_cls, \
             patch("src.grid_search.create_embedder") as mock_create_embedder:
            mock_store_cls.load.return_value = mock_store
            mock_create_embedder.return_value = mock_embedder

            result = sanity_check(qa_pairs, n=3)

        # 2/3 passed (threshold is min(2, n=3) = 2) → True
        assert result is True

    def test_fails_when_gold_chunks_not_retrievable(self, monkeypatch) -> None:
        """Returns False when <2/3 QA pairs retrieve their gold chunks."""
        from unittest.mock import MagicMock, patch
        import numpy as np
        from src.grid_search import sanity_check
        from src.models import QuestionType, QuestionHierarchy, SyntheticQAPair

        qa_pairs = [
            SyntheticQAPair(
                id="q1",
                question="Question 1?",
                question_type=QuestionType.FACTUAL,
                hierarchy=QuestionHierarchy.PARAGRAPH,
                gold_chunk_ids=["B_0_1"],
                expected_answer="Answer 1",
                source_chunk_text="Source 1",
                generation_strategy="per_chunk_chain",
            ),
            SyntheticQAPair(
                id="q2",
                question="Question 2?",
                question_type=QuestionType.COMPARATIVE,
                hierarchy=QuestionHierarchy.SECTION,
                gold_chunk_ids=["B_0_2"],
                expected_answer="Answer 2",
                source_chunk_text="Source 2",
                generation_strategy="multi_chunk",
            ),
            SyntheticQAPair(
                id="q3",
                question="Question 3?",
                question_type=QuestionType.ANALYTICAL,
                hierarchy=QuestionHierarchy.PAGE,
                gold_chunk_ids=["B_0_3"],
                expected_answer="Answer 3",
                source_chunk_text="Source 3",
                generation_strategy="overlap_region",
            ),
        ]

        # Mock FAISS store - only 1 query retrieves gold chunk
        mock_store = MagicMock()
        mock_store.search.side_effect = [
            [("B_0_1", 0.9), ("B_0_5", 0.8)],  # Hit: B_0_1 in results
            [("B_0_7", 0.85), ("B_0_8", 0.75)],  # Miss: B_0_2 not in results
            [("B_0_9", 0.80), ("B_0_10", 0.70)],  # Miss: B_0_3 not in results
        ]

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(3, 384).astype(np.float32)

        with patch("src.grid_search.FAISSVectorStore") as mock_store_cls, \
             patch("src.grid_search.create_embedder") as mock_create_embedder:
            mock_store_cls.load.return_value = mock_store
            mock_create_embedder.return_value = mock_embedder

            result = sanity_check(qa_pairs, n=3)

        # Only 1/3 passed (threshold is 2) → False
        assert result is False
