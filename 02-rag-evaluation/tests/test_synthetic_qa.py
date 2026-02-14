"""Tests for synthetic_qa.py — helper functions, QA quality, and gold mapping.

Tests the pure helper functions (_sample_diverse_chunks, _find_overlap_pairs,
_find_semantically_similar_chunks) and compute_qa_quality without LLM calls.
Strategy functions are tested via mock Instructor client.

Java/TS parallel: like unit tests for a service with @MockBean for the LLM
client. Pure helpers are tested directly, LLM-dependent functions are tested
with mocked responses.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import (
    Chunk,
    QADatasetReport,
    QuestionHierarchy,
    QuestionType,
    SyntheticQAPair,
)
from src.synthetic_qa import (
    _find_overlap_pairs,
    _find_semantically_similar_chunks,
    _sample_diverse_chunks,
    compute_qa_quality,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_chunk(
    chunk_id: str,
    text: str = "Some meaningful text content here",
    start_char: int = 0,
    end_char: int = 100,
    config_name: str = "B",
) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text=text,
        token_count=max(1, len(text.split())),
        start_char=start_char,
        end_char=end_char,
        page_numbers=[1],
        config_name=config_name,
    )


def _make_qa_pair(
    qa_id: str = "q1",
    gold_ids: list[str] | None = None,
    question_type: QuestionType = QuestionType.FACTUAL,
    hierarchy: QuestionHierarchy = QuestionHierarchy.PARAGRAPH,
    strategy: str = "per_chunk_chain",
    is_overlap: bool = False,
) -> SyntheticQAPair:
    """Create a minimal SyntheticQAPair for testing."""
    return SyntheticQAPair(
        id=qa_id,
        question="What is the primary purpose of this content?",
        question_type=question_type,
        hierarchy=hierarchy,
        gold_chunk_ids=gold_ids or ["B_0_1"],
        expected_answer="The primary purpose is described in detail.",
        source_chunk_text="Some meaningful text content here.",
        is_overlap_region=is_overlap,
        generation_strategy=strategy,
    )


# ===========================================================================
# _sample_diverse_chunks Tests
# ===========================================================================

class TestSampleDiverseChunks:
    """Tests for _sample_diverse_chunks."""

    def test_returns_n_chunks(self) -> None:
        """Samples exactly n chunks when enough are available."""
        chunks = [_make_chunk(f"B_0_{i}") for i in range(20)]
        result = _sample_diverse_chunks(chunks, n=8)
        assert len(result) == 8

    def test_spreads_across_documents(self) -> None:
        """Samples from all available documents, not just one."""
        # 3 docs with 10 chunks each
        chunks = []
        for doc in range(3):
            for i in range(10):
                chunks.append(_make_chunk(f"B_{doc}_{i}"))

        result = _sample_diverse_chunks(chunks, n=6)

        # Should have chunks from all 3 documents
        doc_ids = {c.id.split("_")[1] for c in result}
        assert len(doc_ids) == 3

    def test_handles_fewer_chunks_than_n(self) -> None:
        """Returns all chunks when fewer than n are available."""
        chunks = [_make_chunk(f"B_0_{i}") for i in range(3)]
        result = _sample_diverse_chunks(chunks, n=8)
        assert len(result) == 3

    def test_samples_from_single_doc(self) -> None:
        """Works with chunks from a single document."""
        chunks = [_make_chunk(f"B_0_{i}") for i in range(20)]
        result = _sample_diverse_chunks(chunks, n=5)
        assert len(result) == 5


# ===========================================================================
# _find_overlap_pairs Tests
# ===========================================================================

class TestFindOverlapPairs:
    """Tests for _find_overlap_pairs."""

    def test_finds_overlapping_consecutive_chunks(self) -> None:
        """Detects overlap between consecutive chunks from the same doc."""
        c1 = _make_chunk("B_0_0", start_char=0, end_char=200)
        c2 = _make_chunk("B_0_1", start_char=150, end_char=350)
        c3 = _make_chunk("B_0_2", start_char=300, end_char=500)

        pairs = _find_overlap_pairs([c1, c2, c3])

        # c1-c2 overlap: 150-200 = 50 chars
        # c2-c3 overlap: 300-350 = 50 chars
        assert len(pairs) == 2

    def test_skips_different_documents(self) -> None:
        """No overlap between chunks from different documents."""
        c1 = _make_chunk("B_0_0", start_char=0, end_char=200)
        c2 = _make_chunk("B_1_0", start_char=150, end_char=350)

        pairs = _find_overlap_pairs([c1, c2])
        assert len(pairs) == 0

    def test_no_overlap_when_no_char_overlap(self) -> None:
        """Non-overlapping consecutive chunks produce no pairs."""
        c1 = _make_chunk("B_0_0", start_char=0, end_char=100)
        c2 = _make_chunk("B_0_1", start_char=100, end_char=200)

        pairs = _find_overlap_pairs([c1, c2])
        assert len(pairs) == 0

    def test_sorted_by_overlap_descending(self) -> None:
        """Pairs are sorted by overlap size, largest first."""
        c1 = _make_chunk("B_0_0", start_char=0, end_char=200)
        c2 = _make_chunk("B_0_1", start_char=100, end_char=400)   # 100 overlap with c1
        c3 = _make_chunk("B_0_2", start_char=350, end_char=500)   # 50 overlap with c2

        pairs = _find_overlap_pairs([c1, c2, c3])

        assert len(pairs) == 2
        # First pair should have larger overlap (100 > 50)
        assert pairs[0] == (c1, c2)
        assert pairs[1] == (c2, c3)

    def test_empty_chunks(self) -> None:
        """Empty chunk list produces no pairs."""
        assert _find_overlap_pairs([]) == []


# ===========================================================================
# _find_semantically_similar_chunks Tests
# ===========================================================================

class TestFindSemanticallySimilarChunks:
    """Tests for _find_semantically_similar_chunks."""

    def test_returns_top_k_indices(self) -> None:
        """Returns exactly top_k indices."""
        # 5 unit vectors where 0 and 1 are similar, 0 and 4 are different
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)
        # L2 normalize for proper cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        chunk_ids = [f"B_0_{i}" for i in range(5)]

        result = _find_semantically_similar_chunks(0, embeddings, chunk_ids, top_k=2)

        assert len(result) == 2

    def test_excludes_self(self) -> None:
        """Source chunk is never in the results."""
        embeddings = np.eye(5, dtype=np.float32)
        chunk_ids = [f"B_0_{i}" for i in range(5)]

        result = _find_semantically_similar_chunks(0, embeddings, chunk_ids, top_k=3)

        assert 0 not in result

    def test_most_similar_first(self) -> None:
        """Results are ordered by descending similarity."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.99, 0.1],   # Most similar to 0
            [0.5, 0.5],    # Moderately similar
            [0.0, 1.0],    # Least similar
        ], dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        chunk_ids = [f"B_0_{i}" for i in range(4)]

        result = _find_semantically_similar_chunks(0, embeddings, chunk_ids, top_k=3)

        # Index 1 should be first (most similar to index 0)
        assert result[0] == 1


# ===========================================================================
# compute_qa_quality Tests
# ===========================================================================

class TestComputeQAQuality:
    """Tests for compute_qa_quality."""

    def test_total_question_count(self) -> None:
        """Reports the correct total number of questions."""
        pairs = [_make_qa_pair(qa_id=f"q{i}") for i in range(10)]
        report = compute_qa_quality(pairs, total_chunks=100)
        assert report.total_questions == 10

    def test_strategy_distribution(self) -> None:
        """Correctly counts questions per strategy."""
        pairs = [
            _make_qa_pair(qa_id="q1", strategy="per_chunk_chain"),
            _make_qa_pair(qa_id="q2", strategy="per_chunk_chain"),
            _make_qa_pair(qa_id="q3", strategy="multi_chunk"),
        ]
        report = compute_qa_quality(pairs, total_chunks=100)

        assert report.questions_per_strategy["per_chunk_chain"] == 2
        assert report.questions_per_strategy["multi_chunk"] == 1

    def test_type_distribution(self) -> None:
        """Correctly counts questions per QuestionType."""
        pairs = [
            _make_qa_pair(qa_id="q1", question_type=QuestionType.FACTUAL),
            _make_qa_pair(qa_id="q2", question_type=QuestionType.FACTUAL),
            _make_qa_pair(qa_id="q3", question_type=QuestionType.ANALYTICAL),
        ]
        report = compute_qa_quality(pairs, total_chunks=100)

        assert report.questions_per_type[QuestionType.FACTUAL] == 2
        assert report.questions_per_type[QuestionType.ANALYTICAL] == 1

    def test_hierarchy_distribution(self) -> None:
        """Correctly counts questions per QuestionHierarchy."""
        pairs = [
            _make_qa_pair(qa_id="q1", hierarchy=QuestionHierarchy.PARAGRAPH),
            _make_qa_pair(qa_id="q2", hierarchy=QuestionHierarchy.SECTION),
        ]
        report = compute_qa_quality(pairs, total_chunks=100)

        assert report.questions_per_hierarchy[QuestionHierarchy.PARAGRAPH] == 1
        assert report.questions_per_hierarchy[QuestionHierarchy.SECTION] == 1

    def test_chunk_coverage_percent(self) -> None:
        """Coverage = unique gold chunk IDs / total chunks × 100."""
        pairs = [
            _make_qa_pair(qa_id="q1", gold_ids=["B_0_1"]),
            _make_qa_pair(qa_id="q2", gold_ids=["B_0_1", "B_0_2"]),
            _make_qa_pair(qa_id="q3", gold_ids=["B_0_3"]),
        ]
        # Unique gold IDs: B_0_1, B_0_2, B_0_3 = 3 out of 10
        report = compute_qa_quality(pairs, total_chunks=10)
        assert report.chunk_coverage_percent == 30.0

    def test_overlap_question_count(self) -> None:
        """Counts questions where is_overlap_region=True."""
        pairs = [
            _make_qa_pair(qa_id="q1", is_overlap=True),
            _make_qa_pair(qa_id="q2", is_overlap=False),
            _make_qa_pair(qa_id="q3", is_overlap=True),
        ]
        report = compute_qa_quality(pairs, total_chunks=100)
        assert report.overlap_question_count == 2

    def test_avg_questions_per_chunk(self) -> None:
        """Average = total_questions / total_chunks."""
        pairs = [_make_qa_pair(qa_id=f"q{i}") for i in range(5)]
        report = compute_qa_quality(pairs, total_chunks=50)
        assert report.avg_questions_per_chunk == 0.1

    def test_zero_chunks_returns_zero_coverage(self) -> None:
        """No chunks → 0% coverage, 0 avg per chunk."""
        pairs = [_make_qa_pair(qa_id="q1")]
        report = compute_qa_quality(pairs, total_chunks=0)
        assert report.chunk_coverage_percent == 0.0
        assert report.avg_questions_per_chunk == 0.0


# ===========================================================================
# Grid Search: map_gold_chunks Tests
# ===========================================================================

class TestMapGoldChunks:
    """Tests for grid_search.map_gold_chunks — cross-config gold mapping."""

    def test_exact_overlap_maps_correctly(self) -> None:
        """Target chunk fully covers gold chunk → mapped."""
        from src.grid_search import map_gold_chunks

        gold_b = _make_chunk("B_0_5", start_char=100, end_char=300)
        target = _make_chunk("A_0_10", start_char=100, end_char=300, config_name="A")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_5"],
            b_chunks_lookup={"B_0_5": gold_b},
            target_chunks=[target],
        )
        assert "A_0_10" in mapped

    def test_partial_overlap_above_threshold(self) -> None:
        """≥50% overlap → mapped."""
        from src.grid_search import map_gold_chunks

        gold_b = _make_chunk("B_0_5", start_char=100, end_char=300)
        # Overlaps 150 out of 200 chars = 75%
        target = _make_chunk("C_0_2", start_char=150, end_char=400, config_name="C")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_5"],
            b_chunks_lookup={"B_0_5": gold_b},
            target_chunks=[target],
        )
        assert "C_0_2" in mapped

    def test_partial_overlap_below_threshold(self) -> None:
        """<50% overlap → not mapped."""
        from src.grid_search import map_gold_chunks

        gold_b = _make_chunk("B_0_5", start_char=100, end_char=300)
        # Overlaps 50 out of 200 chars = 25%
        target = _make_chunk("C_0_2", start_char=250, end_char=500, config_name="C")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_5"],
            b_chunks_lookup={"B_0_5": gold_b},
            target_chunks=[target],
        )
        assert "C_0_2" not in mapped

    def test_different_docs_dont_match(self) -> None:
        """Chunks from different documents never match, even with same offsets."""
        from src.grid_search import map_gold_chunks

        gold_b = _make_chunk("B_0_5", start_char=100, end_char=300)
        # Same offsets but doc_idx=1 (gold is doc_idx=0)
        target = _make_chunk("A_1_5", start_char=100, end_char=300, config_name="A")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_5"],
            b_chunks_lookup={"B_0_5": gold_b},
            target_chunks=[target],
        )
        assert len(mapped) == 0

    def test_missing_gold_chunk_skipped(self) -> None:
        """Gold ID not found in B lookup → skipped gracefully."""
        from src.grid_search import map_gold_chunks

        target = _make_chunk("A_0_5", start_char=100, end_char=300, config_name="A")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_99"],  # Doesn't exist in lookup
            b_chunks_lookup={},
            target_chunks=[target],
        )
        assert len(mapped) == 0

    def test_multiple_gold_chunks_mapped(self) -> None:
        """Multiple gold B chunks can map to different target chunks."""
        from src.grid_search import map_gold_chunks

        gold_b1 = _make_chunk("B_0_1", start_char=0, end_char=200)
        gold_b2 = _make_chunk("B_0_2", start_char=200, end_char=400)
        target1 = _make_chunk("A_0_1", start_char=0, end_char=200, config_name="A")
        target2 = _make_chunk("A_0_3", start_char=200, end_char=400, config_name="A")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_1", "B_0_2"],
            b_chunks_lookup={"B_0_1": gold_b1, "B_0_2": gold_b2},
            target_chunks=[target1, target2],
        )
        assert set(mapped) == {"A_0_1", "A_0_3"}

    def test_deduplicates_mapped_ids(self) -> None:
        """Same target chunk matched by multiple gold chunks → appears once."""
        from src.grid_search import map_gold_chunks

        gold_b1 = _make_chunk("B_0_1", start_char=0, end_char=200)
        gold_b2 = _make_chunk("B_0_2", start_char=100, end_char=300)
        # Large target covers both gold chunks
        target = _make_chunk("C_0_0", start_char=0, end_char=500, config_name="C")

        mapped = map_gold_chunks(
            gold_b_ids=["B_0_1", "B_0_2"],
            b_chunks_lookup={"B_0_1": gold_b1, "B_0_2": gold_b2},
            target_chunks=[target],
        )
        # Should appear exactly once even though both gold chunks map to it
        assert mapped.count("C_0_0") == 1
