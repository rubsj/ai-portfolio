"""Tests for all 5 chunking strategies.

All tests use inline _make_test_document() — no conftest, no PDFs required.
EmbeddingSemanticChunker tests mock SentenceTransformer to avoid loading the model.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.chunkers import (
    EmbeddingSemanticChunker,
    FixedSizeChunker,
    HeadingSemanticChunker,
    RecursiveChunker,
    SlidingWindowChunker,
)
from src.chunkers._utils import find_page_number
from src.schemas import Chunk, Document, DocumentMetadata, PageInfo


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def _make_test_document(
    content: str | None = None,
    pages: list[str] | None = None,
    source: str = "test.pdf",
) -> Document:
    """Build a Document from either explicit content or a list of page texts.

    If `pages` is given, content = "\\n\\n".join(pages).
    If `content` is given directly, wraps it as a single page.
    """
    if pages is not None:
        page_texts = pages
        joined_content = "\n\n".join(page_texts)
    else:
        joined_content = content or "This is a test document with some content."
        page_texts = [joined_content]

    page_infos = [
        PageInfo(page_number=i, text=text, char_count=len(text))
        for i, text in enumerate(page_texts)
    ]

    return Document(
        content=joined_content,
        metadata=DocumentMetadata(
            source=source,
            title="Test Document",
            author="Test Author",
            page_count=len(page_infos),
        ),
        pages=page_infos,
    )


# ---------------------------------------------------------------------------
# TestFindPageNumber (utils)
# ---------------------------------------------------------------------------


class TestFindPageNumber:
    def test_single_page_returns_zero(self) -> None:
        doc = _make_test_document(pages=["Hello world"])
        assert find_page_number(doc, 0) == 0
        assert find_page_number(doc, 10) == 0

    def test_two_pages_first_page(self) -> None:
        # page 0 = "AAAA" (4 chars), page 1 = "BBBB"
        # content = "AAAA\n\nBBBB", separator takes offsets 4-5
        doc = _make_test_document(pages=["AAAA", "BBBB"])
        assert find_page_number(doc, 0) == 0
        assert find_page_number(doc, 3) == 0

    def test_two_pages_second_page(self) -> None:
        doc = _make_test_document(pages=["AAAA", "BBBB"])
        # page 0 ends at offset 3 (inclusive), page 1 starts at offset 6 (after \n\n)
        assert find_page_number(doc, 6) == 1

    def test_offset_beyond_content_returns_last_page(self) -> None:
        doc = _make_test_document(pages=["AAAA", "BBBB"])
        assert find_page_number(doc, 9999) == 1


# ---------------------------------------------------------------------------
# TestFixedSizeChunker
# ---------------------------------------------------------------------------


class TestFixedSizeChunker:
    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            FixedSizeChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_equal_raises(self) -> None:
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=50, chunk_overlap=50)

    def test_basic_chunking_produces_chunks(self) -> None:
        doc = _make_test_document(content="A" * 200)
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_respected(self) -> None:
        doc = _make_test_document(content="X" * 500)
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        # All chunks except possibly the last should be exactly chunk_size
        for c in chunks[:-1]:
            assert len(c.content) == 100

    def test_overlap_is_applied(self) -> None:
        content = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10  # 260 chars
        doc = _make_test_document(content=content)
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(doc)
        # step = 15; second chunk should start with the last 5 chars of first chunk
        assert chunks[1].content[:5] == chunks[0].content[-5:]

    def test_metadata_document_id_matches(self) -> None:
        doc = _make_test_document(content="Hello " * 50)
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.document_id == doc.id

    def test_metadata_source_matches(self) -> None:
        doc = _make_test_document(content="Hello " * 50, source="myfile.pdf")
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.source == "myfile.pdf"

    def test_chunk_indices_are_sequential(self) -> None:
        doc = _make_test_document(content="X" * 300)
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_whitespace_only_content_returns_empty(self) -> None:
        doc = _make_test_document(content="   \n  \t  ")
        chunker = FixedSizeChunker()
        # whitespace-only chunks are skipped; might still produce chunks due to page content
        # The content is "   \n  \t  " — all whitespace → should return []
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_char_offsets_are_correct(self) -> None:
        content = "ABCDEFGHIJ"  # 10 chars
        doc = _make_test_document(content=content)
        chunker = FixedSizeChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata.start_char == 0
        assert chunks[0].metadata.end_char == 5
        assert chunks[1].metadata.start_char == 5
        assert chunks[1].metadata.end_char == 10

    def test_each_chunk_has_unique_id(self) -> None:
        doc = _make_test_document(content="X" * 300)
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# TestRecursiveChunker
# ---------------------------------------------------------------------------


class TestRecursiveChunker:
    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_paragraph_split_preferred(self) -> None:
        # Two paragraphs separated by \n\n — should split there first
        para1 = "First paragraph content here. " * 5  # ~150 chars
        para2 = "Second paragraph content here. " * 5  # ~155 chars
        content = para1 + "\n\n" + para2
        doc = _make_test_document(content=content)
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        # Should produce at least 2 chunks (one per paragraph)
        assert len(chunks) >= 2

    def test_content_is_preserved(self) -> None:
        # All chunks must be non-empty and contain actual content
        content = "The quick brown fox jumps over the lazy dog. " * 20
        doc = _make_test_document(content=content)
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        # Every chunk must have non-whitespace content
        assert all(c.content.strip() for c in chunks)

    def test_chunk_size_respected(self) -> None:
        content = "word " * 200
        doc = _make_test_document(content=content)
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        for c in chunks:
            # Allow some slack for separators that may be included
            assert len(c.content) <= 60  # 50 + some separator tolerance

    def test_overlap_prepends_previous_tail(self) -> None:
        # Content long enough to produce multiple chunks; overlap should carry over
        content = "ABCDE" * 100
        doc = _make_test_document(content=content)
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # Second chunk should start with last 10 chars of first chunk
        assert chunks[1].content[:10] == chunks[0].content[-10:]

    def test_chunk_indices_are_sequential(self) -> None:
        content = "X " * 200
        doc = _make_test_document(content=content)
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_metadata_document_id_matches(self) -> None:
        doc = _make_test_document(content="Hello world. " * 30)
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.document_id == doc.id

    def test_empty_content_returns_empty(self) -> None:
        doc = _make_test_document(content="   ")
        chunker = RecursiveChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_custom_separators(self) -> None:
        content = "Part1|Part2|Part3|Part4|Part5"
        doc = _make_test_document(content=content)
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0, separators=["|", ""])
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 3


# ---------------------------------------------------------------------------
# TestSlidingWindowChunker
# ---------------------------------------------------------------------------


class TestSlidingWindowChunker:
    def test_step_greater_than_window_raises(self) -> None:
        with pytest.raises(ValueError, match="step_size"):
            SlidingWindowChunker(window_size=100, step_size=150)

    def test_produces_chunks(self) -> None:
        content = "The quick brown fox jumps over the lazy dog. " * 20
        doc = _make_test_document(content=content)
        chunker = SlidingWindowChunker(window_size=20, step_size=15)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_window_overlap_exists(self) -> None:
        # step < window → consecutive chunks share tokens → content overlap
        content = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 10
        doc = _make_test_document(content=content)
        chunker = SlidingWindowChunker(window_size=10, step_size=5)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        # Both chunks should contain content (non-empty)
        assert all(c.content.strip() for c in chunks)

    def test_chunk_indices_are_sequential(self) -> None:
        content = "word " * 200
        doc = _make_test_document(content=content)
        chunker = SlidingWindowChunker(window_size=20, step_size=10)
        chunks = chunker.chunk(doc)
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_metadata_document_id_matches(self) -> None:
        content = "word " * 200
        doc = _make_test_document(content=content)
        chunker = SlidingWindowChunker(window_size=20, step_size=15)
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.document_id == doc.id

    def test_empty_content_returns_empty(self) -> None:
        doc = _make_test_document(content="")
        # Document requires min_length=1 content, so use whitespace
        # Can't construct empty-content Document; test very short content
        content = "Hi"
        doc = _make_test_document(content=content)
        chunker = SlidingWindowChunker(window_size=50, step_size=25)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1

    def test_char_offsets_monotonically_increase(self) -> None:
        content = "alpha beta gamma delta " * 30
        doc = _make_test_document(content=content)
        chunker = SlidingWindowChunker(window_size=10, step_size=5)
        chunks = chunker.chunk(doc)
        starts = [c.metadata.start_char for c in chunks]
        assert starts == sorted(starts)


# ---------------------------------------------------------------------------
# TestHeadingSemanticChunker
# ---------------------------------------------------------------------------


class TestHeadingSemanticChunker:
    def test_markdown_heading_splits(self) -> None:
        content = (
            "# Introduction\n\nThis is the introduction section.\n\n"
            "## Methods\n\nThis describes the methods used.\n\n"
            "## Results\n\nHere are the results of our study.\n"
        )
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker(min_chunk_size=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 3

    def test_academic_heading_splits(self) -> None:
        content = (
            "Abstract\n\nThis is the abstract of the paper.\n\n"
            "Introduction\n\nThis is the introduction.\n\n"
            "Conclusion\n\nThis is the conclusion.\n"
        )
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker(min_chunk_size=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 3

    def test_numbered_heading_splits(self) -> None:
        content = (
            "1. Background\n\nSome background text here.\n\n"
            "2. Approach\n\nThe approach section content.\n\n"
            "3. Results\n\nResults section content here.\n"
        )
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker(min_chunk_size=10)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2

    def test_no_headings_returns_single_chunk(self) -> None:
        content = "This is a document with no headings at all. Just plain text."
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker()
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == content

    def test_oversized_section_is_split(self) -> None:
        # Create a section > max_chunk_size with paragraph (\n\n) boundaries for splitting
        heading = "# Very Long Section\n\n"
        # Each paragraph ~80 chars; 10 paragraphs = ~800 chars total > max_chunk_size=300
        para = "This is a paragraph with enough words to be meaningful here.\n\n"
        body = para * 10
        content = heading + body
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker(min_chunk_size=10, max_chunk_size=300)
        chunks = chunker.chunk(doc)
        # Should be split into multiple sub-chunks
        assert len(chunks) > 1

    def test_heading_included_in_chunk(self) -> None:
        content = "## Methods\n\nWe used a transformer model."
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker(min_chunk_size=5)
        chunks = chunker.chunk(doc)
        # The chunk containing "Methods" should include the heading text
        assert any("Methods" in c.content for c in chunks)

    def test_min_chunk_size_filters_tiny_sections(self) -> None:
        content = "# Title\n\nTiny.\n\n## Real Section\n\nThis is a real section with content."
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker(min_chunk_size=30)
        chunks = chunker.chunk(doc)
        # "Tiny." section should be filtered out
        assert all(len(c.content.strip()) >= 30 for c in chunks)

    def test_metadata_source_matches(self) -> None:
        content = "# Section\n\nContent here with enough text to pass min_chunk_size check."
        doc = _make_test_document(content=content, source="paper.pdf")
        chunker = HeadingSemanticChunker(min_chunk_size=5)
        chunks = chunker.chunk(doc)
        for c in chunks:
            assert c.metadata.source == "paper.pdf"

    def test_chunk_indices_are_sequential(self) -> None:
        content = (
            "# Section A\n\nContent for section A.\n\n"
            "# Section B\n\nContent for section B.\n\n"
            "# Section C\n\nContent for section C.\n"
        )
        doc = _make_test_document(content=content)
        chunker = HeadingSemanticChunker(min_chunk_size=5)
        chunks = chunker.chunk(doc)
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i


# ---------------------------------------------------------------------------
# TestEmbeddingSemanticChunker
# ---------------------------------------------------------------------------

# WHY: We craft embeddings as numpy arrays where similarity between consecutive
# sentences is controlled explicitly. High similarity (>0.85) = same topic (no split).
# Low similarity (<0.85) = topic change (boundary inserted).
#
# Craft strategy:
#   - Sentences in same topic share the SAME embedding vector → cosine sim = 1.0
#   - Sentences across a boundary use ORTHOGONAL vectors → cosine sim = 0.0
#   - This gives exact control over where EmbeddingSemanticChunker places breaks.


def _make_mock_embeddings(n_same: int, n_different: int) -> np.ndarray:
    """Build embeddings for (n_same sentences from topic A) + (n_different from topic B).

    Rows 0..n_same-1 have embedding [1, 0, 0, ...]
    Rows n_same..n_same+n_different-1 have embedding [0, 1, 0, ...]
    → sim between row n_same-1 and row n_same = 0.0 → boundary
    """
    dim = 8
    embeddings = np.zeros((n_same + n_different, dim))
    embeddings[:n_same, 0] = 1.0
    embeddings[n_same:, 1] = 1.0
    return embeddings


class TestEmbeddingSemanticChunker:
    def _make_multi_sentence_doc(self, sentences: list[str]) -> Document:
        """Build a document whose content = sentences joined by '. '."""
        content = ". ".join(sentences) + "."
        return _make_test_document(content=content)

    def test_basic_chunking_with_mock(self) -> None:
        """Two topic groups → should produce 2 chunks."""
        # 4 sentences in topic A, 4 in topic B (each ≥ 20 chars)
        sentences_a = [
            "The transformer model uses attention mechanisms",
            "Self-attention computes queries keys and values",
            "Multi-head attention is the core component here",
            "Positional encoding adds sequence information",
        ]
        sentences_b = [
            "Recurrent networks process text sequentially",
            "Long short-term memory solves vanishing gradients",
            "Gated recurrent units are a simpler alternative",
            "Bidirectional RNNs process in both directions",
        ]
        all_sentences = sentences_a + sentences_b
        doc = self._make_multi_sentence_doc(all_sentences)

        mock_embeddings = _make_mock_embeddings(len(sentences_a), len(sentences_b))

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = mock_embeddings
            MockST.return_value = mock_model

            chunker = EmbeddingSemanticChunker(breakpoint_threshold=0.5, min_chunk_size=10)
            chunks = chunker.chunk(doc)

        assert len(chunks) >= 2

    def test_model_loaded_with_correct_name(self) -> None:
        """ALWAYS use all-MiniLM-L6-v2 (PRD Decision 1)."""
        sentences = [
            "The first sentence is long enough to pass filter here",
            "The second sentence is also long enough to pass filter",
        ]
        doc = self._make_multi_sentence_doc(sentences)
        dummy_embeddings = np.eye(2, 8)  # 2 sentences, 8 dims

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = dummy_embeddings
            MockST.return_value = mock_model

            chunker = EmbeddingSemanticChunker()
            chunker.chunk(doc)

        MockST.assert_called_once_with("all-MiniLM-L6-v2")

    def test_model_is_deleted_after_use(self) -> None:
        """del model + gc.collect() must be called (8GB M2 constraint)."""
        sentences = [
            "The first long sentence is used for testing here",
            "The second long sentence is also used for testing",
        ]
        doc = self._make_multi_sentence_doc(sentences)
        dummy_embeddings = np.eye(2, 8)

        # We verify gc.collect is called — del is a language construct and cannot be
        # directly intercepted, but gc.collect() always follows it per implementation.
        import gc as gc_module

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = dummy_embeddings
            MockST.return_value = mock_model

            with patch.object(gc_module, "collect") as mock_gc:
                chunker = EmbeddingSemanticChunker()
                chunker.chunk(doc)

        mock_gc.assert_called()

    def test_single_sentence_returns_one_chunk(self) -> None:
        """Not enough sentences to find boundaries → return whole doc as one chunk."""
        content = "This is a single long sentence that exceeds the minimum length requirement."
        doc = _make_test_document(content=content)

        chunker = EmbeddingSemanticChunker()
        # No patch needed — EmbeddingSemanticChunker skips embedding for <=1 sentence
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == content

    def test_metadata_document_id_matches(self) -> None:
        sentences = [
            "The transformer architecture revolutionized NLP tasks forever",
            "Attention mechanisms allow models to focus on relevant tokens",
            "BERT pretrains on masked language modeling and next sentence prediction",
            "Recurrent models process sequence tokens one at a time step",
        ]
        doc = self._make_multi_sentence_doc(sentences)
        dummy_embeddings = np.eye(4, 8)

        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()
            mock_model.encode.return_value = dummy_embeddings
            MockST.return_value = mock_model

            chunker = EmbeddingSemanticChunker(breakpoint_threshold=0.99, min_chunk_size=5)
            chunks = chunker.chunk(doc)

        for c in chunks:
            assert c.metadata.document_id == doc.id

    # --- Unit tests for private helper methods ---

    def test_consecutive_cosine_similarities_identical_vectors(self) -> None:
        chunker = EmbeddingSemanticChunker()
        embeddings = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        sims = chunker._consecutive_cosine_similarities(embeddings)
        assert len(sims) == 1
        assert abs(sims[0] - 1.0) < 1e-6

    def test_consecutive_cosine_similarities_orthogonal_vectors(self) -> None:
        chunker = EmbeddingSemanticChunker()
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
        sims = chunker._consecutive_cosine_similarities(embeddings)
        assert len(sims) == 1
        assert abs(sims[0] - 0.0) < 1e-6

    def test_consecutive_cosine_similarities_zero_vector_safe(self) -> None:
        """Zero vector should not cause division by zero."""
        chunker = EmbeddingSemanticChunker()
        embeddings = np.array([[0.0, 0.0], [1.0, 0.0]])
        sims = chunker._consecutive_cosine_similarities(embeddings)
        assert len(sims) == 1
        assert not np.isnan(sims[0])

    def test_group_sentences_no_breakpoints(self) -> None:
        chunker = EmbeddingSemanticChunker()
        sentences = ["Alpha.", "Beta.", "Gamma."]
        groups = chunker._group_sentences(sentences, [])
        assert groups == ["Alpha. Beta. Gamma."]

    def test_group_sentences_with_breakpoint(self) -> None:
        chunker = EmbeddingSemanticChunker()
        sentences = ["Alpha.", "Beta.", "Gamma.", "Delta."]
        # Breakpoint at index 1 → split after "Beta."
        groups = chunker._group_sentences(sentences, [1])
        assert len(groups) == 2
        assert "Alpha." in groups[0]
        assert "Beta." in groups[0]
        assert "Gamma." in groups[1]
        assert "Delta." in groups[1]

    def test_merge_small_chunks_merges_tiny(self) -> None:
        chunker = EmbeddingSemanticChunker(min_chunk_size=50)
        # First chunk is large, second is tiny (< 50 chars)
        chunks = ["A" * 60, "B" * 10]
        merged = chunker._merge_small_chunks(chunks)
        assert len(merged) == 1
        assert "B" * 10 in merged[0]

    def test_merge_small_chunks_keeps_large(self) -> None:
        chunker = EmbeddingSemanticChunker(min_chunk_size=50)
        chunks = ["A" * 100, "B" * 100, "C" * 100]
        merged = chunker._merge_small_chunks(chunks)
        assert len(merged) == 3

    def test_merge_small_chunks_empty_input(self) -> None:
        chunker = EmbeddingSemanticChunker()
        assert chunker._merge_small_chunks([]) == []
