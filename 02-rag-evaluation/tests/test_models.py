"""Tests for Pydantic models — validation, edge cases, JSON roundtrip.

Covers: ChunkConfig, Chunk, SyntheticQAPair, enums, and cross-field validators.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import (
    BloomLevel,
    Chunk,
    ChunkConfig,
    ConfigEvaluation,
    EmbeddingModel,
    QuestionHierarchy,
    QuestionType,
    RetrievalMethod,
    SyntheticQAPair,
)


# ===========================================================================
# Fixtures — reusable valid data
# ===========================================================================

@pytest.fixture()
def valid_chunk_config() -> ChunkConfig:
    return ChunkConfig(
        name="B",
        chunk_size=256,
        overlap=64,
        chunking_goal="Test config",
        is_semantic=False,
    )


@pytest.fixture()
def valid_chunk() -> Chunk:
    return Chunk(
        id="B_0",
        text="This is a test chunk with enough content.",
        token_count=9,
        start_char=0,
        end_char=42,
        page_numbers=[1],
        config_name="B",
        section_header=None,
    )


@pytest.fixture()
def valid_qa_pair() -> SyntheticQAPair:
    return SyntheticQAPair(
        id="q_001",
        question="What is the capital of France and why is it important?",
        question_type=QuestionType.FACTUAL,
        hierarchy=QuestionHierarchy.PARAGRAPH,
        gold_chunk_ids=["B_0"],
        expected_answer="Paris is the capital of France.",
        source_chunk_text="Paris is the capital of France. It is known for the Eiffel Tower.",
        is_overlap_region=False,
        generation_strategy="per_chunk_chain",
    )


# ===========================================================================
# StrEnum Tests
# ===========================================================================

class TestEnums:
    """Verify StrEnum values serialize to plain strings."""

    def test_embedding_model_values(self) -> None:
        assert EmbeddingModel.MINILM == "all-MiniLM-L6-v2"
        assert EmbeddingModel.MPNET == "all-mpnet-base-v2"
        assert EmbeddingModel.OPENAI == "text-embedding-3-small"

    def test_retrieval_method_values(self) -> None:
        assert RetrievalMethod.VECTOR == "vector"
        assert RetrievalMethod.BM25 == "bm25"

    def test_question_type_membership(self) -> None:
        assert len(QuestionType) == 5
        assert "factual" in [qt.value for qt in QuestionType]

    def test_bloom_level_count(self) -> None:
        assert len(BloomLevel) == 6

    def test_enum_iteration(self) -> None:
        """StrEnum supports iteration — used in config.py for ALL_EMBEDDING_MODELS."""
        models = list(EmbeddingModel)
        assert len(models) == 3
        assert all(isinstance(m, str) for m in models)


# ===========================================================================
# ChunkConfig Tests
# ===========================================================================

class TestChunkConfig:
    """Validate ChunkConfig creation and cross-field validators."""

    def test_valid_config(self, valid_chunk_config: ChunkConfig) -> None:
        assert valid_chunk_config.name == "B"
        assert valid_chunk_config.chunk_size == 256
        assert valid_chunk_config.overlap == 64

    def test_overlap_equals_chunk_size_raises(self) -> None:
        with pytest.raises(ValidationError, match="overlap.*must be less than chunk_size"):
            ChunkConfig(
                name="bad",
                chunk_size=256,
                overlap=256,
                chunking_goal="Invalid",
            )

    def test_overlap_exceeds_chunk_size_raises(self) -> None:
        with pytest.raises(ValidationError, match="overlap.*must be less than chunk_size"):
            ChunkConfig(
                name="bad",
                chunk_size=128,
                overlap=200,
                chunking_goal="Invalid",
            )

    def test_chunk_size_zero_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(
                name="bad",
                chunk_size=0,
                overlap=0,
                chunking_goal="Invalid",
            )

    def test_chunk_size_exceeds_max_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(
                name="bad",
                chunk_size=3000,
                overlap=0,
                chunking_goal="Invalid",
            )

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValidationError):
            ChunkConfig(
                name="bad",
                chunk_size=256,
                overlap=-1,
                chunking_goal="Invalid",
            )

    def test_semantic_flag_default_false(self) -> None:
        cfg = ChunkConfig(
            name="test",
            chunk_size=128,
            overlap=32,
            chunking_goal="Test",
        )
        assert cfg.is_semantic is False

    def test_semantic_flag_explicit_true(self) -> None:
        cfg = ChunkConfig(
            name="E",
            chunk_size=512,
            overlap=0,
            chunking_goal="Semantic test",
            is_semantic=True,
        )
        assert cfg.is_semantic is True


# ===========================================================================
# Chunk Tests
# ===========================================================================

class TestChunk:
    """Validate Chunk creation and field/model validators."""

    def test_valid_chunk(self, valid_chunk: Chunk) -> None:
        assert valid_chunk.id == "B_0"
        assert valid_chunk.token_count == 9
        assert valid_chunk.page_numbers == [1]
        assert valid_chunk.section_header is None

    def test_end_char_equals_start_char_raises(self) -> None:
        with pytest.raises(ValidationError, match="end_char.*must be greater than start_char"):
            Chunk(
                id="B_0",
                text="Hello world",
                token_count=2,
                start_char=100,
                end_char=100,
                page_numbers=[1],
                config_name="B",
            )

    def test_end_char_less_than_start_char_raises(self) -> None:
        with pytest.raises(ValidationError, match="end_char.*must be greater than start_char"):
            Chunk(
                id="B_0",
                text="Hello world",
                token_count=2,
                start_char=100,
                end_char=50,
                page_numbers=[1],
                config_name="B",
            )

    def test_whitespace_only_text_raises(self) -> None:
        with pytest.raises(ValidationError, match="non-whitespace"):
            Chunk(
                id="B_0",
                text="   \n\t  ",
                token_count=1,
                start_char=0,
                end_char=7,
                page_numbers=[1],
                config_name="B",
            )

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                id="B_0",
                text="",
                token_count=1,
                start_char=0,
                end_char=1,
                page_numbers=[1],
                config_name="B",
            )

    def test_zero_token_count_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                id="B_0",
                text="Hello",
                token_count=0,
                start_char=0,
                end_char=5,
                page_numbers=[1],
                config_name="B",
            )

    def test_section_header_for_semantic_chunk(self) -> None:
        chunk = Chunk(
            id="E_3",
            text="Some section content here.",
            token_count=5,
            start_char=200,
            end_char=226,
            page_numbers=[2, 3],
            config_name="E",
            section_header="Strategic Report",
        )
        assert chunk.section_header == "Strategic Report"
        assert chunk.page_numbers == [2, 3]

    def test_multiple_page_numbers(self) -> None:
        chunk = Chunk(
            id="C_10",
            text="Long chunk spanning pages.",
            token_count=5,
            start_char=1000,
            end_char=1025,
            page_numbers=[3, 4, 5],
            config_name="C",
        )
        assert len(chunk.page_numbers) == 3


# ===========================================================================
# SyntheticQAPair Tests
# ===========================================================================

class TestSyntheticQAPair:
    """Validate SyntheticQAPair creation and constraints."""

    def test_valid_qa_pair(self, valid_qa_pair: SyntheticQAPair) -> None:
        assert valid_qa_pair.id == "q_001"
        assert valid_qa_pair.question_type == QuestionType.FACTUAL
        assert valid_qa_pair.is_overlap_region is False

    def test_short_question_raises(self) -> None:
        """Question must be >= 10 characters."""
        with pytest.raises(ValidationError):
            SyntheticQAPair(
                id="q_bad",
                question="Why?",
                question_type=QuestionType.FACTUAL,
                hierarchy=QuestionHierarchy.PARAGRAPH,
                gold_chunk_ids=["B_0"],
                expected_answer="Because.",
                source_chunk_text="Some source text.",
                generation_strategy="test",
            )

    def test_empty_gold_chunk_ids_raises(self) -> None:
        """At least one gold chunk ID required."""
        with pytest.raises(ValidationError):
            SyntheticQAPair(
                id="q_bad",
                question="What is the meaning of life in the universe?",
                question_type=QuestionType.ANALYTICAL,
                hierarchy=QuestionHierarchy.SECTION,
                gold_chunk_ids=[],
                expected_answer="42",
                source_chunk_text="Some source.",
                generation_strategy="test",
            )

    @pytest.mark.parametrize("q_type", list(QuestionType))
    def test_all_question_types_valid(self, q_type: QuestionType) -> None:
        """Every QuestionType variant creates a valid QA pair."""
        pair = SyntheticQAPair(
            id=f"q_{q_type.value}",
            question=f"A valid question about {q_type.value} topics here?",
            question_type=q_type,
            hierarchy=QuestionHierarchy.PARAGRAPH,
            gold_chunk_ids=["B_0"],
            expected_answer="An answer.",
            source_chunk_text="Source.",
            generation_strategy="test",
        )
        assert pair.question_type == q_type


# ===========================================================================
# JSON Roundtrip Tests
# ===========================================================================

class TestJSONRoundtrip:
    """Verify models survive JSON serialization/deserialization."""

    def test_chunk_config_roundtrip(self, valid_chunk_config: ChunkConfig) -> None:
        json_str = valid_chunk_config.model_dump_json()
        restored = ChunkConfig.model_validate_json(json_str)
        assert restored == valid_chunk_config

    def test_chunk_roundtrip(self, valid_chunk: Chunk) -> None:
        json_str = valid_chunk.model_dump_json()
        restored = Chunk.model_validate_json(json_str)
        assert restored == valid_chunk

    def test_qa_pair_roundtrip(self, valid_qa_pair: SyntheticQAPair) -> None:
        json_str = valid_qa_pair.model_dump_json()
        restored = SyntheticQAPair.model_validate_json(json_str)
        assert restored == valid_qa_pair

    def test_config_evaluation_roundtrip(self) -> None:
        """ConfigEvaluation is the most complex nested model — test it roundtrips."""
        eval_result = ConfigEvaluation(
            config_id="B-mpnet",
            chunk_config="B",
            embedding_model="all-mpnet-base-v2",
            retrieval_method=RetrievalMethod.VECTOR,
            num_chunks=125,
            num_questions=50,
            avg_recall_at_1=0.6,
            avg_recall_at_3=0.8,
            avg_recall_at_5=0.9,
            avg_precision_at_1=0.6,
            avg_precision_at_3=0.4,
            avg_precision_at_5=0.3,
            avg_mrr_at_5=0.7,
            metrics_by_question_type={
                QuestionType.FACTUAL: {"recall_at_5": 0.95},
            },
            individual_results=[],
        )
        json_str = eval_result.model_dump_json()
        restored = ConfigEvaluation.model_validate_json(json_str)
        assert restored.config_id == "B-mpnet"
        assert restored.avg_recall_at_5 == 0.9


# ===========================================================================
# Parametrized Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Boundary value tests using parametrize."""

    @pytest.mark.parametrize(
        ("chunk_size", "overlap", "should_pass"),
        [
            (1, 0, True),        # Minimum valid
            (2048, 0, True),     # Maximum valid chunk_size
            (2048, 2047, True),  # Max overlap just under chunk_size
            (100, 99, True),     # Overlap one less than chunk_size
            (100, 100, False),   # Overlap equals chunk_size
            (1, 1, False),       # Overlap equals chunk_size (min)
        ],
    )
    def test_chunk_config_boundary_values(
        self, chunk_size: int, overlap: int, should_pass: bool,
    ) -> None:
        if should_pass:
            cfg = ChunkConfig(
                name="test",
                chunk_size=chunk_size,
                overlap=overlap,
                chunking_goal="Boundary test",
            )
            assert cfg.chunk_size == chunk_size
        else:
            with pytest.raises(ValidationError):
                ChunkConfig(
                    name="test",
                    chunk_size=chunk_size,
                    overlap=overlap,
                    chunking_goal="Boundary test",
                )

    @pytest.mark.parametrize(
        ("start", "end", "should_pass"),
        [
            (0, 1, True),         # Minimum valid span
            (0, 10000, True),     # Large span
            (999, 1000, True),    # Adjacent
            (100, 100, False),    # Equal (invalid)
            (100, 50, False),     # Reversed (invalid)
        ],
    )
    def test_chunk_char_offset_boundaries(
        self, start: int, end: int, should_pass: bool,
    ) -> None:
        if should_pass:
            chunk = Chunk(
                id="test_0",
                text="Valid text",
                token_count=2,
                start_char=start,
                end_char=end,
                page_numbers=[0],
                config_name="test",
            )
            assert chunk.start_char == start
        else:
            with pytest.raises(ValidationError):
                Chunk(
                    id="test_0",
                    text="Valid text",
                    token_count=2,
                    start_char=start,
                    end_char=end,
                    page_numbers=[0],
                    config_name="test",
                )
