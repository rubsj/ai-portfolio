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
    GridSearchReport,
    JudgeResult,
    QADatasetReport,
    QuestionHierarchy,
    QuestionType,
    RAGASResult,
    RerankingComparison,
    RetrievalMethod,
    RetrievalResult,
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


# ===========================================================================
# RetrievalResult Tests
# ===========================================================================

class TestRetrievalResult:
    """Validate RetrievalResult creation and metric constraints."""

    @pytest.fixture()
    def valid_retrieval_result(self) -> RetrievalResult:
        return RetrievalResult(
            query_id="q_001",
            question="What are the key components of cloud infrastructure?",
            question_type=QuestionType.FACTUAL,
            gold_chunk_ids=["B_0", "B_1"],
            retrieved_chunk_ids=["B_0", "B_2", "B_3", "B_1", "B_4"],
            retrieved_scores=[0.95, 0.8, 0.75, 0.7, 0.6],
            recall_at_1=0.5,
            recall_at_3=0.5,
            recall_at_5=1.0,
            precision_at_1=1.0,
            precision_at_3=0.33,
            precision_at_5=0.4,
            mrr_at_1=1.0,
            mrr_at_3=1.0,
            mrr_at_5=1.0,
        )

    def test_valid_creation(self, valid_retrieval_result: RetrievalResult) -> None:
        assert valid_retrieval_result.query_id == "q_001"
        assert valid_retrieval_result.recall_at_5 == 1.0

    def test_recall_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalResult(
                query_id="q_bad",
                question="A sufficiently long question?",
                question_type=QuestionType.FACTUAL,
                gold_chunk_ids=["B_0"],
                retrieved_chunk_ids=["B_0"],
                retrieved_scores=[0.9],
                recall_at_1=1.5,  # Out of range
                recall_at_3=0.0,
                recall_at_5=0.0,
                precision_at_1=0.0,
                precision_at_3=0.0,
                precision_at_5=0.0,
                mrr_at_1=0.0,
                mrr_at_3=0.0,
                mrr_at_5=0.0,
            )

    def test_negative_precision_raises(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalResult(
                query_id="q_bad",
                question="A sufficiently long question?",
                question_type=QuestionType.FACTUAL,
                gold_chunk_ids=["B_0"],
                retrieved_chunk_ids=["B_0"],
                retrieved_scores=[0.9],
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                precision_at_1=-0.1,  # Negative
                precision_at_3=0.0,
                precision_at_5=0.0,
                mrr_at_1=0.0,
                mrr_at_3=0.0,
                mrr_at_5=0.0,
            )

    def test_json_roundtrip(self, valid_retrieval_result: RetrievalResult) -> None:
        json_str = valid_retrieval_result.model_dump_json()
        restored = RetrievalResult.model_validate_json(json_str)
        assert restored == valid_retrieval_result

    def test_zero_scores_valid(self) -> None:
        """All metrics at 0.0 is valid (no relevant docs retrieved)."""
        result = RetrievalResult(
            query_id="q_zero",
            question="A question with no matching results?",
            question_type=QuestionType.ANALYTICAL,
            gold_chunk_ids=["B_0"],
            retrieved_chunk_ids=["B_99"],
            retrieved_scores=[0.1],
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.0,
            precision_at_1=0.0,
            precision_at_3=0.0,
            precision_at_5=0.0,
            mrr_at_1=0.0,
            mrr_at_3=0.0,
            mrr_at_5=0.0,
        )
        assert result.recall_at_5 == 0.0


# ===========================================================================
# QADatasetReport Tests
# ===========================================================================

class TestQADatasetReport:
    """Validate QADatasetReport creation and constraints."""

    @pytest.fixture()
    def valid_qa_report(self) -> QADatasetReport:
        return QADatasetReport(
            total_questions=50,
            questions_per_strategy={"per_chunk_chain": 30, "cross_chunk": 20},
            questions_per_type={
                QuestionType.FACTUAL: 15,
                QuestionType.ANALYTICAL: 10,
                QuestionType.COMPARATIVE: 10,
                QuestionType.SUMMARIZATION: 10,
                QuestionType.MULTI_HOP: 5,
            },
            questions_per_hierarchy={
                QuestionHierarchy.PARAGRAPH: 30,
                QuestionHierarchy.SECTION: 15,
                QuestionHierarchy.PAGE: 5,
            },
            chunk_coverage_percent=85.0,
            overlap_question_count=5,
            avg_questions_per_chunk=0.4,
        )

    def test_valid_creation(self, valid_qa_report: QADatasetReport) -> None:
        assert valid_qa_report.total_questions == 50
        assert valid_qa_report.chunk_coverage_percent == 85.0

    def test_coverage_above_100_raises(self) -> None:
        with pytest.raises(ValidationError):
            QADatasetReport(
                total_questions=50,
                questions_per_strategy={},
                questions_per_type={},
                questions_per_hierarchy={},
                chunk_coverage_percent=101.0,  # Above 100
                overlap_question_count=0,
                avg_questions_per_chunk=1.0,
            )

    def test_negative_avg_questions_raises(self) -> None:
        with pytest.raises(ValidationError):
            QADatasetReport(
                total_questions=50,
                questions_per_strategy={},
                questions_per_type={},
                questions_per_hierarchy={},
                chunk_coverage_percent=50.0,
                overlap_question_count=0,
                avg_questions_per_chunk=-1.0,  # Negative
            )

    def test_json_roundtrip(self, valid_qa_report: QADatasetReport) -> None:
        json_str = valid_qa_report.model_dump_json()
        restored = QADatasetReport.model_validate_json(json_str)
        assert restored == valid_qa_report


# ===========================================================================
# RerankingComparison Tests
# ===========================================================================

class TestRerankingComparison:
    """Validate RerankingComparison creation and constraints."""

    @pytest.fixture()
    def valid_reranking(self) -> RerankingComparison:
        return RerankingComparison(
            config_id="B-mpnet",
            precision_at_5_before=0.3,
            precision_at_5_after=0.5,
            recall_at_5_before=0.7,
            recall_at_5_after=0.85,
            mrr_at_5_before=0.6,
            mrr_at_5_after=0.75,
            precision_improvement_pct=66.7,
            recall_improvement_pct=21.4,
            mrr_improvement_pct=25.0,
        )

    def test_valid_creation(self, valid_reranking: RerankingComparison) -> None:
        assert valid_reranking.config_id == "B-mpnet"
        assert valid_reranking.precision_at_5_after == 0.5

    def test_negative_improvement_valid(self) -> None:
        """Negative improvement means degradation — should still be valid."""
        result = RerankingComparison(
            config_id="C-minilm",
            precision_at_5_before=0.5,
            precision_at_5_after=0.4,
            recall_at_5_before=0.8,
            recall_at_5_after=0.7,
            mrr_at_5_before=0.7,
            mrr_at_5_after=0.6,
            precision_improvement_pct=-20.0,
            recall_improvement_pct=-12.5,
            mrr_improvement_pct=-14.3,
        )
        assert result.precision_improvement_pct == -20.0

    def test_before_metric_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            RerankingComparison(
                config_id="bad",
                precision_at_5_before=1.5,  # Out of range
                precision_at_5_after=0.5,
                recall_at_5_before=0.7,
                recall_at_5_after=0.85,
                mrr_at_5_before=0.6,
                mrr_at_5_after=0.75,
                precision_improvement_pct=0.0,
                recall_improvement_pct=0.0,
                mrr_improvement_pct=0.0,
            )

    def test_json_roundtrip(self, valid_reranking: RerankingComparison) -> None:
        json_str = valid_reranking.model_dump_json()
        restored = RerankingComparison.model_validate_json(json_str)
        assert restored == valid_reranking


# ===========================================================================
# RAGASResult Tests
# ===========================================================================

class TestRAGASResult:
    """Validate RAGASResult creation and score constraints."""

    def test_valid_creation(self) -> None:
        result = RAGASResult(
            config_id="B-mpnet",
            faithfulness=0.85,
            answer_relevancy=0.9,
            context_recall=0.8,
            context_precision=0.75,
        )
        assert result.faithfulness == 0.85

    def test_all_scores_at_boundaries(self) -> None:
        """0.0 and 1.0 are both valid boundary values."""
        result = RAGASResult(
            config_id="zero",
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_recall=0.0,
            context_precision=0.0,
        )
        assert result.faithfulness == 0.0

        perfect = RAGASResult(
            config_id="perfect",
            faithfulness=1.0,
            answer_relevancy=1.0,
            context_recall=1.0,
            context_precision=1.0,
        )
        assert perfect.faithfulness == 1.0

    def test_score_above_1_raises(self) -> None:
        with pytest.raises(ValidationError):
            RAGASResult(
                config_id="bad",
                faithfulness=1.1,
                answer_relevancy=0.9,
                context_recall=0.8,
                context_precision=0.75,
            )

    def test_negative_score_raises(self) -> None:
        with pytest.raises(ValidationError):
            RAGASResult(
                config_id="bad",
                faithfulness=0.5,
                answer_relevancy=-0.1,
                context_recall=0.8,
                context_precision=0.75,
            )

    def test_json_roundtrip(self) -> None:
        result = RAGASResult(
            config_id="B-mpnet",
            faithfulness=0.85,
            answer_relevancy=0.9,
            context_recall=0.8,
            context_precision=0.75,
        )
        json_str = result.model_dump_json()
        restored = RAGASResult.model_validate_json(json_str)
        assert restored == result


# ===========================================================================
# JudgeResult Tests
# ===========================================================================

class TestJudgeResult:
    """Validate JudgeResult creation."""

    @pytest.fixture()
    def valid_judge_result(self) -> JudgeResult:
        return JudgeResult(
            question_id="q_001",
            question="What is containerization in cloud computing?",
            generated_answer="Containerization packages applications with dependencies.",
            expected_answer="Docker containers package apps with dependencies.",
            correctness_score=True,
            correctness_reasoning="Answer captures key concept of dependency packaging.",
            has_hallucination=False,
            hallucination_reasoning="All claims supported by context.",
            relevance_grade="Highly",
            relevance_reasoning="Directly addresses the question about containerization.",
            bloom_level=BloomLevel.UNDERSTAND,
            bloom_reasoning="Question tests comprehension of containerization concept.",
        )

    def test_valid_creation(self, valid_judge_result: JudgeResult) -> None:
        assert valid_judge_result.correctness_score is True
        assert valid_judge_result.has_hallucination is False
        assert valid_judge_result.bloom_level == BloomLevel.UNDERSTAND

    @pytest.mark.parametrize("level", list(BloomLevel))
    def test_all_bloom_levels_valid(self, level: BloomLevel) -> None:
        result = JudgeResult(
            question_id="q_bloom",
            question="A question to test bloom level classification?",
            generated_answer="Some answer.",
            expected_answer="Expected answer.",
            correctness_score=True,
            correctness_reasoning="Correct.",
            has_hallucination=False,
            hallucination_reasoning="No hallucination.",
            relevance_grade="Perfectly",
            relevance_reasoning="Relevant.",
            bloom_level=level,
            bloom_reasoning=f"Classified as {level.value}.",
        )
        assert result.bloom_level == level

    def test_json_roundtrip(self, valid_judge_result: JudgeResult) -> None:
        json_str = valid_judge_result.model_dump_json()
        restored = JudgeResult.model_validate_json(json_str)
        assert restored == valid_judge_result

    def test_hallucination_true(self) -> None:
        result = JudgeResult(
            question_id="q_hal",
            question="What is the capital of France in this document?",
            generated_answer="London is the capital of France.",
            expected_answer="Paris.",
            correctness_score=False,
            correctness_reasoning="Wrong city.",
            has_hallucination=True,
            hallucination_reasoning="Claims London, not supported by context.",
            relevance_grade="Irrelevant",
            relevance_reasoning="Answer is factually wrong.",
            bloom_level=BloomLevel.REMEMBER,
            bloom_reasoning="Simple recall question.",
        )
        assert result.has_hallucination is True
        assert result.correctness_score is False


# ===========================================================================
# GridSearchReport Tests
# ===========================================================================

class TestGridSearchReport:
    """Validate GridSearchReport creation and optional field handling."""

    def _make_config_eval(self, config_id: str = "B-mpnet") -> ConfigEvaluation:
        """Helper to create a minimal ConfigEvaluation."""
        return ConfigEvaluation(
            config_id=config_id,
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
            metrics_by_question_type={},
            individual_results=[],
        )

    def test_valid_creation(self) -> None:
        from datetime import datetime, timezone

        report = GridSearchReport(
            pdf_name="financial_services.md",
            total_configs=16,
            config_evaluations=[self._make_config_eval()],
            best_retrieval_config="B-mpnet",
            timestamp=datetime.now(tz=timezone.utc),
            total_runtime_seconds=120.5,
            estimated_api_cost_usd=1.50,
        )
        assert report.pdf_name == "financial_services.md"
        assert report.total_configs == 16

    def test_optional_fields_default_to_none_or_empty(self) -> None:
        from datetime import datetime, timezone

        report = GridSearchReport(
            pdf_name="test.md",
            total_configs=1,
            config_evaluations=[self._make_config_eval()],
            best_retrieval_config="B-mpnet",
            timestamp=datetime.now(tz=timezone.utc),
            total_runtime_seconds=10.0,
            estimated_api_cost_usd=0.0,
        )
        assert report.bm25_baseline is None
        assert report.best_generation_config is None
        assert report.qa_dataset_report is None
        assert report.reranking_comparisons == []
        assert report.ragas_results == []
        assert report.judge_results == []

    def test_negative_cost_raises(self) -> None:
        from datetime import datetime, timezone

        with pytest.raises(ValidationError):
            GridSearchReport(
                pdf_name="test.md",
                total_configs=1,
                config_evaluations=[],
                best_retrieval_config="B-mpnet",
                timestamp=datetime.now(tz=timezone.utc),
                total_runtime_seconds=10.0,
                estimated_api_cost_usd=-1.0,  # Negative cost
            )

    def test_json_roundtrip(self) -> None:
        from datetime import datetime, timezone

        report = GridSearchReport(
            pdf_name="test.md",
            total_configs=2,
            config_evaluations=[self._make_config_eval("A-minilm"), self._make_config_eval("B-mpnet")],
            best_retrieval_config="B-mpnet",
            timestamp=datetime(2026, 2, 12, tzinfo=timezone.utc),
            total_runtime_seconds=300.0,
            estimated_api_cost_usd=5.25,
        )
        json_str = report.model_dump_json()
        restored = GridSearchReport.model_validate_json(json_str)
        assert restored.pdf_name == "test.md"
        assert restored.total_configs == 2
        assert len(restored.config_evaluations) == 2

    def test_with_nested_optional_objects(self) -> None:
        from datetime import datetime, timezone

        eval_obj = self._make_config_eval()
        qa_report = QADatasetReport(
            total_questions=50,
            questions_per_strategy={"chain": 50},
            questions_per_type={QuestionType.FACTUAL: 50},
            questions_per_hierarchy={QuestionHierarchy.PARAGRAPH: 50},
            chunk_coverage_percent=90.0,
            overlap_question_count=3,
            avg_questions_per_chunk=0.5,
        )
        report = GridSearchReport(
            pdf_name="test.md",
            total_configs=1,
            config_evaluations=[eval_obj],
            bm25_baseline=eval_obj,
            best_retrieval_config="B-mpnet",
            best_generation_config="B-mpnet",
            qa_dataset_report=qa_report,
            timestamp=datetime(2026, 2, 12, tzinfo=timezone.utc),
            total_runtime_seconds=100.0,
            estimated_api_cost_usd=2.0,
        )
        assert report.bm25_baseline is not None
        assert report.qa_dataset_report is not None
        assert report.best_generation_config == "B-mpnet"
