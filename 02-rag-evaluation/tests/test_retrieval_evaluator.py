"""Tests for retrieval_evaluator.py — pure computation, no mocking needed.

Tests the three core metric functions (Recall@K, Precision@K, MRR@K) and
the aggregation functions (evaluate_single_question, evaluate_config).

Java/TS parallel: like JUnit tests for a utility class — all assertions
are input/output checks with no side effects or external dependencies.
"""

from __future__ import annotations

import pytest

from src.models import (
    QuestionHierarchy,
    QuestionType,
    RetrievalMethod,
    SyntheticQAPair,
)
from src.retrieval_evaluator import (
    compute_mrr_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_config,
    evaluate_single_question,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_qa_pair(
    qa_id: str = "q1",
    question_type: QuestionType = QuestionType.FACTUAL,
    gold_ids: list[str] | None = None,
) -> SyntheticQAPair:
    """Create a minimal SyntheticQAPair for testing."""
    return SyntheticQAPair(
        id=qa_id,
        question="What is the meaning of life?",
        question_type=question_type,
        hierarchy=QuestionHierarchy.PARAGRAPH,
        gold_chunk_ids=gold_ids or ["B_0_1"],
        expected_answer="42 according to the guide",
        source_chunk_text="The meaning of life is 42.",
        generation_strategy="per_chunk_chain",
    )


# ===========================================================================
# Recall@K Tests
# ===========================================================================

class TestRecallAtK:
    """Tests for compute_recall_at_k."""

    def test_perfect_recall_at_1(self) -> None:
        """Gold chunk is the top result."""
        assert compute_recall_at_k(["A"], ["A", "B", "C"], k=1) == 1.0

    def test_zero_recall_at_1(self) -> None:
        """Gold chunk not in top-1."""
        assert compute_recall_at_k(["A"], ["B", "A", "C"], k=1) == 0.0

    def test_partial_recall_multi_gold(self) -> None:
        """Two gold chunks, only one found in top-3."""
        assert compute_recall_at_k(["A", "D"], ["A", "B", "C"], k=3) == 0.5

    def test_full_recall_multi_gold(self) -> None:
        """Both gold chunks in top-3."""
        assert compute_recall_at_k(["A", "C"], ["A", "B", "C"], k=3) == 1.0

    def test_empty_gold_returns_zero(self) -> None:
        """No gold chunks — recall is 0."""
        assert compute_recall_at_k([], ["A", "B"], k=3) == 0.0

    def test_empty_retrieved_returns_zero(self) -> None:
        """No retrieved results — recall is 0."""
        assert compute_recall_at_k(["A"], [], k=3) == 0.0

    @pytest.mark.parametrize("k,expected", [(1, 1.0), (3, 1.0), (5, 1.0)])
    def test_recall_monotonic_with_k(self, k: int, expected: float) -> None:
        """Recall doesn't decrease as K increases when gold is at rank 1."""
        assert compute_recall_at_k(["A"], ["A", "B", "C", "D", "E"], k=k) == expected

    @pytest.mark.parametrize("k,expected", [(1, 0.0), (3, 0.5), (5, 1.0)])
    def test_recall_increases_with_k(self, k: int, expected: float) -> None:
        """Recall increases as K captures more gold chunks."""
        assert compute_recall_at_k(
            ["B", "E"], ["A", "B", "C", "D", "E"], k=k,
        ) == expected


# ===========================================================================
# Precision@K Tests
# ===========================================================================

class TestPrecisionAtK:
    """Tests for compute_precision_at_k."""

    def test_perfect_precision_at_1(self) -> None:
        """Top-1 is a gold chunk."""
        assert compute_precision_at_k(["A"], ["A", "B", "C"], k=1) == 1.0

    def test_zero_precision(self) -> None:
        """No gold chunks in top-3."""
        assert compute_precision_at_k(["D"], ["A", "B", "C"], k=3) == 0.0

    def test_precision_decreases_with_k(self) -> None:
        """P@1 > P@3 > P@5 when only 1 gold chunk at rank 1."""
        gold = ["A"]
        retrieved = ["A", "B", "C", "D", "E"]
        assert compute_precision_at_k(gold, retrieved, k=1) == 1.0
        assert compute_precision_at_k(gold, retrieved, k=3) == pytest.approx(1 / 3)
        assert compute_precision_at_k(gold, retrieved, k=5) == pytest.approx(1 / 5)

    def test_k_zero_returns_zero(self) -> None:
        """Edge case: K=0 returns 0 (no division by zero)."""
        assert compute_precision_at_k(["A"], ["A"], k=0) == 0.0

    def test_multi_gold_precision(self) -> None:
        """Two gold in top-3 → P@3 = 2/3."""
        assert compute_precision_at_k(
            ["A", "C"], ["A", "B", "C"], k=3,
        ) == pytest.approx(2 / 3)


# ===========================================================================
# MRR@K Tests
# ===========================================================================

class TestMRRAtK:
    """Tests for compute_mrr_at_k."""

    def test_gold_at_rank_1(self) -> None:
        """First result is gold → MRR = 1.0."""
        assert compute_mrr_at_k(["A"], ["A", "B", "C"], k=5) == 1.0

    def test_gold_at_rank_2(self) -> None:
        """Gold at rank 2 → MRR = 0.5."""
        assert compute_mrr_at_k(["B"], ["A", "B", "C"], k=5) == 0.5

    def test_gold_at_rank_3(self) -> None:
        """Gold at rank 3 → MRR = 1/3."""
        assert compute_mrr_at_k(["C"], ["A", "B", "C"], k=5) == pytest.approx(1 / 3)

    def test_no_gold_in_top_k(self) -> None:
        """Gold not in top-K → MRR = 0."""
        assert compute_mrr_at_k(["D"], ["A", "B", "C"], k=3) == 0.0

    def test_multi_gold_returns_first_hit(self) -> None:
        """Multiple gold chunks — MRR uses the first one found."""
        # B is at rank 2, C at rank 3 → MRR = 1/2 (first gold hit)
        assert compute_mrr_at_k(["B", "C"], ["A", "B", "C"], k=5) == 0.5

    @pytest.mark.parametrize("k,expected", [
        (1, 0.0),   # Gold at rank 3, K=1 doesn't reach it
        (3, 1 / 3), # K=3 reaches rank 3
        (5, 1 / 3), # K=5 still finds it at rank 3
    ])
    def test_mrr_respects_k_cutoff(self, k: int, expected: float) -> None:
        """MRR only considers ranks up to K."""
        assert compute_mrr_at_k(["C"], ["A", "B", "C", "D", "E"], k=k) == pytest.approx(expected)


# ===========================================================================
# evaluate_single_question Tests
# ===========================================================================

class TestEvaluateSingleQuestion:
    """Tests for evaluate_single_question."""

    def test_perfect_retrieval(self) -> None:
        """Gold chunk at rank 1 → perfect scores."""
        qa = _make_qa_pair(gold_ids=["A"])
        retrieved = [("A", 0.95), ("B", 0.8), ("C", 0.7)]

        result = evaluate_single_question(qa, retrieved, gold_chunk_ids=["A"])

        assert result.recall_at_1 == 1.0
        assert result.precision_at_1 == 1.0
        assert result.mrr_at_5 == 1.0

    def test_gold_not_retrieved(self) -> None:
        """Gold chunk not in results → all zeros."""
        qa = _make_qa_pair(gold_ids=["D"])
        retrieved = [("A", 0.9), ("B", 0.8), ("C", 0.7)]

        result = evaluate_single_question(qa, retrieved, gold_chunk_ids=["D"])

        assert result.recall_at_1 == 0.0
        assert result.recall_at_3 == 0.0
        assert result.recall_at_5 == 0.0
        assert result.mrr_at_5 == 0.0

    def test_preserves_query_metadata(self) -> None:
        """Result contains question ID, text, and type."""
        qa = _make_qa_pair(qa_id="test_q1", question_type=QuestionType.ANALYTICAL)
        retrieved = [("A", 0.9)]

        result = evaluate_single_question(qa, retrieved, gold_chunk_ids=["A"])

        assert result.query_id == "test_q1"
        assert result.question_type == QuestionType.ANALYTICAL

    def test_stores_retrieved_ids_and_scores(self) -> None:
        """Result stores both IDs and scores from retrieval."""
        qa = _make_qa_pair()
        retrieved = [("X", 0.9), ("Y", 0.7)]

        result = evaluate_single_question(qa, retrieved, gold_chunk_ids=["X"])

        assert result.retrieved_chunk_ids == ["X", "Y"]
        assert result.retrieved_scores == [0.9, 0.7]


# ===========================================================================
# evaluate_config Tests
# ===========================================================================

class TestEvaluateConfig:
    """Tests for evaluate_config — aggregation across multiple questions."""

    def test_averages_across_questions(self) -> None:
        """Two questions: one perfect, one miss → averages are 0.5."""
        qa1 = _make_qa_pair(qa_id="q1", gold_ids=["A"])
        qa2 = _make_qa_pair(qa_id="q2", gold_ids=["D"])

        results1 = [("A", 0.9), ("B", 0.8)]  # Perfect for q1
        results2 = [("A", 0.9), ("B", 0.8)]  # Miss for q2 (gold=D)

        eval_result = evaluate_config(
            qa_pairs=[qa1, qa2],
            retrieval_results=[results1, results2],
            gold_ids_per_question=[["A"], ["D"]],
            config_id="test",
            chunk_config="B",
            embedding_model="minilm",
            retrieval_method=RetrievalMethod.VECTOR,
            num_chunks=100,
        )

        assert eval_result.num_questions == 2
        assert eval_result.avg_recall_at_1 == pytest.approx(0.5)
        assert eval_result.avg_precision_at_1 == pytest.approx(0.5)
        assert eval_result.avg_mrr_at_5 == pytest.approx(0.5)

    def test_per_question_type_breakdown(self) -> None:
        """Metrics are broken down by question type."""
        qa1 = _make_qa_pair(qa_id="q1", question_type=QuestionType.FACTUAL, gold_ids=["A"])
        qa2 = _make_qa_pair(qa_id="q2", question_type=QuestionType.ANALYTICAL, gold_ids=["B"])

        results1 = [("A", 0.9)]  # Hit for factual
        results2 = [("X", 0.9)]  # Miss for analytical

        eval_result = evaluate_config(
            qa_pairs=[qa1, qa2],
            retrieval_results=[results1, results2],
            gold_ids_per_question=[["A"], ["B"]],
            config_id="test",
            chunk_config="B",
            embedding_model="minilm",
            retrieval_method=RetrievalMethod.VECTOR,
            num_chunks=100,
        )

        assert QuestionType.FACTUAL in eval_result.metrics_by_question_type
        assert QuestionType.ANALYTICAL in eval_result.metrics_by_question_type

        factual_metrics = eval_result.metrics_by_question_type[QuestionType.FACTUAL]
        assert factual_metrics["avg_recall_at_1"] == 1.0

        analytical_metrics = eval_result.metrics_by_question_type[QuestionType.ANALYTICAL]
        assert analytical_metrics["avg_recall_at_1"] == 0.0

    def test_stores_config_metadata(self) -> None:
        """ConfigEvaluation stores its identifying metadata."""
        qa = _make_qa_pair(gold_ids=["A"])

        eval_result = evaluate_config(
            qa_pairs=[qa],
            retrieval_results=[[("A", 0.9)]],
            gold_ids_per_question=[["A"]],
            config_id="B-minilm",
            chunk_config="B",
            embedding_model="minilm",
            retrieval_method=RetrievalMethod.VECTOR,
            num_chunks=589,
        )

        assert eval_result.config_id == "B-minilm"
        assert eval_result.chunk_config == "B"
        assert eval_result.embedding_model == "minilm"
        assert eval_result.retrieval_method == RetrievalMethod.VECTOR
        assert eval_result.num_chunks == 589

    def test_empty_questions_returns_zeros(self) -> None:
        """No questions → all-zero metrics."""
        eval_result = evaluate_config(
            qa_pairs=[],
            retrieval_results=[],
            gold_ids_per_question=[],
            config_id="empty",
            chunk_config="B",
            embedding_model="minilm",
            retrieval_method=RetrievalMethod.VECTOR,
            num_chunks=0,
        )

        assert eval_result.num_questions == 0
        assert eval_result.avg_recall_at_5 == 0.0
        assert eval_result.avg_mrr_at_5 == 0.0

    def test_individual_results_stored(self) -> None:
        """ConfigEvaluation contains per-question RetrievalResult objects."""
        qa = _make_qa_pair(gold_ids=["A"])

        eval_result = evaluate_config(
            qa_pairs=[qa],
            retrieval_results=[[("A", 0.9), ("B", 0.5)]],
            gold_ids_per_question=[["A"]],
            config_id="test",
            chunk_config="B",
            embedding_model="minilm",
            retrieval_method=RetrievalMethod.VECTOR,
            num_chunks=100,
        )

        assert len(eval_result.individual_results) == 1
        assert eval_result.individual_results[0].query_id == qa.id
