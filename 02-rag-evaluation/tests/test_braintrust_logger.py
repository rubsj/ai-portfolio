"""Tests for src/braintrust_logger.py — Braintrust experiment logging (Task 23)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.braintrust_logger import (
    _check_api_key,
    log_feedback,
    log_generation_experiment,
    log_reranking_experiment,
    log_retrieval_experiment,
)
from src.models import (
    BloomLevel,
    ConfigEvaluation,
    JudgeResult,
    QuestionHierarchy,
    QuestionType,
    RAGASResult,
    RerankingComparison,
    RetrievalMethod,
    RetrievalResult,
    SyntheticQAPair,
)


# ===========================================================================
# Test Helpers
# ===========================================================================

def _make_qa_pair(
    qa_id: str = "q1",
    question: str = "What causes pipe corrosion?",
    expected_answer: str = "Corrosion is caused by chemical reactions.",
    gold_chunk_ids: list[str] | None = None,
) -> SyntheticQAPair:
    """Create a minimal SyntheticQAPair for testing."""
    return SyntheticQAPair(
        id=qa_id,
        question=question,
        question_type=QuestionType.FACTUAL,
        hierarchy=QuestionHierarchy.PARAGRAPH,
        gold_chunk_ids=gold_chunk_ids or ["c1"],
        expected_answer=expected_answer,
        source_chunk_text="Source text about pipe corrosion.",
        generation_strategy="per_chunk_chain",
    )


def _make_judge_result(
    question_id: str = "q1",
    correctness_score: bool = True,
    has_hallucination: bool = False,
) -> JudgeResult:
    """Create a minimal JudgeResult for testing."""
    return JudgeResult(
        question_id=question_id,
        question="What causes pipe corrosion?",
        generated_answer="Corrosion is caused by oxidation.",
        expected_answer="Corrosion is caused by chemical reactions.",
        correctness_score=correctness_score,
        correctness_reasoning="Answer is correct.",
        has_hallucination=has_hallucination,
        hallucination_reasoning="No hallucination.",
        relevance_grade="Perfectly Relevant",
        relevance_reasoning="Directly answers the question.",
        bloom_level=BloomLevel.UNDERSTAND,
        bloom_reasoning="Explain-type question.",
    )


def _make_retrieval_result(query_id: str = "q1") -> RetrievalResult:
    """Create a minimal RetrievalResult for testing."""
    return RetrievalResult(
        query_id=query_id,
        question="What causes pipe corrosion?",
        question_type=QuestionType.FACTUAL,
        gold_chunk_ids=["c1"],
        retrieved_chunk_ids=["c1", "c2", "c3"],
        retrieved_scores=[0.9, 0.8, 0.7],
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        precision_at_1=1.0,
        precision_at_3=0.33,
        precision_at_5=0.2,
        mrr_at_1=1.0,
        mrr_at_3=1.0,
        mrr_at_5=1.0,
    )


def _make_config_eval(config_id: str = "E-openai") -> ConfigEvaluation:
    """Create a minimal ConfigEvaluation for testing."""
    return ConfigEvaluation(
        config_id=config_id,
        chunk_config="E",
        embedding_model="openai",
        retrieval_method=RetrievalMethod.VECTOR,
        num_chunks=100,
        num_questions=1,
        avg_recall_at_1=1.0,
        avg_recall_at_3=1.0,
        avg_recall_at_5=1.0,
        avg_precision_at_1=1.0,
        avg_precision_at_3=0.33,
        avg_precision_at_5=0.2,
        avg_mrr_at_5=1.0,
        metrics_by_question_type={},
        individual_results=[_make_retrieval_result()],
    )


def _make_ragas_result(config_id: str = "E-openai") -> RAGASResult:
    """Create a minimal RAGASResult for testing."""
    return RAGASResult(
        config_id=config_id,
        faithfulness=0.9,
        answer_relevancy=0.85,
        context_recall=0.8,
        context_precision=0.75,
    )


def _make_reranking_comparison(config_id: str = "E-openai") -> RerankingComparison:
    """Create a minimal RerankingComparison for testing."""
    return RerankingComparison(
        config_id=config_id,
        recall_at_5_before=0.6,
        recall_at_5_after=0.7,
        precision_at_5_before=0.3,
        precision_at_5_after=0.4,
        mrr_at_5_before=0.5,
        mrr_at_5_after=0.6,
        recall_improvement_pct=16.67,
        precision_improvement_pct=33.33,
        mrr_improvement_pct=20.0,
    )


# ===========================================================================
# _check_api_key Tests
# ===========================================================================

class TestCheckApiKey:
    """Tests for API key validation."""

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test-key")
    def test_returns_true_when_key_set(self) -> None:
        """Returns True when BRAINTRUST_API_KEY config constant is set."""
        assert _check_api_key() is True

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "")
    @patch.dict("os.environ", {}, clear=True)
    def test_returns_false_when_key_empty(self) -> None:
        """Returns False and logs warning when no API key."""
        assert _check_api_key() is False

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "")
    @patch.dict("os.environ", {"BRAINTRUST_API_KEY": "env-key"})
    def test_falls_back_to_env_var(self) -> None:
        """Falls back to os.environ when config constant is empty."""
        assert _check_api_key() is True


# ===========================================================================
# log_retrieval_experiment Tests
# ===========================================================================

class TestLogRetrievalExperiment:
    """Tests for retrieval experiment logging."""

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "")
    @patch.dict("os.environ", {}, clear=True)
    def test_skips_when_no_api_key(self) -> None:
        """Does nothing when BRAINTRUST_API_KEY is not set."""
        config_eval = _make_config_eval()
        # Should not raise
        log_retrieval_experiment(config_eval)

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_logs_per_question_results(self, mock_bt) -> None:
        """Logs each individual result as a separate event."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        config_eval = _make_config_eval()
        log_retrieval_experiment(config_eval)

        # init called with project and experiment name
        mock_bt.init.assert_called_once_with(
            project="p2-rag-evaluation",
            experiment="E-openai",
        )

        # One log call per individual result
        assert mock_experiment.log.call_count == 1

        call_kwargs = mock_experiment.log.call_args[1]
        assert call_kwargs["input"]["question"] == "What causes pipe corrosion?"
        assert call_kwargs["scores"]["recall_at_5"] == 1.0
        assert call_kwargs["id"] == "q1"

        mock_experiment.flush.assert_called_once()

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_handles_api_error_gracefully(self, mock_bt) -> None:
        """Catches and logs exceptions without raising."""
        mock_bt.init.side_effect = ConnectionError("API unreachable")

        config_eval = _make_config_eval()
        # Should not raise
        log_retrieval_experiment(config_eval)


# ===========================================================================
# log_generation_experiment Tests
# ===========================================================================

class TestLogGenerationExperiment:
    """Tests for generation + judge experiment logging."""

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "")
    @patch.dict("os.environ", {}, clear=True)
    def test_skips_when_no_api_key(self) -> None:
        """Does nothing when BRAINTRUST_API_KEY is not set."""
        log_generation_experiment(
            "E-openai",
            [_make_qa_pair()],
            ["answer"],
            _make_ragas_result(),
            [_make_judge_result()],
        )

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_logs_with_correct_experiment_name(self, mock_bt) -> None:
        """Experiment name follows '{config_id}-generation' pattern."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_generation_experiment(
            "E-openai",
            [_make_qa_pair()],
            ["Generated answer."],
            _make_ragas_result(),
            [_make_judge_result()],
        )

        mock_bt.init.assert_called_once_with(
            project="p2-rag-evaluation",
            experiment="E-openai-generation",
        )

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_logs_ragas_and_judge_scores(self, mock_bt) -> None:
        """Each event has RAGAS scores, correctness, hallucination, and feedback."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_generation_experiment(
            "E-openai",
            [_make_qa_pair()],
            ["Generated answer."],
            _make_ragas_result(),
            [_make_judge_result(correctness_score=True, has_hallucination=False)],
        )

        call_kwargs = mock_experiment.log.call_args[1]
        scores = call_kwargs["scores"]

        # RAGAS scores
        assert scores["faithfulness"] == 0.9
        assert scores["answer_relevancy"] == 0.85

        # Judge scores
        assert scores["correctness"] == 1.0
        assert scores["hallucination"] == 1.0  # inverted: 1.0 = no hallucination

        # Feedback: correct + no hallucination = thumbs up
        assert scores["feedback"] == 1.0

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_feedback_thumbs_down_when_incorrect(self, mock_bt) -> None:
        """Feedback is 0.0 when answer is incorrect."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_generation_experiment(
            "E-openai",
            [_make_qa_pair()],
            ["Bad answer."],
            _make_ragas_result(),
            [_make_judge_result(correctness_score=False, has_hallucination=False)],
        )

        scores = mock_experiment.log.call_args[1]["scores"]
        assert scores["feedback"] == 0.0

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_feedback_thumbs_down_when_hallucinated(self, mock_bt) -> None:
        """Feedback is 0.0 when answer has hallucinations."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_generation_experiment(
            "E-openai",
            [_make_qa_pair()],
            ["Hallucinated answer."],
            _make_ragas_result(),
            [_make_judge_result(correctness_score=True, has_hallucination=True)],
        )

        scores = mock_experiment.log.call_args[1]["scores"]
        assert scores["feedback"] == 0.0

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_handles_missing_judge_result(self, mock_bt) -> None:
        """Gracefully handles QA pairs without matching judge results."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_generation_experiment(
            "E-openai",
            [_make_qa_pair(qa_id="q99")],  # No matching judge result
            ["Answer."],
            _make_ragas_result(),
            [_make_judge_result(question_id="q1")],  # Different ID
        )

        call_kwargs = mock_experiment.log.call_args[1]
        scores = call_kwargs["scores"]

        # RAGAS scores still present
        assert scores["faithfulness"] == 0.9
        # Feedback defaults to thumbs down (no judge result)
        assert scores["feedback"] == 0.0
        # No judge-specific scores
        assert "correctness" not in scores

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_includes_bloom_and_relevance_metadata(self, mock_bt) -> None:
        """Metadata includes bloom_level and relevance_grade from judge."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_generation_experiment(
            "E-openai",
            [_make_qa_pair()],
            ["Answer."],
            _make_ragas_result(),
            [_make_judge_result()],
        )

        metadata = mock_experiment.log.call_args[1]["metadata"]
        assert metadata["bloom_level"] == "Understand"
        assert metadata["relevance_grade"] == "Perfectly Relevant"

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_handles_api_error_gracefully(self, mock_bt) -> None:
        """Catches and logs exceptions without raising."""
        mock_bt.init.side_effect = RuntimeError("API error")

        log_generation_experiment(
            "E-openai",
            [_make_qa_pair()],
            ["Answer."],
            _make_ragas_result(),
            [_make_judge_result()],
        )


# ===========================================================================
# log_reranking_experiment Tests
# ===========================================================================

class TestLogRerankingExperiment:
    """Tests for reranking comparison logging."""

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "")
    @patch.dict("os.environ", {}, clear=True)
    def test_skips_when_no_api_key(self) -> None:
        """Does nothing when BRAINTRUST_API_KEY is not set."""
        log_reranking_experiment([_make_reranking_comparison()])

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_logs_all_comparisons(self, mock_bt) -> None:
        """One log call per config comparison."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        comparisons = [
            _make_reranking_comparison("E-openai"),
            _make_reranking_comparison("B-openai"),
        ]
        log_reranking_experiment(comparisons)

        mock_bt.init.assert_called_once_with(
            project="p2-rag-evaluation",
            experiment="reranking-comparison",
        )
        assert mock_experiment.log.call_count == 2
        mock_experiment.flush.assert_called_once()

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_logs_before_after_metrics(self, mock_bt) -> None:
        """Output has 'after' metrics, expected has 'before' metrics."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_reranking_experiment([_make_reranking_comparison()])

        call_kwargs = mock_experiment.log.call_args[1]

        # Before metrics in 'expected'
        assert call_kwargs["expected"]["recall_at_5_before"] == 0.6
        # After metrics in 'output'
        assert call_kwargs["output"]["recall_at_5_after"] == 0.7

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_improvement_scores_clamped_to_0_1(self, mock_bt) -> None:
        """Improvement percentages are normalized to [0, 1] for scores."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_reranking_experiment([_make_reranking_comparison()])

        scores = mock_experiment.log.call_args[1]["scores"]
        # 16.67% → 0.1667
        assert 0.0 <= scores["recall_improvement"] <= 1.0
        assert scores["recall_improvement"] == pytest.approx(0.1667, abs=0.001)

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_handles_api_error_gracefully(self, mock_bt) -> None:
        """Catches and logs exceptions without raising."""
        mock_bt.init.side_effect = TimeoutError("Slow API")
        log_reranking_experiment([_make_reranking_comparison()])


# ===========================================================================
# log_feedback Tests
# ===========================================================================

class TestLogFeedback:
    """Tests for standalone feedback logging."""

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "")
    @patch.dict("os.environ", {}, clear=True)
    def test_skips_when_no_api_key(self) -> None:
        """Does nothing when BRAINTRUST_API_KEY is not set."""
        log_feedback("exp-name", "q1", thumbs_up=True)

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_logs_thumbs_up(self, mock_bt) -> None:
        """Thumbs up logs feedback score of 1.0."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_feedback("E-openai-generation", "q1", thumbs_up=True)

        mock_bt.init.assert_called_once_with(
            project="p2-rag-evaluation",
            experiment="E-openai-generation",
            update=True,
        )
        call_kwargs = mock_experiment.log.call_args[1]
        assert call_kwargs["scores"]["feedback"] == 1.0
        assert call_kwargs["id"] == "q1"

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_logs_thumbs_down(self, mock_bt) -> None:
        """Thumbs down logs feedback score of 0.0."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_feedback("E-openai-generation", "q1", thumbs_up=False)

        call_kwargs = mock_experiment.log.call_args[1]
        assert call_kwargs["scores"]["feedback"] == 0.0

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_includes_comment_in_metadata(self, mock_bt) -> None:
        """Comment is included in metadata when provided."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_feedback("exp", "q1", thumbs_up=True, comment="Good answer!")

        call_kwargs = mock_experiment.log.call_args[1]
        assert call_kwargs["metadata"]["feedback_comment"] == "Good answer!"

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_no_metadata_when_no_comment(self, mock_bt) -> None:
        """Metadata is None when no comment provided."""
        mock_experiment = MagicMock()
        mock_bt.init.return_value = mock_experiment

        log_feedback("exp", "q1", thumbs_up=True)

        call_kwargs = mock_experiment.log.call_args[1]
        assert call_kwargs["metadata"] is None

    @patch("src.braintrust_logger.BRAINTRUST_API_KEY", "sk-test")
    @patch("src.braintrust_logger.braintrust")
    def test_handles_api_error_gracefully(self, mock_bt) -> None:
        """Catches and logs exceptions without raising."""
        mock_bt.init.side_effect = Exception("Network error")
        log_feedback("exp", "q1", thumbs_up=True)
