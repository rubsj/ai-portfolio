"""Tests for src/generation_evaluator.py — RAGAS generation evaluation (Task 21)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.generation_evaluator import (
    _evaluate_manually,
    _parse_manual_scores,
    _safe_score,
    evaluate_with_ragas,
    generate_answer,
    run_generation_evaluation,
    save_generation_results,
)
from src.models import Chunk, QuestionHierarchy, QuestionType, RAGASResult, SyntheticQAPair


# ===========================================================================
# Test Helpers
# ===========================================================================

def _make_chunk(chunk_id: str, text: str = "Some chunk text.") -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text=text,
        page_numbers=[1],
        start_char=0,
        end_char=len(text),
        token_count=10,
        config_name="B",
    )


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


# ===========================================================================
# _safe_score Tests
# ===========================================================================

class TestSafeScore:
    """Tests for NaN/None handling in _safe_score."""

    def test_normal_float(self) -> None:
        assert _safe_score(0.85) == 0.85

    def test_nan_returns_zero(self) -> None:
        assert _safe_score(float("nan")) == 0.0

    def test_none_returns_zero(self) -> None:
        assert _safe_score(None) == 0.0

    def test_integer_converted(self) -> None:
        assert _safe_score(1) == 1.0


# ===========================================================================
# _parse_manual_scores Tests
# ===========================================================================

class TestParseManualScores:
    """Tests for parsing the 4-line manual evaluation format."""

    def test_valid_format(self) -> None:
        text = (
            "faithfulness: 0.9\n"
            "answer_relevancy: 0.85\n"
            "context_recall: 0.7\n"
            "context_precision: 0.65\n"
        )
        scores = _parse_manual_scores(text)
        assert scores == {
            "faithfulness": 0.9,
            "answer_relevancy": 0.85,
            "context_recall": 0.7,
            "context_precision": 0.65,
        }

    def test_clamps_to_zero_one(self) -> None:
        """Scores outside [0, 1] are clamped."""
        text = "faithfulness: 1.5\ncontext_recall: -0.3\n"
        scores = _parse_manual_scores(text)
        assert scores["faithfulness"] == 1.0
        assert scores["context_recall"] == 0.0

    def test_ignores_extra_lines(self) -> None:
        """Extra text from LLM doesn't break parsing."""
        text = (
            "Here are the scores:\n"
            "faithfulness: 0.8\n"
            "answer_relevancy: 0.7\n"
            "context_recall: 0.6\n"
            "context_precision: 0.5\n"
            "Overall the answer is good.\n"
        )
        scores = _parse_manual_scores(text)
        assert len(scores) == 4

    def test_ignores_non_numeric_values(self) -> None:
        """Non-numeric values are skipped."""
        text = "faithfulness: high\ncontext_recall: 0.5\n"
        scores = _parse_manual_scores(text)
        assert "faithfulness" not in scores
        assert scores["context_recall"] == 0.5

    def test_empty_string(self) -> None:
        assert _parse_manual_scores("") == {}


# ===========================================================================
# generate_answer Tests
# ===========================================================================

class TestGenerateAnswer:
    """Tests for RAG answer generation."""

    @patch("src.generation_evaluator.load_cached", return_value=None)
    @patch("src.generation_evaluator.save_cached")
    @patch("src.generation_evaluator.OpenAI")
    def test_generates_answer_and_caches(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Calls OpenAI, returns answer text, and caches the response."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Pipe corrosion is caused by water."
        mock_client.chat.completions.create.return_value = mock_response

        result = generate_answer("What causes corrosion?", ["Context about pipes."])

        assert result == "Pipe corrosion is caused by water."
        mock_client.chat.completions.create.assert_called_once()
        mock_save.assert_called_once()

    @patch("src.generation_evaluator.load_cached", return_value="Cached answer.")
    def test_returns_cached_answer(self, mock_load) -> None:
        """Returns cached answer without calling OpenAI."""
        result = generate_answer("Any question?", ["Any context."])
        assert result == "Cached answer."

    @patch("src.generation_evaluator.load_cached", return_value=None)
    @patch("src.generation_evaluator.save_cached")
    @patch("src.generation_evaluator.OpenAI")
    def test_prompt_includes_context_and_question(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """RAG prompt contains both context chunks and the question."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer."
        mock_client.chat.completions.create.return_value = mock_response

        generate_answer("My question?", ["Chunk A text.", "Chunk B text."])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        prompt_content = call_kwargs["messages"][0]["content"]
        assert "My question?" in prompt_content
        assert "Chunk A text." in prompt_content
        assert "Chunk B text." in prompt_content

    @patch("src.generation_evaluator.load_cached", return_value=None)
    @patch("src.generation_evaluator.save_cached")
    @patch("src.generation_evaluator.OpenAI")
    def test_temperature_is_zero(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Temperature is set to 0.0 for deterministic output."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer."
        mock_client.chat.completions.create.return_value = mock_response

        generate_answer("Question?", ["Context."])

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0


# ===========================================================================
# evaluate_with_ragas Tests
# ===========================================================================

class TestEvaluateWithRagas:
    """Tests for RAGAS evaluation with fallback."""

    @patch("src.generation_evaluator._evaluate_with_ragas_library")
    def test_uses_ragas_library_when_available(self, mock_ragas) -> None:
        """Primary path: delegates to RAGAS library."""
        expected = RAGASResult(
            config_id="",
            faithfulness=0.9,
            answer_relevancy=0.8,
            context_recall=0.7,
            context_precision=0.6,
        )
        mock_ragas.return_value = expected

        qa_pairs = [_make_qa_pair()]
        result = evaluate_with_ragas(qa_pairs, ["answer"], [["context"]])

        assert result.faithfulness == 0.9
        mock_ragas.assert_called_once()

    @patch("src.generation_evaluator._evaluate_manually")
    @patch("src.generation_evaluator._evaluate_with_ragas_library")
    def test_falls_back_to_manual_on_error(
        self, mock_ragas, mock_manual,
    ) -> None:
        """Fallback: uses manual evaluation when RAGAS raises."""
        mock_ragas.side_effect = ImportError("langchain issue")
        mock_manual.return_value = RAGASResult(
            config_id="",
            faithfulness=0.7,
            answer_relevancy=0.6,
            context_recall=0.5,
            context_precision=0.4,
        )

        qa_pairs = [_make_qa_pair()]
        result = evaluate_with_ragas(qa_pairs, ["answer"], [["context"]])

        assert result.faithfulness == 0.7
        mock_manual.assert_called_once()


# ===========================================================================
# _evaluate_manually Tests
# ===========================================================================

class TestEvaluateManually:
    """Tests for the manual GPT-4o-mini fallback."""

    @patch("src.generation_evaluator.load_cached", return_value=None)
    @patch("src.generation_evaluator.save_cached")
    @patch("src.generation_evaluator.OpenAI")
    def test_manual_scores_averaged(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Manual evaluation averages scores across all questions."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # Two questions with different scores
        responses = [
            "faithfulness: 0.8\nanswer_relevancy: 0.7\ncontext_recall: 0.6\ncontext_precision: 0.5",
            "faithfulness: 1.0\nanswer_relevancy: 0.9\ncontext_recall: 0.8\ncontext_precision: 0.7",
        ]
        mock_client.chat.completions.create.side_effect = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=r))])
            for r in responses
        ]

        qa_pairs = [_make_qa_pair(qa_id="q1"), _make_qa_pair(qa_id="q2")]
        result = _evaluate_manually(
            qa_pairs,
            ["answer1", "answer2"],
            [["ctx1"], ["ctx2"]],
        )

        assert result.faithfulness == pytest.approx(0.9)  # (0.8 + 1.0) / 2
        assert result.answer_relevancy == pytest.approx(0.8)  # (0.7 + 0.9) / 2
        assert result.context_recall == pytest.approx(0.7)  # (0.6 + 0.8) / 2
        assert result.context_precision == pytest.approx(0.6)  # (0.5 + 0.7) / 2

    def test_empty_qa_pairs(self) -> None:
        """Empty input returns zero scores."""
        result = _evaluate_manually([], [], [])
        assert result.faithfulness == 0.0
        assert result.answer_relevancy == 0.0

    @patch("src.generation_evaluator.load_cached")
    def test_uses_cached_manual_scores(self, mock_load) -> None:
        """Cached manual evaluation responses are reused."""
        mock_load.return_value = (
            "faithfulness: 0.9\nanswer_relevancy: 0.8\n"
            "context_recall: 0.7\ncontext_precision: 0.6"
        )

        qa_pairs = [_make_qa_pair()]
        result = _evaluate_manually(qa_pairs, ["answer"], [["ctx"]])

        assert result.faithfulness == pytest.approx(0.9)


# ===========================================================================
# run_generation_evaluation Tests
# ===========================================================================

class TestRunGenerationEvaluation:
    """Tests for the full pipeline: retrieve → generate → evaluate."""

    @patch("src.generation_evaluator.evaluate_with_ragas")
    @patch("src.generation_evaluator.generate_answer")
    def test_full_pipeline(self, mock_generate, mock_evaluate) -> None:
        """Pipeline retrieves, generates, and evaluates."""
        # Mock FAISS store
        mock_store = MagicMock()
        mock_store.search.return_value = [("c1", 0.9), ("c2", 0.8)]

        # Mock chunk lookup
        chunks = {
            "c1": _make_chunk("c1", "Chunk 1 text."),
            "c2": _make_chunk("c2", "Chunk 2 text."),
        }

        # Mock answer generation
        mock_generate.return_value = "Generated answer."

        # Mock RAGAS evaluation
        mock_evaluate.return_value = RAGASResult(
            config_id="",
            faithfulness=0.9,
            answer_relevancy=0.8,
            context_recall=0.7,
            context_precision=0.6,
        )

        qa_pairs = [_make_qa_pair()]
        query_embs = np.random.randn(1, 1536).astype(np.float32)

        ragas_result, answers, contexts = run_generation_evaluation(
            config_id="E-openai",
            qa_pairs=qa_pairs,
            store=mock_store,
            chunk_lookup=chunks,
            query_embeddings=query_embs,
        )

        assert ragas_result.config_id == "E-openai"
        assert ragas_result.faithfulness == 0.9
        assert len(answers) == 1
        assert answers[0] == "Generated answer."
        assert len(contexts) == 1
        assert len(contexts[0]) == 2  # Two chunks retrieved

    @patch("src.generation_evaluator.evaluate_with_ragas")
    @patch("src.generation_evaluator.generate_answer")
    def test_handles_missing_chunks_in_lookup(
        self, mock_generate, mock_evaluate,
    ) -> None:
        """Chunks not in lookup are silently skipped (no KeyError)."""
        mock_store = MagicMock()
        # FAISS returns 3 IDs but only 2 are in the lookup
        mock_store.search.return_value = [
            ("c1", 0.9), ("missing", 0.85), ("c2", 0.8),
        ]

        chunks = {
            "c1": _make_chunk("c1", "Chunk 1."),
            "c2": _make_chunk("c2", "Chunk 2."),
        }

        mock_generate.return_value = "Answer."
        mock_evaluate.return_value = RAGASResult(
            config_id="", faithfulness=0.5,
            answer_relevancy=0.5, context_recall=0.5, context_precision=0.5,
        )

        qa_pairs = [_make_qa_pair()]
        query_embs = np.random.randn(1, 1536).astype(np.float32)

        _, _, contexts = run_generation_evaluation(
            "E-openai", qa_pairs, mock_store, chunks, query_embs,
        )

        # Only 2 context chunks (the missing one was skipped)
        assert len(contexts[0]) == 2


# ===========================================================================
# save_generation_results Tests
# ===========================================================================

class TestSaveGenerationResults:
    """Tests for result persistence."""

    @patch("src.generation_evaluator.METRICS_DIR")
    def test_saves_json(self, mock_dir, tmp_path) -> None:
        """Results are saved as JSON."""
        mock_dir.__truediv__ = lambda self, name: tmp_path / name
        mock_dir.mkdir = MagicMock()

        result = RAGASResult(
            config_id="E-openai",
            faithfulness=0.9,
            answer_relevancy=0.8,
            context_recall=0.7,
            context_precision=0.6,
        )

        save_generation_results(result)

        import json
        saved = json.loads((tmp_path / "ragas_results.json").read_text())
        assert saved["config_id"] == "E-openai"
        assert saved["faithfulness"] == 0.9
