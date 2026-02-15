"""Tests for src/judge.py — LLM-as-Judge evaluation (Task 22)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.judge import (
    _classify_bloom,
    _fallback_correctness,
    _fallback_hallucination,
    _fallback_relevance,
    _judge_correctness,
    _judge_hallucination,
    _judge_relevance,
    _parse_bloom_response,
    _RELEVANCE_GRADES,
    evaluate_single_with_judges,
    run_judge_evaluation,
    save_judge_results,
)
from src.models import BloomLevel, JudgeResult, QuestionHierarchy, QuestionType, SyntheticQAPair


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


# ===========================================================================
# _parse_bloom_response Tests
# ===========================================================================

class TestParseBloomResponse:
    """Tests for parsing the 2-line Bloom classification response."""

    def test_valid_format(self) -> None:
        text = "level: Analyze\nreasoning: The question asks to compare approaches."
        level, reasoning = _parse_bloom_response(text)
        assert level == BloomLevel.ANALYZE
        assert "compare" in reasoning

    def test_case_insensitive(self) -> None:
        text = "level: REMEMBER\nreasoning: Simple recall."
        level, _ = _parse_bloom_response(text)
        assert level == BloomLevel.REMEMBER

    def test_extra_text_ignored(self) -> None:
        text = (
            "Here is my classification:\n"
            "level: Evaluate\n"
            "reasoning: Asks to judge effectiveness.\n"
            "Hope this helps!"
        )
        level, reasoning = _parse_bloom_response(text)
        assert level == BloomLevel.EVALUATE
        assert "judge" in reasoning.lower() or "effectiveness" in reasoning.lower()

    def test_invalid_level_defaults_to_remember(self) -> None:
        text = "level: Synthesize\nreasoning: Old taxonomy term."
        level, _ = _parse_bloom_response(text)
        assert level == BloomLevel.REMEMBER  # fallback

    def test_empty_string_defaults(self) -> None:
        level, reasoning = _parse_bloom_response("")
        assert level == BloomLevel.REMEMBER
        assert "Could not parse" in reasoning

    @pytest.mark.parametrize(
        "bloom_value",
        ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"],
    )
    def test_all_bloom_levels(self, bloom_value: str) -> None:
        text = f"level: {bloom_value}\nreasoning: Test."
        level, _ = _parse_bloom_response(text)
        assert level.value == bloom_value


# ===========================================================================
# _classify_bloom Tests
# ===========================================================================

class TestClassifyBloom:
    """Tests for the Bloom taxonomy classifier."""

    @patch("src.judge.load_cached", return_value=None)
    @patch("src.judge.save_cached")
    @patch("src.judge.OpenAI")
    def test_calls_openai_and_caches(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Calls GPT-4o and caches the Bloom classification."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "level: Analyze\nreasoning: Asks to break down components."
        )
        mock_client.chat.completions.create.return_value = mock_response

        level, reasoning = _classify_bloom("What are the differences between X and Y?")

        assert level == BloomLevel.ANALYZE
        assert "break down" in reasoning
        mock_save.assert_called_once()

    @patch("src.judge.load_cached")
    def test_returns_cached_bloom(self, mock_load) -> None:
        """Returns cached Bloom classification without calling OpenAI."""
        mock_load.return_value = "level: Create\nreasoning: Asks to design."

        level, reasoning = _classify_bloom("Design a solution for...")

        assert level == BloomLevel.CREATE
        assert "design" in reasoning.lower()


# ===========================================================================
# _judge_correctness Tests
# ===========================================================================

class TestJudgeCorrectness:
    """Tests for the RAFTCorrectness wrapper."""

    def test_returns_judgment(self) -> None:
        """RAFTCorrectness returns (bool, str) tuple."""
        # WHY sys.modules mock: _judge_correctness does a lazy import
        # (`from judges.classifiers.correctness import RAFTCorrectness`).
        # Must inject a mock module so the import resolves to our mock.
        mock_module = MagicMock()
        mock_judge = MagicMock()
        mock_module.RAFTCorrectness.return_value = mock_judge
        mock_judgment = MagicMock()
        mock_judgment.score = True
        mock_judgment.reasoning = "Correct answer."
        mock_judge.judge.return_value = mock_judgment

        with patch.dict("sys.modules", {"judges.classifiers.correctness": mock_module}):
            score, reasoning = _judge_correctness("Q?", "A.", "Expected A.")

        assert score is True
        assert reasoning == "Correct answer."

    @patch("src.judge._fallback_correctness")
    def test_falls_back_on_import_error(self, mock_fallback) -> None:
        """Falls back to OpenAI structured outputs when judges library fails."""
        mock_fallback.return_value = (False, "Fallback: incorrect.")

        # Force the import inside _judge_correctness to fail
        with patch.dict("sys.modules", {"judges.classifiers.correctness": None}):
            score, reasoning = _judge_correctness("Q?", "A.", "Expected A.")

        assert score is False
        mock_fallback.assert_called_once()


# ===========================================================================
# _judge_hallucination Tests
# ===========================================================================

class TestJudgeHallucination:
    """Tests for the HaluEvalAnswerNonFactual wrapper."""

    def test_returns_judgment(self) -> None:
        """HaluEval returns (has_hallucination, reasoning)."""
        mock_module = MagicMock()
        mock_judge = MagicMock()
        mock_module.HaluEvalAnswerNonFactual.return_value = mock_judge
        mock_judgment = MagicMock()
        mock_judgment.score = False
        mock_judgment.reasoning = "No hallucination detected."
        mock_judge.judge.return_value = mock_judgment

        with patch.dict("sys.modules", {"judges.classifiers.hallucination": mock_module}):
            has_hallucination, reasoning = _judge_hallucination("Q?", "A.")

        assert has_hallucination is False
        assert reasoning == "No hallucination detected."

    @patch("src.judge._fallback_hallucination")
    def test_falls_back_on_error(self, mock_fallback) -> None:
        """Falls back when judges library raises."""
        mock_fallback.return_value = (True, "Fallback: hallucinated.")

        with patch.dict("sys.modules", {"judges.classifiers.hallucination": None}):
            has_hallucination, reasoning = _judge_hallucination("Q?", "A.")

        assert has_hallucination is True
        mock_fallback.assert_called_once()


# ===========================================================================
# _judge_relevance Tests
# ===========================================================================

class TestJudgeRelevance:
    """Tests for the ReliableCIRelevance wrapper."""

    def test_returns_grade_string(self) -> None:
        """ReliableCIRelevance int score is mapped to grade string."""
        mock_module = MagicMock()
        mock_judge = MagicMock()
        mock_module.ReliableCIRelevance.return_value = mock_judge
        mock_judgment = MagicMock()
        mock_judgment.score = 3
        mock_judgment.reasoning = "Perfectly relevant."
        mock_judge.judge.return_value = mock_judgment

        with patch.dict("sys.modules", {"judges.graders.relevance": mock_module}):
            grade, reasoning = _judge_relevance("Q?", "A.")

        assert grade == "Perfectly Relevant"
        assert reasoning == "Perfectly relevant."

    @patch("src.judge._fallback_relevance")
    def test_falls_back_on_error(self, mock_fallback) -> None:
        """Falls back when judges library raises."""
        mock_fallback.return_value = ("Related", "Fallback relevance.")

        with patch.dict("sys.modules", {"judges.graders.relevance": None}):
            grade, reasoning = _judge_relevance("Q?", "A.")

        assert grade == "Related"
        mock_fallback.assert_called_once()

    @pytest.mark.parametrize(
        ("score", "expected_grade"),
        [(0, "Irrelevant"), (1, "Related"), (2, "Highly Relevant"), (3, "Perfectly Relevant")],
    )
    def test_all_grade_mappings(self, score: int, expected_grade: str) -> None:
        """All 4 relevance scores map to correct grade strings."""
        assert _RELEVANCE_GRADES[score] == expected_grade


# ===========================================================================
# Fallback Tests
# ===========================================================================

class TestFallbacks:
    """Tests for OpenAI structured output fallback implementations."""

    @patch("src.judge.load_cached", return_value=None)
    @patch("src.judge.save_cached")
    @patch("src.judge.OpenAI")
    def test_fallback_correctness(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Fallback correctness uses structured outputs."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_parsed = MagicMock()
        mock_parsed.is_correct = True
        mock_parsed.reasoning = "Matches expected."
        mock_parsed.model_dump.return_value = {"is_correct": True, "reasoning": "Matches expected."}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_parsed
        mock_client.beta.chat.completions.parse.return_value = mock_response

        score, reasoning = _fallback_correctness("Q?", "A.", "Expected.")

        assert score is True
        assert reasoning == "Matches expected."
        mock_save.assert_called_once()

    @patch("src.judge.load_cached", return_value=None)
    @patch("src.judge.save_cached")
    @patch("src.judge.OpenAI")
    def test_fallback_hallucination(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Fallback hallucination uses structured outputs."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_parsed = MagicMock()
        mock_parsed.has_hallucination = False
        mock_parsed.reasoning = "Factually accurate."
        mock_parsed.model_dump.return_value = {"has_hallucination": False, "reasoning": "Factually accurate."}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_parsed
        mock_client.beta.chat.completions.parse.return_value = mock_response

        has_hall, reasoning = _fallback_hallucination("Q?", "A.")

        assert has_hall is False
        assert reasoning == "Factually accurate."

    @patch("src.judge.load_cached", return_value=None)
    @patch("src.judge.save_cached")
    @patch("src.judge.OpenAI")
    def test_fallback_relevance(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Fallback relevance uses structured outputs."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_parsed = MagicMock()
        mock_parsed.score = 2
        mock_parsed.reasoning = "Somewhat relevant."
        mock_parsed.model_dump.return_value = {"score": 2, "reasoning": "Somewhat relevant."}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_parsed
        mock_client.beta.chat.completions.parse.return_value = mock_response

        grade, reasoning = _fallback_relevance("Q?", "A.")

        assert grade == "Highly Relevant"
        assert reasoning == "Somewhat relevant."

    @patch("src.judge.load_cached")
    def test_fallback_correctness_uses_cache(self, mock_load) -> None:
        """Cached fallback results are reused."""
        mock_load.return_value = {"is_correct": False, "reasoning": "Cached: wrong."}

        score, reasoning = _fallback_correctness("Q?", "A.", "Expected.")

        assert score is False
        assert reasoning == "Cached: wrong."

    @patch("src.judge.load_cached", return_value=None)
    @patch("src.judge.save_cached")
    @patch("src.judge.OpenAI")
    def test_fallback_correctness_handles_none_parsed(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Returns safe defaults when parsing fails."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = None
        mock_client.beta.chat.completions.parse.return_value = mock_response

        score, reasoning = _fallback_correctness("Q?", "A.", "Expected.")

        assert score is False
        assert "Failed to parse" in reasoning

    @patch("src.judge.load_cached", return_value=None)
    @patch("src.judge.save_cached")
    @patch("src.judge.OpenAI")
    def test_fallback_relevance_clamps_score(
        self, mock_openai_cls, mock_save, mock_load,
    ) -> None:
        """Scores outside 0-3 are clamped."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_parsed = MagicMock()
        mock_parsed.score = 5  # out of range
        mock_parsed.reasoning = "Very relevant."
        mock_parsed.model_dump.return_value = {"score": 5, "reasoning": "Very relevant."}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed = mock_parsed
        mock_client.beta.chat.completions.parse.return_value = mock_response

        grade, _ = _fallback_relevance("Q?", "A.")

        assert grade == "Perfectly Relevant"  # clamped to 3


# ===========================================================================
# evaluate_single_with_judges Tests
# ===========================================================================

class TestEvaluateSingleWithJudges:
    """Tests for the single QA evaluation orchestrator."""

    @patch("src.judge._classify_bloom")
    @patch("src.judge._judge_relevance")
    @patch("src.judge._judge_hallucination")
    @patch("src.judge._judge_correctness")
    def test_returns_complete_judge_result(
        self, mock_correct, mock_halluc, mock_relev, mock_bloom,
    ) -> None:
        """All 4 judges contribute to the JudgeResult."""
        mock_correct.return_value = (True, "Correct.")
        mock_halluc.return_value = (False, "No hallucination.")
        mock_relev.return_value = ("Perfectly Relevant", "Direct answer.")
        mock_bloom.return_value = (BloomLevel.ANALYZE, "Asks to compare.")

        result = evaluate_single_with_judges(
            question_id="q1",
            question="Compare X and Y?",
            generated_answer="X differs from Y in...",
            expected_answer="X and Y differ...",
            context="Context about X and Y.",
        )

        assert isinstance(result, JudgeResult)
        assert result.question_id == "q1"
        assert result.correctness_score is True
        assert result.has_hallucination is False
        assert result.relevance_grade == "Perfectly Relevant"
        assert result.bloom_level == BloomLevel.ANALYZE

    @patch("src.judge._classify_bloom")
    @patch("src.judge._judge_relevance")
    @patch("src.judge._judge_hallucination")
    @patch("src.judge._judge_correctness")
    def test_passes_correct_args_to_each_judge(
        self, mock_correct, mock_halluc, mock_relev, mock_bloom,
    ) -> None:
        """Each judge receives the appropriate arguments."""
        mock_correct.return_value = (True, "OK.")
        mock_halluc.return_value = (False, "OK.")
        mock_relev.return_value = ("Related", "OK.")
        mock_bloom.return_value = (BloomLevel.REMEMBER, "OK.")

        evaluate_single_with_judges(
            question_id="q1",
            question="What is X?",
            generated_answer="X is...",
            expected_answer="X means...",
            context="Context.",
        )

        mock_correct.assert_called_once_with("What is X?", "X is...", "X means...")
        mock_halluc.assert_called_once_with("What is X?", "X is...")
        mock_relev.assert_called_once_with("What is X?", "X is...")
        mock_bloom.assert_called_once_with("What is X?")


# ===========================================================================
# run_judge_evaluation Tests
# ===========================================================================

class TestRunJudgeEvaluation:
    """Tests for the full judge pipeline."""

    @patch("src.judge.evaluate_single_with_judges")
    def test_processes_all_qa_pairs(self, mock_eval) -> None:
        """Runs judges on every QA pair."""
        mock_eval.return_value = JudgeResult(
            question_id="q1",
            question="Q?",
            generated_answer="A.",
            expected_answer="E.",
            correctness_score=True,
            correctness_reasoning="Correct.",
            has_hallucination=False,
            hallucination_reasoning="No hallucination.",
            relevance_grade="Perfectly Relevant",
            relevance_reasoning="Direct.",
            bloom_level=BloomLevel.REMEMBER,
            bloom_reasoning="Recall.",
        )

        qa_pairs = [_make_qa_pair(qa_id="q1"), _make_qa_pair(qa_id="q2")]
        results = run_judge_evaluation(
            qa_pairs,
            ["answer1", "answer2"],
            [["ctx1"], ["ctx2"]],
        )

        assert len(results) == 2
        assert mock_eval.call_count == 2

    @patch("src.judge.evaluate_single_with_judges")
    def test_empty_qa_pairs(self, mock_eval) -> None:
        """Empty input returns empty list."""
        results = run_judge_evaluation([], [], [])

        assert results == []
        mock_eval.assert_not_called()


# ===========================================================================
# save_judge_results Tests
# ===========================================================================

class TestSaveJudgeResults:
    """Tests for result persistence."""

    @patch("src.config.METRICS_DIR")
    def test_saves_json(self, mock_dir, tmp_path) -> None:
        """Results are saved as JSON array."""
        mock_dir.__truediv__ = lambda self, name: tmp_path / name
        mock_dir.mkdir = MagicMock()

        results = [
            JudgeResult(
                question_id="q1",
                question="Q?",
                generated_answer="A.",
                expected_answer="E.",
                correctness_score=True,
                correctness_reasoning="OK.",
                has_hallucination=False,
                hallucination_reasoning="OK.",
                relevance_grade="Perfectly Relevant",
                relevance_reasoning="OK.",
                bloom_level=BloomLevel.ANALYZE,
                bloom_reasoning="OK.",
            ),
        ]

        save_judge_results(results)

        import json
        saved = json.loads((tmp_path / "judge_results.json").read_text())
        assert len(saved) == 1
        assert saved[0]["question_id"] == "q1"
        assert saved[0]["bloom_level"] == "Analyze"
