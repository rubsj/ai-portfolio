"""LLM-as-Judge evaluation — Task 22.

Runs 4 judges on each generated QA pair:
1. RAFTCorrectness — is the answer correct vs the gold answer? (bool)
2. HaluEvalAnswerNonFactual — does the answer hallucinate? (bool, True=HAS hallucination)
3. ReliableCIRelevance — how relevant is the answer to the question? (0-3 int → grade string)
4. BloomTaxonomyClassifier — what cognitive level does the question target? (BloomLevel enum)

WHY LLM-as-Judge after RAGAS: RAGAS gives aggregate scores (faithfulness=0.85 across 56 samples).
Judges give per-question verdicts — "Q17 is hallucinated", "Q32 is incorrect". This granularity
finds specific failure modes that aggregate metrics hide.

Java/TS parallel: RAGAS is like a Jest coverage report (percentage). Judges are like individual
test assertions ("this specific output is wrong").
"""

from __future__ import annotations

import logging
from textwrap import dedent

from openai import OpenAI
from pydantic import BaseModel as PydanticBaseModel

from src.cache import compute_cache_key, load_cached, save_cached
from src.config import JUDGE_MODEL
from src.models import (
    BloomLevel,
    JudgeResult,
    SyntheticQAPair,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Relevance Score → Grade Mapping
# ===========================================================================

# WHY a mapping dict: ReliableCIRelevance returns int 0-3. JudgeResult.relevance_grade
# is a human-readable string. This centralizes the conversion.
_RELEVANCE_GRADES: dict[int, str] = {
    0: "Irrelevant",
    1: "Related",
    2: "Highly Relevant",
    3: "Perfectly Relevant",
}


# ===========================================================================
# Bloom Taxonomy Classifier — custom judge
# ===========================================================================

# WHY custom judge: the judges library doesn't include a Bloom classifier.
# We need it to analyze question complexity distribution — are we testing
# recall-level questions or higher-order thinking?

_BLOOM_PROMPT = dedent("""\
    Classify the following question by Bloom's Taxonomy cognitive level.

    The levels are:
    - Remember: Recall facts, definitions (e.g., "What is...?", "List the...")
    - Understand: Explain concepts, summarize (e.g., "Explain why...", "Describe...")
    - Apply: Use knowledge in new situations (e.g., "How would you use...", "Calculate...")
    - Analyze: Break down, compare, find relationships (e.g., "Compare...", "What are the differences...")
    - Evaluate: Judge, justify, critique (e.g., "Which is better...", "Is this approach valid...")
    - Create: Design, propose, construct (e.g., "Design a...", "Propose a solution...")

    Question: {question}

    Respond in EXACTLY this format (2 lines, nothing else):
    level: <one of: Remember, Understand, Apply, Analyze, Evaluate, Create>
    reasoning: <brief explanation>\
""")


def _classify_bloom(question: str) -> tuple[BloomLevel, str]:
    """Classify a question by Bloom's taxonomy using GPT-4o.

    WHY GPT-4o (not 4o-mini): Bloom classification requires nuanced understanding
    of cognitive levels. Matches JUDGE_MODEL used by other judges for consistency.

    WHY not using judges library BaseJudge: the _judge() method returns (reasoning, score)
    which maps to bool/int via Judgment. Bloom needs a string enum, so direct OpenAI
    is simpler than fighting the type system.

    Returns:
        (BloomLevel, reasoning) tuple.
    """
    prompt = _BLOOM_PROMPT.format(question=question)

    cache_key = compute_cache_key(JUDGE_MODEL, f"bloom:{prompt}")
    cached = load_cached(cache_key)

    if cached is not None:
        raw_text = cached
    else:
        client = OpenAI()
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        raw_text = response.choices[0].message.content or ""
        save_cached(
            cache_key,
            raw_text,
            model=JUDGE_MODEL,
            metadata={"question": question, "method": "bloom_classifier"},
        )

    return _parse_bloom_response(raw_text)


def _parse_bloom_response(text: str) -> tuple[BloomLevel, str]:
    """Parse the 2-line Bloom classification response.

    WHY robust parsing: LLM might add extra text. We look for lines matching
    'level:' and 'reasoning:' patterns, ignoring everything else.
    """
    level = BloomLevel.REMEMBER  # default fallback
    reasoning = "Could not parse Bloom classification."

    for line in text.strip().splitlines():
        line = line.strip()
        if line.lower().startswith("level:"):
            value = line.split(":", 1)[1].strip()
            # Try to match against BloomLevel enum values
            for bl in BloomLevel:
                if bl.value.lower() == value.lower():
                    level = bl
                    break
        elif line.lower().startswith("reasoning:"):
            reasoning = line.split(":", 1)[1].strip()

    return level, reasoning


# ===========================================================================
# Judges Library Wrappers — each wrapped in try/except individually
# ===========================================================================

def _judge_correctness(
    question: str,
    generated_answer: str,
    expected_answer: str,
) -> tuple[bool, str]:
    """Run RAFTCorrectness judge. Returns (is_correct, reasoning).

    WHY individual try/except: one judge failing shouldn't block others.
    Each judge has independent value — correctness failing doesn't mean
    we can't still check for hallucinations.
    """
    try:
        from judges.classifiers.correctness import RAFTCorrectness

        judge = RAFTCorrectness(model=f"openai/{JUDGE_MODEL}")
        judgment = judge.judge(
            input=question,
            output=generated_answer,
            expected=expected_answer,
        )
        return bool(judgment.score), judgment.reasoning
    except Exception as exc:
        logger.warning("RAFTCorrectness failed: %s — using fallback", exc)
        return _fallback_correctness(question, generated_answer, expected_answer)


def _judge_hallucination(
    question: str,
    generated_answer: str,
) -> tuple[bool, str]:
    """Run HaluEvalAnswerNonFactual judge. Returns (has_hallucination, reasoning).

    WHY HaluEval doesn't use expected_answer: it checks if the answer fabricates
    information not grounded in world knowledge, independent of the gold answer.
    """
    try:
        from judges.classifiers.hallucination import HaluEvalAnswerNonFactual

        judge = HaluEvalAnswerNonFactual(model=f"openai/{JUDGE_MODEL}")
        judgment = judge.judge(
            input=question,
            output=generated_answer,
        )
        # WHY bool(): judgment.score is True when answer IS hallucinated
        return bool(judgment.score), judgment.reasoning
    except Exception as exc:
        logger.warning("HaluEvalAnswerNonFactual failed: %s — using fallback", exc)
        return _fallback_hallucination(question, generated_answer)


def _judge_relevance(
    question: str,
    generated_answer: str,
) -> tuple[str, str]:
    """Run ReliableCIRelevance judge. Returns (grade_string, reasoning).

    WHY grade string not int: JudgeResult.relevance_grade is human-readable
    ("Perfectly Relevant" not 3). Easier to scan in reports.
    """
    try:
        from judges.graders.relevance import ReliableCIRelevance

        judge = ReliableCIRelevance(model=f"openai/{JUDGE_MODEL}")
        judgment = judge.judge(
            input=question,
            output=generated_answer,
        )
        score = int(judgment.score) if isinstance(judgment.score, (int, float)) else 0
        grade = _RELEVANCE_GRADES.get(score, "Irrelevant")
        return grade, judgment.reasoning
    except Exception as exc:
        logger.warning("ReliableCIRelevance failed: %s — using fallback", exc)
        return _fallback_relevance(question, generated_answer)


# ===========================================================================
# Fallback Implementations — when judges library fails
# ===========================================================================

# WHY fallbacks: judges library uses instructor.from_provider() which may have
# compatibility issues. Falling back to direct OpenAI structured outputs ensures
# we always get results.

class _CorrectnessResponse(PydanticBaseModel):
    """Structured output for correctness fallback."""
    is_correct: bool
    reasoning: str


class _HallucinationResponse(PydanticBaseModel):
    """Structured output for hallucination fallback."""
    has_hallucination: bool
    reasoning: str


class _RelevanceResponse(PydanticBaseModel):
    """Structured output for relevance fallback."""
    score: int
    reasoning: str


def _fallback_correctness(
    question: str,
    generated_answer: str,
    expected_answer: str,
) -> tuple[bool, str]:
    """Fallback: use OpenAI structured outputs for correctness check."""
    prompt = dedent(f"""\
        Is the generated answer correct compared to the expected answer?

        Question: {question}
        Generated Answer: {generated_answer}
        Expected Answer: {expected_answer}

        Check for keyword accuracy, numerical accuracy, and factual alignment.\
    """)

    cache_key = compute_cache_key(JUDGE_MODEL, f"fallback_correctness:{prompt}")
    cached = load_cached(cache_key)
    if cached is not None:
        parsed = _CorrectnessResponse.model_validate(cached)
        return parsed.is_correct, parsed.reasoning

    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=_CorrectnessResponse,
        temperature=0.0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        return False, "Failed to parse correctness response."

    save_cached(
        cache_key,
        result.model_dump(),
        model=JUDGE_MODEL,
        metadata={"method": "fallback_correctness"},
    )
    return result.is_correct, result.reasoning


def _fallback_hallucination(
    question: str,
    generated_answer: str,
) -> tuple[bool, str]:
    """Fallback: use OpenAI structured outputs for hallucination check."""
    prompt = dedent(f"""\
        Does the following answer contain hallucinated (non-factual) information?

        Question: {question}
        Answer: {generated_answer}

        Check for: context misunderstanding, factual contradictions, specificity issues,
        and invalid inferences.\
    """)

    cache_key = compute_cache_key(JUDGE_MODEL, f"fallback_hallucination:{prompt}")
    cached = load_cached(cache_key)
    if cached is not None:
        parsed = _HallucinationResponse.model_validate(cached)
        return parsed.has_hallucination, parsed.reasoning

    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=_HallucinationResponse,
        temperature=0.0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        return True, "Failed to parse hallucination response."

    save_cached(
        cache_key,
        result.model_dump(),
        model=JUDGE_MODEL,
        metadata={"method": "fallback_hallucination"},
    )
    return result.has_hallucination, result.reasoning


def _fallback_relevance(
    question: str,
    generated_answer: str,
) -> tuple[str, str]:
    """Fallback: use OpenAI structured outputs for relevance grading."""
    prompt = dedent(f"""\
        Rate the relevance of this answer to the question on a 0-3 scale:
        0 = Irrelevant, 1 = Related, 2 = Highly Relevant, 3 = Perfectly Relevant

        Question: {question}
        Answer: {generated_answer}\
    """)

    cache_key = compute_cache_key(JUDGE_MODEL, f"fallback_relevance:{prompt}")
    cached = load_cached(cache_key)
    if cached is not None:
        parsed = _RelevanceResponse.model_validate(cached)
        score = max(0, min(3, parsed.score))
        return _RELEVANCE_GRADES.get(score, "Irrelevant"), parsed.reasoning

    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=_RelevanceResponse,
        temperature=0.0,
    )
    result = response.choices[0].message.parsed
    if result is None:
        return "Irrelevant", "Failed to parse relevance response."

    save_cached(
        cache_key,
        result.model_dump(),
        model=JUDGE_MODEL,
        metadata={"method": "fallback_relevance"},
    )
    score = max(0, min(3, result.score))
    return _RELEVANCE_GRADES.get(score, "Irrelevant"), result.reasoning


# ===========================================================================
# Single QA Evaluation — runs all 4 judges
# ===========================================================================

def evaluate_single_with_judges(
    question_id: str,
    question: str,
    generated_answer: str,
    expected_answer: str,
    context: str,
) -> JudgeResult:
    """Run all 4 judges on one QA pair.

    WHY context parameter: currently unused by the 3 library judges (they only
    need question + answer), but the Bloom classifier uses the question text.
    Kept for potential future judges that evaluate context quality.

    Each judge is independently wrapped — one failing doesn't block others.

    Java/TS parallel: like running multiple independent assertions in a test case.
    If assertEquals fails, we still want assertTrue to run.
    """
    # 1. Correctness (RAFTCorrectness)
    correctness_score, correctness_reasoning = _judge_correctness(
        question, generated_answer, expected_answer,
    )

    # 2. Hallucination (HaluEvalAnswerNonFactual)
    has_hallucination, hallucination_reasoning = _judge_hallucination(
        question, generated_answer,
    )

    # 3. Relevance (ReliableCIRelevance)
    relevance_grade, relevance_reasoning = _judge_relevance(
        question, generated_answer,
    )

    # 4. Bloom Taxonomy (custom classifier)
    bloom_level, bloom_reasoning = _classify_bloom(question)

    return JudgeResult(
        question_id=question_id,
        question=question,
        generated_answer=generated_answer,
        expected_answer=expected_answer,
        correctness_score=correctness_score,
        correctness_reasoning=correctness_reasoning,
        has_hallucination=has_hallucination,
        hallucination_reasoning=hallucination_reasoning,
        relevance_grade=relevance_grade,
        relevance_reasoning=relevance_reasoning,
        bloom_level=bloom_level,
        bloom_reasoning=bloom_reasoning,
    )


# ===========================================================================
# Full Pipeline — run judges on all QA pairs
# ===========================================================================

def run_judge_evaluation(
    qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str],
    retrieved_contexts: list[list[str]],
) -> list[JudgeResult]:
    """Run all 4 judges on every QA pair for the best config.

    WHY sequential (not parallel): each judge call is an LLM API call. Running
    4 judges × 56 questions = 224 calls. Sequential keeps rate limits safe
    and makes debugging easier. Total time ~5min at 0.7s/call.

    Returns:
        List of JudgeResult, one per QA pair.
    """
    results: list[JudgeResult] = []

    for i, qa in enumerate(qa_pairs):
        context = "\n".join(retrieved_contexts[i])

        logger.info(
            "Judging %d/%d: %s",
            i + 1, len(qa_pairs), qa.id,
        )

        result = evaluate_single_with_judges(
            question_id=qa.id,
            question=qa.question,
            generated_answer=generated_answers[i],
            expected_answer=qa.expected_answer,
            context=context,
        )
        results.append(result)

    # Log summary
    correct_count = sum(1 for r in results if r.correctness_score)
    hallucination_count = sum(1 for r in results if r.has_hallucination)
    logger.info(
        "Judge summary: %d/%d correct, %d/%d hallucinated",
        correct_count, len(results),
        hallucination_count, len(results),
    )

    return results


# ===========================================================================
# Result Persistence
# ===========================================================================

def save_judge_results(results: list[JudgeResult]) -> None:
    """Save judge results to JSON."""
    import json

    from src.config import METRICS_DIR

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "judge_results.json"
    path.write_text(
        json.dumps([r.model_dump(mode="json") for r in results], indent=2),
    )
    logger.info("Saved %d judge results to %s", len(results), path)
