"""RAGAS generation evaluation — Task 21.

Takes the best config's top-5 retrieved chunks, generates answers with GPT-4o-mini,
and evaluates with RAGAS metrics: faithfulness, answer_relevancy, context_recall,
context_precision.

WHY RAGAS after retrieval: retrieval metrics (R@5, P@5, MRR) only measure if the
right chunks were found. RAGAS measures if the LLM can actually USE those chunks
to produce correct, faithful, relevant answers. A system could have perfect
retrieval but terrible generation (hallucinations, irrelevant answers).

Java/TS parallel: retrieval eval is like testing that a SQL query returns the right
rows (unit test). RAGAS is like testing that the UI renders those rows correctly
and the user sees the right information (integration test).
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
from openai import OpenAI

from src.cache import compute_cache_key, load_cached, save_cached
from src.config import (
    GENERATION_MODEL,
    METRICS_DIR,
    RERANK_TOP_N,
)
from src.models import (
    Chunk,
    RAGASResult,
    SyntheticQAPair,
)
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ===========================================================================
# RAG Prompt — used to generate answers from retrieved context
# ===========================================================================

_RAG_PROMPT = """Answer the following question based ONLY on the provided context.
If the context does not contain enough information, say "I don't have enough context."

Context:
{context}

Question: {question}

Answer:"""


# ===========================================================================
# Answer Generation
# ===========================================================================

def generate_answer(
    question: str,
    context_chunks: list[str],
    *,
    model: str = GENERATION_MODEL,
) -> str:
    """Build RAG prompt, call GPT-4o-mini, cache response. Returns answer text.

    WHY direct OpenAI (not Instructor): we need free-text answers, not structured
    output. Instructor's auto-validation overhead is unnecessary here.

    WHY temperature=0.0: deterministic output for reproducibility — same question +
    context should always produce the same answer. Makes caching reliable.

    Java/TS parallel: like calling fetch() to a REST API with a text/plain response
    instead of JSON. No schema validation needed, just the raw string.
    """
    context = "\n\n".join(context_chunks)
    prompt = _RAG_PROMPT.format(context=context, question=question)

    # Cache by model + full prompt (includes question and context)
    cache_key = compute_cache_key(model, prompt)
    cached = load_cached(cache_key)
    if cached is not None:
        return cached

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    answer = response.choices[0].message.content or ""

    save_cached(
        cache_key,
        answer,
        model=model,
        metadata={"question": question},
    )

    return answer


# ===========================================================================
# RAGAS Evaluation (with fallback)
# ===========================================================================

def _safe_score(value: object) -> float:
    """Convert a RAGAS score to float, treating NaN as 0.0.

    WHY: RAGAS returns NaN when a metric fails for a sample (e.g., empty context).
    We treat failures as 0.0 to avoid NaN propagation in averages.
    """
    if value is None:
        return 0.0
    fval = float(value)  # type: ignore[arg-type]
    if math.isnan(fval):
        return 0.0
    return fval


def evaluate_with_ragas(
    qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str],
    retrieved_contexts: list[list[str]],
) -> RAGASResult:
    """Run RAGAS evaluate() on all samples. Returns averaged metrics.

    WHY try/except: RAGAS has known issues with langchain Pydantic V1/V2 conflicts.
    If it fails, fall back to manual GPT-4o-mini-based evaluation that scores
    the same 4 dimensions with simpler prompts.
    """
    try:
        return _evaluate_with_ragas_library(
            qa_pairs, generated_answers, retrieved_contexts,
        )
    except Exception as exc:
        logger.warning("RAGAS library failed: %s — using manual fallback", exc)
        return _evaluate_manually(
            qa_pairs, generated_answers, retrieved_contexts,
        )


def _evaluate_with_ragas_library(
    qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str],
    retrieved_contexts: list[list[str]],
) -> RAGASResult:
    """Use RAGAS evaluate() directly.

    WHY lazy import: avoids loading langchain/ragas at module level, keeping
    import time fast and allowing tests to mock without heavy deps.

    WHY LangchainLLMWrapper: RAGAS requires its own LLM interface. The wrapper
    adapts langchain's ChatOpenAI to RAGAS's BaseRagasLLM.

    WHY explicit gpt-4o-mini: RAGAS defaults to gpt-4o which is ~17x more
    expensive. Overriding to gpt-4o-mini keeps cost at ~$0.03 for 56 samples.
    """
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    # Build RAGAS dataset from our QA pairs + generated answers
    # WHY SingleTurnSample: RAGAS v0.4 data model. Each sample has user_input,
    # response (generated), retrieved_contexts (from FAISS), and reference
    # (gold answer for context_recall/precision).
    samples = [
        SingleTurnSample(
            user_input=qa.question,
            response=generated_answers[i],
            retrieved_contexts=retrieved_contexts[i],
            reference=qa.expected_answer,
        )
        for i, qa in enumerate(qa_pairs)
    ]
    dataset = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

    logger.info("Running RAGAS evaluate() on %d samples ...", len(samples))
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=evaluator_llm,
        raise_exceptions=False,
    )

    # WHY dict(result): EvaluationResult supports dict-like access for aggregated
    # scores. Keys are metric names (e.g., "faithfulness", "context_recall").
    scores = dict(result)
    logger.info("RAGAS scores: %s", scores)

    return RAGASResult(
        config_id="",  # filled by caller
        faithfulness=_safe_score(scores.get("faithfulness", 0.0)),
        answer_relevancy=_safe_score(
            scores.get("answer_relevancy", scores.get("response_relevancy", 0.0)),
        ),
        context_recall=_safe_score(scores.get("context_recall", 0.0)),
        context_precision=_safe_score(scores.get("context_precision", 0.0)),
    )


# ===========================================================================
# Manual Fallback — when RAGAS has dependency conflicts
# ===========================================================================

# WHY a single prompt for all 4 dimensions: 1 API call per question (56 total)
# instead of 4 calls per question (224 total). Saves cost and latency.
_MANUAL_EVAL_PROMPT = """Score the following RAG system output on 4 dimensions (0.0 to 1.0 each).

Question: {question}
Generated Answer: {answer}
Expected Answer: {expected}
Retrieved Context:
{context}

Score each dimension:
1. Faithfulness: Are all claims in the generated answer supported by the retrieved context? (1.0 = fully supported, 0.0 = unsupported)
2. Answer Relevancy: Does the generated answer directly address the question? (1.0 = perfectly relevant, 0.0 = irrelevant)
3. Context Recall: Does the retrieved context contain the key information from the expected answer? (1.0 = full coverage, 0.0 = no coverage)
4. Context Precision: Is the retrieved context focused and relevant to the question? (1.0 = all context relevant, 0.0 = all noise)

Reply in EXACTLY this format (4 lines, nothing else):
faithfulness: <score>
answer_relevancy: <score>
context_recall: <score>
context_precision: <score>"""


def _parse_manual_scores(text: str) -> dict[str, float]:
    """Parse the 4-line score format from the manual evaluation LLM response.

    WHY parse line-by-line: simple, deterministic parsing. If the LLM adds
    extra text, we still extract scores from lines matching the pattern.
    """
    scores: dict[str, float] = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower().replace(" ", "_")
        try:
            score = max(0.0, min(1.0, float(value.strip())))
        except ValueError:
            continue
        if key in ("faithfulness", "answer_relevancy", "context_recall", "context_precision"):
            scores[key] = score
    return scores


def _evaluate_manually(
    qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str],
    retrieved_contexts: list[list[str]],
) -> RAGASResult:
    """Manual GPT-4o-mini-based evaluation fallback.

    WHY fallback exists: RAGAS has known issues with langchain Pydantic V1/V2
    conflicts. This manual approach scores the same 4 dimensions using direct
    OpenAI calls — less sophisticated than RAGAS (no claim extraction, no
    embedding-based relevancy) but produces comparable scores.

    Java/TS parallel: like a polyfill — when the library fails, we implement
    the same interface with simpler logic.
    """
    client = OpenAI()

    all_scores: list[dict[str, float]] = []

    for i, qa in enumerate(qa_pairs):
        context = "\n".join(retrieved_contexts[i])
        prompt = _MANUAL_EVAL_PROMPT.format(
            question=qa.question,
            answer=generated_answers[i],
            expected=qa.expected_answer,
            context=context,
        )

        cache_key = compute_cache_key(GENERATION_MODEL, f"manual_eval:{prompt}")
        cached = load_cached(cache_key)

        if cached is not None:
            scores = _parse_manual_scores(cached)
        else:
            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw_text = response.choices[0].message.content or ""
            scores = _parse_manual_scores(raw_text)
            save_cached(
                cache_key,
                raw_text,
                model=GENERATION_MODEL,
                metadata={"question": qa.question, "method": "manual_fallback"},
            )

        all_scores.append(scores)

    n = len(qa_pairs)
    if n == 0:
        return RAGASResult(
            config_id="",
            faithfulness=0.0,
            answer_relevancy=0.0,
            context_recall=0.0,
            context_precision=0.0,
        )

    return RAGASResult(
        config_id="",  # filled by caller
        faithfulness=sum(s.get("faithfulness", 0.0) for s in all_scores) / n,
        answer_relevancy=sum(s.get("answer_relevancy", 0.0) for s in all_scores) / n,
        context_recall=sum(s.get("context_recall", 0.0) for s in all_scores) / n,
        context_precision=sum(s.get("context_precision", 0.0) for s in all_scores) / n,
    )


# ===========================================================================
# Full Pipeline — retrieve → generate → evaluate
# ===========================================================================

def run_generation_evaluation(
    config_id: str,
    qa_pairs: list[SyntheticQAPair],
    store: FAISSVectorStore,
    chunk_lookup: dict[str, Chunk],
    query_embeddings: np.ndarray,
) -> tuple[RAGASResult, list[str], list[list[str]]]:
    """Full pipeline: retrieve top-5 → generate answers → RAGAS eval.

    WHY return answers + contexts: downstream steps (judges, Braintrust logging)
    need these — avoid recomputing.

    Args:
        config_id: Config identifier (e.g., "E-openai").
        qa_pairs: All synthetic QA pairs.
        store: FAISS vector store for this config.
        chunk_lookup: Dict mapping chunk_id → Chunk for text lookup.
        query_embeddings: Pre-computed query embeddings (OpenAI).

    Returns:
        (RAGASResult, generated_answers, retrieved_contexts) tuple.
    """
    generated_answers: list[str] = []
    retrieved_contexts: list[list[str]] = []

    for i, qa in enumerate(qa_pairs):
        # Retrieve top-5 chunks from FAISS
        results = store.search(query_embeddings[i], k=RERANK_TOP_N)
        retrieved_ids = [cid for cid, _ in results]

        # Look up chunk texts (FAISS only stores vectors)
        context_texts = []
        for cid in retrieved_ids:
            chunk = chunk_lookup.get(cid)
            if chunk:
                context_texts.append(chunk.text)

        retrieved_contexts.append(context_texts)

        # Generate answer using RAG prompt
        answer = generate_answer(qa.question, context_texts)
        generated_answers.append(answer)

    logger.info("Generated %d answers for %s", len(generated_answers), config_id)

    # Evaluate with RAGAS (or manual fallback)
    ragas_result = evaluate_with_ragas(qa_pairs, generated_answers, retrieved_contexts)
    # WHY model_copy: fill in config_id that evaluate_with_ragas leaves empty
    ragas_result = ragas_result.model_copy(update={"config_id": config_id})

    logger.info(
        "RAGAS %s: faith=%.3f relevancy=%.3f recall=%.3f precision=%.3f",
        config_id,
        ragas_result.faithfulness,
        ragas_result.answer_relevancy,
        ragas_result.context_recall,
        ragas_result.context_precision,
    )

    return ragas_result, generated_answers, retrieved_contexts


# ===========================================================================
# Result Persistence
# ===========================================================================

def save_generation_results(ragas_result: RAGASResult) -> None:
    """Save RAGAS evaluation results to JSON."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "ragas_results.json"
    path.write_text(json.dumps(ragas_result.model_dump(mode="json"), indent=2))
    logger.info("Saved RAGAS results to %s", path)
