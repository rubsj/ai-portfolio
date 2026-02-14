"""Retrieval evaluation metrics — Recall, Precision, MRR at K=1,3,5.

Pure computation module. No LLM calls, no I/O, no external dependencies
beyond Pydantic (for the data models). Takes gold chunk IDs + retrieved
chunk IDs and produces RetrievalResult and ConfigEvaluation objects.

Java/TS parallel: like a pure utility class with static methods — no state,
no side effects, just input → output. Similar to a JUnit assertion helper
or a metrics calculator in a test framework.

WHY separate module: retrieval evaluation is orthogonal to the retrieval
itself. Keeping metrics computation pure makes it easy to test (no mocking)
and reuse across grid search, reranking comparison, etc.
"""

from __future__ import annotations

from collections import defaultdict

from src.models import (
    ConfigEvaluation,
    QuestionType,
    RetrievalMethod,
    RetrievalResult,
    SyntheticQAPair,
)


# ===========================================================================
# Core Metric Functions
# ===========================================================================

def compute_recall_at_k(
    gold_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """Recall@K = |gold ∩ top_k| / |gold|.

    Measures: what fraction of relevant chunks did we retrieve in top-K?

    WHY divide by |gold| not |top_k|: recall is about coverage of the
    ground truth. If gold has 3 chunks and we found 2 in top-5, recall=0.67
    regardless of K. Higher K → more chances → higher recall (monotonic).

    Java/TS parallel: like computing set intersection size / set A size.
    """
    if not gold_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    return len(gold_set & top_k) / len(gold_set)


def compute_precision_at_k(
    gold_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """Precision@K = |gold ∩ top_k| / K.

    Measures: what fraction of top-K results are relevant?

    WHY divide by K (not |top_k|): standard definition uses K, not the
    actual number retrieved. If retriever returns fewer than K results,
    the missing slots count as non-relevant (precision drops).

    Java/TS parallel: like computing hit rate in a cache — hits / attempts.
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    return len(gold_set & top_k) / k


def compute_mrr_at_k(
    gold_ids: list[str],
    retrieved_ids: list[str],
    k: int,
) -> float:
    """MRR@K = 1 / rank of first gold chunk in top-K, else 0.

    Measures: how quickly does the retriever surface a relevant result?

    WHY MRR over MAP: MRR is simpler and directly answers "how many results
    does a user need to scan before finding something useful?" — key UX metric
    for RAG pipelines where the LLM typically reads top-1 or top-3.

    Java/TS parallel: like finding indexOf(goldItem) in a ranked list and
    returning 1/(index+1). If not found → 0.
    """
    gold_set = set(gold_ids)
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in gold_set:
            return 1.0 / rank
    return 0.0


# ===========================================================================
# Single Question Evaluation
# ===========================================================================

def evaluate_single_question(
    qa_pair: SyntheticQAPair,
    retrieved: list[tuple[str, float]],
    gold_chunk_ids: list[str],
) -> RetrievalResult:
    """Evaluate retrieval for one question — computes all 9 metrics.

    Args:
        qa_pair: The synthetic QA pair being evaluated.
        retrieved: List of (chunk_id, score) tuples from the retriever,
            in ranked order (best first).
        gold_chunk_ids: Ground-truth chunk IDs for this question in the
            TARGET config's namespace (already mapped from Config B if needed).

    WHY gold_chunk_ids as parameter (not from qa_pair): for non-B configs,
    gold IDs must be mapped from Config B namespace to the target config's
    namespace. The caller (grid_search.py) handles this mapping.
    """
    retrieved_ids = [chunk_id for chunk_id, _ in retrieved]
    retrieved_scores = [score for _, score in retrieved]

    return RetrievalResult(
        query_id=qa_pair.id,
        question=qa_pair.question,
        question_type=qa_pair.question_type,
        gold_chunk_ids=gold_chunk_ids,
        retrieved_chunk_ids=retrieved_ids,
        retrieved_scores=retrieved_scores,
        recall_at_1=compute_recall_at_k(gold_chunk_ids, retrieved_ids, 1),
        recall_at_3=compute_recall_at_k(gold_chunk_ids, retrieved_ids, 3),
        recall_at_5=compute_recall_at_k(gold_chunk_ids, retrieved_ids, 5),
        precision_at_1=compute_precision_at_k(gold_chunk_ids, retrieved_ids, 1),
        precision_at_3=compute_precision_at_k(gold_chunk_ids, retrieved_ids, 3),
        precision_at_5=compute_precision_at_k(gold_chunk_ids, retrieved_ids, 5),
        mrr_at_1=compute_mrr_at_k(gold_chunk_ids, retrieved_ids, 1),
        mrr_at_3=compute_mrr_at_k(gold_chunk_ids, retrieved_ids, 3),
        mrr_at_5=compute_mrr_at_k(gold_chunk_ids, retrieved_ids, 5),
    )


# ===========================================================================
# Config-Level Evaluation (aggregate across all questions)
# ===========================================================================

def evaluate_config(
    qa_pairs: list[SyntheticQAPair],
    retrieval_results: list[list[tuple[str, float]]],
    gold_ids_per_question: list[list[str]],
    *,
    config_id: str,
    chunk_config: str,
    embedding_model: str,
    retrieval_method: RetrievalMethod,
    num_chunks: int,
) -> ConfigEvaluation:
    """Evaluate retrieval across all questions for one config.

    Args:
        qa_pairs: All synthetic QA pairs.
        retrieval_results: For each question, list of (chunk_id, score) from search.
        gold_ids_per_question: For each question, mapped gold chunk IDs in this
            config's namespace.
        config_id: Identifier like 'B-minilm' or 'bm25'.
        chunk_config: Chunk config name (A, B, C, D, E).
        embedding_model: Model name or 'bm25'.
        retrieval_method: VECTOR or BM25.
        num_chunks: Total chunks in this config's index.

    WHY parallel lists (not a dict): qa_pairs, retrieval_results, and
    gold_ids_per_question are aligned by index. This avoids dict key
    mismatches and keeps the API simple for the grid search loop.
    """
    # Evaluate each question individually
    individual_results: list[RetrievalResult] = []
    for qa_pair, retrieved, gold_ids in zip(
        qa_pairs, retrieval_results, gold_ids_per_question, strict=True,
    ):
        result = evaluate_single_question(qa_pair, retrieved, gold_ids)
        individual_results.append(result)

    n = len(individual_results)

    # WHY guard against n=0: shouldn't happen in practice, but division by
    # zero would crash the pipeline. Return zeros instead.
    if n == 0:
        return ConfigEvaluation(
            config_id=config_id,
            chunk_config=chunk_config,
            embedding_model=embedding_model,
            retrieval_method=retrieval_method,
            num_chunks=num_chunks,
            num_questions=0,
            avg_recall_at_1=0.0,
            avg_recall_at_3=0.0,
            avg_recall_at_5=0.0,
            avg_precision_at_1=0.0,
            avg_precision_at_3=0.0,
            avg_precision_at_5=0.0,
            avg_mrr_at_5=0.0,
            metrics_by_question_type={},
            individual_results=[],
        )

    # Compute averages across all questions
    avg_recall_at_1 = sum(r.recall_at_1 for r in individual_results) / n
    avg_recall_at_3 = sum(r.recall_at_3 for r in individual_results) / n
    avg_recall_at_5 = sum(r.recall_at_5 for r in individual_results) / n
    avg_precision_at_1 = sum(r.precision_at_1 for r in individual_results) / n
    avg_precision_at_3 = sum(r.precision_at_3 for r in individual_results) / n
    avg_precision_at_5 = sum(r.precision_at_5 for r in individual_results) / n
    avg_mrr_at_5 = sum(r.mrr_at_5 for r in individual_results) / n

    # Per-question-type breakdown — groups results by QuestionType, averages each
    # WHY defaultdict(list): avoids KeyError when a question type has zero results.
    by_type: dict[QuestionType, list[RetrievalResult]] = defaultdict(list)
    for r in individual_results:
        by_type[r.question_type].append(r)

    metrics_by_question_type: dict[QuestionType, dict[str, float]] = {}
    for qtype, results in by_type.items():
        m = len(results)
        metrics_by_question_type[qtype] = {
            "count": m,
            "avg_recall_at_1": sum(r.recall_at_1 for r in results) / m,
            "avg_recall_at_3": sum(r.recall_at_3 for r in results) / m,
            "avg_recall_at_5": sum(r.recall_at_5 for r in results) / m,
            "avg_precision_at_1": sum(r.precision_at_1 for r in results) / m,
            "avg_precision_at_3": sum(r.precision_at_3 for r in results) / m,
            "avg_precision_at_5": sum(r.precision_at_5 for r in results) / m,
            "avg_mrr_at_5": sum(r.mrr_at_5 for r in results) / m,
        }

    return ConfigEvaluation(
        config_id=config_id,
        chunk_config=chunk_config,
        embedding_model=embedding_model,
        retrieval_method=retrieval_method,
        num_chunks=num_chunks,
        num_questions=n,
        avg_recall_at_1=avg_recall_at_1,
        avg_recall_at_3=avg_recall_at_3,
        avg_recall_at_5=avg_recall_at_5,
        avg_precision_at_1=avg_precision_at_1,
        avg_precision_at_3=avg_precision_at_3,
        avg_precision_at_5=avg_precision_at_5,
        avg_mrr_at_5=avg_mrr_at_5,
        metrics_by_question_type=metrics_by_question_type,
        individual_results=individual_results,
    )
