"""Cohere cross-encoder reranking — Task 20.

Takes FAISS top-20 candidates and uses Cohere Rerank API to narrow down to
top-5 with higher relevance. Computes before/after metrics to quantify the
reranking improvement.

WHY cross-encoder reranking: FAISS uses bi-encoder embeddings (query and doc
embedded independently). Bi-encoders are fast but less accurate because they
can't model token-level query-doc interactions. Cross-encoders (like Cohere
rerank-v3.5) score query+doc pairs jointly — more accurate but slower.
The 2-stage pipeline (FAISS narrow → Cohere refine) combines speed and quality.

Java/TS parallel: like a database query with a coarse filter (WHERE clause)
followed by a fine-grained scorer (ORDER BY with UDF). FAISS is the WHERE,
Cohere is the ORDER BY.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np

from src.cache import compute_cache_key, load_cached, save_cached
from src.config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    INDICES_DIR,
    METRICS_DIR,
    RERANK_RETRIEVAL_TOP_N,
    RERANK_TOP_N,
    model_key,
)
from src.embedder import create_embedder
from src.models import (
    Chunk,
    EmbeddingModel,
    RerankingComparison,
    SyntheticQAPair,
)
from src.retrieval_evaluator import (
    compute_mrr_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ===========================================================================
# Core Reranking
# ===========================================================================

def rerank_chunks(
    query: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    *,
    top_n: int = RERANK_TOP_N,
) -> list[tuple[str, float]]:
    """Call Cohere Rerank API. Returns (chunk_id, relevance_score) sorted desc.

    WHY cache: Cohere free tier has 1000 calls/month. Caching prevents
    duplicate calls during development and reruns.

    Args:
        query: The search query.
        chunk_ids: IDs of candidate chunks (aligned with chunk_texts).
        chunk_texts: Text content of candidate chunks.
        top_n: Number of top results to return after reranking.

    Returns:
        List of (chunk_id, relevance_score) tuples, sorted by relevance desc.
    """
    if not chunk_ids:
        return []

    # WHY cache by query+docs: same query+docs should produce same reranking.
    cache_prompt = f"rerank:{query}\n---\n" + "\n".join(chunk_texts)
    cache_key = compute_cache_key(COHERE_RERANK_MODEL, cache_prompt)
    cached = load_cached(cache_key)

    if cached is not None:
        # Cached response is list of {"chunk_id": ..., "relevance_score": ...}
        return [(r["chunk_id"], r["relevance_score"]) for r in cached]

    # WHY lazy import: avoids loading cohere at module level, which would
    # require COHERE_API_KEY to be set even for tests that mock this function.
    import cohere

    # WHY rate limit: Cohere trial key allows 10 API calls/minute.
    # 6.5s between calls keeps us safely under the limit.
    # Only affects uncached calls — cached responses bypass this.
    time.sleep(6.5)

    client = cohere.ClientV2(api_key=COHERE_API_KEY)
    response = client.rerank(
        model=COHERE_RERANK_MODEL,
        query=query,
        documents=chunk_texts,
        top_n=top_n,
    )

    # WHY map via index: Cohere returns indices into the documents list,
    # not the chunk IDs. Map back using the aligned chunk_ids list.
    results: list[tuple[str, float]] = [
        (chunk_ids[r.index], r.relevance_score)
        for r in response.results
    ]

    # Cache for reuse
    save_cached(
        cache_key,
        [{"chunk_id": cid, "relevance_score": score} for cid, score in results],
        model=COHERE_RERANK_MODEL,
        metadata={"query": query, "top_n": top_n},
    )

    return results


# ===========================================================================
# Per-Config Reranking Evaluation
# ===========================================================================

def rerank_config(
    config_id: str,
    qa_pairs: list[SyntheticQAPair],
    store: FAISSVectorStore,
    chunk_lookup: dict[str, Chunk],
    gold_ids_per_question: list[list[str]],
    query_embeddings: np.ndarray,
) -> RerankingComparison:
    """For one config: FAISS top-20 → Cohere rerank → top-5.

    Computes before/after metrics and improvement percentages.

    Args:
        config_id: Config identifier (e.g., "E-openai").
        qa_pairs: All synthetic QA pairs.
        store: FAISS vector store for this config.
        chunk_lookup: Dict mapping chunk_id → Chunk for text lookup.
        gold_ids_per_question: Gold chunk IDs per question (already mapped).
        query_embeddings: Pre-computed query embeddings (OpenAI).

    Returns:
        RerankingComparison with before/after metrics.
    """
    # Accumulators for averaging across all questions
    before_recall = []
    before_precision = []
    before_mrr = []
    after_recall = []
    after_precision = []
    after_mrr = []

    for i, qa in enumerate(qa_pairs):
        gold_ids = gold_ids_per_question[i]

        # FAISS top-20 retrieval
        faiss_results = store.search(query_embeddings[i], k=RERANK_RETRIEVAL_TOP_N)
        retrieved_ids = [cid for cid, _ in faiss_results]

        # Before: metrics on FAISS top-5 (retrieved_ids[:5] of the 20 candidates)
        before_recall.append(compute_recall_at_k(gold_ids, retrieved_ids, 5))
        before_precision.append(compute_precision_at_k(gold_ids, retrieved_ids, 5))
        before_mrr.append(compute_mrr_at_k(gold_ids, retrieved_ids, 5))

        # Look up chunk texts for Cohere (FAISS only stores vectors)
        chunk_texts = []
        valid_ids = []
        for cid in retrieved_ids:
            chunk = chunk_lookup.get(cid)
            if chunk:
                chunk_texts.append(chunk.text)
                valid_ids.append(cid)

        # Cohere rerank → top-5
        reranked = rerank_chunks(
            query=qa.question,
            chunk_ids=valid_ids,
            chunk_texts=chunk_texts,
            top_n=RERANK_TOP_N,
        )
        reranked_ids = [cid for cid, _ in reranked]

        # After: metrics on Cohere's reranked top-5
        after_recall.append(compute_recall_at_k(gold_ids, reranked_ids, 5))
        after_precision.append(compute_precision_at_k(gold_ids, reranked_ids, 5))
        after_mrr.append(compute_mrr_at_k(gold_ids, reranked_ids, 5))

    n = len(qa_pairs)

    # Average across all questions
    avg_recall_before = sum(before_recall) / n if n else 0.0
    avg_recall_after = sum(after_recall) / n if n else 0.0
    avg_precision_before = sum(before_precision) / n if n else 0.0
    avg_precision_after = sum(after_precision) / n if n else 0.0
    avg_mrr_before = sum(before_mrr) / n if n else 0.0
    avg_mrr_after = sum(after_mrr) / n if n else 0.0

    # WHY guard against zero: division by zero if before-metric is 0.
    # 0→0 improvement = 0%, 0→positive improvement = "infinite" capped at 0.
    def _improvement_pct(before: float, after: float) -> float:
        if before == 0.0:
            return 0.0
        return ((after - before) / before) * 100

    return RerankingComparison(
        config_id=config_id,
        recall_at_5_before=avg_recall_before,
        recall_at_5_after=avg_recall_after,
        precision_at_5_before=avg_precision_before,
        precision_at_5_after=avg_precision_after,
        mrr_at_5_before=avg_mrr_before,
        mrr_at_5_after=avg_mrr_after,
        recall_improvement_pct=_improvement_pct(avg_recall_before, avg_recall_after),
        precision_improvement_pct=_improvement_pct(avg_precision_before, avg_precision_after),
        mrr_improvement_pct=_improvement_pct(avg_mrr_before, avg_mrr_after),
    )


# ===========================================================================
# Entry Point — Rerank Top-3 Configs
# ===========================================================================

def run_reranking(
    top_config_ids: list[str],
    qa_pairs: list[SyntheticQAPair],
    chunks_by_config: dict[str, list[Chunk]],
    b_chunks_lookup: dict[str, Chunk],
) -> list[RerankingComparison]:
    """Rerank top-3 configs. Embeds queries once (all 3 use OpenAI).

    Args:
        top_config_ids: Config IDs to rerank (e.g., ["E-openai", "B-openai", "D-openai"]).
        qa_pairs: All synthetic QA pairs.
        chunks_by_config: Dict mapping config name → list of Chunk objects.
        b_chunks_lookup: Dict mapping Config B chunk ID → Chunk for gold mapping.

    Returns:
        List of RerankingComparison, one per config.
    """
    from src.grid_search import map_gold_chunks

    # WHY embed once: all top-3 configs use OpenAI embeddings.
    # Embedding 56 queries costs ~$0.01 — don't waste it by re-embedding per config.
    logger.info("Embedding %d queries with OpenAI for reranking ...", len(qa_pairs))
    embedder = create_embedder(EmbeddingModel.OPENAI)
    query_embeddings = embedder.embed([qa.question for qa in qa_pairs])

    comparisons: list[RerankingComparison] = []

    for config_id in top_config_ids:
        # Parse config_id: "E-openai" → config_name="E", model_key="openai"
        config_name, mkey = config_id.split("-", 1)

        # Load FAISS index
        index_path = INDICES_DIR / f"{mkey}_{config_name}"
        store = FAISSVectorStore.load(index_path)

        # Build chunk lookup for this config
        config_chunks = chunks_by_config[config_name]
        chunk_lookup = {chunk.id: chunk for chunk in config_chunks}

        # Map gold chunk IDs to this config's namespace
        if config_name == "B":
            gold_ids_per_question = [qa.gold_chunk_ids for qa in qa_pairs]
        else:
            gold_ids_per_question = [
                map_gold_chunks(qa.gold_chunk_ids, b_chunks_lookup, config_chunks)
                for qa in qa_pairs
            ]

        logger.info("Reranking config %s ...", config_id)
        comparison = rerank_config(
            config_id=config_id,
            qa_pairs=qa_pairs,
            store=store,
            chunk_lookup=chunk_lookup,
            gold_ids_per_question=gold_ids_per_question,
            query_embeddings=query_embeddings,
        )
        comparisons.append(comparison)

        logger.info(
            "  %s: R@5 %.3f → %.3f (%+.1f%%)  P@5 %.3f → %.3f (%+.1f%%)",
            config_id,
            comparison.recall_at_5_before,
            comparison.recall_at_5_after,
            comparison.recall_improvement_pct,
            comparison.precision_at_5_before,
            comparison.precision_at_5_after,
            comparison.precision_improvement_pct,
        )

    return comparisons


# ===========================================================================
# Result Persistence
# ===========================================================================

def save_reranking_results(comparisons: list[RerankingComparison]) -> None:
    """Save reranking comparison results to JSON."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "reranking_results.json"
    data = [c.model_dump(mode="json") for c in comparisons]
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %d reranking comparisons to %s", len(comparisons), path)


# TODO(Day 5): Extract orchestration into cli.py when building Click commands

# ===========================================================================
# Day 4 Orchestration Runner — Tasks 20-24
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from src.braintrust_logger import (
        log_feedback,
        log_generation_experiment,
        log_reranking_experiment,
        log_retrieval_experiment,
    )
    from src.config import OUTPUT_DIR
    from src.generation_evaluator import (
        run_generation_evaluation,
        save_generation_results,
    )
    from src.judge import run_judge_evaluation, save_judge_results
    from src.models import ConfigEvaluation
    from src.synthetic_qa import load_qa_pairs
    from src.visualization import load_evaluations

    # -----------------------------------------------------------------------
    # Step 1: Load grid search results → identify top-3 configs by Recall@5
    # -----------------------------------------------------------------------
    logger.info("Loading grid search results ...")
    evaluations = load_evaluations()
    evaluations.sort(key=lambda e: e.avg_recall_at_5, reverse=True)
    top_3 = evaluations[:3]
    top_config_ids = [e.config_id for e in top_3]
    best_config_id = top_config_ids[0]

    logger.info(
        "Top-3 by Recall@5: %s",
        ", ".join(f"{e.config_id} ({e.avg_recall_at_5:.3f})" for e in top_3),
    )

    # -----------------------------------------------------------------------
    # Step 2: Load QA pairs
    # -----------------------------------------------------------------------
    logger.info("Loading QA pairs ...")
    qa_pairs = load_qa_pairs()
    logger.info("Loaded %d QA pairs", len(qa_pairs))

    # -----------------------------------------------------------------------
    # Step 3: Load chunk files → chunk_lookup dicts
    # -----------------------------------------------------------------------
    logger.info("Loading chunk files ...")
    config_names = {"A", "B", "C", "D", "E"}
    chunks_by_config: dict[str, list[Chunk]] = {}
    for name in sorted(config_names):
        path = OUTPUT_DIR / f"chunks_{name}.json"
        raw = json.loads(path.read_text())
        chunks_by_config[name] = [Chunk.model_validate(item) for item in raw]
        logger.info("  Config %s: %d chunks", name, len(chunks_by_config[name]))

    b_chunks_lookup = {c.id: c for c in chunks_by_config["B"]}

    # -----------------------------------------------------------------------
    # Step 4: RERANKING — top-3 configs (Task 20)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("RERANKING — top-3 configs")
    logger.info("=" * 60)

    comparisons = run_reranking(
        top_config_ids=top_config_ids,
        qa_pairs=qa_pairs,
        chunks_by_config=chunks_by_config,
        b_chunks_lookup=b_chunks_lookup,
    )
    save_reranking_results(comparisons)

    # -----------------------------------------------------------------------
    # Step 5: GENERATION + RAGAS — best config (Task 21)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("GENERATION + RAGAS — %s", best_config_id)
    logger.info("=" * 60)

    # Parse best config for FAISS index loading
    best_config_name, best_mkey = best_config_id.split("-", 1)
    best_index_path = INDICES_DIR / f"{best_mkey}_{best_config_name}"
    best_store = FAISSVectorStore.load(best_index_path)
    best_chunk_lookup = {c.id: c for c in chunks_by_config[best_config_name]}

    # WHY embed queries here: generation_evaluator needs OpenAI embeddings.
    # Reuse the same embedder — queries are the same 56 questions.
    logger.info("Embedding %d queries with OpenAI ...", len(qa_pairs))
    embedder = create_embedder(EmbeddingModel.OPENAI)
    query_embeddings = embedder.embed([qa.question for qa in qa_pairs])

    ragas_result, generated_answers, retrieved_contexts = run_generation_evaluation(
        config_id=best_config_id,
        qa_pairs=qa_pairs,
        store=best_store,
        chunk_lookup=best_chunk_lookup,
        query_embeddings=query_embeddings,
    )
    save_generation_results(ragas_result)

    logger.info(
        "RAGAS: faithfulness=%.3f  relevancy=%.3f  recall=%.3f  precision=%.3f",
        ragas_result.faithfulness,
        ragas_result.answer_relevancy,
        ragas_result.context_recall,
        ragas_result.context_precision,
    )

    # -----------------------------------------------------------------------
    # Step 6: JUDGES — best config (Task 22)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("JUDGES — %s", best_config_id)
    logger.info("=" * 60)

    judge_results = run_judge_evaluation(
        qa_pairs=qa_pairs,
        generated_answers=generated_answers,
        retrieved_contexts=retrieved_contexts,
    )
    save_judge_results(judge_results)

    # -----------------------------------------------------------------------
    # Step 7: BRAINTRUST LOGGING (Task 23)
    # -----------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BRAINTRUST LOGGING")
    logger.info("=" * 60)

    # Log top-3 retrieval experiments
    for eval_ in top_3:
        log_retrieval_experiment(eval_)

    # Log reranking comparison
    log_reranking_experiment(comparisons)

    # Log generation + judge results for best config
    log_generation_experiment(
        config_id=best_config_id,
        qa_pairs=qa_pairs,
        generated_answers=generated_answers,
        ragas_result=ragas_result,
        judge_results=judge_results,
    )

    # Auto-classify feedback: thumbs up if correct AND no hallucination
    judge_lookup = {jr.question_id: jr for jr in judge_results}
    for qa in qa_pairs:
        jr = judge_lookup.get(qa.id)
        thumbs_up = jr is not None and jr.correctness_score and not jr.has_hallucination
        log_feedback(
            experiment_name=f"{best_config_id}-generation",
            question_id=qa.id,
            thumbs_up=thumbs_up,
        )

    # -----------------------------------------------------------------------
    # Step 8: Summary
    # -----------------------------------------------------------------------
    correct_count = sum(1 for jr in judge_results if jr.correctness_score)
    hallucination_count = sum(1 for jr in judge_results if jr.has_hallucination)
    thumbs_up_count = sum(
        1 for jr in judge_results
        if jr.correctness_score and not jr.has_hallucination
    )

    print(f"\n{'=' * 60}")
    print("Day 4 Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"\nTop-3 Configs (by Recall@5):")
    for e in top_3:
        print(f"  {e.config_id}: {e.avg_recall_at_5:.3f}")

    print(f"\nReranking Results:")
    for comp in comparisons:
        print(
            f"  {comp.config_id}: R@5 {comp.recall_at_5_before:.3f} → "
            f"{comp.recall_at_5_after:.3f} ({comp.recall_improvement_pct:+.1f}%)"
        )

    print(f"\nRAGAS Scores ({best_config_id}):")
    print(f"  Faithfulness:      {ragas_result.faithfulness:.3f}")
    print(f"  Answer Relevancy:  {ragas_result.answer_relevancy:.3f}")
    print(f"  Context Recall:    {ragas_result.context_recall:.3f}")
    print(f"  Context Precision: {ragas_result.context_precision:.3f}")

    print(f"\nJudge Verdicts ({best_config_id}):")
    print(f"  Correct:       {correct_count}/{len(judge_results)}")
    print(f"  Hallucinated:  {hallucination_count}/{len(judge_results)}")
    print(f"  Thumbs Up:     {thumbs_up_count}/{len(judge_results)}")
    print(f"\nResults saved to: results/metrics/")
