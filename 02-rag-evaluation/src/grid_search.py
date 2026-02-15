"""Grid search orchestrator — evaluates all 16 retrieval configs.

Runs the full grid search: 15 FAISS configs (5 chunk configs × 3 embedding
models) + 1 BM25 baseline. For each config, retrieves top-K chunks for every
QA pair and computes Recall, Precision, MRR at K=1,3,5.

WHY 16 configs: PRD Section 3c specifies a 5×3+1 grid. The 5 chunk configs
(A-E) test different chunking hypotheses, the 3 embedding models test
representation quality, and BM25 is the lexical baseline.

Memory management: one embedding model at a time. Load → embed all questions
→ search all 5 chunk configs → unload → gc.collect(). Same pattern as
index_builder.py. Peak RAM: ~700MB (largest model mpnet ~420MB + query
embeddings ~0.1MB + chunk data ~100MB).

Java/TS parallel: like a Spring Batch job that orchestrates multiple
processing steps. Each step (embed, search, evaluate) is a separate
function, and the orchestrator handles sequencing, resource cleanup,
and result aggregation.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np

from src.bm25_baseline import BM25Retriever
from src.config import (
    ALL_CHUNK_CONFIGS,
    ALL_EMBEDDING_MODELS,
    BM25_CHUNK_CONFIG,
    INDICES_DIR,
    METRICS_DIR,
    OUTPUT_DIR,
    RETRIEVAL_TOP_N,
    model_key,
)
from src.embedder import create_embedder
from src.models import (
    Chunk,
    ConfigEvaluation,
    RetrievalMethod,
    SyntheticQAPair,
)
from src.retrieval_evaluator import evaluate_config
from src.synthetic_qa import load_qa_pairs
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ===========================================================================
# Cross-Config Gold Chunk Mapping
# ===========================================================================

def map_gold_chunks(
    gold_b_ids: list[str],
    b_chunks_lookup: dict[str, Chunk],
    target_chunks: list[Chunk],
    overlap_threshold: float = 0.5,
) -> list[str]:
    """Map gold chunk IDs from Config B namespace to a target config's namespace.

    Uses character position overlap: a target chunk is considered "gold" if
    ≥50% of a gold B chunk's character span overlaps with the target chunk.

    WHY character overlap (not token overlap): character offsets are stored
    on every Chunk (start_char, end_char). Token counts vary by tokenizer,
    but character offsets are exact and model-agnostic.

    WHY 0.5 threshold: a target chunk needs to cover at least half of the
    gold chunk's content to be considered a match. Lower thresholds produce
    too many false positives (tiny overlaps), higher thresholds miss partial
    matches in fine-grained configs (A has 128-token chunks).

    Args:
        gold_b_ids: Gold chunk IDs in Config B namespace (e.g., ["B_0_42"]).
        b_chunks_lookup: Dict mapping Config B chunk ID → Chunk object.
        target_chunks: All chunks in the target config.
        overlap_threshold: Minimum overlap ratio to consider a match.

    Returns:
        List of chunk IDs in the target config's namespace that match the
        gold B chunks. May be empty if no target chunks overlap sufficiently.
    """
    mapped_ids: list[str] = []

    for gold_id in gold_b_ids:
        gold_chunk = b_chunks_lookup.get(gold_id)
        if gold_chunk is None:
            logger.warning("Gold chunk %s not found in B lookup", gold_id)
            continue

        # WHY extract doc_idx: chunks from different documents can't overlap.
        # Chunk ID format: {config}_{doc_idx}_{chunk_idx}. Only compare
        # target chunks from the same document.
        gold_doc_idx = gold_id.split("_")[1]
        gold_span = gold_chunk.end_char - gold_chunk.start_char

        if gold_span <= 0:
            continue

        for target in target_chunks:
            # Only compare chunks from the same document
            target_doc_idx = target.id.split("_")[1]
            if target_doc_idx != gold_doc_idx:
                continue

            # Compute character overlap
            overlap = max(
                0,
                min(gold_chunk.end_char, target.end_char)
                - max(gold_chunk.start_char, target.start_char),
            )
            overlap_ratio = overlap / gold_span

            if overlap_ratio >= overlap_threshold:
                mapped_ids.append(target.id)

    # WHY deduplicate: multiple gold B chunks might map to the same target chunk
    # (e.g., in Config C where chunks are larger). list(set()) removes dupes.
    return list(set(mapped_ids))


# ===========================================================================
# Chunk Loading
# ===========================================================================

def _load_all_chunks() -> dict[str, list[Chunk]]:
    """Load chunk JSON files for all 5 configs.

    Returns dict mapping config name (A-E) to list of Chunk objects.

    WHY load all at once: grid search needs to map gold chunks across configs.
    Each chunk file is ~1-3MB JSON, total ~10MB — fits easily in memory.
    """
    chunks_by_config: dict[str, list[Chunk]] = {}

    for config in ALL_CHUNK_CONFIGS:
        path = OUTPUT_DIR / f"chunks_{config.name}.json"
        data = json.loads(path.read_text())
        chunks = [Chunk.model_validate(d) for d in data]
        chunks_by_config[config.name] = chunks
        logger.info("Loaded %d chunks for config %s", len(chunks), config.name)

    return chunks_by_config


# ===========================================================================
# Sanity Check
# ===========================================================================

def sanity_check(qa_pairs: list[SyntheticQAPair], n: int = 3) -> bool:
    """Quick sanity check — verify gold chunks are retrievable before full grid search.

    Picks n QA pairs (one per strategy if possible), embeds with MiniLM,
    searches Config B index, checks if any gold chunk appears in top-10.

    Returns True if ≥2/3 pass (Recall@10 > 0).

    WHY this gate: if gold chunks aren't retrievable, the full grid search
    will produce all-zero metrics — a waste of 3-5 minutes. Better to catch
    bad QA pairs early.
    """
    from src.models import EmbeddingModel

    # Pick diverse QA pairs — one from each strategy if possible
    strategies = ["per_chunk_chain", "multi_chunk", "overlap_region"]
    sample: list[SyntheticQAPair] = []
    for strategy in strategies:
        for qa in qa_pairs:
            if qa.generation_strategy == strategy:
                sample.append(qa)
                break
    # Fallback: if fewer than n found, fill from remaining
    while len(sample) < n and len(sample) < len(qa_pairs):
        candidate = qa_pairs[len(sample)]
        if candidate not in sample:
            sample.append(candidate)

    logger.info("Sanity check: testing %d QA pairs on Config B + MiniLM", len(sample))

    # Load Config B index + embed queries
    store = FAISSVectorStore.load(INDICES_DIR / "minilm_B")
    embedder = create_embedder(EmbeddingModel.MINILM)
    query_vecs = embedder.embed([qa.question for qa in sample])

    # WHY cleanup immediately: we only need the embedder for this check.
    del embedder
    gc.collect()

    passes = 0
    for i, qa in enumerate(sample):
        results = store.search(query_vecs[i], k=10)
        retrieved_ids = {chunk_id for chunk_id, _ in results}
        gold_set = set(qa.gold_chunk_ids)
        hit = bool(gold_set & retrieved_ids)
        status = "PASS" if hit else "FAIL"
        logger.info(
            "  [%s] Q: %.60s... | gold=%s | hit=%s",
            status, qa.question, qa.gold_chunk_ids[:2], hit,
        )
        if hit:
            passes += 1

    threshold = min(2, len(sample))
    passed = passes >= threshold
    logger.info(
        "Sanity check: %d/%d passed (threshold=%d) — %s",
        passes, len(sample), threshold, "OK" if passed else "FAILED",
    )
    return passed


# ===========================================================================
# Grid Search
# ===========================================================================

def run_grid_search(
    qa_pairs: list[SyntheticQAPair],
) -> list[ConfigEvaluation]:
    """Run the full 16-config grid search evaluation.

    Phase 1: Vector search (15 configs = 5 chunk configs × 3 embedding models).
    Phase 2: BM25 baseline (1 config, Config B chunks).

    Returns list of 16 ConfigEvaluation objects.
    """
    total_start = time.perf_counter()

    # Load all chunk files and build Config B lookup
    chunks_by_config = _load_all_chunks()
    b_chunks_lookup = {chunk.id: chunk for chunk in chunks_by_config["B"]}

    evaluations: list[ConfigEvaluation] = []

    # ── Phase 1: Vector search (15 configs) ──
    for emb_model in ALL_EMBEDDING_MODELS:
        mkey = model_key(emb_model)
        logger.info("=" * 60)
        logger.info("Embedding questions with %s ...", emb_model.value)

        start = time.perf_counter()
        embedder = create_embedder(emb_model)

        # Embed all questions once per model — amortize the embedding cost
        query_embeddings = embedder.embed([qa.question for qa in qa_pairs])
        logger.info(
            "Embedded %d questions in %.1fs",
            len(qa_pairs), time.perf_counter() - start,
        )

        for config in ALL_CHUNK_CONFIGS:
            config_name = config.name
            config_id = f"{config_name}-{mkey}"

            index_path = INDICES_DIR / f"{mkey}_{config_name}"
            if not index_path.with_suffix(".faiss").exists():
                logger.warning("Index not found: %s, skipping", index_path)
                continue

            store = FAISSVectorStore.load(index_path)

            # Search for every QA pair
            all_results: list[list[tuple[str, float]]] = []
            for i in range(len(qa_pairs)):
                results = store.search(query_embeddings[i], k=RETRIEVAL_TOP_N)
                all_results.append(results)

            # Map gold chunk IDs to this config's namespace
            if config_name == "B":
                # Config B — gold IDs match directly, no mapping needed
                mapped_golds = [qa.gold_chunk_ids for qa in qa_pairs]
            else:
                mapped_golds = [
                    map_gold_chunks(
                        qa.gold_chunk_ids,
                        b_chunks_lookup,
                        chunks_by_config[config_name],
                    )
                    for qa in qa_pairs
                ]

            config_eval = evaluate_config(
                qa_pairs=qa_pairs,
                retrieval_results=all_results,
                gold_ids_per_question=mapped_golds,
                config_id=config_id,
                chunk_config=config_name,
                embedding_model=mkey,
                retrieval_method=RetrievalMethod.VECTOR,
                num_chunks=store.size,
            )
            evaluations.append(config_eval)
            logger.info(
                "  %s: R@5=%.3f  P@5=%.3f  MRR@5=%.3f",
                config_id,
                config_eval.avg_recall_at_5,
                config_eval.avg_precision_at_5,
                config_eval.avg_mrr_at_5,
            )

        # WHY explicit cleanup: free the embedding model's RAM before loading
        # the next one. mpnet is ~420MB, MiniLM ~80MB.
        del embedder
        gc.collect()
        logger.info("Unloaded %s", emb_model.value)

    # ── Phase 2: BM25 baseline (runs exactly once, outside embedding loop) ──
    logger.info("=" * 60)
    logger.info("Running BM25 baseline evaluation ...")

    bm25_path = INDICES_DIR / f"bm25_{BM25_CHUNK_CONFIG.name}"
    bm25 = BM25Retriever.load(bm25_path)

    bm25_results: list[list[tuple[str, float]]] = [
        bm25.search(qa.question, k=RETRIEVAL_TOP_N) for qa in qa_pairs
    ]
    # Config B — gold IDs match directly
    bm25_golds = [qa.gold_chunk_ids for qa in qa_pairs]

    bm25_eval = evaluate_config(
        qa_pairs=qa_pairs,
        retrieval_results=bm25_results,
        gold_ids_per_question=bm25_golds,
        config_id="bm25",
        chunk_config=BM25_CHUNK_CONFIG.name,
        embedding_model="bm25",
        retrieval_method=RetrievalMethod.BM25,
        num_chunks=bm25.size,
    )
    evaluations.append(bm25_eval)
    logger.info(
        "  bm25: R@5=%.3f  P@5=%.3f  MRR@5=%.3f",
        bm25_eval.avg_recall_at_5,
        bm25_eval.avg_precision_at_5,
        bm25_eval.avg_mrr_at_5,
    )

    total_elapsed = time.perf_counter() - total_start
    logger.info("Grid search complete: %d configs in %.1fs", len(evaluations), total_elapsed)

    return evaluations


# ===========================================================================
# Result Persistence
# ===========================================================================

def save_grid_results(evaluations: list[ConfigEvaluation]) -> Path:
    """Save grid search results to JSON.

    WHY separate from run_grid_search: separation of concerns. The orchestrator
    produces results, the caller decides where/how to save. But we provide a
    convenience function for the standard path.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "grid_search_results.json"

    data = [e.model_dump(mode="json") for e in evaluations]
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %d config evaluations to %s", len(evaluations), path)
    return path


# ===========================================================================
# Summary Printer
# ===========================================================================

def print_summary(evaluations: list[ConfigEvaluation]) -> None:
    """Print the 3-section summary: Top-3, BM25 Baseline, Config E Analysis."""
    # Sort by Recall@5 descending
    ranked = sorted(evaluations, key=lambda e: e.avg_recall_at_5, reverse=True)

    print("\n" + "=" * 60)
    print("=== Top-3 Configs by Recall@5 ===")
    print("=" * 60)
    for i, e in enumerate(ranked[:3], 1):
        print(
            f"#{i}  {e.config_id:12s}  Recall@5={e.avg_recall_at_5:.3f}  "
            f"P@5={e.avg_precision_at_5:.3f}  MRR@5={e.avg_mrr_at_5:.3f}"
        )

    # BM25 baseline section
    bm25_eval = next((e for e in evaluations if e.config_id == "bm25"), None)
    if bm25_eval:
        bm25_rank = next(
            i for i, e in enumerate(ranked, 1) if e.config_id == "bm25"
        )
        best_vector = next(
            e for e in ranked if e.retrieval_method == RetrievalMethod.VECTOR
        )
        delta_pct = (
            (best_vector.avg_recall_at_5 - bm25_eval.avg_recall_at_5)
            / max(bm25_eval.avg_recall_at_5, 1e-6)
            * 100
        )

        print(f"\n{'=' * 60}")
        print("=== BM25 Baseline ===")
        print(f"{'=' * 60}")
        print(
            f"bm25:  Recall@5={bm25_eval.avg_recall_at_5:.3f} "
            f"(rank #{bm25_rank}/{len(evaluations)})"
        )
        print(
            f"Best vector config ({best_vector.config_id}) beats BM25 by "
            f"{delta_pct:+.1f}%"
        )

    # Config E analysis — ALWAYS prints regardless of ranking
    print(f"\n{'=' * 60}")
    print("=== Config E (Semantic) Analysis ===")
    print(f"{'=' * 60}")

    e_configs = [e for e in evaluations if e.chunk_config == "E"]
    for e in e_configs:
        e_rank = next(
            i for i, r in enumerate(ranked, 1) if r.config_id == e.config_id
        )
        print(
            f"{e.config_id:12s}  Recall@5={e.avg_recall_at_5:.3f} "
            f"(rank #{e_rank}/{len(evaluations)})"
        )

    # Compare best Config E vs best Config B (same model)
    if e_configs:
        best_e = max(e_configs, key=lambda e: e.avg_recall_at_5)
        # Find the Config B counterpart with the same embedding model
        b_match = next(
            (e for e in evaluations
             if e.chunk_config == "B" and e.embedding_model == best_e.embedding_model),
            None,
        )
        if b_match:
            delta = (
                (best_e.avg_recall_at_5 - b_match.avg_recall_at_5)
                / max(b_match.avg_recall_at_5, 1e-6)
                * 100
            )
            verb = "beats" if delta > 0 else "loses to"
            print(
                f"\nBest fixed-size comparator (B-{b_match.embedding_model}): "
                f"Recall@5={b_match.avg_recall_at_5:.3f}"
            )
            print(f"Delta: Config E {verb} Config B by {abs(delta):.1f}%")

    print()


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":
    # WHY force CPU: MPS (Apple Silicon GPU) causes SIGSEGV when loading
    # a second SentenceTransformer model after gc.collect() frees the first.
    # torch.set_default_device("cpu") must run before faiss C-library loads,
    # but `python -m src.grid_search` runs module-level imports (faiss) first.
    #
    # WORKAROUND: run via inline script that sets torch device before imports:
    #   uv run python -c "import torch; torch.set_default_device('cpu'); \
    #     from src.grid_search import *; from src.synthetic_qa import load_qa_pairs; \
    #     qa_pairs = load_qa_pairs(); evals = run_grid_search(qa_pairs); \
    #     save_grid_results(evals); print_summary(evals)"
    import torch
    torch.set_default_device("cpu")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Step 1: Load QA pairs
    logger.info("Loading QA pairs ...")
    qa_pairs = load_qa_pairs()
    logger.info("Loaded %d QA pairs", len(qa_pairs))

    # Step 2: Sanity check
    logger.info("Running sanity check ...")
    if not sanity_check(qa_pairs):
        print("\nSANITY CHECK FAILED — aborting grid search.")
        print("Debug QA quality or gold chunk assignment before proceeding.")
        raise SystemExit(1)

    # Step 3: Run full grid search
    logger.info("Starting full grid search ...")
    evaluations = run_grid_search(qa_pairs)

    # Step 4: Save results
    result_path = save_grid_results(evaluations)
    logger.info("Results saved to %s", result_path)

    # Step 5: Print summary
    print_summary(evaluations)

    # Step 6: Print final verification
    print(f"Total configs evaluated: {len(evaluations)}")
    print(f"Results at: {result_path}")


def compile_grid_search_report(
    pdf_name: str = "Kaggle Annual Reports 2024",
    runtime_seconds: float | None = None,
) -> GridSearchReport:
    """Compile all evaluation results into a single GridSearchReport JSON.

    WHY: Single source of truth for Streamlit and README. Aggregates 4 separate
    result files into one validated Pydantic artifact.

    Args:
        pdf_name: Name of the evaluated document
        runtime_seconds: Total pipeline runtime (optional)

    Returns:
        GridSearchReport model instance
    """
    from datetime import datetime
    from pathlib import Path

    from src.config import METRICS_DIR, REPORTS_DIR
    from src.models import (
        ConfigEvaluation,
        GridSearchReport,
        JudgeResult,
        QADatasetReport,
        RAGASResult,
        RerankingComparison,
    )

    logger.info("Compiling GridSearchReport from result files...")

    # Load all result files
    grid_path = METRICS_DIR / "grid_search_results.json"
    rerank_path = METRICS_DIR / "reranking_results.json"
    ragas_path = METRICS_DIR / "ragas_results.json"
    judge_path = METRICS_DIR / "judge_results.json"
    qa_report_path = REPORTS_DIR / "qa_dataset_report.json"

    # Parse config evaluations
    grid_data = json.loads(grid_path.read_text())
    config_evaluations = [ConfigEvaluation.model_validate(item) for item in grid_data]

    # Parse reranking comparisons
    rerank_data = json.loads(rerank_path.read_text())
    reranking_comparisons = [RerankingComparison.model_validate(item) for item in rerank_data]

    # Parse RAGAS results
    ragas_data = json.loads(ragas_path.read_text())
    ragas_results = [RAGASResult.model_validate(ragas_data)]

    # Parse judge results
    judge_data = json.loads(judge_path.read_text())
    judge_results = [JudgeResult.model_validate(item) for item in judge_data]

    # Parse QA dataset report
    qa_report_data = json.loads(qa_report_path.read_text())
    qa_dataset_report = QADatasetReport.model_validate(qa_report_data)

    # Identify best configs
    # Best retrieval: max Recall@5 across all vector configs (exclude BM25)
    vector_configs = [ev for ev in config_evaluations if ev.retrieval_method.value == "vector"]
    best_retrieval = max(vector_configs, key=lambda ev: ev.avg_recall_at_5)

    # Best generation: E-openai (only config with RAGAS evaluation)
    best_generation = "E-openai"

    # Identify BM25 baseline
    bm25_baseline = next((ev for ev in config_evaluations if ev.retrieval_method.value == "bm25"), None)

    # Estimate API cost
    # Embeddings: 56 questions × 16 configs × 50 tokens/query × $0.02/1M = $0.009
    # Generation: 56 answers × 200 tokens/answer × $0.60/1M = $0.007
    # Reranking: 3 configs × 56 questions × $1/1K = $0.17
    # Total: ~$0.19
    estimated_cost = 0.009 + 0.007 + 0.17

    # Build report
    report = GridSearchReport(
        pdf_name=pdf_name,
        total_configs=len(config_evaluations),
        config_evaluations=config_evaluations,
        bm25_baseline=bm25_baseline,
        reranking_comparisons=reranking_comparisons,
        ragas_results=ragas_results,
        judge_results=judge_results,
        best_retrieval_config=best_retrieval.config_id,
        best_generation_config=best_generation,
        qa_dataset_report=qa_dataset_report,
        timestamp=datetime.now(),
        total_runtime_seconds=runtime_seconds or 0.0,
        estimated_api_cost_usd=estimated_cost,
    )

    # Save to JSON
    output_path = REPORTS_DIR / "grid_search_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2))

    logger.info(f"✅ GridSearchReport saved to {output_path}")
    logger.info(f"   Best retrieval config: {report.best_retrieval_config}")
    logger.info(f"   Best generation config: {report.best_generation_config}")
    logger.info(f"   Estimated API cost: ${report.estimated_api_cost_usd:.2f}")

    return report
