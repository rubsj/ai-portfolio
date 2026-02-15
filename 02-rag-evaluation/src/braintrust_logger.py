"""Braintrust experiment logging — Task 23.

Logs retrieval, generation, judge, and reranking results to Braintrust for
experiment tracking and comparison. Non-fatal — Braintrust failures never
block local pipeline execution.

WHY Braintrust after local JSON: local JSON (results/metrics/) is the source
of truth. Braintrust adds a dashboard for comparison, filtering, and sharing.
If the API is down or unconfigured, the pipeline still completes.

Java/TS parallel: like sending metrics to Datadog/Grafana after writing local
logs. The primary output is local; the dashboard is a convenience layer.
"""

from __future__ import annotations

import logging
import os

import braintrust

from src.config import BRAINTRUST_API_KEY
from src.models import (
    ConfigEvaluation,
    JudgeResult,
    RAGASResult,
    RerankingComparison,
    SyntheticQAPair,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# API Key Check
# ===========================================================================

def _check_api_key() -> bool:
    """Check if BRAINTRUST_API_KEY is available.

    WHY a helper: called at the top of every public function. Avoids
    duplicating the check and log message.
    """
    if not BRAINTRUST_API_KEY and not os.environ.get("BRAINTRUST_API_KEY"):
        logger.warning("BRAINTRUST_API_KEY not set — skipping Braintrust logging")
        return False
    return True


# ===========================================================================
# Retrieval Experiment Logging
# ===========================================================================

def log_retrieval_experiment(config_eval: ConfigEvaluation) -> None:
    """Log one config's retrieval results to Braintrust.

    WHY one experiment per config: enables side-by-side comparison in the
    Braintrust dashboard (e.g., E-openai vs B-openai).

    Logs each individual question result with per-question scores, plus
    config-level metadata.
    """
    if not _check_api_key():
        return

    try:
        experiment = braintrust.init(
            project="p2-rag-evaluation",
            experiment=config_eval.config_id,
        )

        for result in config_eval.individual_results:
            experiment.log(
                input={"question": result.question, "question_type": result.question_type.value},
                output={"retrieved_chunk_ids": result.retrieved_chunk_ids[:5]},
                expected={"gold_chunk_ids": result.gold_chunk_ids},
                scores={
                    "recall_at_5": result.recall_at_5,
                    "precision_at_5": result.precision_at_5,
                    "mrr_at_5": result.mrr_at_5,
                },
                metadata={
                    "config_id": config_eval.config_id,
                    "chunk_config": config_eval.chunk_config,
                    "embedding_model": config_eval.embedding_model,
                    "question_type": result.question_type.value,
                },
                id=result.query_id,
            )

        experiment.flush()
        logger.info("Logged retrieval experiment '%s' to Braintrust", config_eval.config_id)

    except Exception as exc:
        logger.warning("Braintrust retrieval logging failed: %s", exc)


# ===========================================================================
# Generation + Judge Experiment Logging
# ===========================================================================

def log_generation_experiment(
    config_id: str,
    qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str],
    ragas_result: RAGASResult,
    judge_results: list[JudgeResult],
) -> None:
    """Log generation + judge results to Braintrust.

    WHY combined generation+judges: they evaluate the same QA pairs from
    different angles. One experiment with all scores is easier to analyze
    than two separate experiments.

    Experiment name: '{config_id}-generation' (e.g., 'E-openai-generation').
    """
    if not _check_api_key():
        return

    try:
        experiment = braintrust.init(
            project="p2-rag-evaluation",
            experiment=f"{config_id}-generation",
        )

        # Build a lookup from question_id → JudgeResult for fast access
        judge_lookup: dict[str, JudgeResult] = {
            jr.question_id: jr for jr in judge_results
        }

        for i, qa in enumerate(qa_pairs):
            jr = judge_lookup.get(qa.id)

            # WHY feedback auto-classification: thumbs up if correct AND no
            # hallucination. This enables filtering by "good" vs "bad" answers
            # in the Braintrust dashboard.
            thumbs_up = (
                jr is not None
                and jr.correctness_score
                and not jr.has_hallucination
            )

            scores: dict[str, float] = {
                "faithfulness": ragas_result.faithfulness,
                "answer_relevancy": ragas_result.answer_relevancy,
                "context_recall": ragas_result.context_recall,
                "context_precision": ragas_result.context_precision,
                "feedback": 1.0 if thumbs_up else 0.0,
            }

            metadata: dict[str, object] = {
                "config_id": config_id,
                "question_type": qa.question_type.value,
            }

            if jr is not None:
                scores["correctness"] = 1.0 if jr.correctness_score else 0.0
                scores["hallucination"] = 0.0 if jr.has_hallucination else 1.0
                metadata["relevance_grade"] = jr.relevance_grade
                metadata["bloom_level"] = jr.bloom_level.value
                metadata["correctness_reasoning"] = jr.correctness_reasoning
                metadata["hallucination_reasoning"] = jr.hallucination_reasoning

            experiment.log(
                input={"question": qa.question},
                output={"generated_answer": generated_answers[i]},
                expected={"expected_answer": qa.expected_answer},
                scores=scores,
                metadata=metadata,
                id=qa.id,
            )

        experiment.flush()
        logger.info(
            "Logged generation experiment '%s-generation' to Braintrust",
            config_id,
        )

    except Exception as exc:
        logger.warning("Braintrust generation logging failed: %s", exc)


# ===========================================================================
# Reranking Experiment Logging
# ===========================================================================

def log_reranking_experiment(comparisons: list[RerankingComparison]) -> None:
    """Log reranking before/after comparison to Braintrust.

    WHY one experiment for all configs: reranking comparison is about the
    relative impact across configs, not individual question results.

    Experiment name: 'reranking-comparison'.
    """
    if not _check_api_key():
        return

    try:
        experiment = braintrust.init(
            project="p2-rag-evaluation",
            experiment="reranking-comparison",
        )

        for comp in comparisons:
            experiment.log(
                input={"config_id": comp.config_id},
                output={
                    "recall_at_5_after": comp.recall_at_5_after,
                    "precision_at_5_after": comp.precision_at_5_after,
                    "mrr_at_5_after": comp.mrr_at_5_after,
                },
                expected={
                    "recall_at_5_before": comp.recall_at_5_before,
                    "precision_at_5_before": comp.precision_at_5_before,
                    "mrr_at_5_before": comp.mrr_at_5_before,
                },
                scores={
                    "recall_improvement": max(0.0, min(1.0, comp.recall_improvement_pct / 100)),
                    "precision_improvement": max(0.0, min(1.0, comp.precision_improvement_pct / 100)),
                    "mrr_improvement": max(0.0, min(1.0, comp.mrr_improvement_pct / 100)),
                },
                metadata={
                    "config_id": comp.config_id,
                    "recall_improvement_pct": comp.recall_improvement_pct,
                    "precision_improvement_pct": comp.precision_improvement_pct,
                    "mrr_improvement_pct": comp.mrr_improvement_pct,
                },
                id=comp.config_id,
            )

        experiment.flush()
        logger.info("Logged reranking-comparison experiment to Braintrust")

    except Exception as exc:
        logger.warning("Braintrust reranking logging failed: %s", exc)


# ===========================================================================
# Feedback Classification
# ===========================================================================

def log_feedback(
    experiment_name: str,
    question_id: str,
    thumbs_up: bool,
    *,
    comment: str = "",
) -> None:
    """Log thumbs up/down feedback on a QA result to Braintrust.

    WHY separate from log_generation_experiment: enables manual post-run
    feedback (e.g., from Streamlit on Day 5) without re-running the full
    pipeline.

    Uses experiment.log() with a 'feedback' score key. Auto-classification
    during Day 4 run: thumbs up if correctness=True AND hallucination=False.
    """
    if not _check_api_key():
        return

    try:
        experiment = braintrust.init(
            project="p2-rag-evaluation",
            experiment=experiment_name,
            update=True,
        )

        experiment.log(
            scores={"feedback": 1.0 if thumbs_up else 0.0},
            metadata={"feedback_comment": comment} if comment else None,
            id=question_id,
        )

        experiment.flush()
        logger.info(
            "Logged feedback for %s in '%s': %s",
            question_id,
            experiment_name,
            "thumbs_up" if thumbs_up else "thumbs_down",
        )

    except Exception as exc:
        logger.warning("Braintrust feedback logging failed: %s", exc)
