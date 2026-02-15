"""
P2 Verification Test Suite — "Trust But Verify" for AI-Generated Code
======================================================================

PURPOSE: Validate that Claude Code's P2 RAG evaluation pipeline produces
         correct results across all stages. This is YOUR audit layer.

HOW TO USE:
    1. Copy this file to: 02-rag-evaluation/tests/test_verification.py
    2. Run: cd 02-rag-evaluation && uv run pytest tests/test_verification.py -v
    3. Layer 3 and 4 tests require actual pipeline outputs to exist
       (run after Days 3-4 are complete)
    4. Tests marked @pytest.mark.requires_outputs need actual results files

LAYERS:
    Layer 1: Manual spot-checks with hand-computed expected values
    Layer 2: Invariant checks (mathematical properties that MUST hold)
    Layer 3: End-to-end trace of individual questions through the pipeline
    Layer 4: Statistical smell tests on aggregate results
    Layer 5: Independent re-implementation of core metric functions

WHY THIS FILE EXISTS:
    Claude Code can produce code that runs without errors but computes
    subtly wrong results — e.g., Recall@5 calculated as Precision@5,
    or gold chunk IDs leaking into retrieval. This test suite catches
    those failures by testing properties and invariants, not just "does
    it run."

JAVA/TS PARALLEL:
    Think of this as a combination of:
    - JUnit @ParameterizedTest with hand-computed expected values
    - Property-based testing (like jqwik or QuickCheck)
    - Integration test assertions on production data
    - Double-entry bookkeeping (two implementations, same answer)
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest


# ============================================================================
# CONFIGURATION — Update these paths to match your actual project layout
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
REPORTS_DIR = RESULTS_DIR / "reports"
CHARTS_DIR = RESULTS_DIR / "charts"
DATA_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# WHY: pytest.mark.requires_outputs lets us skip tests that need pipeline
# results to exist. Run Layer 1 & 2 & 5 immediately. Run Layer 3 & 4 after
# the grid search completes.
requires_outputs = pytest.mark.skipif(
    not RESULTS_DIR.exists(),
    reason="Pipeline results not yet generated — run grid search first",
)


# ============================================================================
# LAYER 1: MANUAL SPOT-CHECKS WITH KNOWN INPUTS
# ============================================================================
# First Principle: If you know the correct answer for a specific input,
# and the code gives a different answer, the code is WRONG. No ambiguity.
#
# Java parallel: JUnit assertEquals with hand-computed expected values.
# ============================================================================


class TestLayer1_RetrievalMetrics:
    """
    Tests retrieval metric functions with inputs where we KNOW the correct
    answer because we computed it by hand.

    WHY this is Layer 1 (highest priority): If metric computation is wrong,
    every downstream result — heatmaps, best config selection, reranking
    comparison — is meaningless. This is the foundation.
    """

    def _import_retrieval_evaluator(self):
        """
        Dynamically import the retrieval evaluator module.

        WHY dynamic import: This test file should be usable even if some
        modules haven't been built yet. Importing at test time (not module
        load time) means we get a clear skip/error instead of an ImportError
        that blocks ALL tests.

        PYTHON PATTERN — Java parallel:
            In Java, you'd use @Disabled or assume(). In Python, we use
            pytest.importorskip() or try/except with pytest.skip().
        """
        try:
            from src.retrieval_evaluator import (
                compute_mrr_at_k,
                compute_precision_at_k,
                compute_recall_at_k,
            )
            return compute_recall_at_k, compute_precision_at_k, compute_mrr_at_k
        except ImportError:
            pytest.skip(
                "src.retrieval_evaluator not found — build it first (Day 3, Task 15)"
            )

    # ---- Scenario 1: Gold chunk is at position 2 of 5 ----
    # Retrieved: [chunk_7, chunk_3, chunk_1, chunk_5, chunk_9]
    # Gold:      [chunk_3]
    # 
    # Hand-computed:
    #   Recall@1 = 0/1 = 0.0     (chunk_3 NOT in top-1)
    #   Recall@3 = 1/1 = 1.0     (chunk_3 IS in top-3)
    #   Recall@5 = 1/1 = 1.0     (chunk_3 IS in top-5)
    #   Precision@1 = 0/1 = 0.0  (0 relevant in top-1)
    #   Precision@3 = 1/3 ≈ 0.333 (1 relevant in top-3)
    #   Precision@5 = 1/5 = 0.2  (1 relevant in top-5)
    #   MRR@5 = 1/2 = 0.5        (first relevant at position 2)

    def test_recall_at_k_gold_at_position_2(self):
        compute_recall, _, _ = self._import_retrieval_evaluator()

        gold = ["chunk_3"]
        retrieved = ["chunk_7", "chunk_3", "chunk_1", "chunk_5", "chunk_9"]

        assert compute_recall(gold, retrieved, k=1) == pytest.approx(0.0)
        assert compute_recall(gold, retrieved, k=3) == pytest.approx(1.0)
        assert compute_recall(gold, retrieved, k=5) == pytest.approx(1.0)

    def test_precision_at_k_gold_at_position_2(self):
        _, compute_precision, _ = self._import_retrieval_evaluator()

        gold = ["chunk_3"]
        retrieved = ["chunk_7", "chunk_3", "chunk_1", "chunk_5", "chunk_9"]

        assert compute_precision(gold, retrieved, k=1) == pytest.approx(0.0)
        assert compute_precision(gold, retrieved, k=3) == pytest.approx(1 / 3)
        assert compute_precision(gold, retrieved, k=5) == pytest.approx(1 / 5)

    def test_mrr_at_k_gold_at_position_2(self):
        _, _, compute_mrr = self._import_retrieval_evaluator()

        gold = ["chunk_3"]
        retrieved = ["chunk_7", "chunk_3", "chunk_1", "chunk_5", "chunk_9"]

        # WHY: MRR = 1/rank_of_first_relevant. chunk_3 is at index 1 → rank 2.
        assert compute_mrr(gold, retrieved, k=1) == pytest.approx(0.0)
        assert compute_mrr(gold, retrieved, k=3) == pytest.approx(0.5)
        assert compute_mrr(gold, retrieved, k=5) == pytest.approx(0.5)

    # ---- Scenario 2: Multiple gold chunks ----
    # Retrieved: [chunk_1, chunk_5, chunk_2, chunk_8, chunk_3]
    # Gold:      [chunk_2, chunk_5]
    #
    # Hand-computed:
    #   Recall@1 = 0/2 = 0.0     (neither gold in top-1)
    #   Recall@3 = 2/2 = 1.0     (both chunk_5 AND chunk_2 in top-3)
    #   Recall@5 = 2/2 = 1.0     (both in top-5)
    #   Precision@1 = 0/1 = 0.0
    #   Precision@3 = 2/3 ≈ 0.667
    #   Precision@5 = 2/5 = 0.4
    #   MRR@5 = 1/2 = 0.5        (first relevant = chunk_5 at position 2)

    def test_recall_multiple_gold_chunks(self):
        compute_recall, _, _ = self._import_retrieval_evaluator()

        gold = ["chunk_2", "chunk_5"]
        retrieved = ["chunk_1", "chunk_5", "chunk_2", "chunk_8", "chunk_3"]

        assert compute_recall(gold, retrieved, k=1) == pytest.approx(0.0)
        assert compute_recall(gold, retrieved, k=3) == pytest.approx(1.0)
        assert compute_recall(gold, retrieved, k=5) == pytest.approx(1.0)

    def test_precision_multiple_gold_chunks(self):
        _, compute_precision, _ = self._import_retrieval_evaluator()

        gold = ["chunk_2", "chunk_5"]
        retrieved = ["chunk_1", "chunk_5", "chunk_2", "chunk_8", "chunk_3"]

        assert compute_precision(gold, retrieved, k=1) == pytest.approx(0.0)
        assert compute_precision(gold, retrieved, k=3) == pytest.approx(2 / 3)
        assert compute_precision(gold, retrieved, k=5) == pytest.approx(2 / 5)

    def test_mrr_multiple_gold_chunks(self):
        _, _, compute_mrr = self._import_retrieval_evaluator()

        gold = ["chunk_2", "chunk_5"]
        retrieved = ["chunk_1", "chunk_5", "chunk_2", "chunk_8", "chunk_3"]

        # WHY: MRR looks at the FIRST relevant result only.
        # chunk_5 at position 2 → 1/2 = 0.5
        assert compute_mrr(gold, retrieved, k=5) == pytest.approx(0.5)

    # ---- Scenario 3: Perfect retrieval (gold at position 1) ----
    # Retrieved: [chunk_3, chunk_1, chunk_5, chunk_8, chunk_9]
    # Gold:      [chunk_3]

    def test_perfect_retrieval_gold_at_position_1(self):
        compute_recall, compute_precision, compute_mrr = (
            self._import_retrieval_evaluator()
        )

        gold = ["chunk_3"]
        retrieved = ["chunk_3", "chunk_1", "chunk_5", "chunk_8", "chunk_9"]

        assert compute_recall(gold, retrieved, k=1) == pytest.approx(1.0)
        assert compute_precision(gold, retrieved, k=1) == pytest.approx(1.0)
        assert compute_mrr(gold, retrieved, k=1) == pytest.approx(1.0)
        assert compute_mrr(gold, retrieved, k=5) == pytest.approx(1.0)

    # ---- Scenario 4: Total miss (gold not in retrieved at all) ----
    # Retrieved: [chunk_1, chunk_5, chunk_8, chunk_9, chunk_10]
    # Gold:      [chunk_3]

    def test_total_miss_gold_not_in_retrieved(self):
        compute_recall, compute_precision, compute_mrr = (
            self._import_retrieval_evaluator()
        )

        gold = ["chunk_3"]
        retrieved = ["chunk_1", "chunk_5", "chunk_8", "chunk_9", "chunk_10"]

        assert compute_recall(gold, retrieved, k=5) == pytest.approx(0.0)
        assert compute_precision(gold, retrieved, k=5) == pytest.approx(0.0)
        assert compute_mrr(gold, retrieved, k=5) == pytest.approx(0.0)

    # ---- Scenario 5: Edge case — empty retrieved list ----

    def test_empty_retrieved_list(self):
        compute_recall, compute_precision, compute_mrr = (
            self._import_retrieval_evaluator()
        )

        gold = ["chunk_3"]
        retrieved = []

        assert compute_recall(gold, retrieved, k=5) == pytest.approx(0.0)
        assert compute_precision(gold, retrieved, k=5) == pytest.approx(0.0)
        assert compute_mrr(gold, retrieved, k=5) == pytest.approx(0.0)

    # ---- Scenario 6: Retrieved list shorter than K ----
    # WHY: If we only retrieve 3 results but K=5, the function should
    # NOT crash — it should evaluate against what's available.

    def test_retrieved_shorter_than_k(self):
        compute_recall, compute_precision, compute_mrr = (
            self._import_retrieval_evaluator()
        )

        gold = ["chunk_3"]
        retrieved = ["chunk_1", "chunk_3"]  # Only 2 results, but K=5

        assert compute_recall(gold, retrieved, k=5) == pytest.approx(1.0)
        # WHY: Precision@5 with only 2 results — denominator should be
        # min(k, len(retrieved)) = 2, OR k=5 depending on implementation.
        # Both conventions exist. The test checks it doesn't crash and
        # returns a value between 0 and 1.
        precision = compute_precision(gold, retrieved, k=5)
        assert 0.0 <= precision <= 1.0


class TestLayer1_EmbeddingsSanity:
    """
    Verifies that the embedding pipeline produces vectors with correct
    properties: right dimensions, normalized, semantically meaningful.

    WHY: If embeddings are wrong (unnormalized, wrong model, truncated),
    FAISS IndexFlatIP returns garbage rankings. This is a silent failure —
    the pipeline runs fine, metrics look plausible, but results are random.
    """

    def _import_embedder(self):
        try:
            from src.embedder import create_embedder
            return create_embedder
        except ImportError:
            pytest.skip("src.embedder not found — build it first (Day 2, Task 7)")

    def test_minilm_produces_384_dimensions(self):
        """all-MiniLM-L6-v2 must produce 384-dim vectors."""
        create_embedder = self._import_embedder()

        try:
            from src.models import EmbeddingModel
            embedder = create_embedder(EmbeddingModel("all-MiniLM-L6-v2"))
        except Exception:
            # WHY: Enum value might differ in actual implementation.
            # Fall back to trying string-based creation.
            pytest.skip("Could not create MiniLM embedder — check EmbeddingModel enum values")

        vectors = embedder.embed(["Hello world test sentence"])
        assert vectors.shape[1] == 384, (
            f"MiniLM should produce 384-dim vectors, got {vectors.shape[1]}. "
            "Check: is the correct model loaded? Did sentence-transformers "
            "silently fall back to a different model?"
        )

    def test_embeddings_are_normalized(self):
        """
        All embeddings MUST be L2-normalized for FAISS IndexFlatIP.

        WHY: IndexFlatIP computes inner product (dot product). For normalized
        vectors, dot product = cosine similarity. If vectors aren't normalized,
        longer vectors get higher scores regardless of semantic meaning.

        This is the #1 silent bug in RAG pipelines.
        """
        create_embedder = self._import_embedder()

        try:
            from src.models import EmbeddingModel
            embedder = create_embedder(EmbeddingModel("all-MiniLM-L6-v2"))
        except Exception:
            pytest.skip("Could not create embedder")

        vectors = embedder.embed([
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a programming language",
        ])

        # WHY: L2 norm of each row should be ≈ 1.0 (within floating point tolerance)
        for i in range(vectors.shape[0]):
            norm = np.linalg.norm(vectors[i])
            assert abs(norm - 1.0) < 1e-5, (
                f"Vector {i} has L2 norm {norm:.6f}, expected ≈1.0. "
                "Embeddings are NOT normalized. Add faiss.normalize_L2() "
                "or normalize manually before indexing."
            )

    def test_semantic_similarity_makes_sense(self):
        """
        Similar sentences should have higher cosine similarity than
        unrelated sentences. If this fails, the embedding model is
        either wrong or broken.
        """
        create_embedder = self._import_embedder()

        try:
            from src.models import EmbeddingModel
            embedder = create_embedder(EmbeddingModel("all-MiniLM-L6-v2"))
        except Exception:
            pytest.skip("Could not create embedder")

        vectors = embedder.embed([
            "How to fix a leaking kitchen faucet",       # A: plumbing topic
            "Repairing a dripping tap in the kitchen",    # B: same topic, different words
            "The history of the Roman Empire in Europe",  # C: completely unrelated
        ])

        # WHY: For normalized vectors, dot product = cosine similarity
        sim_ab = float(np.dot(vectors[0], vectors[1]))
        sim_ac = float(np.dot(vectors[0], vectors[2]))

        assert sim_ab > sim_ac, (
            f"Semantic similarity broken: similar sentences ({sim_ab:.4f}) "
            f"scored LOWER than unrelated sentences ({sim_ac:.4f}). "
            "The embedding model is not encoding meaning correctly."
        )
        # WHY: We also check the gap is meaningful, not just barely above
        assert sim_ab - sim_ac > 0.1, (
            f"Semantic gap too small: {sim_ab:.4f} vs {sim_ac:.4f}. "
            "Embeddings may be degenerate (all vectors similar)."
        )


class TestLayer1_ChunkingSanity:
    """
    Verifies chunking produces correct results on a controlled input.
    """

    def _import_chunker(self):
        try:
            from src.chunker import chunk_text
            return chunk_text
        except ImportError:
            try:
                # WHY: Function name might differ. Try alternative signatures.
                from src.chunker import create_chunks, Chunker
                return create_chunks
            except ImportError:
                pytest.skip("src.chunker not found — build it first (Day 1, Task 5)")

    def test_larger_chunks_produce_fewer_count(self):
        """
        Config A (128 tokens) MUST produce more chunks than Config C (512 tokens)
        for the same document. If not, chunk_size parameter is being ignored.

        WHY this test: Verifies the chunk_size parameter is actually affecting output.
        This is a common silent bug where the config is parsed but not used.
        """
        try:
            from src.config import CONFIG_A, CONFIG_C
            from src.chunker import chunk_document
            from src.parser import ParseResult
        except ImportError as e:
            pytest.skip(f"Could not import chunker components: {e}")

        # WHY: We need a document long enough to produce multiple chunks
        # across configs. ~5000 words should produce:
        # Config A (128 tokens): ~40+ chunks
        # Config C (512 tokens): ~10 chunks
        fake_text = "This is a test sentence about home repair and DIY maintenance. " * 1000

        # Create a ParseResult object (what chunk_document expects)
        parse_result = ParseResult(full_text=fake_text, page_map=[], headers=[])

        try:
            chunks_a = chunk_document(parse_result, CONFIG_A)
            chunks_c = chunk_document(parse_result, CONFIG_C)
        except Exception as e:
            pytest.skip(
                f"Chunker failed: {e}. Check chunk_document API."
            )

        assert len(chunks_a) > len(chunks_c), (
            f"Config A (128 tokens) produced {len(chunks_a)} chunks, "
            f"Config C (512 tokens) produced {len(chunks_c)} chunks. "
            "Smaller chunk_size (Config A: 128) should produce MORE chunks than larger chunk_size (Config C: 512). "
            "Is chunk_size actually being used by the splitter?"
        )


# ============================================================================
# LAYER 2: INVARIANT CHECKS
# ============================================================================
# First Principle: Certain mathematical properties MUST hold regardless
# of the input data. If they don't, the implementation has a bug.
#
# Java parallel: Property-based testing with jqwik, or design-by-contract
# with @Invariant annotations.
# ============================================================================


class TestLayer2_MetricInvariants:
    """
    Tests mathematical properties that must ALWAYS hold for retrieval metrics,
    regardless of the actual data.

    These are not testing specific values — they're testing that the metric
    functions obey the mathematical definitions.
    """

    def _import_retrieval_evaluator(self):
        try:
            from src.retrieval_evaluator import (
                compute_mrr_at_k,
                compute_precision_at_k,
                compute_recall_at_k,
            )
            return compute_recall_at_k, compute_precision_at_k, compute_mrr_at_k
        except ImportError:
            pytest.skip("src.retrieval_evaluator not found")

    # WHY parametrize: Tests the invariant across many different input
    # configurations, not just one cherry-picked example. If ANY combination
    # violates the invariant, there's a bug.
    #
    # PYTHON PATTERN — @pytest.mark.parametrize:
    #   Java equivalent: @ParameterizedTest + @MethodSource
    #   Each tuple becomes a separate test case in the report.
    @pytest.mark.parametrize(
        "gold,retrieved",
        [
            (["c1"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c3"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c5"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c1", "c2"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c1", "c3", "c5"], ["c2", "c1", "c5", "c3", "c4"]),
            (["c99"], ["c1", "c2", "c3", "c4", "c5"]),  # Total miss
        ],
    )
    def test_recall_is_monotonically_nondecreasing_with_k(self, gold, retrieved):
        """
        Recall@1 ≤ Recall@3 ≤ Recall@5

        WHY this MUST hold: As K increases, you're checking more retrieved
        results. You can only FIND more relevant docs, never fewer.
        Recall can stay flat (already found all golds) but NEVER decrease.

        If this fails: the function is likely dividing by K instead of
        by len(gold_ids), or is re-computing the set incorrectly.
        """
        compute_recall, _, _ = self._import_retrieval_evaluator()

        r1 = compute_recall(gold, retrieved, k=1)
        r3 = compute_recall(gold, retrieved, k=3)
        r5 = compute_recall(gold, retrieved, k=5)

        assert r1 <= r3 + 1e-9, f"Recall@1 ({r1}) > Recall@3 ({r3}) — IMPOSSIBLE"
        assert r3 <= r5 + 1e-9, f"Recall@3 ({r3}) > Recall@5 ({r5}) — IMPOSSIBLE"

    @pytest.mark.parametrize(
        "gold,retrieved",
        [
            (["c1"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c3"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c1", "c2"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c99"], ["c1", "c2", "c3", "c4", "c5"]),
        ],
    )
    def test_all_metrics_bounded_between_0_and_1(self, gold, retrieved):
        """
        All retrieval metrics must be in [0.0, 1.0].

        WHY: These are ratios. A value > 1.0 means the numerator exceeds
        the denominator, which is a counting bug. A value < 0.0 means
        a sign error.
        """
        compute_recall, compute_precision, compute_mrr = (
            self._import_retrieval_evaluator()
        )

        for k in [1, 3, 5]:
            recall = compute_recall(gold, retrieved, k=k)
            precision = compute_precision(gold, retrieved, k=k)
            mrr = compute_mrr(gold, retrieved, k=k)

            assert 0.0 <= recall <= 1.0, f"Recall@{k} = {recall} — out of bounds"
            assert 0.0 <= precision <= 1.0, f"Precision@{k} = {precision} — out of bounds"
            assert 0.0 <= mrr <= 1.0, f"MRR@{k} = {mrr} — out of bounds"

    @pytest.mark.parametrize(
        "gold,retrieved",
        [
            (["c1"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c1", "c2"], ["c1", "c2", "c3", "c4", "c5"]),
            (["c99"], ["c1", "c2", "c3", "c4", "c5"]),
        ],
    )
    def test_mrr_is_monotonically_nondecreasing_with_k(self, gold, retrieved):
        """
        MRR@1 ≤ MRR@3 ≤ MRR@5

        WHY: MRR = 1/rank_of_first_relevant. With larger K, you can only
        find the first relevant doc at the same rank or discover it in
        a previously unchecked position. MRR can never decrease.
        """
        _, _, compute_mrr = self._import_retrieval_evaluator()

        mrr1 = compute_mrr(gold, retrieved, k=1)
        mrr3 = compute_mrr(gold, retrieved, k=3)
        mrr5 = compute_mrr(gold, retrieved, k=5)

        assert mrr1 <= mrr3 + 1e-9, f"MRR@1 ({mrr1}) > MRR@3 ({mrr3})"
        assert mrr3 <= mrr5 + 1e-9, f"MRR@3 ({mrr3}) > MRR@5 ({mrr5})"


class TestLayer2_QADataInvariants:
    """
    Tests invariants on the synthetic QA dataset that must hold
    regardless of what questions were generated.
    """

    def _load_qa_pairs(self) -> list[dict]:
        """Load synthetic QA pairs from output directory."""
        qa_files = list(DATA_OUTPUT_DIR.glob("*qa*"))
        if not qa_files:
            qa_files = list(DATA_OUTPUT_DIR.glob("*question*"))
        if not qa_files:
            pytest.skip("No QA dataset files found in data/output/")

        # WHY: Try multiple possible filenames Claude Code might have used
        for f in qa_files:
            if f.suffix == ".json":
                with open(f) as fh:
                    data = json.load(fh)
                    if isinstance(data, list) and len(data) > 0:
                        return data
        pytest.skip("Could not load QA pairs from any JSON file")

    def _load_all_chunk_ids(self) -> set[str]:
        """Load all chunk IDs from all config outputs."""
        all_ids = set()
        chunk_files = list(DATA_OUTPUT_DIR.glob("*chunk*"))
        for f in chunk_files:
            if f.suffix == ".json":
                with open(f) as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        for chunk in data:
                            if isinstance(chunk, dict) and "id" in chunk:
                                all_ids.add(chunk["id"])
        return all_ids

    @requires_outputs
    def test_every_question_has_gold_chunk_ids(self):
        """
        Every synthetic QA pair MUST have at least one gold_chunk_id.

        WHY: A question without gold chunk IDs cannot be evaluated for
        retrieval quality. Recall/Precision/MRR are all undefined.
        This is a data integrity issue that corrupts all downstream metrics.
        """
        qa_pairs = self._load_qa_pairs()

        for i, qa in enumerate(qa_pairs):
            gold_ids = qa.get("gold_chunk_ids", [])
            assert len(gold_ids) > 0, (
                f"QA pair {i} ('{qa.get('question', '???')[:60]}...') "
                f"has no gold_chunk_ids. This question is unevaluable."
            )

    @requires_outputs
    def test_gold_chunk_ids_reference_real_chunks(self):
        """
        Every gold_chunk_id MUST correspond to an actual chunk in the dataset.

        WHY: If a gold chunk ID is "B_42" but Config B only has 40 chunks,
        Recall will always be 0 for that question — not because retrieval
        failed, but because the gold reference is invalid. This is the most
        common silent data integrity bug in RAG evaluation.
        """
        qa_pairs = self._load_qa_pairs()
        all_chunk_ids = self._load_all_chunk_ids()

        if not all_chunk_ids:
            pytest.skip("No chunk data found to validate against")

        orphaned = []
        for qa in qa_pairs:
            for gid in qa.get("gold_chunk_ids", []):
                if gid not in all_chunk_ids:
                    orphaned.append((qa.get("id", "?"), gid))

        assert len(orphaned) == 0, (
            f"Found {len(orphaned)} gold_chunk_ids that don't match any actual chunk. "
            f"First 5 orphans: {orphaned[:5]}. "
            "This means Recall will be 0 for these questions regardless of "
            "retrieval quality — a data integrity bug, not a retrieval failure."
        )

    @requires_outputs
    def test_minimum_50_questions_generated(self):
        """PRD requires ≥50 questions."""
        qa_pairs = self._load_qa_pairs()
        assert len(qa_pairs) >= 50, (
            f"Only {len(qa_pairs)} QA pairs generated. PRD requires ≥50. "
            "Check synthetic_qa.py — did all 5 strategies run?"
        )

    @requires_outputs
    def test_question_type_diversity(self):
        """
        All 5 question types should be represented.

        WHY: If all questions are 'factual', we can't assess how the
        pipeline handles analytical or multi-hop reasoning. The PRD
        requires 5 types for comprehensive evaluation.
        """
        qa_pairs = self._load_qa_pairs()
        types_found = {qa.get("question_type") for qa in qa_pairs}

        expected_types = {"factual", "comparative", "analytical", "summarization", "multi_hop"}
        missing = expected_types - types_found
        assert len(missing) == 0, (
            f"Missing question types: {missing}. "
            f"Found types: {types_found}. "
            "Check synthetic_qa.py strategy distribution."
        )


# ============================================================================
# LAYER 3: END-TO-END TRACE OF INDIVIDUAL QUESTIONS
# ============================================================================
# First Principle: Follow one data point through the entire pipeline and
# verify each stage makes sense. Aggregate metrics can hide individual
# failures — this catches them.
#
# Java parallel: Integration test that traces a transaction from API call
# through service layer to database and back.
# ============================================================================


class TestLayer3_EndToEndTrace:
    """
    Picks individual questions from the QA dataset and traces them through
    the entire pipeline, verifying each stage.

    RUN THIS AFTER: Grid search has completed and results exist in results/
    """

    def _load_qa_pairs(self) -> list[dict]:
        qa_files = list(DATA_OUTPUT_DIR.glob("*qa*")) + list(
            DATA_OUTPUT_DIR.glob("*question*")
        )
        for f in qa_files:
            if f.suffix == ".json":
                with open(f) as fh:
                    data = json.load(fh)
                    if isinstance(data, list) and len(data) > 0:
                        return data
        pytest.skip("No QA dataset found")

    def _load_chunks_for_config(self, config_name: str) -> list[dict]:
        """Load chunks for a specific config (e.g., 'B')."""
        chunk_files = list(DATA_OUTPUT_DIR.glob(f"*{config_name}*chunk*"))
        if not chunk_files:
            chunk_files = list(DATA_OUTPUT_DIR.glob(f"*chunk*{config_name}*"))
        for f in chunk_files:
            if f.suffix == ".json":
                with open(f) as fh:
                    return json.load(fh)
        return []

    def _load_grid_search_report(self) -> dict | None:
        # WHY: Try both paths. Legacy reports in reports/, new results in metrics/
        report_path = REPORTS_DIR / "grid_search_report.json"
        metrics_path = PROJECT_ROOT / "results" / "metrics" / "grid_search_results.json"
        ragas_path = PROJECT_ROOT / "results" / "metrics" / "ragas_results.json"

        report = None
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
        elif metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
                # WHY: Metrics file is a list, but tests expect dict with config_evaluations key
                if isinstance(data, list):
                    report = {"config_evaluations": data}
                else:
                    report = data

        # WHY: Load RAGAS results separately if available (common when eval done separately)
        if report and ragas_path.exists():
            with open(ragas_path) as f:
                ragas_data = json.load(f)
                report["ragas_results"] = ragas_data

        return report

    @requires_outputs
    def test_gold_chunk_text_actually_exists_in_document(self):
        """
        For 3 random questions, verify the source_chunk_text field
        contains text that actually appears in the original document.

        WHY: Claude Code might have generated plausible-sounding chunk
        text that doesn't actually come from the source document. This
        would mean gold chunk IDs are fabricated references.
        """
        qa_pairs = self._load_qa_pairs()

        # WHY: Check 3 random questions (not just the first — avoids
        # the "first N are fine, rest are garbage" failure mode)
        import random
        random.seed(42)  # WHY: Deterministic for reproducibility
        sample = random.sample(qa_pairs, min(3, len(qa_pairs)))

        # Try to find the source document
        input_dir = PROJECT_ROOT / "data" / "input"
        source_files = list(input_dir.glob("*.md")) + list(input_dir.glob("*.pdf"))

        if not source_files:
            pytest.skip("No source document found in data/input/")

        # WHY: Read the source document to verify chunks came from it
        source_text = ""
        for sf in source_files:
            if sf.suffix == ".md":
                source_text += sf.read_text(encoding="utf-8")

        if not source_text:
            pytest.skip("Could not read source document text")

        for qa in sample:
            chunk_text = qa.get("source_chunk_text", "")
            if not chunk_text:
                continue

            # WHY: Check that a meaningful substring of the chunk text
            # appears in the source document. We use a 50-char window
            # because the full chunk might have been slightly modified
            # during chunking (whitespace normalization, etc.)
            search_snippet = chunk_text[20:70].strip() if len(chunk_text) > 70 else chunk_text.strip()

            assert search_snippet in source_text, (
                f"QA '{qa.get('id', '?')}': chunk text snippet not found in source document. "
                f"Snippet: '{search_snippet[:80]}...'. "
                "This suggests the gold chunk reference is fabricated."
            )

    @requires_outputs
    def test_retrieval_results_contain_real_chunk_ids(self):
        """
        Verify that retrieved_chunk_ids in results reference actual chunks,
        not hallucinated IDs.
        """
        report = self._load_grid_search_report()
        if not report:
            pytest.skip("Grid search report not found")

        # WHY: Check at least one config's individual results
        config_evals = report.get("config_evaluations", [])
        if not config_evals:
            pytest.skip("No config evaluations in report")

        eval_data = config_evals[0]
        individual_results = eval_data.get("individual_results", [])

        if not individual_results:
            pytest.skip("No individual results found in first config evaluation")

        # Collect all chunk IDs that should exist for this config
        config_name = eval_data.get("config_id", "").split("-")[0]  # e.g., "B" from "B-minilm"
        chunks = self._load_chunks_for_config(config_name)
        valid_ids = {c["id"] for c in chunks if isinstance(c, dict) and "id" in c}

        if not valid_ids:
            pytest.skip(f"No chunks found for config {config_name}")

        for result in individual_results[:5]:  # Check first 5
            for rid in result.get("retrieved_chunk_ids", []):
                assert rid in valid_ids, (
                    f"Retrieved chunk ID '{rid}' does not exist in "
                    f"config {config_name}'s chunk set. "
                    "FAISS index may be out of sync with chunk data."
                )


# ============================================================================
# LAYER 4: STATISTICAL SMELL TESTS
# ============================================================================
# First Principle: If aggregate results violate expected statistical
# properties, something is systematically wrong — even if individual
# computations pass.
#
# Java parallel: Like asserting that a load test's p99 latency is under
# threshold — testing system-level behavior, not unit behavior.
# ============================================================================


class TestLayer4_StatisticalSmellTests:
    """
    Examines aggregate pipeline results for patterns that indicate
    systematic bugs, even when individual computations are correct.

    These tests catch:
    - Data leakage (gold chunks leaking into retrieval)
    - Pipeline short-circuits (all configs using same index)
    - Degenerate results (all zeros, all ones, all identical)
    """

    def _load_grid_search_report(self) -> dict:
        # WHY: Try both paths. Legacy reports in reports/, new results in metrics/
        report_path = REPORTS_DIR / "grid_search_report.json"
        metrics_path = PROJECT_ROOT / "results" / "metrics" / "grid_search_results.json"
        ragas_path = PROJECT_ROOT / "results" / "metrics" / "ragas_results.json"

        report = None
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
        elif metrics_path.exists():
            with open(metrics_path) as f:
                data = json.load(f)
                # WHY: Metrics file is a list, but tests expect dict with config_evaluations key
                if isinstance(data, list):
                    report = {"config_evaluations": data}
                else:
                    report = data

        if not report:
            pytest.skip("Grid search report not found — run grid search first")

        # WHY: Load RAGAS results separately if available (common when eval done separately)
        if ragas_path.exists():
            with open(ragas_path) as f:
                ragas_data = json.load(f)
                report["ragas_results"] = ragas_data

        return report

    @requires_outputs
    def test_not_all_configs_have_identical_metrics(self):
        """
        Different configs (chunk size + embedding model) MUST produce
        different retrieval metrics. If all are identical, the pipeline
        is likely using the same index for every config.

        WHY: This is the most common "looks fine, is broken" bug.
        The grid search runs, produces a nice heatmap, but every cell
        has the same value because a config variable isn't being used.
        """
        report = self._load_grid_search_report()
        config_evals = report.get("config_evaluations", [])

        if len(config_evals) < 2:
            pytest.skip("Need at least 2 config evaluations")

        # Collect Recall@5 for each config
        recall_values = []
        for eval_data in config_evals:
            # WHY: Field name might vary. Try common patterns.
            r5 = (
                eval_data.get("avg_recall_at_5")
                or eval_data.get("recall_at_5")
                or eval_data.get("recall_5")
            )
            if r5 is not None:
                recall_values.append(r5)

        if len(recall_values) < 2:
            pytest.skip("Could not extract Recall@5 from config evaluations")

        unique_values = set(round(v, 6) for v in recall_values)
        assert len(unique_values) > 1, (
            f"ALL {len(recall_values)} configs have identical Recall@5 = {recall_values[0]}. "
            "This almost certainly means the pipeline is using the same index for "
            "every config. Check grid_search.py — is the config variable actually "
            "being passed to the vector store?"
        )

    @requires_outputs
    def test_recall_at_5_is_not_perfect_for_all_questions(self):
        """
        If Recall@5 = 1.0 for ALL questions across ALL configs, gold
        chunk IDs are likely leaking into the retrieval step.

        WHY: In a real RAG pipeline, some questions should be hard to
        answer with any config. 100% recall across the board indicates
        that the retrieval function is "cheating" — perhaps by including
        gold chunk IDs in the search results directly.
        """
        report = self._load_grid_search_report()
        config_evals = report.get("config_evaluations", [])

        perfect_count = 0
        total_count = 0

        for eval_data in config_evals:
            r5 = (
                eval_data.get("avg_recall_at_5")
                or eval_data.get("recall_at_5")
                or eval_data.get("recall_5")
            )
            if r5 is not None:
                total_count += 1
                if abs(r5 - 1.0) < 1e-6:
                    perfect_count += 1

        if total_count == 0:
            pytest.skip("No Recall@5 values found")

        # WHY: It's OK for SOME configs to hit 100% (especially large
        # chunk sizes where most content is in every chunk). But ALL
        # configs at 100% is a data leakage red flag.
        assert perfect_count < total_count, (
            f"ALL {total_count} configs have perfect Recall@5 = 1.0. "
            "This is a strong indicator of data leakage — gold chunk IDs "
            "may be included in the retrieval results. Check vector_store.py "
            "and grid_search.py for accidental inclusion of gold data."
        )

    @requires_outputs
    def test_bm25_has_nonzero_recall(self):
        """
        BM25 baseline should retrieve SOMETHING. If all metrics are 0,
        the tokenization is probably broken.

        WHY: BM25 uses simple word matching. For any reasonable document
        and questions derived from that document, BM25 should find at
        least some relevant chunks. Zero recall means either:
        1. Tokenization is wrong (empty tokens, wrong tokenizer)
        2. The BM25 index wasn't built from the same chunks
        3. The search function returns empty results
        """
        report = self._load_grid_search_report()

        # Look for BM25 in config evaluations
        bm25_eval = None
        for eval_data in report.get("config_evaluations", []):
            config_id = eval_data.get("config_id", "").lower()
            method = eval_data.get("retrieval_method", "").lower()
            if "bm25" in config_id or "bm25" in method:
                bm25_eval = eval_data
                break

        if bm25_eval is None:
            # WHY: BM25 might be stored separately
            bm25_eval = report.get("bm25_baseline")

        if bm25_eval is None:
            pytest.skip("BM25 baseline results not found in report")

        r5 = (
            bm25_eval.get("avg_recall_at_5")
            or bm25_eval.get("recall_at_5")
            or bm25_eval.get("recall_5")
        )

        assert r5 is not None, "BM25 Recall@5 field not found in results"
        assert r5 > 0.0, (
            "BM25 Recall@5 = 0.0 — the lexical baseline retrieves nothing. "
            "Check bm25_baseline.py: is tokenization working? Try "
            "`.lower().split()` on a chunk and verify tokens aren't empty."
        )

    @requires_outputs
    def test_ragas_scores_not_all_clustered_near_05(self):
        """
        If all RAGAS scores cluster near 0.5, the evaluation prompts
        are likely not receiving proper context.

        WHY: 0.5 is the "I have no idea" score for many metrics.
        It means the LLM evaluator can't distinguish good from bad,
        which happens when the context passed to RAGAS is empty,
        truncated, or malformed.
        """
        report = self._load_grid_search_report()
        ragas = report.get("ragas_results")

        if not ragas:
            pytest.skip("RAGAS results not found — run Day 4 evaluation first")

        scores = []
        for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            val = ragas.get(key)
            if val is not None:
                scores.append(val)

        if not scores:
            pytest.skip("No RAGAS scores found")

        # WHY: Check if all scores are suspiciously clustered around 0.5
        near_05 = sum(1 for s in scores if 0.4 <= s <= 0.6)
        assert near_05 < len(scores), (
            f"All {len(scores)} RAGAS scores are between 0.4-0.6: {scores}. "
            "This suggests the RAGAS evaluator isn't receiving proper context. "
            "Check generation_evaluator.py — are retrieved chunks being "
            "passed to the RAGAS evaluation correctly?"
        )


# ============================================================================
# LAYER 5: INDEPENDENT RE-IMPLEMENTATION
# ============================================================================
# First Principle: Two independent implementations of the same function
# must produce the same output. If they disagree, one is wrong.
#
# Java parallel: Double-entry bookkeeping. Two accountants computing the
# same total independently — if they disagree, there's an error.
# ============================================================================


class TestLayer5_IndependentReimplementation:
    """
    Re-implements core metric functions from scratch using ONLY their
    mathematical definitions. Compares against Claude Code's implementation.

    If results disagree, one implementation is wrong.
    Use this to identify exactly where a bug is.
    """

    # ---- Our independent implementations (from definitions) ----

    @staticmethod
    def _our_recall_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
        """
        Recall@K = |{relevant docs in top-K}| / |{total relevant docs}|

        WHY from scratch: This is the most critical metric in the pipeline.
        If Claude Code implemented it as Precision or MRR by mistake,
        the entire grid search ranking is wrong.
        """
        if not gold_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = set(gold_ids) & set(top_k)
        return len(relevant_in_top_k) / len(gold_ids)

    @staticmethod
    def _our_precision_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
        """
        Precision@K = |{relevant docs in top-K}| / K

        WHY different from Recall: Denominator is K (not total relevant).
        A common bug is swapping the denominators between Recall and Precision.
        """
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        relevant_in_top_k = set(gold_ids) & set(top_k)
        return len(relevant_in_top_k) / k

    @staticmethod
    def _our_mrr_at_k(gold_ids: list[str], retrieved_ids: list[str], k: int) -> float:
        """
        MRR@K = 1 / rank_of_first_relevant_result (within top-K)
                0 if no relevant result in top-K

        WHY: MRR rewards finding the first relevant result early.
        Common bugs: using 0-indexed position instead of 1-indexed rank,
        or computing average reciprocal rank instead of MRR.
        """
        top_k = retrieved_ids[:k]
        gold_set = set(gold_ids)
        for rank_0indexed, doc_id in enumerate(top_k):
            if doc_id in gold_set:
                return 1.0 / (rank_0indexed + 1)  # WHY +1: rank is 1-indexed
        return 0.0

    # ---- Cross-validation tests ----

    def _import_retrieval_evaluator(self):
        try:
            from src.retrieval_evaluator import (
                compute_mrr_at_k,
                compute_precision_at_k,
                compute_recall_at_k,
            )
            return compute_recall_at_k, compute_precision_at_k, compute_mrr_at_k
        except ImportError:
            pytest.skip("src.retrieval_evaluator not found")

    @pytest.mark.parametrize(
        "gold,retrieved,k",
        [
            # WHY these test cases: Cover all interesting positions and edge cases
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 1),
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 3),
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 5),
            (["c1", "c5"], ["c1", "c2", "c3", "c5", "c4"], 1),
            (["c1", "c5"], ["c1", "c2", "c3", "c5", "c4"], 3),
            (["c1", "c5"], ["c1", "c2", "c3", "c5", "c4"], 5),
            (["c99"], ["c1", "c2", "c3", "c4", "c5"], 5),         # Total miss
            (["c1"], ["c1", "c2", "c3", "c4", "c5"], 1),          # Perfect hit
            (["c1", "c2", "c3"], ["c3", "c1", "c2", "c4", "c5"], 5),  # All gold in top-5
            (["c5"], ["c1", "c2", "c3", "c4", "c5"], 5),          # Gold at last position
        ],
    )
    def test_recall_matches_independent_implementation(self, gold, retrieved, k):
        """
        Claude Code's compute_recall_at_k must match our from-scratch version.
        """
        compute_recall, _, _ = self._import_retrieval_evaluator()

        theirs = compute_recall(gold, retrieved, k=k)
        ours = self._our_recall_at_k(gold, retrieved, k)

        assert theirs == pytest.approx(ours, abs=1e-9), (
            f"RECALL MISMATCH at k={k}!\n"
            f"  Claude Code says: {theirs}\n"
            f"  Our calculation:  {ours}\n"
            f"  Gold: {gold}, Retrieved: {retrieved}\n"
            "One implementation is wrong. Check the denominator — "
            "Recall divides by len(gold_ids), not by K."
        )

    @pytest.mark.parametrize(
        "gold,retrieved,k",
        [
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 1),
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 3),
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 5),
            (["c1", "c5"], ["c1", "c2", "c3", "c5", "c4"], 3),
            (["c1", "c5"], ["c1", "c2", "c3", "c5", "c4"], 5),
            (["c99"], ["c1", "c2", "c3", "c4", "c5"], 5),
            (["c1"], ["c1", "c2", "c3", "c4", "c5"], 1),
        ],
    )
    def test_precision_matches_independent_implementation(self, gold, retrieved, k):
        """
        Claude Code's compute_precision_at_k must match our from-scratch version.
        """
        _, compute_precision, _ = self._import_retrieval_evaluator()

        theirs = compute_precision(gold, retrieved, k=k)
        ours = self._our_precision_at_k(gold, retrieved, k)

        assert theirs == pytest.approx(ours, abs=1e-9), (
            f"PRECISION MISMATCH at k={k}!\n"
            f"  Claude Code says: {theirs}\n"
            f"  Our calculation:  {ours}\n"
            f"  Gold: {gold}, Retrieved: {retrieved}\n"
            "One implementation is wrong. Check the denominator — "
            "Precision divides by K, not by len(gold_ids)."
        )

    @pytest.mark.parametrize(
        "gold,retrieved,k",
        [
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 1),
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 3),
            (["c3"], ["c7", "c3", "c1", "c5", "c9"], 5),
            (["c1", "c5"], ["c1", "c2", "c3", "c5", "c4"], 5),
            (["c99"], ["c1", "c2", "c3", "c4", "c5"], 5),
            (["c1"], ["c1", "c2", "c3", "c4", "c5"], 1),
            (["c5"], ["c1", "c2", "c3", "c4", "c5"], 5),
        ],
    )
    def test_mrr_matches_independent_implementation(self, gold, retrieved, k):
        """
        Claude Code's compute_mrr_at_k must match our from-scratch version.
        """
        _, _, compute_mrr = self._import_retrieval_evaluator()

        theirs = compute_mrr(gold, retrieved, k=k)
        ours = self._our_mrr_at_k(gold, retrieved, k)

        assert theirs == pytest.approx(ours, abs=1e-9), (
            f"MRR MISMATCH at k={k}!\n"
            f"  Claude Code says: {theirs}\n"
            f"  Our calculation:  {ours}\n"
            f"  Gold: {gold}, Retrieved: {retrieved}\n"
            "One implementation is wrong. Check: MRR uses 1-indexed rank "
            "(position 0 → rank 1), and only cares about the FIRST relevant result."
        )


# ============================================================================
# LAYER 5 BONUS: AGGREGATION CROSS-CHECK
# ============================================================================
# WHY a separate section: Even if individual metric functions are correct,
# the aggregation (averaging across questions, grouping by type) can be wrong.
# ============================================================================


class TestLayer5_AggregationCrossCheck:
    """
    Loads actual pipeline results and re-computes aggregates independently
    to verify the grid search orchestrator didn't make averaging mistakes.
    """

    def _load_grid_search_report(self) -> dict:
        report_path = REPORTS_DIR / "grid_search_report.json"
        if not report_path.exists():
            pytest.skip("Grid search report not found")
        with open(report_path) as f:
            return json.load(f)

    @requires_outputs
    def test_avg_recall_matches_manual_average(self):
        """
        Re-compute average Recall@5 from individual results and compare
        to the reported aggregate.

        WHY: A common bug is computing the average wrong — e.g., using
        a weighted average when a simple mean is expected, or accidentally
        including NaN values that silently corrupt the mean.

        PYTHON GOTCHA (vs Java):
            In Java, (0 + 1) / 2 = 0 (integer division).
            In Python 3, (0 + 1) / 2 = 0.5 (float division).
            But np.mean([]) returns nan, not 0, which can silently propagate.
        """
        report = self._load_grid_search_report()
        config_evals = report.get("config_evaluations", [])

        if not config_evals:
            pytest.skip("No config evaluations found")

        for eval_data in config_evals[:3]:  # Check first 3 configs
            individual = eval_data.get("individual_results", [])
            if not individual:
                continue

            # Manually compute average Recall@5 from individual results
            recall_values = []
            for result in individual:
                r5 = (
                    result.get("recall_at_5")
                    or result.get("recall_5")
                )
                if r5 is not None:
                    recall_values.append(r5)

            if not recall_values:
                continue

            our_avg = sum(recall_values) / len(recall_values)

            # Compare to reported aggregate
            reported_avg = (
                eval_data.get("avg_recall_at_5")
                or eval_data.get("recall_at_5")
                or eval_data.get("recall_5")
            )

            if reported_avg is None:
                continue

            assert our_avg == pytest.approx(reported_avg, abs=1e-6), (
                f"Config '{eval_data.get('config_id', '?')}': "
                f"reported avg Recall@5 = {reported_avg:.6f}, "
                f"but manual average of {len(recall_values)} individual "
                f"results = {our_avg:.6f}. "
                "The grid search orchestrator's averaging is wrong."
            )


# ============================================================================
# HELPER: Print diagnostic summary (not a test — run manually)
# ============================================================================


def print_diagnostic_summary():
    """
    Run this manually to get a quick overview of pipeline health:
        cd 02-rag-evaluation
        python -c "from tests.test_verification import print_diagnostic_summary; print_diagnostic_summary()"

    WHY: Sometimes you want a quick visual check before running full tests.
    This prints key numbers you can eyeball for sanity.
    """
    print("\n" + "=" * 60)
    print("P2 PIPELINE DIAGNOSTIC SUMMARY")
    print("=" * 60)

    report_path = REPORTS_DIR / "grid_search_report.json"
    if not report_path.exists():
        print("❌ Grid search report not found. Run the pipeline first.")
        return

    with open(report_path) as f:
        report = json.load(f)

    config_evals = report.get("config_evaluations", [])
    print(f"\n📊 Configs evaluated: {len(config_evals)}")

    # Show Recall@5 distribution
    print("\n📈 Recall@5 by config:")
    for eval_data in config_evals:
        config_id = eval_data.get("config_id", "?")
        r5 = eval_data.get("avg_recall_at_5") or eval_data.get("recall_at_5") or "?"
        print(f"   {config_id:20s} → {r5}")

    # Check for identical values (Layer 4 smell test)
    r5_values = [
        eval_data.get("avg_recall_at_5") or eval_data.get("recall_at_5")
        for eval_data in config_evals
    ]
    r5_values = [v for v in r5_values if v is not None]
    unique = len(set(round(v, 6) for v in r5_values))
    if unique == 1 and len(r5_values) > 1:
        print("\n   ⚠️  ALL configs have IDENTICAL Recall@5 — likely a bug!")
    else:
        print(f"\n   ✅ {unique} distinct Recall@5 values across {len(r5_values)} configs")

    # BM25 check
    bm25 = report.get("bm25_baseline", {})
    bm25_r5 = bm25.get("avg_recall_at_5") or bm25.get("recall_at_5")
    if bm25_r5 is not None:
        print(f"\n🔍 BM25 baseline Recall@5: {bm25_r5}")
        if bm25_r5 == 0.0:
            print("   ⚠️  BM25 retrieves NOTHING — check tokenization")
    else:
        print("\n🔍 BM25 baseline: not found in report")

    # RAGAS check
    ragas = report.get("ragas_results", {})
    if ragas:
        print("\n🎯 RAGAS scores:")
        for key in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
            val = ragas.get(key, "?")
            flag = "⚠️" if isinstance(val, (int, float)) and 0.4 <= val <= 0.6 else "  "
            print(f"   {flag} {key:25s} → {val}")
    else:
        print("\n🎯 RAGAS: not found in report (Day 4)")

    # QA dataset
    qa_report = report.get("qa_dataset_report", {})
    if qa_report:
        total = qa_report.get("total_questions", "?")
        coverage = qa_report.get("chunk_coverage_percent", "?")
        print(f"\n📝 QA Dataset: {total} questions, {coverage}% chunk coverage")
    else:
        print("\n📝 QA Dataset report: not found")

    print("\n" + "=" * 60)
    print("Run full verification: pytest tests/test_verification.py -v")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_diagnostic_summary()
