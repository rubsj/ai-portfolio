"""P4 Resume Coach — end-to-end pipeline orchestrator.

Runs all pipeline stages in sequence. Use skip flags for iterative dev
(e.g., --skip-generation once data is on disk, --skip-judge to save API cost).

WHY subprocess for generation/correction/analysis: Those modules parse sys.argv
via argparse, so calling their main() directly from here would fail — pipeline's
own args would leak into their parsers. Subprocess keeps each module's CLI
isolated while still letting us control execution from a single entrypoint.

WHY direct import for labeling, judge, multi-hop, and vector_store: Those
functions accept typed arguments (no argparse), so they compose cleanly.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


def _run_module(module: str, *, extra_args: list[str] | None = None) -> None:
    """Run a Python module as a subprocess.

    WHY sys.executable: Ensures we use the same virtualenv Python that's
    running pipeline.py — no PATH ambiguity across different venv setups.
    WHY check=True: Propagates non-zero exit codes as CalledProcessError so
    the pipeline aborts immediately on any stage failure rather than silently
    continuing with corrupted/missing data.
    """
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def _step_generate(dry_run: bool) -> None:
    """Stage 1: Generate 50 jobs + 250 resumes via run_generation.

    WHY subprocess: run_generation.main() calls argparse on sys.argv[1:].
    Calling it directly from pipeline would break if pipeline's own flags
    are present in sys.argv.
    """
    print("\n[1/7] Generating jobs + resumes...")
    args = ["--dry-run"] if dry_run else []
    _run_module("src.run_generation", extra_args=args)


def _step_label() -> None:
    """Stage 2: Label all pairs with failure modes via run_labeling."""
    print("\n[2/7] Labeling failure modes...")
    # WHY direct import: run_labeling.run() takes no args, no argparse.
    from src.run_labeling import run as label_run

    label_run()


def _step_judge() -> None:
    """Stage 3: GPT-4o evaluation of all pairs via judge.judge_batch.

    WHY direct import: judge_batch() accepts typed args — we load data
    fresh via DataStore (guarantees latest generated files are picked up).
    """
    print("\n[3/7] Running LLM judge (GPT-4o)...")
    from src.data_paths import DataStore
    from src.judge import judge_batch

    store = DataStore()
    if not store.pairs:
        print("  No pairs found — skipping judge (run generation first).")
        return
    judge_batch(store.pairs, store.jobs, store.resumes)


def _step_correct() -> None:
    """Stage 4: Correction loop via corrector.

    WHY subprocess: corrector.py has no callable run()/main() — entry logic
    is embedded in its __main__ block. Subprocess is the safe invocation path.
    """
    print("\n[4/7] Running corrector...")
    _run_module("src.corrector")


def _step_analyze() -> None:
    """Stage 5: Charts + pipeline_summary.json via analyzer.

    WHY subprocess: analyzer.py has no callable run()/main() — same as corrector.
    """
    print("\n[5/7] Running analyzer (charts + pipeline_summary)...")
    _run_module("src.analyzer")


def _step_multi_hop() -> None:
    """Stage 6: Generate multi-hop evaluation questions."""
    print("\n[6/7] Generating multi-hop evaluation questions...")
    # WHY direct import: multi_hop.run() takes no args, no argparse.
    from src.multi_hop import run as multihop_run

    multihop_run()


def _step_vector_store() -> None:
    """Stage 7: Build ChromaDB resume index."""
    print("\n[7/7] Building ChromaDB vector index...")
    # WHY direct import: build_resume_index() is a clean function with
    # a typed signature — no argparse, no sys.argv side effects.
    from src.vector_store import build_resume_index

    build_resume_index()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_pipeline(
    *,
    skip_generation: bool = False,
    skip_judge: bool = False,
    skip_vector_store: bool = False,
    dry_run: bool = False,
) -> None:
    """Run the full P4 pipeline from generation to vector index.

    Args:
        skip_generation: Skip job/resume generation (data already on disk).
        skip_judge:       Skip GPT-4o judge evaluation (saves ~$0.50 API cost).
        skip_vector_store: Skip ChromaDB index build (data already indexed).
        dry_run:          Pass --dry-run to generation (2 jobs × 5 = 10 pairs).
    """
    print("=" * 60)
    print("P4 Resume Coach — Pipeline")
    print("=" * 60)

    if not skip_generation:
        _step_generate(dry_run=dry_run)
    else:
        print("\n[1/7] Generation skipped (--skip-generation).")

    _step_label()

    if not skip_judge:
        _step_judge()
    else:
        print("\n[3/7] Judge skipped (--skip-judge).")

    _step_correct()
    _step_analyze()
    _step_multi_hop()

    if not skip_vector_store:
        _step_vector_store()
    else:
        print("\n[7/7] Vector store skipped (--skip-vector-store).")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="P4 Resume Coach — end-to-end pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (first time):
  python -m src.pipeline

  # Skip generation (data already on disk), skip judge (save API cost):
  python -m src.pipeline --skip-generation --skip-judge

  # Quick dry-run (2 jobs × 5 resumes = 10 pairs):
  python -m src.pipeline --dry-run --skip-judge

  # Re-index only (rebuild ChromaDB from existing resumes):
  python -m src.pipeline --skip-generation --skip-judge --skip-vector-store=false
        """,
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip job/resume generation (data already on disk)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip GPT-4o judge evaluation (saves ~$0.50 API cost)",
    )
    parser.add_argument(
        "--skip-vector-store",
        action="store_true",
        help="Skip ChromaDB index build",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to generation (2 jobs × 5 = 10 pairs only)",
    )
    args = parser.parse_args()

    run_pipeline(
        skip_generation=args.skip_generation,
        skip_judge=args.skip_judge,
        skip_vector_store=args.skip_vector_store,
        dry_run=args.dry_run,
    )
