"""Index builder — orchestrates parsing, chunking, embedding, and index creation.

Builds all 16 retrieval backends (15 FAISS indices + 1 BM25) from the 3 Kaggle
Markdown files. This is the Day 2 orchestration script that ties together
parser → chunker → embedder → vector_store / bm25_baseline.

WHY a separate module (not inline in a notebook): reproducibility. Running
`uv run python -m src.index_builder` rebuilds everything from scratch. No hidden
state, no notebook cell ordering issues.

Java/TS parallel: like a Spring Boot CommandLineRunner that wires up all the
services and executes the pipeline. Or a Gradle build task that orchestrates
compile → test → package.

RAM management strategy (8GB MacBook Air M2):
- Local models (MiniLM, mpnet): load one at a time, embed all 5 configs, then
  `del model; gc.collect()` before loading the next. Never two models in memory.
- API model (OpenAI): no local model loaded, uses ThreadPoolExecutor for I/O.
- FAISS indices: saved to disk immediately after creation, not held in memory.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

from src.bm25_baseline import BM25Retriever
from src.chunker import chunk_document
from src.config import (
    ALL_CHUNK_CONFIGS,
    API_EMBEDDING_MODELS,
    BM25_CHUNK_CONFIG,
    INDICES_DIR,
    INPUT_DIR,
    LOCAL_EMBEDDING_MODELS,
    OUTPUT_DIR,
    model_key,
)
from src.embedder import create_embedder
from src.models import Chunk, EmbeddingModel
from src.parser import parse_document
from src.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


# ===========================================================================
# Parsing + Chunking
# ===========================================================================

def _discover_input_files() -> list[Path]:
    """Find all .md files in data/input/.

    WHY sorted: deterministic order across runs — always financial_services,
    healthcare, technology (alphabetical). Reproducible chunk IDs.
    """
    md_files = sorted(INPUT_DIR.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in {INPUT_DIR}")
    logger.info("Found %d input files: %s", len(md_files), [f.name for f in md_files])
    return md_files


def _parse_and_chunk_all() -> dict[str, list[Chunk]]:
    """Parse all input docs and chunk with each config.

    Returns:
        Dict mapping config name (A-E) to concatenated chunk list from all docs.
        Chunk IDs are globally unique because each doc's chunks are re-indexed
        with a doc prefix: {config}_{doc_idx}_{chunk_idx} (e.g., B_0_42).

    WHY re-parse at runtime (not load Day 1 outputs): ensures consistency.
    Same parser + chunker code produces identical chunks every time. No stale
    artifact risk from a previous run with different config.
    """
    input_files = _discover_input_files()

    # Parse all documents once (shared across all chunk configs)
    parse_results = []
    for f in input_files:
        logger.info("Parsing %s ...", f.name)
        parse_results.append(parse_document(f))
    logger.info(
        "Parsed %d documents (total %d chars)",
        len(parse_results),
        sum(len(pr.full_text) for pr in parse_results),
    )

    # Chunk each document with each config, concatenate per config
    chunks_by_config: dict[str, list[Chunk]] = {}

    for config in ALL_CHUNK_CONFIGS:
        all_chunks: list[Chunk] = []
        for doc_idx, pr in enumerate(parse_results):
            doc_chunks = chunk_document(pr, config)

            # WHY re-ID chunks: chunk_document produces IDs like "B_0", "B_1" per
            # doc. When combining 3 docs, we need globally unique IDs. Prefix with
            # doc index: "B_0_0", "B_0_1" (doc 0), "B_1_0", "B_1_1" (doc 1), etc.
            for i, chunk in enumerate(doc_chunks):
                chunk.id = f"{config.name}_{doc_idx}_{i}"

            all_chunks.extend(doc_chunks)

        chunks_by_config[config.name] = all_chunks
        logger.info(
            "Config %s: %d chunks total (from %d docs)",
            config.name, len(all_chunks), len(parse_results),
        )

    return chunks_by_config


def _save_chunk_lists(chunks_by_config: dict[str, list[Chunk]]) -> None:
    """Save chunk lists as JSON for later use by QA generation and verification.

    WHY save chunks: Day 3 QA generation needs to know which chunks exist and
    their text content. Saving here avoids re-parsing + re-chunking later.
    Also enables the manual checkpoint test to display chunk text.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for config_name, chunks in chunks_by_config.items():
        path = OUTPUT_DIR / f"chunks_{config_name}.json"
        # WHY model_dump: Pydantic v2 serialization — converts Chunk to dict
        # with all fields (text, token_count, page_numbers, etc.)
        data = [chunk.model_dump() for chunk in chunks]
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d chunks to %s", len(chunks), path.name)


# ===========================================================================
# FAISS Index Building
# ===========================================================================

def _build_faiss_indices_local(
    chunks_by_config: dict[str, list[Chunk]],
) -> None:
    """Build FAISS indices for all local embedding models (MiniLM, mpnet).

    WHY process one model at a time: 8GB RAM constraint. MiniLM is ~80MB,
    mpnet is ~420MB. Loading both at once wastes RAM. Sequential loading with
    explicit cleanup between models keeps peak usage under 700MB.

    WHY embed all 5 configs per model before unloading: loading a
    SentenceTransformer model takes 2-5 seconds. Loading it once and
    embedding all configs amortizes the startup cost.
    """
    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    for model in LOCAL_EMBEDDING_MODELS:
        model_key = model_key(model)
        logger.info("=" * 60)
        logger.info("Loading local model: %s", model.value)
        start = time.perf_counter()
        embedder = create_embedder(model)
        logger.info("Model loaded in %.1fs", time.perf_counter() - start)

        for config_name, chunks in chunks_by_config.items():
            _embed_and_save(embedder, model_key, config_name, chunks)

        # WHY explicit cleanup: Python's gc doesn't immediately free the
        # model's ~400MB of tensors. `del` drops the reference, gc.collect()
        # forces the cycle collector. Prevents accumulation between models.
        del embedder
        gc.collect()
        logger.info("Unloaded model: %s", model.value)


def _build_faiss_indices_api(
    chunks_by_config: dict[str, list[Chunk]],
) -> None:
    """Build FAISS indices for API embedding models (OpenAI).

    WHY separate from local: no model to load/unload. LiteLLMEmbedder is
    stateless — just an API key and a model name. ThreadPoolExecutor handles
    the parallelism internally.
    """
    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    for model in API_EMBEDDING_MODELS:
        model_key = model_key(model)
        logger.info("=" * 60)
        logger.info("Using API model: %s", model.value)
        embedder = create_embedder(model)

        for config_name, chunks in chunks_by_config.items():
            _embed_and_save(embedder, model_key, config_name, chunks)


def _embed_and_save(
    embedder,
    model_key: str,
    config_name: str,
    chunks: list[Chunk],
) -> None:
    """Embed chunks and save the FAISS index to disk.

    WHY save immediately: don't hold indices in memory. Each index is ~3MB
    (500 vectors × 1536d × 4 bytes). Saving immediately keeps memory flat.
    """
    texts = [chunk.text for chunk in chunks]
    chunk_ids = [chunk.id for chunk in chunks]

    logger.info(
        "Embedding %d chunks for %s_%s ...",
        len(texts), model_key, config_name,
    )
    start = time.perf_counter()
    embeddings = embedder.embed(texts)
    elapsed = time.perf_counter() - start
    logger.info(
        "Embedded %d chunks in %.1fs (%.0f chunks/sec)",
        len(texts), elapsed, len(texts) / max(elapsed, 0.001),
    )

    # Create FAISS index and add vectors
    store = FAISSVectorStore(dimension=embedder.dimensions)
    store.add(embeddings, chunk_ids)

    # Save to data/output/indices/{model}_{config}.faiss + .json
    index_path = INDICES_DIR / f"{model_key}_{config_name}"
    store.save(index_path)
    logger.info("Saved FAISS index: %s (%d vectors)", index_path.name, store.size)


# ===========================================================================
# BM25 Index Building
# ===========================================================================

def _build_bm25(chunks_by_config: dict[str, list[Chunk]]) -> None:
    """Build BM25 baseline from Config B chunks.

    WHY Config B: PRD Section 3c specifies "BM25 baseline using Config B chunks".
    Config B (256 tokens, 64 overlap) is the industry baseline for dense search —
    using the same chunks for BM25 makes the comparison fair.
    """
    INDICES_DIR.mkdir(parents=True, exist_ok=True)

    config_name = BM25_CHUNK_CONFIG.name
    chunks = chunks_by_config[config_name]

    logger.info("Building BM25 index from %d Config %s chunks", len(chunks), config_name)
    retriever = BM25Retriever(chunks)

    index_path = INDICES_DIR / f"bm25_{config_name}"
    retriever.save(index_path)
    logger.info("Saved BM25 index: %s (%d docs)", index_path.name, retriever.size)


# ===========================================================================
# Checkpoint Verification
# ===========================================================================

def _run_checkpoint_queries(chunks_by_config: dict[str, list[Chunk]]) -> None:
    """Run smoke-test queries and print chunk text snippets + scores.

    WHY print text (not just IDs): visual relevance check. If "total revenue"
    returns chunks about healthcare, something is broken. Showing first 200
    chars lets you quickly verify without reading full chunks.
    """
    query = "total revenue"
    k = 3

    # WHY build a text lookup: search returns chunk IDs, but we need the text
    # to display. Build a flat dict from all configs for O(1) lookup.
    chunk_text_lookup: dict[str, str] = {}
    for chunks in chunks_by_config.values():
        for chunk in chunks:
            chunk_text_lookup[chunk.id] = chunk.text

    print("\n" + "=" * 70)
    print(f"CHECKPOINT: Query = \"{query}\", k = {k}")
    print("=" * 70)

    # Test a few representative indices
    test_indices = [
        ("minilm_B", "FAISS MiniLM Config B"),
        ("mpnet_B", "FAISS mpnet Config B"),
        ("minilm_E", "FAISS MiniLM Config E"),
    ]

    for index_stem, label in test_indices:
        index_path = INDICES_DIR / index_stem
        faiss_file = index_path.with_suffix(".faiss")
        if not faiss_file.exists():
            print(f"\n  [{label}] — index not found, skipping")
            continue

        store = FAISSVectorStore.load(index_path)
        # WHY embed the query: FAISS search needs a vector. Use the same
        # model that built the index for a fair comparison.
        model_name = index_stem.split("_")[0]
        model_enum = _model_enum_from_key(model_name)
        embedder = create_embedder(model_enum)
        query_vec = embedder.embed([query])[0]
        results = store.search(query_vec, k=k)

        print(f"\n  [{label}]")
        for rank, (chunk_id, score) in enumerate(results, 1):
            text_snippet = chunk_text_lookup.get(chunk_id, "(text not found)")[:200]
            print(f"    #{rank} [{chunk_id}] score={score:.4f}")
            print(f"        {text_snippet}...")

        # WHY cleanup: if we loaded a local model for the query, free it
        del embedder
        gc.collect()

    # BM25 checkpoint
    bm25_path = INDICES_DIR / f"bm25_{BM25_CHUNK_CONFIG.name}"
    if bm25_path.with_suffix(".pkl").exists():
        retriever = BM25Retriever.load(bm25_path)
        results = retriever.search(query, k=k)

        print(f"\n  [BM25 Config {BM25_CHUNK_CONFIG.name}]")
        for rank, (chunk_id, score) in enumerate(results, 1):
            text_snippet = chunk_text_lookup.get(chunk_id, "(text not found)")[:200]
            print(f"    #{rank} [{chunk_id}] score={score:.4f}")
            print(f"        {text_snippet}...")

    # Config E vs Config B side-by-side
    print("\n" + "-" * 70)
    print("Config E (semantic) vs Config B (fixed-size) — MiniLM")
    print("-" * 70)

    for config_label, index_stem in [("B (fixed-size)", "minilm_B"), ("E (semantic)", "minilm_E")]:
        index_path = INDICES_DIR / index_stem
        if not index_path.with_suffix(".faiss").exists():
            print(f"\n  Config {config_label}: index not found, skipping")
            continue

        store = FAISSVectorStore.load(index_path)
        embedder = create_embedder(EmbeddingModel.MINILM)
        query_vec = embedder.embed([query])[0]
        results = store.search(query_vec, k=k)

        print(f"\n  Config {config_label}:")
        for rank, (chunk_id, score) in enumerate(results, 1):
            text_snippet = chunk_text_lookup.get(chunk_id, "(text not found)")[:200]
            print(f"    #{rank} [{chunk_id}] score={score:.4f}")
            print(f"        {text_snippet}...")

        del embedder
        gc.collect()

    print("\n" + "=" * 70)


# ===========================================================================
# Helpers
# ===========================================================================

def _model_enum_from_key(key: str) -> EmbeddingModel:
    """Reverse lookup: short key → EmbeddingModel enum."""
    return {
        "minilm": EmbeddingModel.MINILM,
        "mpnet": EmbeddingModel.MPNET,
        "openai": EmbeddingModel.OPENAI,
    }[key]


# ===========================================================================
# Main Entry Point
# ===========================================================================

def build_all_indices() -> None:
    """Build all 16 retrieval backends (15 FAISS + 1 BM25).

    Orchestration order:
    1. Parse + chunk all docs (shared across all models)
    2. Save chunk lists as JSON (for Day 3 QA generation)
    3. Build FAISS indices for local models (sequential, with cleanup)
    4. Build FAISS indices for API models (parallel batches)
    5. Build BM25 baseline from Config B chunks
    6. Run checkpoint verification queries
    """
    total_start = time.perf_counter()

    # Step 1: Parse and chunk
    logger.info("Step 1/6: Parsing and chunking all documents ...")
    chunks_by_config = _parse_and_chunk_all()

    # Step 2: Save chunk lists
    logger.info("Step 2/6: Saving chunk lists ...")
    _save_chunk_lists(chunks_by_config)

    # Step 3: Local FAISS indices
    logger.info("Step 3/6: Building FAISS indices (local models) ...")
    _build_faiss_indices_local(chunks_by_config)

    # Step 4: API FAISS indices
    logger.info("Step 4/6: Building FAISS indices (API models) ...")
    _build_faiss_indices_api(chunks_by_config)

    # Step 5: BM25
    logger.info("Step 5/6: Building BM25 baseline ...")
    _build_bm25(chunks_by_config)

    # Step 6: Checkpoint
    logger.info("Step 6/6: Running checkpoint verification ...")
    _run_checkpoint_queries(chunks_by_config)

    total_elapsed = time.perf_counter() - total_start
    logger.info("All 16 indices built in %.1fs", total_elapsed)

    # Summary
    faiss_count = len(list(INDICES_DIR.glob("*.faiss")))
    bm25_count = len(list(INDICES_DIR.glob("*.pkl")))
    chunk_count = len(list(OUTPUT_DIR.glob("chunks_*.json")))
    print(f"\nDone! {faiss_count} FAISS indices + {bm25_count} BM25 = "
          f"{faiss_count + bm25_count} retrieval backends")
    print(f"Chunk files: {chunk_count}")
    print(f"Total time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    # WHY basicConfig here (not in config.py): only configure logging when
    # running as a script. When imported as a module, let the caller decide.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    build_all_indices()
