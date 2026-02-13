"""FAISS vector store — wraps IndexFlatIP for similarity search.

Provides add/search/save/load over a FAISS flat inner-product index.
Since embedder.py L2-normalizes all vectors, inner product = cosine similarity.

WHY IndexFlatIP (not IndexFlatL2 or IndexIVFFlat):
- IndexFlatIP: brute-force inner product. O(n) per query but exact results.
- For <10K vectors (our case: ~500 chunks × 5 configs) brute-force is fast enough
  (<1ms per query). No training needed, no quantization loss.
- Approximate indices (IVF, HNSW) are for millions of vectors — overkill here.

WHY separate .faiss + .json files: FAISS native serialization only stores vectors,
not metadata. We store chunk ID mapping in a sidecar JSON file so we can map
FAISS integer indices back to chunk IDs like "B_42".

Java/TS parallel: like a DAO wrapping a database client. The FAISS index is the
storage engine, this class adds ID mapping and validation on top.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """Vector store backed by FAISS IndexFlatIP.

    Stores L2-normalized embeddings and maps FAISS integer indices to chunk IDs.
    Since all vectors are unit-length, inner product = cosine similarity.

    Java/TS parallel: like a typed wrapper around a Map<Integer, String> + a
    vector database client. The chunk_ids list IS the mapping — chunk_ids[i]
    corresponds to FAISS vector at position i.
    """

    def __init__(self, dimension: int) -> None:
        """Create an empty FAISS index for vectors of the given dimension.

        WHY store dimension explicitly: needed for validation in add() and for
        recreating the index on load().
        """
        self._dimension = dimension
        # WHY IndexFlatIP: inner product on L2-normalized vectors = cosine similarity.
        # Brute-force search — exact results, no training step needed.
        self._index = faiss.IndexFlatIP(dimension)
        # WHY list (not dict): FAISS returns integer indices (0, 1, 2, ...),
        # and we need O(1) lookup by position. chunk_ids[i] = ID of vector i.
        self._chunk_ids: list[str] = []
        logger.info("Created FAISS IndexFlatIP with dimension=%d", dimension)

    @property
    def dimension(self) -> int:
        """Dimensionality of stored vectors."""
        return self._dimension

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    def add(self, embeddings: np.ndarray, chunk_ids: list[str]) -> None:
        """Add embeddings and their corresponding chunk IDs to the index.

        Validates that:
        1. Embedding dimensions match the index dimension
        2. Number of embeddings matches number of chunk IDs

        WHY validate here (not in search): bad data in the index is silent —
        wrong dimensions cause segfaults, mismatched IDs cause wrong results.
        Fail fast on add() so the error is close to the cause.
        """
        if embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D array, got {embeddings.ndim}D"
            )
        if embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} doesn't match "
                f"index dimension {self._dimension}"
            )
        if embeddings.shape[0] != len(chunk_ids):
            raise ValueError(
                f"Got {embeddings.shape[0]} embeddings but {len(chunk_ids)} chunk IDs"
            )

        # WHY ascontiguousarray + float32: FAISS requires C-contiguous float32.
        # numpy slicing can produce non-contiguous views, and some embedders
        # may return float64. This ensures FAISS gets exactly what it needs.
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(vectors)
        self._chunk_ids.extend(chunk_ids)
        logger.info("Added %d vectors to index (total: %d)", len(chunk_ids), self.size)

    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Search for the k most similar vectors to the query.

        Returns (chunk_id, score) pairs sorted by descending similarity.
        If k > stored vectors, returns all stored vectors (no crash).

        WHY clamp k: FAISS raises an error if k > ntotal. We silently clamp
        because callers shouldn't need to know the index size — just ask for
        top-k and get back up to k results.
        """
        if self.size == 0:
            return []

        # WHY reshape: FAISS expects (n_queries, dimension). A single query
        # vector might be (dimension,) — reshape to (1, dimension).
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Clamp k to index size — FAISS requires k <= ntotal
        effective_k = min(k, self.size)

        query = np.ascontiguousarray(query, dtype=np.float32)
        # WHY scores, indices: FAISS returns parallel arrays of scores and
        # integer indices. scores[0][i] is the similarity for indices[0][i].
        scores, indices = self._index.search(query, effective_k)

        # Build (chunk_id, score) pairs from the first (and only) query row.
        # WHY filter idx != -1: FAISS uses -1 as a sentinel for "no result"
        # when the index has fewer vectors than k (shouldn't happen after
        # clamping, but defensive).
        results: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:
                results.append((self._chunk_ids[idx], float(score)))

        return results

    def save(self, path: Path) -> None:
        """Save the FAISS index and chunk ID mapping to disk.

        Creates two files:
        - {path}.faiss — the FAISS index binary
        - {path}.json — the chunk ID list (JSON array)

        WHY two files: FAISS has its own binary format (write_index/read_index).
        Chunk IDs are metadata that FAISS doesn't know about — store separately.
        The .json sidecar pattern keeps them paired by filename stem.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss_path = path.with_suffix(".faiss")
        json_path = path.with_suffix(".json")

        faiss.write_index(self._index, str(faiss_path))
        json_path.write_text(json.dumps(self._chunk_ids, indent=2))

        logger.info(
            "Saved FAISS index (%d vectors, %dd) to %s",
            self.size, self._dimension, faiss_path,
        )

    @classmethod
    def load(cls, path: Path) -> FAISSVectorStore:
        """Load a FAISS index and chunk ID mapping from disk.

        WHY classmethod (not staticmethod): returns a new FAISSVectorStore
        instance. Java parallel: like a static factory method.

        The path should be the stem — method appends .faiss and .json suffixes.
        """
        faiss_path = path.with_suffix(".faiss")
        json_path = path.with_suffix(".json")

        index = faiss.read_index(str(faiss_path))
        chunk_ids = json.loads(json_path.read_text())

        # WHY construct manually instead of cls(dimension): we already have
        # the loaded index, don't want to create a new empty one.
        store = cls.__new__(cls)
        store._dimension = index.d
        store._index = index
        store._chunk_ids = chunk_ids

        logger.info(
            "Loaded FAISS index (%d vectors, %dd) from %s",
            store.size, store._dimension, faiss_path,
        )
        return store
