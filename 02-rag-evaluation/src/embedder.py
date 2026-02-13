"""Embedding factory — wraps SentenceTransformers (local) and LiteLLM (API).

Provides a uniform interface for embedding text chunks into vectors. Two backends:
1. SentenceTransformerEmbedder: loads model locally, sequential batch processing
2. LiteLLMEmbedder: calls OpenAI API via LiteLLM, parallel batch processing

Java/TS parallel: like a Strategy pattern — BaseEmbedder is the interface,
concrete classes implement different embedding strategies, and the factory
function is a simple switch on the enum value (like a Spring @Bean method).

WHY L2 normalization: FAISS IndexFlatIP computes inner product (dot product).
For unit vectors, dot product = cosine similarity. Normalizing here means the
vector store doesn't need to know about similarity metrics.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.config import API_EMBEDDING_MAX_WORKERS, EMBEDDING_DIMENSIONS
from src.models import EmbeddingModel

logger = logging.getLogger(__name__)

# WHY 32: SentenceTransformers default. Controls per-batch tensor allocation —
# higher values use more RAM but fewer forward passes. 32 is safe for 8GB.
_ST_BATCH_SIZE = 32

# WHY 50: OpenAI embedding API accepts up to 2048 texts per call, but smaller
# batches (20-50) keep per-request latency low and allow better parallelism.
_API_BATCH_SIZE = 50


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors to unit L2 norm (in-place where possible).

    WHY not sklearn.preprocessing.normalize: avoid importing sklearn just for
    one operation. numpy is already loaded and this is 3 lines.
    """
    # Compute L2 norms per row. keepdims=True for broadcasting.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors (shouldn't happen with real text,
    # but defensive coding for edge cases like empty embeddings).
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class BaseEmbedder(ABC):
    """Abstract base class for all embedding backends.

    Java/TS parallel: like an interface with two required members —
    `embed()` method and `dimensions` property.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into L2-normalized vectors.

        Returns:
            np.ndarray of shape (len(texts), self.dimensions), dtype float32.
            All vectors have unit L2 norm.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embedding via sentence-transformers library.

    Loads the model into RAM on init. Processes texts sequentially in batches
    of 32 — no threading needed since the bottleneck is the model forward pass
    (CPU/GPU-bound, not I/O-bound).

    WHY lazy import: sentence_transformers is heavy (~500MB with torch).
    Only import when this class is actually instantiated, not when embedder.py
    is imported (which happens during tests that mock this class).
    """

    def __init__(self, model: EmbeddingModel) -> None:
        # WHY lazy import: avoids loading torch/sentence_transformers at module
        # level, which would slow down imports and break mocked tests.
        from sentence_transformers import SentenceTransformer

        self._model_name = model.value
        self._dimensions = EMBEDDING_DIMENSIONS[model]
        logger.info("Loading SentenceTransformer model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        logger.info("Model loaded: %s (%d dimensions)", self._model_name, self._dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the local SentenceTransformer model.

        WHY batch_size=32: controls peak tensor RAM per encode() call.
        SentenceTransformer.encode() handles batching internally, but we
        pass batch_size explicitly for predictable memory behavior.
        """
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)

        logger.info(
            "Embedding %d texts with %s (batch_size=%d)",
            len(texts), self._model_name, _ST_BATCH_SIZE,
        )
        # WHY convert_to_numpy=True: returns np.ndarray directly (no torch tensor).
        # show_progress_bar=False: we log progress ourselves, bar clutters CI output.
        vectors = self._model.encode(
            texts,
            batch_size=_ST_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = vectors.astype(np.float32)
        return _l2_normalize(vectors)


class LiteLLMEmbedder(BaseEmbedder):
    """API-based embedding via LiteLLM (wraps OpenAI, Cohere, etc.).

    WHY LiteLLM over raw openai SDK: unified interface across providers.
    If we ever swap to Cohere or another provider, only the model name changes.

    WHY ThreadPoolExecutor: embedding API calls are I/O-bound (network latency).
    We batch texts into groups of 50, then send all batches in parallel via
    a thread pool. This minimizes total wall-clock time.

    Java/TS parallel: like calling an async API with CompletableFuture.allOf()
    in Java, or Promise.all() in TypeScript.
    """

    def __init__(self, model: EmbeddingModel) -> None:
        self._model_name = model.value
        self._dimensions = EMBEDDING_DIMENSIONS[model]
        logger.info("Initialized LiteLLM embedder: %s (%d dimensions)",
                     self._model_name, self._dimensions)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        """Embed a single batch via LiteLLM API call.

        WHY separate method: each batch becomes one API call. The thread pool
        executor calls this method in parallel across batches.
        """
        import litellm

        # WHY litellm.embedding (not litellm.aembedding): we're using threads,
        # not asyncio. The sync version is correct here.
        response = litellm.embedding(model=self._model_name, input=batch)

        # Extract embedding vectors from response.
        # WHY sorted by index: LiteLLM/OpenAI API may return embeddings
        # out of order (each has an 'index' field). Sort to guarantee
        # input-output alignment.
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        vectors = np.array(
            [item["embedding"] for item in sorted_data],
            dtype=np.float32,
        )
        return vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts via parallel API calls using ThreadPoolExecutor.

        Splits texts into batches of 50, sends all batches in parallel,
        then concatenates results in order.
        """
        if not texts:
            return np.empty((0, self._dimensions), dtype=np.float32)

        # Split into batches of _API_BATCH_SIZE
        batches = [
            texts[i : i + _API_BATCH_SIZE]
            for i in range(0, len(texts), _API_BATCH_SIZE)
        ]
        logger.info(
            "Embedding %d texts with %s (%d batches, max_workers=%d)",
            len(texts), self._model_name, len(batches), API_EMBEDDING_MAX_WORKERS,
        )

        # WHY ThreadPoolExecutor: I/O-bound work (waiting for API responses).
        # max_workers=8 balances throughput vs rate limits.
        with ThreadPoolExecutor(max_workers=API_EMBEDDING_MAX_WORKERS) as executor:
            # map() preserves input order — batch_results[i] corresponds to batches[i]
            batch_results = list(executor.map(self._embed_batch, batches))

        # Concatenate all batch results into one array
        vectors = np.concatenate(batch_results, axis=0)
        return _l2_normalize(vectors)


def create_embedder(model: EmbeddingModel) -> BaseEmbedder:
    """Factory function — returns the correct embedder for the given model.

    WHY a factory over direct instantiation: callers don't need to know which
    class to use. The EmbeddingModel enum is the only public API for selecting
    an embedding backend.

    Java/TS parallel: like a static factory method (Effective Java Item 1) or
    a simple DI container that resolves by type.
    """
    if model in (EmbeddingModel.MINILM, EmbeddingModel.MPNET):
        return SentenceTransformerEmbedder(model)
    elif model == EmbeddingModel.OPENAI:
        return LiteLLMEmbedder(model)
    else:
        raise ValueError(f"Unknown embedding model: {model}")
