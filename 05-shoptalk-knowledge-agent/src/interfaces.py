"""Abstract base classes for swappable pipeline components.

Factory swaps implementations via YAML config without code changes.
Same idea as Java's interface + @Component — define the contract here,
concrete classes live elsewhere.

ABCs:
    BaseChunker     — chunk(document) → list[Chunk]
    BaseEmbedder    — embed(texts) → np.ndarray, embed_query(query) → np.ndarray
    BaseVectorStore — add(), search(), save(), load()
    BaseRetriever   — retrieve(query, top_k) → list[RetrievalResult]
    BaseReranker    — rerank(query, results, top_k) → list[RetrievalResult]
    BaseLLM         — generate(prompt, system_prompt, temperature) → str
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.schemas import Chunk, Document, RetrievalResult


class BaseChunker(ABC):
    """Chunking strategy interface."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into retrieval units.

        Returns non-empty list. Each chunk.metadata.document_id == document.id.
        """


class BaseEmbedder(ABC):
    """Dense embedding model interface.

    Swappable (MiniLM → mpnet → OpenAI) so retriever and vector store
    stay model-agnostic.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Batch embed. Returns shape (len(texts), dimensions), L2-normalised."""

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Single query embed — may apply query-specific prefix.

        Returns shape (dimensions,), L2-normalised.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensionality. MiniLM=384, mpnet=768, OpenAI=1536.

        FAISS IndexFlatIP must be initialized with this value.
        """


class BaseVectorStore(ABC):
    """Vector index storage and retrieval.

    P5 uses FAISS (vs ChromaDB in P4) for explicit index management
    across 35+ configs with different embedding dimensions. See ADR-001.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks + pre-computed embeddings. Expects L2-normalised vectors."""

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Top-K similarity search. Returns (Chunk, score) descending."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist to disk. Creates <path>.faiss + <path>.json."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load from disk (same path used in save())."""


class BaseRetriever(ABC):
    """Retrieval interface (dense, BM25, hybrid)."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve top-K chunks. Returns descending by score, len <= top_k."""


class BaseReranker(ABC):
    """Reranking interface.

    CrossEncoder and Cohere have very different APIs but same contract:
    query + candidates in, re-scored subset out. Pipeline doesn't care which.
    """

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank candidates (typically top-20 from retriever). Returns top_k."""


class BaseLLM(ABC):
    """LLM text generation interface.

    Default is LiteLLM (OpenAI, Anthropic, Cohere behind one API).
    ABC allows swapping to raw SDK if LiteLLM is problematic. See ADR-004.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Generate completion. temperature=0.0 means deterministic/greedy."""