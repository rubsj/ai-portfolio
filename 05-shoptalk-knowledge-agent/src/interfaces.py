"""Abstract base classes (strategy pattern) for all swappable pipeline components.

Every component implements an ABC so the factory can swap implementations
via YAML config without code changes. Same pattern as Java's interface + @Component.

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
    """Strategy interface for all chunking implementations.

    Java parallel: interface Chunker { List<Chunk> chunk(Document doc); }

    Each implementation (FixedSizeChunker, RecursiveChunker, etc.) is a concrete
    strategy. The factory selects one at runtime via the 'chunking_strategy' config field.
    """

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a document into atomic retrieval units.

        Args:
            document: Fully extracted Document with content and page metadata.

        Returns:
            Non-empty list of Chunks. Each chunk.metadata.document_id == document.id.
        """


class BaseEmbedder(ABC):
    """Strategy interface for dense embedding models.

    Java parallel: interface Embedder { float[][] embed(List<String> texts); }

    WHY: The embedder is swappable (MiniLM → mpnet → OpenAI) so the retriever
    and vector store don't need to know which model is active.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into dense vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            2D array of shape (len(texts), self.dimensions), L2-normalised.
        """

    @abstractmethod
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string (may apply query-specific prompt).

        Args:
            query: The user query text.

        Returns:
            1D array of shape (self.dimensions,), L2-normalised.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors produced by this model.

        MiniLM = 384, mpnet = 768, OpenAI text-embedding-3-small = 1536.
        FAISS IndexFlatIP must be initialised with this value.
        """


class BaseVectorStore(ABC):
    """Strategy interface for vector index storage and retrieval.

    Java parallel: interface VectorStore<T> { void add(List<T> chunks, float[][] embeddings); }

    WHY: FAISS is used in P5 (vs ChromaDB in P4) because we need explicit index
    management for 35+ experiment configs with different embedding dimensions.
    See ADR-001.
    """

    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their pre-computed embeddings to the index.

        Args:
            chunks: List of Chunk objects.
            embeddings: 2D array (len(chunks), dimensions), L2-normalised.
        """

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, top_k: int
    ) -> list[tuple[Chunk, float]]:
        """Find the top-K most similar chunks for a query embedding.

        Args:
            query_embedding: 1D array (dimensions,), L2-normalised.
            top_k: Number of results to return.

        Returns:
            List of (Chunk, score) tuples, descending by score.
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the index and chunk metadata to disk.

        Args:
            path: Directory path. Creates two files: <path>.faiss + <path>.json.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved index and chunk metadata from disk.

        Args:
            path: Directory path (same as used in save()).
        """


class BaseRetriever(ABC):
    """Strategy interface for all retrieval methods (dense, BM25, hybrid).

    Java parallel: interface Retriever { List<RetrievalResult> retrieve(String query, int topK); }
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Retrieve top-K relevant chunks for a query.

        Args:
            query: The user query text.
            top_k: Number of results to return.

        Returns:
            List of RetrievalResult, descending by score, len <= top_k.
        """


class BaseReranker(ABC):
    """Strategy interface for reranking a candidate set.

    Java parallel: interface Reranker { List<RetrievalResult> rerank(String query, List<RetrievalResult> candidates, int topK); }

    WHY: CrossEncoder and Cohere are very different APIs but have the same contract —
    take a query + candidate list, return a re-scored + re-ranked subset.
    The pipeline doesn't need to know which reranker is active.
    """

    @abstractmethod
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int
    ) -> list[RetrievalResult]:
        """Rerank a list of retrieval results using a more expensive model.

        Args:
            query: The original user query.
            results: Candidate RetrievalResults from the retriever (typically top-20).
            top_k: Number of results to return after reranking.

        Returns:
            Re-scored and re-ranked list of RetrievalResult, len <= top_k.
        """


class BaseLLM(ABC):
    """Strategy interface for LLM text generation.

    Java parallel: interface LLM { String generate(String prompt, String systemPrompt, double temperature); }

    WHY: LiteLLM is the default (supports OpenAI, Anthropic, Cohere behind one API),
    but the ABC allows swapping to raw OpenAI SDK if LiteLLM is problematic. See ADR-004.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user/assistant prompt.
            system_prompt: Optional system instruction. Empty string = no system prompt.
            temperature: Sampling temperature. 0.0 = deterministic greedy decoding.

        Returns:
            The generated text string.
        """
