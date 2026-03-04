"""Embedding-semantic chunker — splits where consecutive sentence similarity drops.

Embeds every sentence, computes cosine similarity between consecutive sentences,
and splits where similarity falls below a threshold (semantic breakpoint).

ALWAYS uses MiniLM (all-MiniLM-L6-v2) for boundary detection regardless of the
indexing embedder — MiniLM is smallest (22.7M params), boundary detection only
needs relative similarity comparison, not absolute quality.

After chunking: del model + gc.collect() before indexing phase begins.

Config params: breakpoint_threshold (default 0.85), min_chunk_size (default 100).
"""

from __future__ import annotations

import gc
import re
import uuid

import numpy as np

from src.chunkers._utils import find_page_number
from src.interfaces import BaseChunker
from src.schemas import Chunk, ChunkMetadata, Document

# WHY: sentence boundary regex — split on ". ", "! ", "? " followed by a capital letter
# or end-of-string. More robust than naive ". " split for abbreviations like "Dr. Smith".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Minimum sentence length in characters to include in embedding
# WHY: very short "sentences" like "Fig." or "et al." confuse boundary detection
_MIN_SENTENCE_LEN = 20

# MiniLM model name — hard-coded per PRD Decision 1 (ALWAYS use MiniLM for boundary detection)
_BOUNDARY_MODEL = "all-MiniLM-L6-v2"


class EmbeddingSemanticChunker(BaseChunker):
    """Semantic chunker that finds topic boundaries via embedding similarity drops.

    Algorithm:
        1. Split document content into sentences.
        2. Filter sentences shorter than _MIN_SENTENCE_LEN.
        3. Embed all sentences with MiniLM.
        4. Compute cosine similarity between consecutive sentence embeddings.
        5. Mark a boundary after sentence[i] if similarity[i] < breakpoint_threshold.
        6. Group sentences into chunks at boundaries.
        7. Merge chunks smaller than min_chunk_size with their neighbour.
        8. del model + gc.collect() (8GB M2 RAM constraint — RULE 4 in CLAUDE.md).

    Java parallel: like a Lucene Analyzer that segments text using semantic
    clustering rather than character counts.
    """

    def __init__(
        self,
        breakpoint_threshold: float = 0.85,
        min_chunk_size: int = 100,
    ) -> None:
        """
        Args:
            breakpoint_threshold: Cosine similarity below which a chunk boundary is inserted.
                Lower values = fewer, larger chunks. Higher values = more, smaller chunks.
            min_chunk_size: Minimum chunk length in characters. Smaller chunks are merged.
        """
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document at semantic topic boundaries.

        Args:
            document: Fully extracted Document.

        Returns:
            List of Chunks grouped by semantic coherence.
        """
        content = document.content

        # Step 1-2: split and filter sentences
        raw_sentences = _SENTENCE_SPLIT_RE.split(content)
        sentences = [s.strip() for s in raw_sentences if len(s.strip()) >= _MIN_SENTENCE_LEN]

        if len(sentences) <= 1:
            # Not enough sentences to find boundaries — return whole doc as one chunk
            if content.strip():
                return [self._make_chunk(document, content, 0, len(content), 0)]
            return []

        # Step 3: embed with MiniLM (load → use → del → gc)
        # WHY: lazy import here so the module can be imported without loading the model
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        model = SentenceTransformer(_BOUNDARY_MODEL)
        embeddings: np.ndarray = model.encode(sentences, convert_to_numpy=True)  # type: ignore[assignment]

        # Step 4: consecutive cosine similarities
        similarities = self._consecutive_cosine_similarities(embeddings)

        # Step 5: find breakpoints (indices i where we split AFTER sentence[i])
        breakpoints = [i for i, sim in enumerate(similarities) if sim < self.breakpoint_threshold]

        # Step 6: group sentences into chunks
        grouped_texts = self._group_sentences(sentences, breakpoints)

        # Step 7: merge small chunks
        merged_texts = self._merge_small_chunks(grouped_texts)

        # Step 8: free model memory (8GB M2 constraint)
        del model
        gc.collect()

        # Build Chunk objects with provenance metadata
        chunks: list[Chunk] = []
        for i, text in enumerate(merged_texts):
            if not text.strip():
                continue
            # Find char offset in original content
            start = content.find(text[:50])  # use first 50 chars as search key
            if start == -1:
                start = 0
            end = start + len(text)
            page_number = find_page_number(document, start)
            chunks.append(
                self._make_chunk(document, text, start, end, i)
            )

        return chunks

    def _consecutive_cosine_similarities(self, embeddings: np.ndarray) -> list[float]:
        """Compute cosine similarity between each consecutive pair of sentence embeddings.

        WHY L2-normalise before dot product: this converts inner product to cosine
        similarity. Equivalent to sklearn's cosine_similarity but without the import.

        Args:
            embeddings: 2D array of shape (n_sentences, dimensions).

        Returns:
            List of n_sentences - 1 similarity scores.
        """
        # L2-normalise rows: each row divided by its L2 norm
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # WHY: clamp norms to avoid division by zero for zero vectors
        norms = np.where(norms == 0, 1.0, norms)
        normalised = embeddings / norms

        # Cosine similarity for consecutive pairs = dot product of normalised vectors
        similarities: list[float] = []
        for i in range(len(normalised) - 1):
            sim = float(np.dot(normalised[i], normalised[i + 1]))
            similarities.append(sim)
        return similarities

    def _group_sentences(self, sentences: list[str], breakpoints: list[int]) -> list[str]:
        """Group sentences into text blocks at the specified breakpoint indices.

        A breakpoint at index i means: insert a chunk boundary AFTER sentence[i].
        So sentences [0..i] form one chunk, [i+1..next_breakpoint] form the next, etc.

        Args:
            sentences: All filtered sentences.
            breakpoints: Indices after which to split.

        Returns:
            List of text blocks (each block = joined sentences for one chunk).
        """
        if not breakpoints:
            return [" ".join(sentences)]

        breakpoint_set = set(breakpoints)
        groups: list[str] = []
        current: list[str] = []

        for i, sentence in enumerate(sentences):
            current.append(sentence)
            if i in breakpoint_set:
                groups.append(" ".join(current))
                current = []

        if current:
            groups.append(" ".join(current))

        return groups

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge chunks smaller than min_chunk_size with their neighbour.

        Strategy: scan left to right; if a chunk is too small, append it to the
        previous chunk (or prepend to the next if it's the first chunk).

        Args:
            chunks: Text blocks from _group_sentences.

        Returns:
            List of merged text blocks where each is >= min_chunk_size characters
            (except possibly the very last chunk which absorbs all small remainders).
        """
        if not chunks:
            return chunks

        merged: list[str] = []
        for chunk in chunks:
            if merged and len(chunk) < self.min_chunk_size:
                # Merge with previous: append with a space separator
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        return merged

    def _make_chunk(
        self,
        document: Document,
        text: str,
        start: int,
        end: int,
        chunk_index: int,
    ) -> Chunk:
        """Helper to construct a Chunk with consistent metadata."""
        page_number = find_page_number(document, start)
        return Chunk(
            id=str(uuid.uuid4()),
            content=text,
            metadata=ChunkMetadata(
                document_id=document.id,
                source=document.metadata.source,
                page_number=page_number,
                start_char=start,
                end_char=end,
                chunk_index=chunk_index,
            ),
        )
