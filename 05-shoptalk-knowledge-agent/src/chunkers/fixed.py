"""Fixed-size chunker — splits document into chunks of exactly N characters.

Simplest strategy. Ignores sentence/paragraph boundaries.
Useful as a baseline: any retrieval improvement over fixed-size is meaningful.

Config params: chunk_size (default 512), chunk_overlap (default 50).
"""

from __future__ import annotations

import uuid

from src.chunkers._utils import find_page_number
from src.interfaces import BaseChunker
from src.schemas import Chunk, ChunkMetadata, Document


class FixedSizeChunker(BaseChunker):
    """Character-based sliding window chunker.

    Splits document.content into overlapping windows of `chunk_size` characters,
    stepping forward by (chunk_size - chunk_overlap) each iteration.

    Java parallel: like Apache Commons' StringUtils.split() but with overlap.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50) -> None:
        """
        Args:
            chunk_size: Target chunk size in characters.
            chunk_overlap: Number of characters from the previous chunk to prepend.

        Raises:
            ValueError: If chunk_overlap >= chunk_size (would cause infinite loop).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size character chunks with overlap.

        Args:
            document: Fully extracted Document.

        Returns:
            List of Chunks. Empty list if document.content is all whitespace.
        """
        content = document.content
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        chunk_index = 0
        start = 0

        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            chunk_text = content[start:end]

            # WHY: skip whitespace-only chunks that can appear at document boundaries
            if chunk_text.strip():
                page_number = find_page_number(document, start)
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        content=chunk_text,
                        metadata=ChunkMetadata(
                            document_id=document.id,
                            source=document.metadata.source,
                            page_number=page_number,
                            start_char=start,
                            end_char=end,
                            chunk_index=chunk_index,
                        ),
                    )
                )
                chunk_index += 1

            start += step

        return chunks
