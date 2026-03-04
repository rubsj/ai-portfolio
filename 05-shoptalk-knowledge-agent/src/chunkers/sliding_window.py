"""Sliding window chunker — overlapping windows of N tokens.

Every chunk overlaps with the previous by overlap_size tokens. Ensures no
information is lost at chunk boundaries — a fact split across a boundary
will appear in at least one complete chunk.

Why useful: dense retrieval can miss facts that land on chunk boundaries.
Sliding window trades index size for coverage completeness.

Config params: window_size (tokens), step_size (tokens, controls overlap).
"""

from __future__ import annotations

import uuid

import tiktoken

from src.chunkers._utils import find_page_number
from src.interfaces import BaseChunker
from src.schemas import Chunk, ChunkMetadata, Document


class SlidingWindowChunker(BaseChunker):
    """Token-based sliding window chunker using tiktoken for accurate token counts.

    Unlike FixedSizeChunker (character-based), this operates on LLM tokens so
    chunks align with the model's actual context window.

    WHY tiktoken over a simple split: LLM tokens ≠ characters or words.
    "transformer" = 1 token, "GPT-4" = 3 tokens. Accurate token counts matter
    when the downstream LLM has a strict context window.

    Java parallel: like a Scanner with a fixed-size sliding buffer.
    """

    def __init__(
        self,
        window_size: int = 200,
        step_size: int = 150,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """
        Args:
            window_size: Number of tokens per chunk.
            step_size: Number of tokens to advance per step (overlap = window - step).
            encoding_name: tiktoken encoding. cl100k_base = GPT-3.5/4 tokenizer.

        Raises:
            ValueError: If step_size > window_size (no overlap, would lose coverage).
        """
        if step_size > window_size:
            raise ValueError(
                f"step_size ({step_size}) must be <= window_size ({window_size})"
            )
        self.window_size = window_size
        self.step_size = step_size
        # WHY: get_encoding() is cached by tiktoken — safe to call per-instance
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into token-based sliding window chunks.

        Algorithm:
            1. Encode full content to token IDs.
            2. Slide a window of window_size tokens, stepping by step_size.
            3. Decode each window back to text.
            4. Compute char offset by decoding the prefix up to the window start.

        Args:
            document: Fully extracted Document.

        Returns:
            List of Chunks with token-aligned boundaries.
        """
        content = document.content
        tokens = self.encoding.encode(content)

        if not tokens:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        start_token = 0

        while start_token < len(tokens):
            end_token = min(start_token + self.window_size, len(tokens))
            window_tokens = tokens[start_token:end_token]
            chunk_text = self.encoding.decode(window_tokens)

            if chunk_text.strip():
                # Map token offset → char offset by decoding the prefix
                # WHY: no direct token→char mapping in tiktoken, so we decode the prefix.
                # This is O(n) per chunk but acceptable for academic papers (~25K tokens).
                char_start = len(self.encoding.decode(tokens[:start_token]))
                char_end = char_start + len(chunk_text)
                page_number = find_page_number(document, char_start)

                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        content=chunk_text,
                        metadata=ChunkMetadata(
                            document_id=document.id,
                            source=document.metadata.source,
                            page_number=page_number,
                            start_char=char_start,
                            end_char=char_end,
                            chunk_index=chunk_index,
                        ),
                    )
                )
                chunk_index += 1

            start_token += self.step_size

        return chunks
