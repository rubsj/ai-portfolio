"""Heading-semantic chunker — splits on document section headings.

Detects section boundaries using regex patterns for common heading formats
(ALL CAPS lines, numbered sections like "3.1", lines followed by dashes).
Each section becomes one chunk, preserving semantic coherence.

Requires no embeddings during chunking (unlike EmbeddingSemanticChunker).
Best for well-structured academic papers with clear section hierarchy.
"""

from __future__ import annotations

import re
import uuid

from src.chunkers._utils import find_page_number
from src.interfaces import BaseChunker
from src.schemas import Chunk, ChunkMetadata, Document

# ---------------------------------------------------------------------------
# Pre-compiled heading patterns (module-level — compiled once)
# ---------------------------------------------------------------------------

# Pattern 1: Markdown headings (# Intro, ## Methods, ### 3.1 Results)
_MARKDOWN_HEADING = re.compile(r"^#{1,4}\s+.+", re.MULTILINE)

# Pattern 2: Numbered section headings like "3.", "3.1", "A.1 Introduction"
_NUMBERED_HEADING = re.compile(r"^[A-Z0-9]+(\.[0-9]+)*\.?\s+[A-Z]", re.MULTILINE)

# Pattern 3: ALL-CAPS short lines (conference paper section headers)
# ≥10 chars to avoid matching single-word labels like "ABSTRACT" alone
_ALLCAPS_HEADING = re.compile(r"^[A-Z][A-Z\s]{9,}$", re.MULTILINE)

# Pattern 4: Academic section names (common in NLP/ML papers)
_ACADEMIC_HEADING = re.compile(
    r"^(Abstract|Introduction|Background|Related Work|Methodology|Methods|"
    r"Approach|Model Architecture|Experiments?|Results?|Discussion|"
    r"Conclusion|Conclusions|References|Appendix|Acknowledgements?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_ALL_HEADING_PATTERNS = [
    _MARKDOWN_HEADING,
    _NUMBERED_HEADING,
    _ALLCAPS_HEADING,
    _ACADEMIC_HEADING,
]


class HeadingSemanticChunker(BaseChunker):
    """Splits document at section heading boundaries.

    Algorithm:
        1. Scan content for all heading matches (4 pattern types).
        2. Sort heading positions to get section boundaries.
        3. Slice content between consecutive boundaries.
        4. Split oversized sections at paragraph boundaries ("\n\n").
        5. Skip sections smaller than min_chunk_size.

    WHY prefer heading-based for academic papers: NLP papers have highly
    consistent section structure (Abstract → Introduction → Methods → Results →
    Conclusion). Keeping sections intact means a retrieval hit on "Methods"
    returns the complete methodology, not a fragment mid-paragraph.

    Java parallel: like an XML parser that splits on <section> tags.
    """

    def __init__(
        self,
        heading_patterns: list[re.Pattern] | None = None,
        min_chunk_size: int = 50,
        max_chunk_size: int = 3000,
    ) -> None:
        """
        Args:
            heading_patterns: Custom compiled regex patterns. Defaults to the 4 module-level patterns.
            min_chunk_size: Sections shorter than this are skipped (likely noise).
            max_chunk_size: Sections longer than this are split at paragraph boundaries.
        """
        self.heading_patterns = heading_patterns or _ALL_HEADING_PATTERNS
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document at heading boundaries.

        Args:
            document: Fully extracted Document.

        Returns:
            List of Chunks, one per section (possibly split further if oversized).
        """
        content = document.content
        boundaries = self._find_heading_boundaries(content)

        if not boundaries:
            # No headings found — return entire content as one chunk
            if content.strip():
                return [self._make_chunk(document, content, 0, len(content), 0)]
            return []

        sections = self._split_at_boundaries(content, boundaries)
        chunks: list[Chunk] = []
        chunk_index = 0

        for section_text, section_start in sections:
            if len(section_text.strip()) < self.min_chunk_size:
                continue  # skip tiny fragments (e.g., blank sections)

            if len(section_text) <= self.max_chunk_size:
                chunks.append(
                    self._make_chunk(
                        document, section_text, section_start,
                        section_start + len(section_text), chunk_index,
                    )
                )
                chunk_index += 1
            else:
                # Oversized section — split at paragraph boundaries
                sub_chunks = self._split_oversized(
                    document, section_text, section_start, chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)

        return chunks

    def _find_heading_boundaries(self, content: str) -> list[int]:
        """Return sorted list of character offsets where headings start.

        Args:
            content: Full document text.

        Returns:
            Sorted list of heading start positions (deduplicated).
        """
        positions: set[int] = set()
        for pattern in self.heading_patterns:
            for match in pattern.finditer(content):
                positions.add(match.start())
        return sorted(positions)

    def _split_at_boundaries(
        self, content: str, boundaries: list[int]
    ) -> list[tuple[str, int]]:
        """Slice content between heading positions.

        The heading line is included at the START of its section (not at the
        end of the previous section). This way the retrieved chunk always
        contains the heading that labels its content.

        Args:
            content: Full document text.
            boundaries: Sorted list of heading start positions.

        Returns:
            List of (section_text, start_offset) tuples.
        """
        sections: list[tuple[str, int]] = []

        # Text before the first heading (e.g., title page, abstract preamble)
        if boundaries[0] > 0:
            pre_heading = content[: boundaries[0]]
            if pre_heading.strip():
                sections.append((pre_heading, 0))

        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(content)
            section_text = content[start:end]
            sections.append((section_text, start))

        return sections

    def _split_oversized(
        self,
        document: Document,
        text: str,
        base_offset: int,
        start_index: int,
    ) -> list[Chunk]:
        """Split an oversized section at paragraph boundaries.

        Args:
            document: Source document (for metadata).
            text: Section text that exceeds max_chunk_size.
            base_offset: Character offset of this section within document.content.
            start_index: chunk_index for the first sub-chunk.

        Returns:
            List of sub-chunks from this section.
        """
        paragraphs = text.split("\n\n")
        chunks: list[Chunk] = []
        current_parts: list[str] = []
        current_len = 0
        local_offset = base_offset
        chunk_index = start_index

        for para in paragraphs:
            if current_len + len(para) + 2 > self.max_chunk_size and current_parts:
                # Flush current accumulation
                chunk_text = "\n\n".join(current_parts)
                chunks.append(
                    self._make_chunk(
                        document, chunk_text, local_offset,
                        local_offset + len(chunk_text), chunk_index,
                    )
                )
                chunk_index += 1
                local_offset += len(chunk_text) + 2  # +2 for the "\n\n" separator
                current_parts = []
                current_len = 0

            current_parts.append(para)
            current_len += len(para) + 2

        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            if chunk_text.strip():
                chunks.append(
                    self._make_chunk(
                        document, chunk_text, local_offset,
                        local_offset + len(chunk_text), chunk_index,
                    )
                )

        return chunks

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
