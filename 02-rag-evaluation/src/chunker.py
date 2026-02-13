"""Document chunker — fixed-size (A-D) and semantic header-based (E) strategies.

Splits parsed documents into Chunk objects using two strategies:
1. Fixed-size: RecursiveCharacterTextSplitter with tiktoken token counting (Configs A-D)
2. Semantic: Split on Markdown ## and ### headers (Config E)

Java/TS parallel: like a Strategy pattern — one public function dispatches to
the right chunking strategy based on config.is_semantic. Similar to Java's
TextSplitter interface with multiple implementations.
"""

from __future__ import annotations

import logging

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import SEMANTIC_MAX_TOKENS, SEMANTIC_MIN_TOKENS, SEMANTIC_SUBDIVISION_CONFIG
from src.models import Chunk, ChunkConfig
from src.parser import ParseResult, get_page_for_offset

logger = logging.getLogger(__name__)

# WHY cl100k_base: the encoding used by GPT-4o, GPT-4o-mini, and text-embedding-3-small.
# Token counts align with what the LLM and embedding model actually see, so chunk_size
# in tokens accurately reflects model context usage.
# Java/TS parallel: no built-in equivalent — tokenization is model-specific in LLMs.
_ENCODING = tiktoken.get_encoding("cl100k_base")


# ===========================================================================
# Public API
# ===========================================================================

def chunk_document(parse_result: ParseResult, config: ChunkConfig) -> list[Chunk]:
    """Dispatch to the right chunking strategy based on config.is_semantic.

    Java/TS parallel: like a factory method that picks the right strategy
    based on a config flag — Strategy pattern.

    Args:
        parse_result: Output from parse_document() — full text + metadata.
        config: Which chunk config to use (A-E).

    Returns:
        List of Chunk objects with IDs, text, token counts, and char offsets.
    """
    if config.is_semantic:
        return _chunk_semantic(parse_result, config)
    return _chunk_fixed_size(parse_result, config)


# ===========================================================================
# Token Counting
# ===========================================================================

def _count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding.

    WHY tiktoken over len(text.split()): tokens align with what the LLM sees.
    "don't" is 1 word but 2 tokens. Character count is even worse — "cat" is
    3 chars but 1 token. Using the actual tokenizer prevents chunk size drift.

    Java/TS parallel: no built-in equivalent — closest is OpenAI's tiktoken npm package.
    """
    return len(_ENCODING.encode(text))


# ===========================================================================
# Fixed-Size Chunking (Configs A-D)
# ===========================================================================

def _chunk_fixed_size(parse_result: ParseResult, config: ChunkConfig) -> list[Chunk]:
    """Split text using RecursiveCharacterTextSplitter with tiktoken length.

    WHY RecursiveCharacterTextSplitter over TokenTextSplitter: recursive tries
    larger separators first (paragraphs → sentences → words), producing more
    semantically coherent chunks. TokenTextSplitter just counts tokens and cuts.

    Args:
        parse_result: Parsed document with full_text and page_map.
        config: ChunkConfig with chunk_size (tokens) and overlap (tokens).

    Returns:
        List of Chunk objects ordered by position in the document.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.overlap,
        length_function=_count_tokens,
        # WHY these separators in this order: try paragraph breaks first, then
        # line breaks, then sentence ends, then words. Preserves the most
        # semantic structure possible within the token budget.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(parse_result.full_text)

    chunks: list[Chunk] = []
    search_start = 0

    for i, chunk_text in enumerate(raw_chunks):
        # WHY _find_char_offset: the splitter strips whitespace, so we need to
        # recover where each chunk actually appears in the original document.
        start_char = _find_char_offset(chunk_text, parse_result.full_text, search_start)
        end_char = start_char + len(chunk_text)

        # Advance search past current match to handle overlapping chunks correctly
        search_start = start_char + 1

        page_numbers = _assign_page_numbers(start_char, end_char, parse_result.page_map)
        token_count = _count_tokens(chunk_text)

        chunks.append(Chunk(
            id=f"{config.name}_{i}",
            text=chunk_text,
            token_count=token_count,
            start_char=start_char,
            end_char=end_char,
            page_numbers=page_numbers,
            config_name=config.name,
            section_header=None,  # Fixed-size chunks don't track headers
        ))

    logger.info(
        "Config %s: %d chunks (size=%d, overlap=%d)",
        config.name, len(chunks), config.chunk_size, config.overlap,
    )
    return chunks


# ===========================================================================
# Semantic Chunking (Config E)
# ===========================================================================

def _chunk_semantic(parse_result: ParseResult, config: ChunkConfig) -> list[Chunk]:
    """Split on Markdown ## and ### headers — structure-aware chunking.

    Each chunk is one document section (from one header to the next). Preserves
    the author's intended organization rather than cutting at arbitrary token
    boundaries.

    Handles edge cases:
    - Sections >512 tokens: subdivided using Config B params (256/64)
    - Sections <32 tokens: merged with the next section
    - No ##/### headers found: falls back to Config B fixed-size chunking

    WHY only ## and ###: # headers are too coarse (entire chapters), #### too
    fine (individual definitions). ## and ### hit the sweet spot for RAG — each
    chunk is a coherent topic.

    Args:
        parse_result: Parsed document with full_text, headers, and page_map.
        config: Config E (is_semantic=True). chunk_size used as max token threshold.

    Returns:
        List of Chunk objects aligned to document section boundaries.
    """
    full_text = parse_result.full_text

    # Filter to only ## (level 2) and ### (level 3) headers
    semantic_headers = [h for h in parse_result.headers if h.level in (2, 3)]

    # WHY fallback: PDFs may have no Markdown headers. Without split points,
    # semantic chunking is meaningless — fall back to the baseline fixed-size.
    if not semantic_headers:
        logger.warning(
            "No ##/### headers found in %s — falling back to Config B fixed-size chunking",
            parse_result.source_path,
        )
        return _chunk_fixed_size(parse_result, SEMANTIC_SUBDIVISION_CONFIG)

    # --- Phase 1: Build raw sections from header positions ---
    # Each section spans from one header to the next (or end of document).
    # Section text includes the header line itself.
    sections: list[tuple[str, str, int, int]] = []  # (header_text, text, doc_start, doc_end)

    # Preamble: text before the first ## or ### header
    if semantic_headers[0].char_offset > 0:
        doc_start, doc_end, text = _extract_section_span(
            full_text, 0, semantic_headers[0].char_offset,
        )
        if text and _count_tokens(text) >= SEMANTIC_MIN_TOKENS:
            sections.append(("(Preamble)", text, doc_start, doc_end))

    for i, header in enumerate(semantic_headers):
        section_end_char = (
            semantic_headers[i + 1].char_offset
            if i + 1 < len(semantic_headers)
            else len(full_text)
        )
        doc_start, doc_end, text = _extract_section_span(
            full_text, header.char_offset, section_end_char,
        )
        if text:
            sections.append((header.text, text, doc_start, doc_end))

    # --- Phase 2: Merge small sections (<32 tokens) with the next section ---
    # WHY merge: tiny sections (e.g., a header with just a name below it) produce
    # low-quality embeddings. Merging gives the embedding model enough context.
    merged: list[tuple[str, str, int, int]] = []
    i = 0
    while i < len(sections):
        header, text, doc_start, doc_end = sections[i]
        tokens = _count_tokens(text)

        while tokens < SEMANTIC_MIN_TOKENS and i + 1 < len(sections):
            i += 1
            _, next_text, _, next_end = sections[i]
            text = text + "\n\n" + next_text
            doc_end = next_end
            tokens = _count_tokens(text)

        merged.append((header, text, doc_start, doc_end))
        i += 1

    # --- Phase 3: Build Chunk objects, subdividing oversized sections ---
    chunks: list[Chunk] = []
    chunk_idx = 0

    for header, text, doc_start, doc_end in merged:
        tokens = _count_tokens(text)

        if tokens > SEMANTIC_MAX_TOKENS:
            # WHY subdivide with Config B: a 2000-token section would dominate
            # the embedding space. Splitting at 256/64 keeps sub-chunks comparable
            # to other configs while preserving the section_header link.
            sub_results = _subdivide_section(
                text, full_text, doc_start, doc_end, parse_result.page_map,
            )
            for sub_text, abs_start, abs_end, pages in sub_results:
                chunks.append(Chunk(
                    id=f"{config.name}_{chunk_idx}",
                    text=sub_text,
                    token_count=_count_tokens(sub_text),
                    start_char=abs_start,
                    end_char=abs_end,
                    page_numbers=pages,
                    config_name=config.name,
                    section_header=header,
                ))
                chunk_idx += 1
        else:
            pages = _assign_page_numbers(doc_start, doc_end, parse_result.page_map)
            chunks.append(Chunk(
                id=f"{config.name}_{chunk_idx}",
                text=text,
                token_count=tokens,
                start_char=doc_start,
                end_char=doc_end,
                page_numbers=pages,
                config_name=config.name,
                section_header=header,
            ))
            chunk_idx += 1

    logger.info(
        "Config E: %d chunks from %d sections (%d after merging)",
        len(chunks), len(sections), len(merged),
    )
    return chunks


# ===========================================================================
# Semantic Chunking Helpers
# ===========================================================================

def _extract_section_span(
    full_text: str, raw_start: int, raw_end: int,
) -> tuple[int, int, str]:
    """Extract a section's stripped text and compute precise document offsets.

    WHY manual offset math instead of find(): we know exactly where the raw
    span starts in the document, so adjusting for stripped whitespace is
    deterministic — no risk of matching the wrong position.

    Returns:
        (doc_start, doc_end, stripped_text) where doc_start/doc_end point to
        the first/last non-whitespace characters in the raw span.
    """
    raw = full_text[raw_start:raw_end]
    text = raw.strip()
    if not text:
        return raw_start, raw_end, ""

    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw) - len(raw.rstrip())
    doc_start = raw_start + leading
    doc_end = raw_end - trailing
    return doc_start, doc_end, text


def _subdivide_section(
    section_text: str,
    full_text: str,
    doc_start: int,
    doc_end: int,
    page_map: list[tuple[int, int, int]],
) -> list[tuple[str, int, int, list[int]]]:
    """Subdivide an oversized section using Config B params (256/64).

    For non-merged sections, sub-chunks are direct substrings of full_text and
    exact offsets are recovered via find(). For merged sections (where artificial
    \\n\\n separators cause mismatch), falls back to proportional offset estimation
    within the document span.

    Returns:
        List of (text, abs_start, abs_end, page_numbers) tuples.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SEMANTIC_SUBDIVISION_CONFIG.chunk_size,
        chunk_overlap=SEMANTIC_SUBDIVISION_CONFIG.overlap,
        length_function=_count_tokens,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    sub_texts = splitter.split_text(section_text)

    results: list[tuple[str, int, int, list[int]]] = []
    doc_span = doc_end - doc_start
    search_start = doc_start

    for sub_text in sub_texts:
        # Try exact match in full_text first (works for non-merged sections)
        idx = full_text.find(sub_text, search_start)
        if idx != -1:
            abs_start = idx
        else:
            # WHY proportional fallback: for merged sections, the merge inserts
            # artificial "\n\n" that doesn't match full_text whitespace. Instead
            # of a stale position, estimate using the sub-chunk's position within
            # the section text mapped proportionally to the document span.
            sec_pos = section_text.find(sub_text)
            if sec_pos != -1 and len(section_text) > 0:
                fraction = sec_pos / len(section_text)
                abs_start = doc_start + int(fraction * doc_span)
            else:
                abs_start = search_start

        abs_end = min(abs_start + len(sub_text), doc_end)
        # Ensure end > start (Chunk model validator requirement)
        abs_end = max(abs_end, abs_start + 1)
        search_start = abs_start + 1

        pages = _assign_page_numbers(abs_start, abs_end, page_map)
        results.append((sub_text, abs_start, abs_end, pages))

    return results


# ===========================================================================
# Shared Helper Functions
# ===========================================================================

def _find_char_offset(chunk_text: str, full_text: str, search_start: int) -> int:
    """Find where chunk_text appears in full_text, scanning forward.

    WHY scan forward: chunks are produced in document order by the splitter.
    Searching from the last found position is O(chunk_size) per call instead
    of O(full_text) — much faster for large documents.

    Java/TS parallel: like String.indexOf(target, fromIndex) in Java.

    Args:
        chunk_text: The chunk's text content (may have stripped whitespace).
        full_text: The complete document text.
        search_start: Position to start searching from.

    Returns:
        Character offset where chunk_text starts in full_text.
    """
    # Primary: scan forward from last position
    idx = full_text.find(chunk_text, search_start)
    if idx != -1:
        return idx

    # Fallback: search from beginning (handles edge cases with overlap)
    idx = full_text.find(chunk_text)
    if idx != -1:
        return idx

    # Last resort: use search_start as best estimate. Prevents crashes but
    # produces imprecise offsets — logged for debugging.
    logger.warning(
        "Could not find exact offset for chunk '%s...' — using estimate at %d",
        chunk_text[:50], search_start,
    )
    return search_start


def _assign_page_numbers(
    start_char: int, end_char: int, page_map: list[tuple[int, int, int]],
) -> list[int]:
    """Determine which page(s) a chunk spans based on character offsets.

    WHY check the full range: chunks near page boundaries can straddle two or
    more pages. Using only start_char would lose that information, which matters
    for the Streamlit UI and QA quality analysis.

    Java/TS parallel: like finding all intervals that overlap with [start, end)
    — an interval overlap query.

    Args:
        start_char: Where the chunk starts in the document (inclusive).
        end_char: Where the chunk ends in the document (exclusive).
        page_map: Sorted list of (page_start, page_end, page_num) from parser.

    Returns:
        Sorted list of unique page numbers this chunk spans.
    """
    if not page_map:
        return [0]

    # Check which page spans overlap with [start_char, end_char)
    pages: set[int] = set()
    for page_start, page_end, page_num in page_map:
        if page_start < end_char and page_end > start_char:
            pages.add(page_num)

    # Fallback: if no overlap found (shouldn't happen), use binary search
    if not pages:
        pages.add(get_page_for_offset(page_map, start_char))

    return sorted(pages)
