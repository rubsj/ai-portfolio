"""Document parser — factory pattern for PDF and Kaggle Markdown inputs.

Parses documents into a uniform ParseResult structure that the chunker consumes.
Supports two formats:
  1. Kaggle Enterprise RAG Markdown (.md) — our primary input format
  2. PDF via PyMuPDF (.pdf) — for extensibility ("RAG for Any PDF")

Java/TS parallel: like a factory pattern — one public function dispatches to
the right parser based on file extension, returning the same interface either way.
Similar to Java's DocumentParserFactory with strategy pattern.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# ===========================================================================
# Data Structures — dataclasses, not Pydantic
# WHY dataclass over Pydantic: these are internal pipeline objects, not
# external-facing data contracts. No need for JSON serialization or validation.
# Java/TS parallel: plain Java records or TypeScript interfaces.
# ===========================================================================

@dataclass
class HeaderInfo:
    """A single header found in the document."""
    level: int               # 1 for #, 2 for ##, etc.
    text: str                # Header text (stripped of # and whitespace)
    char_offset: int         # Character offset in the cleaned text
    page_number: int         # Which page this header appears on


@dataclass
class ParseResult:
    """Uniform output from any parser — consumed by the chunker.

    full_text is the cleaned document text (no HTML, no image refs, no page
    separators). page_map maps character ranges to page numbers. headers
    lists all extracted headers with their positions.
    """
    full_text: str
    page_map: list[tuple[int, int, int]] = field(default_factory=list)  # (start_char, end_char, page_num)
    headers: list[HeaderInfo] = field(default_factory=list)
    source_path: Path = field(default_factory=Path)
    num_pages: int = 0


# ===========================================================================
# Public API
# ===========================================================================

def parse_document(file_path: Path) -> ParseResult:
    """Factory dispatcher — routes to the right parser based on file extension.

    Java/TS parallel: like a factory method that returns the same interface
    regardless of which concrete implementation handles the file.

    Args:
        file_path: Path to a .md or .pdf file.

    Returns:
        ParseResult with cleaned text, page map, and headers.

    Raises:
        ValueError: If file extension is not .md or .pdf.
        FileNotFoundError: If the file doesn't exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".md":
        return _parse_markdown(path)
    elif suffix == ".pdf":
        return _parse_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .md or .pdf")


def get_page_for_offset(page_map: list[tuple[int, int, int]], char_offset: int) -> int:
    """Binary search to find which page a character offset belongs to.

    WHY bisect_right: O(log n) lookup instead of O(n) linear scan. For a 50-page
    document with hundreds of chunks, this matters.
    Java/TS parallel: like Collections.binarySearch() on a sorted list.

    Args:
        page_map: Sorted list of (start_char, end_char, page_num) tuples.
        char_offset: The character position to look up.

    Returns:
        Page number containing the offset. Falls back to last page if offset
        is beyond all ranges (shouldn't happen with well-formed data).
    """
    if not page_map:
        return 0

    # Extract start positions for binary search
    starts = [entry[0] for entry in page_map]

    # bisect_right returns the insertion point — subtract 1 to get the page
    # whose range contains char_offset. Example: starts=[0, 500, 1200],
    # char_offset=600 → bisect_right returns 2 → index 1 → page_map[1].
    idx = bisect_right(starts, char_offset) - 1

    # Clamp to valid range
    idx = max(0, min(idx, len(page_map) - 1))
    return page_map[idx][2]


# ===========================================================================
# Kaggle Markdown Parser
# ===========================================================================

# WHY compiled regex: these run once per line across 1000+ line files.
# Compiling avoids re-parsing the pattern on every call.
# Java/TS parallel: like Pattern.compile() in Java.

# Page separator: {N} followed by 48 dashes (Kaggle format)
_PAGE_SEP_RE = re.compile(r"^\{(\d+)\}-{48}$")

# HTML tags to strip: <span ...>, </span>, <br>, <br/>
_HTML_TAG_RE = re.compile(r"</?(?:span|br)\b[^>]*>")

# Image refs: ![alt](_page_N_...) — Kaggle embeds non-existent image paths
_IMAGE_REF_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Markdown headers: 1-6 # characters at start of line
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _parse_markdown(file_path: Path) -> ParseResult:
    """Parse Kaggle Enterprise RAG Markdown format.

    The Kaggle format has:
      - Page separators: {N}------------------------------------------------ (48 dashes)
      - HTML tags: <span id="...">, <br>, </span>
      - Image refs: ![](_page_N_Picture_M.jpeg)
      - Markdown headers: #, ##, ###, ####

    We strip artifacts, extract headers with char offsets, and build a page map
    that lets the chunker assign page numbers to each chunk.
    """
    raw_text = file_path.read_text(encoding="utf-8")
    raw_lines = raw_text.splitlines()

    # Pass 1: identify page boundaries and clean lines
    cleaned_lines: list[str] = []
    # Track which page each cleaned line belongs to
    line_pages: list[int] = []
    current_page = 0

    for line in raw_lines:
        # Check for page separator
        sep_match = _PAGE_SEP_RE.match(line.strip())
        if sep_match:
            current_page = int(sep_match.group(1))
            continue  # Don't include separator in output

        # Strip image references (entire line if it's only an image ref)
        line = _IMAGE_REF_RE.sub("", line)

        # Strip HTML tags
        line = _HTML_TAG_RE.sub("", line)

        # Keep the line (even if blank — preserves paragraph structure)
        cleaned_lines.append(line)
        line_pages.append(current_page)

    # Join cleaned lines into full text
    full_text = "\n".join(cleaned_lines)

    # Pass 2: build page_map and extract headers from the cleaned text.
    # We need char offsets, so we walk through cleaned_lines tracking position.
    page_map: list[tuple[int, int, int]] = []
    headers: list[HeaderInfo] = []

    char_pos = 0
    # Track page spans: accumulate chars per page, emit (start, end, page) entries
    current_page_start = 0
    current_page_num = line_pages[0] if line_pages else 0

    for i, line in enumerate(cleaned_lines):
        line_start = char_pos
        page_num = line_pages[i]

        # Detect page transitions → close previous page span, start new one
        if page_num != current_page_num:
            if char_pos > current_page_start:
                page_map.append((current_page_start, char_pos, current_page_num))
            current_page_start = char_pos
            current_page_num = page_num

        # Extract headers
        header_match = _HEADER_RE.match(line.strip())
        if header_match:
            level = len(header_match.group(1))
            # Strip bold markers (**) from header text — Kaggle wraps headers in bold
            header_text = header_match.group(2).strip().strip("*").strip()
            if header_text:  # Skip empty headers
                headers.append(HeaderInfo(
                    level=level,
                    text=header_text,
                    char_offset=line_start,
                    page_number=page_num,
                ))

        # Advance position: line length + 1 for the \n separator
        char_pos += len(line) + 1  # +1 for newline

    # Close final page span
    if char_pos > current_page_start:
        page_map.append((current_page_start, char_pos, current_page_num))

    # Count unique pages
    unique_pages = {entry[2] for entry in page_map}
    num_pages = len(unique_pages)

    return ParseResult(
        full_text=full_text,
        page_map=page_map,
        headers=headers,
        source_path=file_path,
        num_pages=num_pages,
    )


# ===========================================================================
# PDF Parser (PyMuPDF)
# ===========================================================================

def _parse_pdf(file_path: Path) -> ParseResult:
    """Parse PDF using PyMuPDF (fitz).

    Extracts text page-by-page, builds page_map from character offsets,
    and extracts headers heuristically (lines that look like headings based
    on font size — PyMuPDF exposes this via text blocks).

    WHY PyMuPDF over pdfplumber/PyPDF2: faster extraction, better layout
    handling, and we can access font metadata for header detection.
    """
    doc = fitz.open(file_path)

    full_text_parts: list[str] = []
    page_map: list[tuple[int, int, int]] = []
    headers: list[HeaderInfo] = []
    char_pos = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")

        if not page_text.strip():
            continue

        page_start = char_pos
        full_text_parts.append(page_text)
        char_pos += len(page_text)

        page_map.append((page_start, char_pos, page_num))

        # Heuristic header extraction from PDF: look for short lines in
        # larger font sizes. PyMuPDF's get_text("dict") gives font info.
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # type 0 = text block
                continue
            for line_data in block.get("lines", []):
                for span in line_data.get("spans", []):
                    text = span.get("text", "").strip()
                    font_size = span.get("size", 0)
                    # Heuristic: font size > 14 and line < 200 chars = likely header
                    if font_size > 14 and 0 < len(text) < 200:
                        # Find approximate char offset for this header text
                        header_offset = full_text_parts[-1].find(text)
                        if header_offset >= 0:
                            headers.append(HeaderInfo(
                                level=1 if font_size > 18 else 2 if font_size > 14 else 3,
                                text=text,
                                char_offset=page_start + header_offset,
                                page_number=page_num,
                            ))

    # WHY save page count before close: len(doc) raises ValueError after close()
    num_pages = len(doc) if page_map else 0
    doc.close()

    full_text = "".join(full_text_parts)

    return ParseResult(
        full_text=full_text,
        page_map=page_map,
        headers=headers,
        source_path=file_path,
        num_pages=num_pages,
    )
