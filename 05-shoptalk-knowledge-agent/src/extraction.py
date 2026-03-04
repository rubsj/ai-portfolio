"""PDF extraction with PyMuPDF (fitz) and text cleaning utilities.

Pipeline:
    extract_pdf()  →  Document (all pages, cleaned text, metadata)
    extract_all_pdfs()  →  list[Document]

Text cleaning stages (applied per page in order):
    1. remove_headers_footers()  — drop running headers/page numbers from top+bottom
    2. clean_text()              — fix ligatures, rejoin hyphenated words, collapse whitespace
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF — `pip install pymupdf` installs as `fitz`

from src.schemas import Document, DocumentMetadata, PageInfo

# ---------------------------------------------------------------------------
# Pre-compiled regexes (module-level — compiled once, not per-call)
# WHY: re.compile() at module load avoids recompiling the same pattern for
# every page of every document, which matters for batch processing.
# ---------------------------------------------------------------------------

# Ligature map: unicode ligatures → ASCII equivalents
# Ligatures appear in PDFs because some fonts store fi/fl as a single glyph.
_LIGATURES: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# Match a hyphenated line-break: word-\nword → rejoin as one word
# WHY: PDFs often break "trans-\nformer" across lines; NLP tokenizers see two tokens.
_HYPHENATION_RE = re.compile(r"(\w+)-\n(\w+)")

# Collapse 3+ newlines to 2 (paragraph break)
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")

# Collapse 2+ spaces/tabs to a single space
_EXCESSIVE_SPACES_RE = re.compile(r"[ \t]{2,}")

# Header/footer heuristics
_STANDALONE_DIGIT_RE = re.compile(r"^\s*\d+\s*$")               # standalone page number: "  3  "
_PAGE_N_RE = re.compile(r"^\s*[Pp]age\s+\d+", re.IGNORECASE)    # "Page 3", "page 3 of 10"
_N_OF_M_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE)  # "3 of 10"
_ARXIV_RE = re.compile(r"^\s*arXiv:", re.IGNORECASE)             # "arXiv:1706.03762v5"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalise extracted PDF text.

    Stages (in order):
        1. Ligature expansion: ﬁ→fi, ﬂ→fl, ﬃ→ffi, ﬄ→ffl, ﬀ→ff
        2. Hyphenation rejoin: "trans-\\njoin" → "transjoin"
        3. Collapse excessive newlines (3+ → 2)
        4. Collapse excessive spaces/tabs (2+ → 1)
        5. Strip leading/trailing whitespace

    Args:
        text: Raw text from PyMuPDF (may contain ligatures and hyphenation).

    Returns:
        Cleaned text string.
    """
    # Stage 1: ligatures
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)

    # Stage 2: hyphenated line-breaks
    # r"\1\2" joins the two captured groups without a space — "re-\njoin" → "rejoin"
    text = _HYPHENATION_RE.sub(r"\1\2", text)

    # Stage 3: collapse ≥3 consecutive newlines to exactly 2 (paragraph separator)
    text = _EXCESSIVE_NEWLINES_RE.sub("\n\n", text)

    # Stage 4: collapse repeated spaces/tabs
    text = _EXCESSIVE_SPACES_RE.sub(" ", text)

    return text.strip()


def _is_header_or_footer(line: str, page_number: int, total_pages: int) -> bool:
    """Heuristically determine if a line is a running header or footer.

    Heuristics (any match → True):
        - Standalone digits (page numbers)
        - "Page N" or "N of M" patterns
        - Lines starting with "arXiv:"
        - Short ALL-CAPS lines (≤60 chars, ≥3 chars) — common in academic paper headers

    Args:
        line: A single line of text from the top or bottom of a page.
        page_number: 0-indexed page number (unused currently but kept for future use).
        total_pages: Total number of pages (unused currently but kept for future use).

    Returns:
        True if the line looks like a header or footer.
    """
    stripped = line.strip()
    if not stripped:
        return False

    if _STANDALONE_DIGIT_RE.match(stripped):
        return True
    if _PAGE_N_RE.match(stripped):
        return True
    if _N_OF_M_RE.match(stripped):
        return True
    if _ARXIV_RE.match(stripped):
        return True

    # Short ALL-CAPS line — typical for journal/conference headers like "NEURAL INFORMATION PROCESSING SYSTEMS"
    # WHY: ≥3 chars avoids false-positives on single-letter section labels (e.g., "A")
    if 3 <= len(stripped) <= 60 and stripped.upper() == stripped and stripped.replace(" ", "").isalpha():
        return True

    return False


def remove_headers_footers(text: str, page_number: int, total_pages: int) -> str:
    """Remove running headers and footers from page text.

    Only inspects the first 3 and last 3 lines — avoids removing body text.
    Skips pages with ≤6 lines (not enough content to safely remove lines).

    Args:
        text: Full page text before cleaning.
        page_number: 0-indexed page number.
        total_pages: Total number of pages.

    Returns:
        Page text with header/footer lines removed.
    """
    lines = text.split("\n")

    # WHY: short pages (e.g., chapter headings, section breaks) often have ≤6 lines.
    # Removing 3 top + 3 bottom lines from a 6-line page would leave nothing.
    if len(lines) <= 6:
        return text

    # Check and remove header lines (first 3)
    head_keep = []
    for i, line in enumerate(lines[:3]):
        if not _is_header_or_footer(line, page_number, total_pages):
            # First non-header line found — keep from this line onward
            head_keep = lines[i:]
            break
    else:
        # All first 3 lines are headers — keep everything from line 3 onwards
        head_keep = lines[3:]

    # Check and remove footer lines (last 3 of the possibly-trimmed list)
    foot_keep = []
    for i, line in enumerate(reversed(head_keep[-3:])):
        if not _is_header_or_footer(line, page_number, total_pages):
            cut = len(head_keep) - i  # keep up to this index
            foot_keep = head_keep[:cut]
            break
    else:
        foot_keep = head_keep[:-3] if len(head_keep) > 3 else head_keep

    return "\n".join(foot_keep)


def extract_pdf(pdf_path: str | Path) -> Document:
    """Extract text and metadata from a single PDF using PyMuPDF.

    Pipeline per page:
        1. page.get_text() — raw text extraction
        2. remove_headers_footers() — strip running headers/page numbers
        3. clean_text() — ligatures, hyphenation, whitespace

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Document with all pages and full concatenated content.

    Raises:
        FileNotFoundError: If pdf_path does not exist.
        ValueError: If the PDF has no extractable text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))  # type: ignore[attr-defined]

    # WHY: read metadata BEFORE doc.close() — accessing it after close raises RuntimeError
    raw_meta = doc.metadata or {}
    total_pages = len(doc)

    pages: list[PageInfo] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        raw_text = page.get_text()  # returns "" for image-only pages
        cleaned = clean_text(remove_headers_footers(raw_text, page_num, total_pages))
        pages.append(
            PageInfo(
                page_number=page_num,
                text=cleaned,
                char_count=len(cleaned),
            )
        )

    doc.close()

    # Join all pages with double-newline paragraph separator
    full_content = "\n\n".join(p.text for p in pages)
    if not full_content.strip():
        raise ValueError(f"No extractable text found in PDF: {path}")

    metadata = DocumentMetadata(
        source=str(path),
        title=raw_meta.get("title", "") or "",
        author=raw_meta.get("author", "") or "",
        page_count=total_pages,
    )

    return Document(content=full_content, metadata=metadata, pages=pages)


def extract_all_pdfs(pdf_dir: str | Path) -> list[Document]:
    """Extract all PDFs in a directory.

    Args:
        pdf_dir: Directory containing .pdf files.

    Returns:
        List of Documents, one per PDF. Skips files that fail extraction (logs warning).
    """
    import logging

    logger = logging.getLogger(__name__)
    pdf_dir = Path(pdf_dir)
    documents: list[Document] = []

    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            doc = extract_pdf(pdf_path)
            documents.append(doc)
            logger.info(f"Extracted {pdf_path.name}: {doc.metadata.page_count} pages")
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping {pdf_path.name}: {e}")

    return documents
