"""PDF extraction with PyMuPDF (fitz) and text cleaning.

Pipeline:
    extract_pdf()  →  Document (all pages, cleaned text, metadata)
    extract_all_pdfs()  →  list[Document]

Per-page cleaning (in order):
    1. remove_headers_footers()  — drop running headers/page numbers
    2. clean_text()              — fix ligatures, rejoin hyphenated words, collapse whitespace
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF — `pip install pymupdf` installs as `fitz`

from src.schemas import Document, DocumentMetadata, PageInfo

# ---------------------------------------------------------------------------
# Pre-compiled regexes — compiled once at module load, not per-call.
# Matters when batch-processing hundreds of pages.
# ---------------------------------------------------------------------------

# PDF fonts sometimes store fi/fl as single unicode ligature glyphs
_LIGATURES: dict[str, str] = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

# PDFs break "trans-\nformer" across lines; NLP tokenizers see two tokens
_HYPHENATION_RE = re.compile(r"(\w+)-\n(\w+)")

_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")
_EXCESSIVE_SPACES_RE = re.compile(r"[ \t]{2,}")

# Header/footer heuristics
_STANDALONE_DIGIT_RE = re.compile(r"^\s*\d+\s*$")               # "  3  "
_PAGE_N_RE = re.compile(r"^\s*[Pp]age\s+\d+", re.IGNORECASE)    # "Page 3", "page 3 of 10"
_N_OF_M_RE = re.compile(r"^\s*\d+\s+of\s+\d+\s*$", re.IGNORECASE)  # "3 of 10"
_ARXIV_RE = re.compile(r"^\s*arXiv:", re.IGNORECASE)             # "arXiv:1706.03762v5"


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Fix ligatures, rejoin hyphenated line-breaks, collapse whitespace.

    Returns text safe for NLP tokenization.
    """
    for ligature, replacement in _LIGATURES.items():
        text = text.replace(ligature, replacement)

    # r"\1\2" joins captured groups — "re-\njoin" → "rejoin"
    text = _HYPHENATION_RE.sub(r"\1\2", text)

    text = _EXCESSIVE_NEWLINES_RE.sub("\n\n", text)
    text = _EXCESSIVE_SPACES_RE.sub(" ", text)

    return text.strip()


def _is_header_or_footer(line: str, page_number: int, total_pages: int) -> bool:
    """Check if a line matches common header/footer patterns.

    page_number/total_pages unused but kept for call-site symmetry with
    remove_headers_footers().
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

    # Short ALL-CAPS line — journal/conference headers like "NEURAL INFORMATION PROCESSING SYSTEMS"
    # ≥3 chars avoids false-positives on single-letter section labels ("A")
    if 3 <= len(stripped) <= 60 and stripped.upper() == stripped and stripped.replace(" ", "").isalpha():
        return True

    return False


def remove_headers_footers(text: str, page_number: int, total_pages: int) -> str:
    """Strip running headers/footers by inspecting first 3 and last 3 lines.

    Skips pages with ≤6 lines — removing 6 of 6 lines would nuke the page.
    """
    lines = text.split("\n")

    if len(lines) <= 6:
        return text

    head_keep = []
    for i, line in enumerate(lines[:3]):
        if not _is_header_or_footer(line, page_number, total_pages):
            head_keep = lines[i:]
            break
    else:
        head_keep = lines[3:]

    foot_keep = []
    for i, line in enumerate(reversed(head_keep[-3:])):
        if not _is_header_or_footer(line, page_number, total_pages):
            cut = len(head_keep) - i
            foot_keep = head_keep[:cut]
            break
    else:
        foot_keep = head_keep[:-3] if len(head_keep) > 3 else head_keep

    return "\n".join(foot_keep)


def extract_pdf(pdf_path: str | Path) -> Document:
    """Extract text and metadata from a single PDF.

    Raises FileNotFoundError if path is missing, ValueError if no text found.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    doc = fitz.open(str(path))  # type: ignore[attr-defined]

    # grab metadata before close() — accessing after close raises RuntimeError
    raw_meta = doc.metadata or {}
    total_pages = len(doc)

    pages: list[PageInfo] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        raw_text = page.get_text()  # "" for image-only pages
        cleaned = clean_text(remove_headers_footers(raw_text, page_num, total_pages))
        pages.append(
            PageInfo(
                page_number=page_num,
                text=cleaned,
                char_count=len(cleaned),
            )
        )

    doc.close()

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
    """Extract all PDFs in a directory. Skips files that fail (logs warning)."""
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