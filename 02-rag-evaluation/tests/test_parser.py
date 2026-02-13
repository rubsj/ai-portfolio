"""Tests for parser.py — factory dispatcher, Markdown/PDF parsing, helpers.

Uses synthetic fixtures written to tmp_path for fast, deterministic tests
that don't depend on real Kaggle input files.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF — used to create minimal PDF fixtures
import pytest

from src.parser import (
    _HEADER_RE,
    _HTML_TAG_RE,
    _IMAGE_REF_RE,
    _PAGE_SEP_RE,
    _parse_markdown,
    get_page_for_offset,
    parse_document,
)


# ===========================================================================
# Factory Dispatcher — parse_document()
# ===========================================================================

class TestParseDocument:
    """Test the factory dispatcher routes correctly and handles errors."""

    def test_md_file_dispatches_to_markdown_parser(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("## Section\n\nSome content here.", encoding="utf-8")
        result = parse_document(md_file)
        assert result.full_text.strip() != ""
        assert result.source_path == md_file

    def test_pdf_file_dispatches_to_pdf_parser(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "test.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello PDF world")
        doc.save(str(pdf_file))
        doc.close()

        result = parse_document(pdf_file)
        assert "Hello PDF world" in result.full_text
        assert result.source_path == pdf_file

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("plain text")
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_document(txt_file)

    def test_nonexistent_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Document not found"):
            parse_document(Path("/does/not/exist.md"))

    def test_case_insensitive_extension(self, tmp_path: Path) -> None:
        """File with .MD (uppercase) should still be dispatched correctly."""
        md_file = tmp_path / "test.MD"
        md_file.write_text("## Title\n\nContent.")
        result = parse_document(md_file)
        assert result.full_text.strip() != ""


# ===========================================================================
# Binary Search — get_page_for_offset()
# ===========================================================================

class TestGetPageForOffset:
    """Test binary search for page lookup by character offset."""

    def test_empty_page_map_returns_zero(self) -> None:
        assert get_page_for_offset([], 100) == 0

    def test_offset_in_first_page(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1), (1000, 1500, 2)]
        assert get_page_for_offset(page_map, 250) == 0

    def test_offset_in_middle_page(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1), (1000, 1500, 2)]
        assert get_page_for_offset(page_map, 750) == 1

    def test_offset_in_last_page(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1), (1000, 1500, 2)]
        assert get_page_for_offset(page_map, 1250) == 2

    def test_offset_at_exact_boundary(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1)]
        assert get_page_for_offset(page_map, 500) == 1

    def test_offset_at_start(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1)]
        assert get_page_for_offset(page_map, 0) == 0

    def test_offset_beyond_all_ranges(self) -> None:
        """Offset past last range should return last page (clamp behavior)."""
        page_map = [(0, 500, 0), (500, 1000, 1)]
        assert get_page_for_offset(page_map, 5000) == 1

    def test_single_page_map(self) -> None:
        page_map = [(0, 1000, 0)]
        assert get_page_for_offset(page_map, 500) == 0


# ===========================================================================
# Markdown Parser — _parse_markdown()
# ===========================================================================

class TestParseMarkdown:
    """Test Kaggle Markdown parsing: page seps, HTML stripping, headers."""

    def test_basic_text_extracted(self, tmp_path: Path) -> None:
        md_file = tmp_path / "basic.md"
        md_file.write_text("Hello world\n\nSecond paragraph.", encoding="utf-8")
        result = _parse_markdown(md_file)
        assert "Hello world" in result.full_text
        assert "Second paragraph." in result.full_text

    def test_page_separator_stripped(self, tmp_path: Path) -> None:
        """Page separator line should not appear in output text."""
        content = "Page zero content.\n{1}------------------------------------------------\nPage one content."
        md_file = tmp_path / "pages.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert "------------------------------------------------" not in result.full_text
        assert "{1}" not in result.full_text
        assert "Page zero content." in result.full_text
        assert "Page one content." in result.full_text

    def test_page_map_from_separators(self, tmp_path: Path) -> None:
        """Multiple page separators create correct page_map entries."""
        content = "Page zero.\n{1}------------------------------------------------\nPage one.\n{2}------------------------------------------------\nPage two."
        md_file = tmp_path / "multi_page.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        # Should have 3 pages (0, 1, 2)
        assert result.num_pages == 3
        page_nums = [entry[2] for entry in result.page_map]
        assert 0 in page_nums
        assert 1 in page_nums
        assert 2 in page_nums

    def test_html_span_tags_stripped(self, tmp_path: Path) -> None:
        content = '<span id="anchor">Important</span> text here.'
        md_file = tmp_path / "html.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert "<span" not in result.full_text
        assert "</span>" not in result.full_text
        assert "Important" in result.full_text
        assert "text here." in result.full_text

    def test_html_br_tags_stripped(self, tmp_path: Path) -> None:
        content = "Line one<br>Line two<br/>Line three"
        md_file = tmp_path / "br.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert "<br>" not in result.full_text
        assert "<br/>" not in result.full_text

    def test_image_refs_stripped(self, tmp_path: Path) -> None:
        content = "Before image.\n![alt text](_page_1_Picture_1.jpeg)\nAfter image."
        md_file = tmp_path / "images.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert "![" not in result.full_text
        assert "_page_1_Picture_1.jpeg" not in result.full_text
        assert "Before image." in result.full_text
        assert "After image." in result.full_text

    def test_headers_level_1(self, tmp_path: Path) -> None:
        md_file = tmp_path / "h1.md"
        md_file.write_text("# Main Title\n\nContent.", encoding="utf-8")
        result = _parse_markdown(md_file)
        assert len(result.headers) == 1
        assert result.headers[0].level == 1
        assert result.headers[0].text == "Main Title"

    def test_headers_level_2_and_3(self, tmp_path: Path) -> None:
        content = "## Section\n\nContent.\n\n### Subsection\n\nMore content."
        md_file = tmp_path / "h23.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert len(result.headers) == 2
        assert result.headers[0].level == 2
        assert result.headers[0].text == "Section"
        assert result.headers[1].level == 3
        assert result.headers[1].text == "Subsection"

    def test_header_level_4(self, tmp_path: Path) -> None:
        md_file = tmp_path / "h4.md"
        md_file.write_text("#### Deep Header\n\nContent.", encoding="utf-8")
        result = _parse_markdown(md_file)
        assert len(result.headers) == 1
        assert result.headers[0].level == 4

    def test_header_char_offset_correct(self, tmp_path: Path) -> None:
        """Header char_offset should point to the header line in full_text."""
        content = "Intro text.\n\n## The Section\n\nSection body."
        md_file = tmp_path / "offset.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        header = result.headers[0]
        # The text at char_offset should start with "## The Section"
        assert result.full_text[header.char_offset:].startswith("## The Section")

    def test_header_page_number_correct(self, tmp_path: Path) -> None:
        """Headers after a page separator should have the correct page number."""
        content = "## Page Zero Header\n\nContent.\n{1}------------------------------------------------\n## Page One Header\n\nMore."
        md_file = tmp_path / "header_pages.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert len(result.headers) == 2
        assert result.headers[0].page_number == 0
        assert result.headers[1].page_number == 1

    def test_bold_markers_stripped_from_headers(self, tmp_path: Path) -> None:
        """Kaggle wraps some headers in bold — **Header** should become Header."""
        md_file = tmp_path / "bold.md"
        md_file.write_text("## **Bold Header**\n\nContent.", encoding="utf-8")
        result = _parse_markdown(md_file)
        assert result.headers[0].text == "Bold Header"

    def test_empty_headers_skipped(self, tmp_path: Path) -> None:
        """A header line with only # and whitespace should be ignored."""
        md_file = tmp_path / "empty_h.md"
        md_file.write_text("## \n\nContent.\n\n## Real Header\n\nMore.", encoding="utf-8")
        result = _parse_markdown(md_file)
        # Only the non-empty header should be captured
        assert len(result.headers) == 1
        assert result.headers[0].text == "Real Header"

    def test_blank_lines_preserved(self, tmp_path: Path) -> None:
        """Blank lines maintain paragraph structure in full_text."""
        content = "Paragraph one.\n\nParagraph two."
        md_file = tmp_path / "blanks.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert "\n\n" in result.full_text

    def test_num_pages_counts_unique_pages(self, tmp_path: Path) -> None:
        content = "P0.\n{1}------------------------------------------------\nP1."
        md_file = tmp_path / "numpages.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        assert result.num_pages == 2

    def test_no_page_separators_single_page(self, tmp_path: Path) -> None:
        """File with no page separators → all text on page 0, 1 page."""
        md_file = tmp_path / "single.md"
        md_file.write_text("Just simple content.\n\nMore text.", encoding="utf-8")
        result = _parse_markdown(md_file)
        assert result.num_pages == 1
        assert all(entry[2] == 0 for entry in result.page_map)

    def test_combined_kaggle_format(self, tmp_path: Path) -> None:
        """Integration test: page seps + HTML + images + headers all together."""
        content = (
            '<span id="a1">## **Introduction**</span>\n'
            "\n"
            "Some intro text with <br> line break.\n"
            "![img](_page_0_Picture_1.jpeg)\n"
            "\n"
            "{1}------------------------------------------------\n"
            "## Methods\n"
            "\n"
            "Method details here."
        )
        md_file = tmp_path / "kaggle.md"
        md_file.write_text(content, encoding="utf-8")
        result = _parse_markdown(md_file)
        # HTML stripped
        assert "<span" not in result.full_text
        assert "<br>" not in result.full_text
        # Image stripped
        assert "![img]" not in result.full_text
        # Page separator stripped
        assert "---" not in result.full_text
        # Headers extracted
        assert len(result.headers) == 2
        assert result.headers[0].text == "Introduction"
        assert result.headers[1].text == "Methods"
        # Pages correct
        assert result.num_pages == 2


# ===========================================================================
# PDF Parser
# ===========================================================================

class TestParsePdf:
    """Test PDF parsing using minimal PyMuPDF-created fixtures."""

    def test_basic_pdf_text_extraction(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "basic.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Test document content")
        doc.save(str(pdf_file))
        doc.close()

        result = parse_document(pdf_file)
        assert "Test document content" in result.full_text
        assert len(result.page_map) >= 1

    def test_multi_page_pdf(self, tmp_path: Path) -> None:
        pdf_file = tmp_path / "multi.pdf"
        doc = fitz.open()
        for i in range(3):
            page = doc.new_page()
            page.insert_text((72, 72), f"Content on page {i}")
        doc.save(str(pdf_file))
        doc.close()

        result = parse_document(pdf_file)
        assert result.num_pages == 3
        assert "Content on page 0" in result.full_text
        assert "Content on page 2" in result.full_text

    def test_empty_page_skipped(self, tmp_path: Path) -> None:
        """Pages with no text content should be skipped in page_map."""
        pdf_file = tmp_path / "empty_page.pdf"
        doc = fitz.open()
        doc.new_page()  # Empty page
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Content on page 2")
        doc.save(str(pdf_file))
        doc.close()

        result = parse_document(pdf_file)
        # Only the non-empty page should be in page_map
        assert len(result.page_map) == 1
        assert "Content on page 2" in result.full_text

    def test_pdf_header_extraction_large_font(self, tmp_path: Path) -> None:
        """Text in large font (>14pt) should be extracted as headers."""
        pdf_file = tmp_path / "headers.pdf"
        doc = fitz.open()
        page = doc.new_page()
        # Insert large font text (>18pt → level 1)
        page.insert_text((72, 72), "Big Title", fontsize=20)
        # Insert medium font text (>14pt → level 2)
        page.insert_text((72, 120), "Medium Header", fontsize=16)
        # Insert normal text (<=14pt → not a header)
        page.insert_text((72, 160), "Normal body text that is not a header.", fontsize=11)
        doc.save(str(pdf_file))
        doc.close()

        result = parse_document(pdf_file)
        assert "Big Title" in result.full_text
        # Should extract headers from large font text
        header_texts = [h.text for h in result.headers]
        assert "Big Title" in header_texts
        assert "Medium Header" in header_texts

    def test_pdf_with_image_block(self, tmp_path: Path) -> None:
        """PDF with non-text blocks (images) should skip them gracefully."""
        pdf_file = tmp_path / "mixed.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Text content here")
        # Insert a small image (creates a non-text block, type != 0)
        import struct
        import zlib

        def _make_png() -> bytes:
            """Create minimal 1x1 red PNG."""
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
            ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
            raw = b"\x00\xff\x00\x00"
            idat_data = zlib.compress(raw)
            idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
            idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
            iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
            return sig + ihdr + idat + iend

        img_rect = fitz.Rect(72, 100, 172, 200)
        page.insert_image(img_rect, stream=_make_png())
        doc.save(str(pdf_file))
        doc.close()

        result = parse_document(pdf_file)
        assert "Text content here" in result.full_text


# ===========================================================================
# Regex Pattern Unit Tests
# ===========================================================================

class TestRegexPatterns:
    """Unit tests for compiled regex patterns used in markdown parsing."""

    def test_page_sep_matches_valid(self) -> None:
        line = "{1}" + "-" * 48
        assert _PAGE_SEP_RE.match(line) is not None
        assert _PAGE_SEP_RE.match(line).group(1) == "1"

    def test_page_sep_extracts_multi_digit_page(self) -> None:
        line = "{42}" + "-" * 48
        assert _PAGE_SEP_RE.match(line).group(1) == "42"

    def test_page_sep_rejects_too_few_dashes(self) -> None:
        line = "{1}" + "-" * 30
        assert _PAGE_SEP_RE.match(line) is None

    def test_page_sep_rejects_plain_text(self) -> None:
        assert _PAGE_SEP_RE.match("regular text") is None

    def test_html_tag_matches_span_open(self) -> None:
        assert _HTML_TAG_RE.search('<span id="foo">') is not None

    def test_html_tag_matches_span_close(self) -> None:
        assert _HTML_TAG_RE.search("</span>") is not None

    def test_html_tag_matches_br(self) -> None:
        assert _HTML_TAG_RE.search("<br>") is not None

    def test_html_tag_matches_br_self_closing(self) -> None:
        assert _HTML_TAG_RE.search("<br/>") is not None

    def test_html_tag_no_match_on_plain(self) -> None:
        assert _HTML_TAG_RE.search("no html here") is None

    def test_image_ref_matches(self) -> None:
        ref = "![alt text](_page_1_Picture_1.jpeg)"
        assert _IMAGE_REF_RE.search(ref) is not None

    def test_image_ref_matches_empty_alt(self) -> None:
        ref = "![](_page_0_img.png)"
        assert _IMAGE_REF_RE.search(ref) is not None

    def test_image_ref_no_match_on_link(self) -> None:
        """Regular markdown links [text](url) should NOT match."""
        assert _IMAGE_REF_RE.search("[click here](https://example.com)") is None

    def test_header_re_level_1(self) -> None:
        m = _HEADER_RE.match("# Title")
        assert m is not None
        assert len(m.group(1)) == 1
        assert m.group(2) == "Title"

    def test_header_re_level_3(self) -> None:
        m = _HEADER_RE.match("### Sub Section")
        assert m is not None
        assert len(m.group(1)) == 3
        assert m.group(2) == "Sub Section"

    def test_header_re_no_match_without_space(self) -> None:
        """#NoSpace should not match — header needs space after #."""
        assert _HEADER_RE.match("#NoSpace") is None
