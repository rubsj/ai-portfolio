"""PDF extraction and text cleaning using PyMuPDF.

Loads PDFs via fitz.open(), extracts text page-by-page, applies cleaning
(headers/footers, ligatures, hyphenation) and returns Document objects.

Why PyMuPDF over pdfplumber/pypdf: spec-required; best single-column
extraction accuracy; coordinate access for potential multi-column handling.
"""
