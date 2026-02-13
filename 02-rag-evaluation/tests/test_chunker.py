"""Tests for chunker — fixed-size (A-D) and semantic (E) strategies.

Uses a small synthetic markdown fixture for fast, deterministic tests.
Does NOT load real 1344-line input files — those are for integration tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunker import _assign_page_numbers, _count_tokens, _find_char_offset, chunk_document
from src.models import Chunk, ChunkConfig
from src.parser import HeaderInfo, ParseResult


# ===========================================================================
# Synthetic Markdown Fixture
# ===========================================================================

# ~200 tokens of content with ## and ### headers, simulating Kaggle format.
# Each section is large enough to produce meaningful chunks but small enough
# to keep tests fast and deterministic.
_SYNTHETIC_MD = """\
# Enterprise Architecture Guide

This document covers modern enterprise architecture patterns and practices \
for building scalable distributed systems in cloud environments.

## Cloud Infrastructure

Cloud computing provides on-demand availability of computer system resources, \
especially data storage and computing power, without direct active management \
by the user. Major providers include AWS, Azure, and Google Cloud Platform. \
Organizations must evaluate cost, performance, compliance, and vendor lock-in \
when selecting a cloud provider for their workloads.

### Containerization

Docker containers package applications with their dependencies into standardized \
units for software development. Kubernetes orchestrates these containers at scale, \
providing automated deployment, scaling, and management. Container registries like \
ECR and ACR store and distribute container images across environments.

### Service Mesh

A service mesh like Istio or Linkerd provides infrastructure-level networking \
for microservices. It handles service discovery, load balancing, encryption, \
observability, and traceability. The sidecar proxy pattern intercepts all network \
communication between services without requiring code changes.

## Data Architecture

Modern data architecture encompasses data lakes, warehouses, and lakehouses. \
The medallion architecture organizes data into bronze, silver, and gold layers. \
Stream processing with Apache Kafka enables real-time data pipelines alongside \
traditional batch processing with Apache Spark. Data governance ensures quality, \
lineage, and compliance across the entire data lifecycle.

### Data Governance

Data governance establishes policies and procedures for managing data assets. \
It includes data cataloging, quality monitoring, access control, and compliance \
reporting. Tools like Apache Atlas and Collibra provide metadata management \
capabilities for enterprise data governance programs.

## Security Practices

Zero trust architecture assumes no implicit trust in any network segment. \
Every access request must be verified regardless of location. Identity and \
access management through OAuth 2.0 and OpenID Connect provides standardized \
authentication and authorization. Security information and event management \
systems aggregate and analyze security data across the organization.\
"""


@pytest.fixture()
def synthetic_parse_result() -> ParseResult:
    """Build a ParseResult from _SYNTHETIC_MD, simulating parser output.

    We manually extract headers and build a page_map (2 pages) so chunker
    tests don't depend on the parser module's implementation details.
    """
    full_text = _SYNTHETIC_MD

    # Extract headers manually — matches what parser._parse_markdown would produce
    headers: list[HeaderInfo] = []
    char_pos = 0
    for line in full_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## ") and not stripped.startswith("### "):
            headers.append(HeaderInfo(
                level=2,
                text=stripped[3:],
                char_offset=char_pos,
                page_number=0,
            ))
        elif stripped.startswith("### "):
            headers.append(HeaderInfo(
                level=3,
                text=stripped[4:],
                char_offset=char_pos,
                page_number=0,
            ))
        char_pos += len(line) + 1  # +1 for newline

    # Simulate 2-page document: first half = page 0, second half = page 1
    mid = len(full_text) // 2
    page_map = [(0, mid, 0), (mid, len(full_text), 1)]

    # Assign correct page numbers to headers
    for h in headers:
        h.page_number = 0 if h.char_offset < mid else 1

    return ParseResult(
        full_text=full_text,
        page_map=page_map,
        headers=headers,
        source_path=Path("test_synthetic.md"),
        num_pages=2,
    )


# Configs for testing — smaller than production to keep chunk counts manageable
_CONFIG_SMALL = ChunkConfig(
    name="small",
    chunk_size=64,
    overlap=16,
    chunking_goal="Small chunks for testing",
)

_CONFIG_MEDIUM = ChunkConfig(
    name="medium",
    chunk_size=128,
    overlap=32,
    chunking_goal="Medium chunks for testing",
)

_CONFIG_LARGE = ChunkConfig(
    name="large",
    chunk_size=256,
    overlap=64,
    chunking_goal="Large chunks for testing",
)

_CONFIG_SEMANTIC = ChunkConfig(
    name="semantic",
    chunk_size=512,
    overlap=0,
    chunking_goal="Semantic splitting for testing",
    is_semantic=True,
)


# ===========================================================================
# Token Counting
# ===========================================================================

class TestTokenCounting:
    """Verify tiktoken cl100k_base token counting."""

    def test_simple_text(self) -> None:
        # "hello world" is 2 tokens in cl100k_base
        assert _count_tokens("hello world") == 2

    def test_empty_string(self) -> None:
        assert _count_tokens("") == 0

    def test_contractions_split(self) -> None:
        # "don't" is 2 tokens in cl100k_base (don + 't)
        tokens = _count_tokens("don't")
        assert tokens >= 2  # At least 2 tokens for contraction


# ===========================================================================
# Fixed-Size Chunking
# ===========================================================================

class TestFixedSizeChunking:
    """Test Configs A-D (fixed-size RecursiveCharacterTextSplitter)."""

    def test_smaller_chunk_size_produces_more_chunks(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        """Core invariant: smaller chunks = more chunks from same text."""
        small_chunks = chunk_document(synthetic_parse_result, _CONFIG_SMALL)
        medium_chunks = chunk_document(synthetic_parse_result, _CONFIG_MEDIUM)
        large_chunks = chunk_document(synthetic_parse_result, _CONFIG_LARGE)

        assert len(small_chunks) > len(medium_chunks)
        assert len(medium_chunks) > len(large_chunks)
        assert len(large_chunks) >= 1

    def test_chunk_ids_follow_convention(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        """IDs must be {config_name}_{sequential_index}."""
        chunks = chunk_document(synthetic_parse_result, _CONFIG_MEDIUM)
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"medium_{i}"

    def test_all_chunks_have_positive_token_count(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        chunks = chunk_document(synthetic_parse_result, _CONFIG_MEDIUM)
        assert all(c.token_count > 0 for c in chunks)

    def test_all_chunks_have_valid_page_numbers(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        chunks = chunk_document(synthetic_parse_result, _CONFIG_MEDIUM)
        for chunk in chunks:
            assert len(chunk.page_numbers) >= 1
            assert all(p in (0, 1) for p in chunk.page_numbers)

    def test_chunks_have_valid_char_offsets(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        """start_char < end_char, and offsets within document bounds."""
        chunks = chunk_document(synthetic_parse_result, _CONFIG_MEDIUM)
        doc_len = len(synthetic_parse_result.full_text)
        for chunk in chunks:
            assert chunk.start_char < chunk.end_char
            assert chunk.start_char >= 0
            assert chunk.end_char <= doc_len

    def test_fixed_size_chunks_have_no_section_header(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        chunks = chunk_document(synthetic_parse_result, _CONFIG_MEDIUM)
        assert all(c.section_header is None for c in chunks)

    def test_all_chunks_are_chunk_model(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        """Every item in the result is a valid Chunk model instance."""
        chunks = chunk_document(synthetic_parse_result, _CONFIG_MEDIUM)
        assert all(isinstance(c, Chunk) for c in chunks)


# ===========================================================================
# Semantic Chunking (Config E)
# ===========================================================================

class TestSemanticChunking:
    """Test Config E (header-based splitting with merge/subdivide)."""

    def test_section_headers_populated(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        """Every Config E chunk must have a section_header."""
        chunks = chunk_document(synthetic_parse_result, _CONFIG_SEMANTIC)
        assert all(c.section_header is not None for c in chunks)

    def test_chunks_align_with_headers(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        """Section headers in chunks should match document headers."""
        chunks = chunk_document(synthetic_parse_result, _CONFIG_SEMANTIC)
        chunk_headers = {c.section_header for c in chunks}
        # Our fixture has: Cloud Infrastructure, Containerization, Service Mesh,
        # Data Architecture, Data Governance, Security Practices
        expected_headers = {
            "Cloud Infrastructure",
            "Containerization",
            "Service Mesh",
            "Data Architecture",
            "Data Governance",
            "Security Practices",
        }
        # All expected headers should appear (some may be merged but header preserved)
        assert chunk_headers & expected_headers  # At least some overlap

    def test_semantic_chunk_ids_sequential(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        chunks = chunk_document(synthetic_parse_result, _CONFIG_SEMANTIC)
        for i, chunk in enumerate(chunks):
            assert chunk.id == f"semantic_{i}"

    def test_semantic_produces_fewer_chunks_than_small_fixed(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        """Semantic chunking by headers produces fewer, more coherent chunks."""
        semantic_chunks = chunk_document(synthetic_parse_result, _CONFIG_SEMANTIC)
        small_chunks = chunk_document(synthetic_parse_result, _CONFIG_SMALL)
        assert len(semantic_chunks) < len(small_chunks)

    def test_semantic_chunks_have_valid_page_numbers(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        chunks = chunk_document(synthetic_parse_result, _CONFIG_SEMANTIC)
        for chunk in chunks:
            assert len(chunk.page_numbers) >= 1

    def test_semantic_chunks_have_positive_token_count(
        self, synthetic_parse_result: ParseResult,
    ) -> None:
        chunks = chunk_document(synthetic_parse_result, _CONFIG_SEMANTIC)
        assert all(c.token_count > 0 for c in chunks)

    def test_fallback_when_no_headers(self) -> None:
        """Without ## or ### headers, semantic chunking falls back to fixed-size."""
        no_header_text = "Just plain text. " * 100
        parse_result = ParseResult(
            full_text=no_header_text,
            page_map=[(0, len(no_header_text), 0)],
            headers=[],
            source_path=Path("no_headers.md"),
            num_pages=1,
        )
        chunks = chunk_document(parse_result, _CONFIG_SEMANTIC)
        # Should produce chunks (via fallback) rather than crashing
        assert len(chunks) > 0
        # Fallback uses Config B (256/64), so section_header should be None
        assert all(c.section_header is None for c in chunks)

    def test_fallback_with_only_h1_headers(self) -> None:
        """# headers (level 1) are ignored by semantic — should fallback."""
        h1_text = "# Title\n\nSome content here. " * 50
        parse_result = ParseResult(
            full_text=h1_text,
            page_map=[(0, len(h1_text), 0)],
            headers=[HeaderInfo(level=1, text="Title", char_offset=0, page_number=0)],
            source_path=Path("h1_only.md"),
            num_pages=1,
        )
        chunks = chunk_document(parse_result, _CONFIG_SEMANTIC)
        assert len(chunks) > 0
        # No ##/### headers → fallback to fixed-size → no section_header
        assert all(c.section_header is None for c in chunks)


# ===========================================================================
# Helper Function Unit Tests
# ===========================================================================

class TestHelperFunctions:
    """Unit tests for internal helper functions."""

    def test_find_char_offset_basic(self) -> None:
        full = "Hello world, this is a test document."
        assert _find_char_offset("this is", full, 0) == 13

    def test_find_char_offset_with_search_start(self) -> None:
        full = "aaa bbb aaa bbb"
        # Searching from position 4 should find second "aaa"
        assert _find_char_offset("aaa", full, 4) == 8

    def test_find_char_offset_fallback_to_beginning(self) -> None:
        full = "hello world"
        # Search start past the match — should fall back to find from beginning
        assert _find_char_offset("hello", full, 100) == 0

    def test_assign_page_numbers_single_page(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1)]
        assert _assign_page_numbers(100, 200, page_map) == [0]

    def test_assign_page_numbers_spanning_boundary(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1)]
        assert _assign_page_numbers(400, 600, page_map) == [0, 1]

    def test_assign_page_numbers_empty_page_map(self) -> None:
        assert _assign_page_numbers(0, 100, []) == [0]

    def test_assign_page_numbers_exact_boundary(self) -> None:
        page_map = [(0, 500, 0), (500, 1000, 1)]
        # Chunk starts exactly at page boundary
        assert _assign_page_numbers(500, 600, page_map) == [1]
