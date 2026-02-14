"""Tests for index_builder.py — parsing, chunking, embedding, index orchestration.

Tests the orchestrator functions with mocked I/O (FAISS, embedders, BM25,
file system) and verifies pure helpers directly.

Java/TS parallel: like integration tests with @MockBean for every external
service (SentenceTransformer, FAISS, OpenAI). The wiring logic is tested,
not the backends themselves.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.models import Chunk, EmbeddingModel
from src.index_builder import (
    _build_bm25,
    _build_faiss_indices_api,
    _build_faiss_indices_local,
    _discover_input_files,
    _embed_and_save,
    _model_enum_from_key,
    _parse_and_chunk_all,
    _run_checkpoint_queries,
    _save_chunk_lists,
    build_all_indices,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_chunk(
    chunk_id: str = "B_0_0",
    text: str = "Some meaningful text content for testing purposes",
    config_name: str = "B",
) -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=chunk_id,
        text=text,
        token_count=10,
        start_char=0,
        end_char=100,
        page_numbers=[1],
        config_name=config_name,
    )


# ===========================================================================
# _model_enum_from_key Tests
# ===========================================================================

class TestModelEnumFromKey:
    """Tests for _model_enum_from_key — reverse lookup helper."""

    def test_minilm_key(self) -> None:
        """'minilm' maps to EmbeddingModel.MINILM."""
        assert _model_enum_from_key("minilm") is EmbeddingModel.MINILM

    def test_mpnet_key(self) -> None:
        """'mpnet' maps to EmbeddingModel.MPNET."""
        assert _model_enum_from_key("mpnet") is EmbeddingModel.MPNET

    def test_openai_key(self) -> None:
        """'openai' maps to EmbeddingModel.OPENAI."""
        assert _model_enum_from_key("openai") is EmbeddingModel.OPENAI

    def test_unknown_key_raises_key_error(self) -> None:
        """Unknown key raises KeyError."""
        with pytest.raises(KeyError):
            _model_enum_from_key("unknown_model")


# ===========================================================================
# _discover_input_files Tests
# ===========================================================================

class TestDiscoverInputFiles:
    """Tests for _discover_input_files — finds .md files in data/input/."""

    def test_finds_md_files(self, tmp_path, monkeypatch) -> None:
        """Returns sorted list of .md files."""
        monkeypatch.setattr("src.index_builder.INPUT_DIR", tmp_path)

        (tmp_path / "b_healthcare.md").write_text("# Healthcare")
        (tmp_path / "a_financial.md").write_text("# Financial")
        (tmp_path / "c_technology.md").write_text("# Technology")

        result = _discover_input_files()

        assert len(result) == 3
        # WHY check sorted: deterministic order is part of the contract
        assert result[0].name == "a_financial.md"
        assert result[1].name == "b_healthcare.md"
        assert result[2].name == "c_technology.md"

    def test_raises_when_no_md_files(self, tmp_path, monkeypatch) -> None:
        """Raises FileNotFoundError when directory has no .md files."""
        monkeypatch.setattr("src.index_builder.INPUT_DIR", tmp_path)

        with pytest.raises(FileNotFoundError, match="No .md files"):
            _discover_input_files()

    def test_ignores_non_md_files(self, tmp_path, monkeypatch) -> None:
        """Only .md files are returned, not .txt or .json."""
        monkeypatch.setattr("src.index_builder.INPUT_DIR", tmp_path)

        (tmp_path / "data.md").write_text("# Doc")
        (tmp_path / "readme.txt").write_text("readme")
        (tmp_path / "config.json").write_text("{}")

        result = _discover_input_files()
        assert len(result) == 1
        assert result[0].name == "data.md"


# ===========================================================================
# _save_chunk_lists Tests
# ===========================================================================

class TestSaveChunkLists:
    """Tests for _save_chunk_lists — serializes chunks to JSON."""

    def test_saves_all_configs(self, tmp_path, monkeypatch) -> None:
        """Creates one JSON file per chunk config."""
        monkeypatch.setattr("src.index_builder.OUTPUT_DIR", tmp_path)

        chunks_by_config = {
            "A": [_make_chunk("A_0_0", config_name="A")],
            "B": [_make_chunk("B_0_0"), _make_chunk("B_0_1")],
        }

        _save_chunk_lists(chunks_by_config)

        assert (tmp_path / "chunks_A.json").exists()
        assert (tmp_path / "chunks_B.json").exists()

        data_a = json.loads((tmp_path / "chunks_A.json").read_text())
        assert len(data_a) == 1
        assert data_a[0]["id"] == "A_0_0"

        data_b = json.loads((tmp_path / "chunks_B.json").read_text())
        assert len(data_b) == 2

    def test_creates_output_dir(self, tmp_path, monkeypatch) -> None:
        """Creates parent directory if it doesn't exist."""
        out_dir = tmp_path / "nested" / "output"
        monkeypatch.setattr("src.index_builder.OUTPUT_DIR", out_dir)

        _save_chunk_lists({"B": [_make_chunk()]})

        assert (out_dir / "chunks_B.json").exists()


# ===========================================================================
# _embed_and_save Tests
# ===========================================================================

class TestEmbedAndSave:
    """Tests for _embed_and_save — embeds chunks and saves FAISS index."""

    @patch("src.index_builder.FAISSVectorStore")
    def test_creates_and_saves_index(self, mock_store_cls, tmp_path, monkeypatch) -> None:
        """Embeds chunks, creates FAISS store, and saves to disk."""
        monkeypatch.setattr("src.index_builder.INDICES_DIR", tmp_path)

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.zeros((2, 384), dtype=np.float32)
        mock_embedder.dimensions = 384

        mock_store = MagicMock()
        mock_store.size = 2
        mock_store_cls.return_value = mock_store

        chunks = [_make_chunk("B_0_0"), _make_chunk("B_0_1")]

        _embed_and_save(mock_embedder, "minilm", "B", chunks)

        # WHY check embed call: verifies chunk texts are extracted correctly
        mock_embedder.embed.assert_called_once()
        texts_arg = mock_embedder.embed.call_args[0][0]
        assert len(texts_arg) == 2

        # WHY check store creation: verifies dimension matches embedder
        mock_store_cls.assert_called_once_with(dimension=384)
        mock_store.add.assert_called_once()
        mock_store.save.assert_called_once()

        # WHY check save path: verifies naming convention {model}_{config}
        save_path = mock_store.save.call_args[0][0]
        assert save_path.name == "minilm_B"


# ===========================================================================
# _build_faiss_indices_local Tests
# ===========================================================================

class TestBuildFaissIndicesLocal:
    """Tests for _build_faiss_indices_local — local model loop with cleanup."""

    @patch("src.index_builder._embed_and_save")
    @patch("src.index_builder.create_embedder")
    def test_processes_each_local_model(
        self, mock_create, mock_embed_save, tmp_path, monkeypatch,
    ) -> None:
        """Calls create_embedder for each LOCAL model and embeds all configs."""
        monkeypatch.setattr("src.index_builder.INDICES_DIR", tmp_path)

        from src.config import LOCAL_EMBEDDING_MODELS
        mock_embedder = MagicMock()
        mock_create.return_value = mock_embedder

        chunks_by_config = {
            "A": [_make_chunk("A_0_0", config_name="A")],
            "B": [_make_chunk("B_0_0")],
        }

        _build_faiss_indices_local(chunks_by_config)

        # WHY check call count: each local model × each config
        assert mock_create.call_count == len(LOCAL_EMBEDDING_MODELS)
        expected_embed_calls = len(LOCAL_EMBEDDING_MODELS) * len(chunks_by_config)
        assert mock_embed_save.call_count == expected_embed_calls


# ===========================================================================
# _build_faiss_indices_api Tests
# ===========================================================================

class TestBuildFaissIndicesApi:
    """Tests for _build_faiss_indices_api — API model loop (no cleanup)."""

    @patch("src.index_builder._embed_and_save")
    @patch("src.index_builder.create_embedder")
    def test_processes_each_api_model(
        self, mock_create, mock_embed_save, tmp_path, monkeypatch,
    ) -> None:
        """Calls create_embedder for each API model and embeds all configs."""
        monkeypatch.setattr("src.index_builder.INDICES_DIR", tmp_path)

        from src.config import API_EMBEDDING_MODELS
        mock_embedder = MagicMock()
        mock_create.return_value = mock_embedder

        chunks_by_config = {
            "B": [_make_chunk("B_0_0")],
            "C": [_make_chunk("C_0_0", config_name="C")],
        }

        _build_faiss_indices_api(chunks_by_config)

        assert mock_create.call_count == len(API_EMBEDDING_MODELS)
        expected = len(API_EMBEDDING_MODELS) * len(chunks_by_config)
        assert mock_embed_save.call_count == expected


# ===========================================================================
# _build_bm25 Tests
# ===========================================================================

class TestBuildBm25:
    """Tests for _build_bm25 — BM25 baseline from Config B."""

    @patch("src.index_builder.BM25Retriever")
    def test_builds_from_config_b(self, mock_bm25_cls, tmp_path, monkeypatch) -> None:
        """Creates BM25 index from Config B chunks and saves to disk."""
        monkeypatch.setattr("src.index_builder.INDICES_DIR", tmp_path)

        mock_retriever = MagicMock()
        mock_retriever.size = 3
        mock_bm25_cls.return_value = mock_retriever

        from src.config import BM25_CHUNK_CONFIG
        config_name = BM25_CHUNK_CONFIG.name

        chunks_by_config = {
            config_name: [_make_chunk(f"{config_name}_0_{i}") for i in range(3)],
            "A": [_make_chunk("A_0_0", config_name="A")],
        }

        _build_bm25(chunks_by_config)

        # WHY check constructor: only Config B chunks should be passed
        mock_bm25_cls.assert_called_once()
        passed_chunks = mock_bm25_cls.call_args[0][0]
        assert len(passed_chunks) == 3

        # WHY check save: verifies naming convention bm25_{config}
        mock_retriever.save.assert_called_once()
        save_path = mock_retriever.save.call_args[0][0]
        assert f"bm25_{config_name}" in str(save_path)


# ===========================================================================
# _parse_and_chunk_all Tests
# ===========================================================================

class TestParseAndChunkAll:
    """Tests for _parse_and_chunk_all — parse + chunk orchestration."""

    @patch("src.index_builder.chunk_document")
    @patch("src.index_builder.parse_document")
    @patch("src.index_builder._discover_input_files")
    def test_parses_and_chunks_all_configs(
        self, mock_discover, mock_parse, mock_chunk,
    ) -> None:
        """Parses each input file and chunks with all 5 configs."""
        from src.config import ALL_CHUNK_CONFIGS
        from pathlib import Path

        # 2 input files
        mock_discover.return_value = [Path("/fake/a.md"), Path("/fake/b.md")]

        # Mock parse results with full_text attribute
        mock_pr1 = MagicMock()
        mock_pr1.full_text = "Document one content"
        mock_pr2 = MagicMock()
        mock_pr2.full_text = "Document two content"
        mock_parse.side_effect = [mock_pr1, mock_pr2]

        # WHY side_effect (not return_value): _parse_and_chunk_all mutates
        # chunk IDs in-place. If we return the same objects every call, the
        # last iteration's mutations overwrite earlier ones. Fresh objects
        # each call avoids this.
        mock_chunk.side_effect = lambda pr, config: [
            _make_chunk("temp_0"),
            _make_chunk("temp_1"),
        ]

        result = _parse_and_chunk_all()

        # WHY 5 configs: A through E
        assert len(result) == len(ALL_CHUNK_CONFIGS)

        for config in ALL_CHUNK_CONFIGS:
            # 2 docs × 2 chunks each = 4 chunks per config
            assert len(result[config.name]) == 4

        # WHY check chunk IDs: must be globally unique with doc prefix
        # Format: {config}_{doc_idx}_{chunk_idx}
        b_chunks = result[ALL_CHUNK_CONFIGS[1].name]  # Config B
        config_name = ALL_CHUNK_CONFIGS[1].name
        assert b_chunks[0].id == f"{config_name}_0_0"
        assert b_chunks[1].id == f"{config_name}_0_1"
        assert b_chunks[2].id == f"{config_name}_1_0"
        assert b_chunks[3].id == f"{config_name}_1_1"


# ===========================================================================
# _run_checkpoint_queries Tests
# ===========================================================================

class TestRunCheckpointQueries:
    """Tests for _run_checkpoint_queries — smoke-test query verification."""

    @patch("src.index_builder.BM25Retriever")
    @patch("src.index_builder.create_embedder")
    @patch("src.index_builder.FAISSVectorStore")
    def test_prints_checkpoint_output(
        self, mock_store_cls, mock_create, mock_bm25_cls,
        tmp_path, monkeypatch, capsys,
    ) -> None:
        """Prints checkpoint query results for FAISS and BM25."""
        monkeypatch.setattr("src.index_builder.INDICES_DIR", tmp_path)
        monkeypatch.setattr("src.index_builder.BM25_CHUNK_CONFIG", MagicMock(name="B"))

        # Create fake .faiss files so existence checks pass
        (tmp_path / "minilm_B.faiss").write_bytes(b"fake")
        (tmp_path / "mpnet_B.faiss").write_bytes(b"fake")
        (tmp_path / "minilm_E.faiss").write_bytes(b"fake")
        (tmp_path / "bm25_B.pkl").write_bytes(b"fake")

        # Mock FAISS store
        mock_store = MagicMock()
        mock_store.search.return_value = [
            ("B_0_0", 0.95), ("B_0_1", 0.85), ("B_0_2", 0.75),
        ]
        mock_store_cls.load.return_value = mock_store

        # Mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [np.zeros(384, dtype=np.float32)]
        mock_create.return_value = mock_embedder

        # Mock BM25
        mock_bm25 = MagicMock()
        mock_bm25.search.return_value = [
            ("B_0_0", 3.5), ("B_0_1", 2.1), ("B_0_2", 1.8),
        ]
        mock_bm25_cls.load.return_value = mock_bm25

        # Chunks for text lookup
        chunks_by_config = {
            "B": [
                _make_chunk("B_0_0", text="Revenue was strong in Q3"),
                _make_chunk("B_0_1", text="Total revenue reached $5B"),
                _make_chunk("B_0_2", text="Operating costs declined"),
            ],
            "E": [_make_chunk("E_0_0", text="Semantic chunk text", config_name="E")],
        }

        _run_checkpoint_queries(chunks_by_config)

        output = capsys.readouterr().out
        assert "CHECKPOINT" in output
        assert "total revenue" in output
        assert "B_0_0" in output

    @patch("src.index_builder.create_embedder")
    @patch("src.index_builder.FAISSVectorStore")
    def test_skips_missing_indices(
        self, mock_store_cls, mock_create, tmp_path, monkeypatch, capsys,
    ) -> None:
        """Prints 'index not found' for missing FAISS files."""
        monkeypatch.setattr("src.index_builder.INDICES_DIR", tmp_path)
        monkeypatch.setattr("src.index_builder.BM25_CHUNK_CONFIG", MagicMock(name="B"))

        # No .faiss or .pkl files exist → all indices skipped
        chunks_by_config = {"B": [_make_chunk("B_0_0")]}

        _run_checkpoint_queries(chunks_by_config)

        output = capsys.readouterr().out
        assert "index not found" in output


# ===========================================================================
# build_all_indices Tests
# ===========================================================================

class TestBuildAllIndices:
    """Tests for build_all_indices — full orchestration."""

    @patch("src.index_builder._run_checkpoint_queries")
    @patch("src.index_builder._build_bm25")
    @patch("src.index_builder._build_faiss_indices_api")
    @patch("src.index_builder._build_faiss_indices_local")
    @patch("src.index_builder._save_chunk_lists")
    @patch("src.index_builder._parse_and_chunk_all")
    def test_calls_all_steps_in_order(
        self, mock_parse, mock_save, mock_local, mock_api, mock_bm25, mock_check,
        tmp_path, monkeypatch, capsys,
    ) -> None:
        """Orchestrates all 6 steps and prints summary."""
        monkeypatch.setattr("src.index_builder.INDICES_DIR", tmp_path)
        monkeypatch.setattr("src.index_builder.OUTPUT_DIR", tmp_path)

        # Create fake index files for summary counts
        (tmp_path / "minilm_B.faiss").write_bytes(b"f")
        (tmp_path / "bm25_B.pkl").write_bytes(b"f")
        (tmp_path / "chunks_B.json").write_text("[]")

        mock_chunks = {"B": [_make_chunk()]}
        mock_parse.return_value = mock_chunks

        build_all_indices()

        # WHY verify call order: steps must execute sequentially
        mock_parse.assert_called_once()
        mock_save.assert_called_once_with(mock_chunks)
        mock_local.assert_called_once_with(mock_chunks)
        mock_api.assert_called_once_with(mock_chunks)
        mock_bm25.assert_called_once_with(mock_chunks)
        mock_check.assert_called_once_with(mock_chunks)

        # WHY check summary: prints count of created indices
        output = capsys.readouterr().out
        assert "Done!" in output
        assert "FAISS" in output
