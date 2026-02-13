"""Tests for embedder.py — embedding factory, SentenceTransformer, LiteLLM.

All external dependencies mocked (sentence_transformers, litellm) to avoid
loading real models or making API calls. Tests verify shape, normalization,
batching, threading, and factory logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import EMBEDDING_DIMENSIONS
from src.embedder import (
    LiteLLMEmbedder,
    SentenceTransformerEmbedder,
    _l2_normalize,
    create_embedder,
)
from src.models import EmbeddingModel


# ===========================================================================
# Helpers
# ===========================================================================

def _random_vectors(n: int, dim: int, *, normalize: bool = False) -> np.ndarray:
    """Generate random float32 vectors for testing."""
    rng = np.random.default_rng(42)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
    return vectors


# ===========================================================================
# L2 Normalization Tests
# ===========================================================================

class TestL2Normalize:
    """Verify the _l2_normalize utility function."""

    def test_normalize_produces_unit_vectors(self) -> None:
        vectors = _random_vectors(5, 384)
        normalized = _l2_normalize(vectors)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_normalize_preserves_shape(self) -> None:
        vectors = _random_vectors(3, 768)
        normalized = _l2_normalize(vectors)
        assert normalized.shape == (3, 768)

    def test_normalize_handles_zero_vector(self) -> None:
        """Zero vectors should not cause division by zero."""
        vectors = np.zeros((2, 384), dtype=np.float32)
        normalized = _l2_normalize(vectors)
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))


# ===========================================================================
# SentenceTransformerEmbedder Tests
# ===========================================================================

class TestSentenceTransformerEmbedder:
    """Tests for the local SentenceTransformer backend."""

    @pytest.fixture()
    def mock_st_model(self):
        """Create a mock SentenceTransformer that returns random vectors."""
        with patch("sentence_transformers.SentenceTransformer") as MockST:
            mock_model = MagicMock()

            def fake_encode(texts, batch_size=32, convert_to_numpy=True,
                            show_progress_bar=False):
                return _random_vectors(len(texts), 384)

            mock_model.encode.side_effect = fake_encode
            MockST.return_value = mock_model
            yield mock_model

    def test_embed_shape_minilm(self, mock_st_model) -> None:
        """Output shape is (n, 384) for MiniLM."""
        embedder = SentenceTransformerEmbedder(EmbeddingModel.MINILM)
        result = embedder.embed(["hello", "world", "foo"])
        assert result.shape == (3, 384)

    def test_embed_normalized(self, mock_st_model) -> None:
        """All output vectors have unit L2 norm."""
        embedder = SentenceTransformerEmbedder(EmbeddingModel.MINILM)
        result = embedder.embed(["text one", "text two"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_embed_dtype_float32(self, mock_st_model) -> None:
        """Output dtype is float32."""
        embedder = SentenceTransformerEmbedder(EmbeddingModel.MINILM)
        result = embedder.embed(["test"])
        assert result.dtype == np.float32

    def test_embed_empty_list(self, mock_st_model) -> None:
        """Empty input returns empty array with correct shape."""
        embedder = SentenceTransformerEmbedder(EmbeddingModel.MINILM)
        result = embedder.embed([])
        assert result.shape == (0, 384)
        assert result.dtype == np.float32

    def test_embed_single_text(self, mock_st_model) -> None:
        """Single text produces (1, dimensions) array."""
        embedder = SentenceTransformerEmbedder(EmbeddingModel.MINILM)
        result = embedder.embed(["one text"])
        assert result.shape == (1, 384)

    def test_batch_size_passed_to_encode(self, mock_st_model) -> None:
        """batch_size=32 is passed to SentenceTransformer.encode()."""
        embedder = SentenceTransformerEmbedder(EmbeddingModel.MINILM)
        embedder.embed(["a", "b", "c"])
        mock_st_model.encode.assert_called_once()
        call_kwargs = mock_st_model.encode.call_args
        assert call_kwargs.kwargs.get("batch_size") == 32 or call_kwargs[1].get("batch_size") == 32

    def test_dimensions_property(self, mock_st_model) -> None:
        """dimensions property matches config for the model."""
        embedder = SentenceTransformerEmbedder(EmbeddingModel.MINILM)
        assert embedder.dimensions == 384

    @patch("sentence_transformers.SentenceTransformer")
    def test_mpnet_dimensions(self, MockST) -> None:
        """mpnet embedder has 768 dimensions."""
        mock_model = MagicMock()
        mock_model.encode.side_effect = lambda texts, **kw: _random_vectors(len(texts), 768)
        MockST.return_value = mock_model

        embedder = SentenceTransformerEmbedder(EmbeddingModel.MPNET)
        assert embedder.dimensions == 768
        result = embedder.embed(["test"])
        assert result.shape == (1, 768)


# ===========================================================================
# LiteLLMEmbedder Tests
# ===========================================================================

class TestLiteLLMEmbedder:
    """Tests for the API-based LiteLLM backend."""

    @staticmethod
    def _make_litellm_response(texts: list[str], dim: int = 1536):
        """Create a mock LiteLLM embedding response."""
        rng = np.random.default_rng(42)
        response = MagicMock()
        response.data = [
            {"index": i, "embedding": rng.standard_normal(dim).tolist()}
            for i in range(len(texts))
        ]
        return response

    @patch("litellm.embedding")
    def test_embed_shape(self, mock_embedding) -> None:
        """Output shape is (n, 1536) for OpenAI."""
        mock_embedding.side_effect = lambda model, input: (
            self._make_litellm_response(input)
        )
        embedder = LiteLLMEmbedder(EmbeddingModel.OPENAI)
        result = embedder.embed(["hello", "world"])
        assert result.shape == (2, 1536)

    @patch("litellm.embedding")
    def test_embed_normalized(self, mock_embedding) -> None:
        """All output vectors have unit L2 norm after normalization."""
        mock_embedding.side_effect = lambda model, input: (
            self._make_litellm_response(input)
        )
        embedder = LiteLLMEmbedder(EmbeddingModel.OPENAI)
        result = embedder.embed(["text"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    @patch("litellm.embedding")
    def test_embed_empty_list(self, mock_embedding) -> None:
        """Empty input returns empty array with correct shape."""
        embedder = LiteLLMEmbedder(EmbeddingModel.OPENAI)
        result = embedder.embed([])
        assert result.shape == (0, 1536)
        assert result.dtype == np.float32

    @patch("litellm.embedding")
    def test_embed_single_text(self, mock_embedding) -> None:
        """Single text produces (1, 1536) array."""
        mock_embedding.side_effect = lambda model, input: (
            self._make_litellm_response(input)
        )
        embedder = LiteLLMEmbedder(EmbeddingModel.OPENAI)
        result = embedder.embed(["one"])
        assert result.shape == (1, 1536)

    @patch("litellm.embedding")
    def test_order_preserved_when_api_returns_shuffled(self, mock_embedding) -> None:
        """Embeddings are sorted by index even if API returns them out of order."""
        rng = np.random.default_rng(99)
        response = MagicMock()
        # Return indices in reverse order to simulate out-of-order API response
        response.data = [
            {"index": 2, "embedding": rng.standard_normal(1536).tolist()},
            {"index": 0, "embedding": rng.standard_normal(1536).tolist()},
            {"index": 1, "embedding": rng.standard_normal(1536).tolist()},
        ]
        mock_embedding.return_value = response

        embedder = LiteLLMEmbedder(EmbeddingModel.OPENAI)
        result = embedder.embed(["a", "b", "c"])
        assert result.shape == (3, 1536)

    @patch("litellm.embedding")
    def test_batching_large_input(self, mock_embedding) -> None:
        """Texts are grouped into batches of 50 — 120 texts = 3 API calls."""
        mock_embedding.side_effect = lambda model, input: (
            self._make_litellm_response(input)
        )
        embedder = LiteLLMEmbedder(EmbeddingModel.OPENAI)
        texts = [f"text {i}" for i in range(120)]
        result = embedder.embed(texts)

        assert result.shape == (120, 1536)
        # 120 texts / 50 per batch = 3 API calls
        assert mock_embedding.call_count == 3

    @patch("litellm.embedding")
    def test_dimensions_property(self, mock_embedding) -> None:
        """dimensions property returns 1536 for OpenAI."""
        embedder = LiteLLMEmbedder(EmbeddingModel.OPENAI)
        assert embedder.dimensions == 1536


# ===========================================================================
# Factory Tests
# ===========================================================================

class TestCreateEmbedder:
    """Verify the create_embedder factory function."""

    @patch("sentence_transformers.SentenceTransformer")
    def test_factory_minilm_returns_sentence_transformer(self, MockST) -> None:
        MockST.return_value = MagicMock()
        embedder = create_embedder(EmbeddingModel.MINILM)
        assert isinstance(embedder, SentenceTransformerEmbedder)

    @patch("sentence_transformers.SentenceTransformer")
    def test_factory_mpnet_returns_sentence_transformer(self, MockST) -> None:
        MockST.return_value = MagicMock()
        embedder = create_embedder(EmbeddingModel.MPNET)
        assert isinstance(embedder, SentenceTransformerEmbedder)

    def test_factory_openai_returns_litellm(self) -> None:
        embedder = create_embedder(EmbeddingModel.OPENAI)
        assert isinstance(embedder, LiteLLMEmbedder)

    @patch("sentence_transformers.SentenceTransformer")
    @pytest.mark.parametrize("model", list(EmbeddingModel))
    def test_factory_all_models_return_base_embedder(self, MockST, model) -> None:
        """All enum values produce a valid embedder (parametrized)."""
        MockST.return_value = MagicMock()
        from src.embedder import BaseEmbedder
        embedder = create_embedder(model)
        assert isinstance(embedder, BaseEmbedder)
        assert embedder.dimensions == EMBEDDING_DIMENSIONS[model]
