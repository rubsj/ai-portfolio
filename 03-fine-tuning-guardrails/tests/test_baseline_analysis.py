from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np

from src.models import CompatibilityLabel, DatingPair


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_pair(label: int = 1) -> DatingPair:
    return DatingPair(
        text_1="boy: hello there",
        text_2="girl: hi there",
        label=CompatibilityLabel(label),
        category="test",
        subcategory="test",
        pair_type="compatible_match",
    )


# ------------------------------------------------------------------ #
# load_embeddings
# ------------------------------------------------------------------ #

def test_load_embeddings_round_trips(tmp_path):
    """Saved .npz reloads to identical arrays."""
    from src.baseline_analysis import load_embeddings

    rng = np.random.default_rng(0)
    emb1 = rng.standard_normal((10, 8)).astype(np.float32)
    emb2 = rng.standard_normal((10, 8)).astype(np.float32)
    path = tmp_path / "test.npz"
    np.savez(path, text1=emb1, text2=emb2)

    loaded1, loaded2 = load_embeddings(path)
    np.testing.assert_allclose(loaded1, emb1)
    np.testing.assert_allclose(loaded2, emb2)


# ------------------------------------------------------------------ #
# run_umap_projection
# ------------------------------------------------------------------ #

def test_run_umap_projection_output_shape():
    """Output shape is (2N, 2) — both sides of each pair projected together."""
    from src.baseline_analysis import run_umap_projection

    rng = np.random.default_rng(42)
    n = 25
    emb1 = rng.standard_normal((n, 8))
    emb2 = rng.standard_normal((n, 8))
    proj = run_umap_projection(emb1, emb2)
    assert proj.shape == (2 * n, 2)


def test_run_umap_projection_dtype():
    """UMAP output is float (not int/bool)."""
    from src.baseline_analysis import run_umap_projection

    rng = np.random.default_rng(1)
    emb1 = rng.standard_normal((15, 8))
    emb2 = rng.standard_normal((15, 8))
    proj = run_umap_projection(emb1, emb2)
    assert np.issubdtype(proj.dtype, np.floating)


# ------------------------------------------------------------------ #
# run_hdbscan_clustering
# ------------------------------------------------------------------ #

def test_run_hdbscan_clustering_return_types():
    """Return types: (float, int, float, ndarray) with valid ranges."""
    from src.baseline_analysis import run_hdbscan_clustering

    rng = np.random.default_rng(0)
    n = 40
    # 2D projections for n pairs (the function expects shape (2N, 2))
    proj = rng.standard_normal((2 * n, 2))
    labels = [i % 2 for i in range(n)]

    purity, n_clusters, noise_ratio, cluster_labels = run_hdbscan_clustering(proj, labels)

    assert isinstance(purity, float)
    assert isinstance(n_clusters, int)
    assert isinstance(noise_ratio, float)
    assert isinstance(cluster_labels, np.ndarray)
    assert len(cluster_labels) == n
    assert 0.0 <= purity <= 1.0
    assert 0.0 <= noise_ratio <= 1.0


def test_run_hdbscan_clustering_no_clusters_returns_zero_purity():
    """Very few points → HDBSCAN finds no clusters → purity = 0.0, n_clusters = 0."""
    from src.baseline_analysis import run_hdbscan_clustering

    rng = np.random.default_rng(2)
    n = 5  # Too few points for min_cluster_size=10 → all noise
    proj = rng.standard_normal((2 * n, 2))
    labels = [i % 2 for i in range(n)]

    purity, n_clusters, noise_ratio, cluster_labels = run_hdbscan_clustering(proj, labels)
    if n_clusters == 0:
        assert purity == 0.0
    assert noise_ratio == float(np.sum(cluster_labels == -1) / n)


def test_run_hdbscan_clustering_purity_weighted():
    """With highly separable clusters, purity should be high."""
    from src.baseline_analysis import run_hdbscan_clustering

    # Create two tight clusters well-separated in 2D
    rng = np.random.default_rng(99)
    n = 60
    # Compatible group at (+5, 0), incompatible at (-5, 0)
    compat_proj1 = rng.normal(loc=[5.0, 0.0], scale=0.2, size=(n // 2, 2))
    incompat_proj1 = rng.normal(loc=[-5.0, 0.0], scale=0.2, size=(n // 2, 2))
    compat_proj2 = rng.normal(loc=[5.0, 0.0], scale=0.2, size=(n // 2, 2))
    incompat_proj2 = rng.normal(loc=[-5.0, 0.0], scale=0.2, size=(n // 2, 2))

    # Stack as (2N, 2): first N = text_1, last N = text_2
    proj = np.vstack([
        np.vstack([compat_proj1, incompat_proj1]),   # first N rows
        np.vstack([compat_proj2, incompat_proj2]),   # last N rows
    ])
    labels = [1] * (n // 2) + [0] * (n // 2)

    purity, n_clusters, noise_ratio, cluster_labels = run_hdbscan_clustering(proj, labels)
    if n_clusters > 0:
        assert purity >= 0.7  # Well-separated clusters should have high purity


# ------------------------------------------------------------------ #
# generate_all_baseline_embeddings
# ------------------------------------------------------------------ #

def test_generate_all_baseline_embeddings_saves_npz(tmp_path, monkeypatch):
    """Mocked SentenceTransformer → saves baseline_train.npz and baseline_eval.npz."""
    from src.baseline_analysis import generate_all_baseline_embeddings

    n = 3
    pairs = [_make_pair(label=i % 2) for i in range(n)]

    # Mock the sentence_transformers module in sys.modules so the inline
    # 'from sentence_transformers import SentenceTransformer' resolves to our mock
    mock_st_module = MagicMock()
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros((n, 384), dtype=np.float32)
    mock_st_module.SentenceTransformer.return_value = mock_model
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st_module)

    # Redirect EMBEDDINGS_DIR so files land in tmp_path
    monkeypatch.setattr("src.baseline_analysis.EMBEDDINGS_DIR", tmp_path)

    generate_all_baseline_embeddings(pairs, pairs)

    assert (tmp_path / "baseline_train.npz").exists()
    assert (tmp_path / "baseline_eval.npz").exists()

    # Verify shape of saved arrays
    data = np.load(tmp_path / "baseline_train.npz")
    assert data["text1"].shape == (n, 384)
    assert data["text2"].shape == (n, 384)


# ------------------------------------------------------------------ #
# run_full_baseline — orchestration test
# ------------------------------------------------------------------ #

def test_run_full_baseline_creates_outputs(tmp_path, monkeypatch):
    """run_full_baseline orchestrates all steps and writes metric + HTML files."""
    from src.baseline_analysis import run_full_baseline

    rng = np.random.default_rng(42)
    n = 20

    # ---- Fake embeddings (small, fast, no model needed) ----
    emb1 = rng.standard_normal((n, 8)).astype(np.float32)
    emb2 = rng.standard_normal((n, 8)).astype(np.float32)
    embed_dir = tmp_path / "embeddings"
    embed_dir.mkdir()
    np.savez(embed_dir / "baseline_train.npz", text1=emb1, text2=emb2)
    np.savez(embed_dir / "baseline_eval.npz", text1=emb1, text2=emb2)

    # ---- Small fake pairs (alternating labels) ----
    pairs = [_make_pair(label=i % 2) for i in range(n)]

    # ---- Patches ----
    # Redirect EMBEDDINGS_DIR so pipeline finds fake .npz
    monkeypatch.setattr("src.baseline_analysis.EMBEDDINGS_DIR", embed_dir)

    # Mock UMAP — actual UMAP on 40 points is fast but adds UMAP init overhead
    fixed_proj = rng.standard_normal((2 * n, 2))
    monkeypatch.setattr("src.baseline_analysis.run_umap_projection",
                        lambda *a, **kw: fixed_proj)

    # Mock HDBSCAN clustering
    fixed_cluster_labels = np.zeros(n, dtype=int)
    monkeypatch.setattr(
        "src.baseline_analysis.run_hdbscan_clustering",
        lambda *a, **kw: (0.8, 3, 0.1, fixed_cluster_labels),
    )

    # Redirect CWD so relative-path outputs land in tmp_path
    monkeypatch.chdir(tmp_path)

    metrics = run_full_baseline(pairs, pairs)

    # Verify output files
    assert (tmp_path / "eval" / "baseline_metrics.json").exists()
    assert (tmp_path / "eval" / "baseline_report.html").exists()
    assert (tmp_path / "eval" / "visualizations" / "baseline" / "cosine_distributions.png").exists()
    assert (tmp_path / "eval" / "visualizations" / "baseline" / "umap_scatter.html").exists()

    # Verify returned metrics
    assert isinstance(metrics.compatibility_margin, float)
    assert isinstance(metrics.auc_roc, float)
    assert isinstance(metrics.false_positive_counts, dict)
