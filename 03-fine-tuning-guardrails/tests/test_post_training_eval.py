from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import DatingPair
from src.post_training_eval import (
    evaluate_from_embeddings,
    resolve_baseline_embeddings_path,
    resolve_baseline_path,
)


def test_resolve_baseline_path_finds_eval_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_baseline_path finds file in eval/ directory first."""
    # WHY: Correction #2 requires checking eval/ before data/baseline/
    monkeypatch.chdir(tmp_path)

    # Create eval/ directory with the file
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "baseline_metrics.json").write_text("{}")

    result = resolve_baseline_path("baseline_metrics.json")

    # Should find it in eval/ (first candidate checked)
    assert result == Path("eval/baseline_metrics.json")


def test_resolve_baseline_path_falls_back_to_data_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_baseline_path falls back to data/baseline/ if eval/ not found."""
    monkeypatch.chdir(tmp_path)

    # Create only data/baseline/ directory with the file
    data_baseline_dir = tmp_path / "data" / "baseline"
    data_baseline_dir.mkdir(parents=True)
    (data_baseline_dir / "baseline_metrics.json").write_text("{}")

    result = resolve_baseline_path("baseline_metrics.json")

    # Should find it in data/baseline/ (second candidate)
    assert result == Path("data/baseline/baseline_metrics.json")


def test_resolve_baseline_path_raises_if_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_baseline_path raises FileNotFoundError if not in either location."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Cannot find baseline_metrics.json"):
        resolve_baseline_path("baseline_metrics.json")


def test_resolve_baseline_embeddings_path_raises_if_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resolve_baseline_embeddings_path raises if baseline_eval.npz not found."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError, match="Baseline embeddings not found"):
        resolve_baseline_embeddings_path()


def test_evaluate_from_embeddings_returns_correct_bundle(tmp_path: Path) -> None:
    """Test evaluate_from_embeddings returns EvaluationBundle with correct shapes."""
    # WHY: This is the core evaluation function that calls all 8 metrics

    # Create fake embeddings (N=10 pairs, dim=384)
    n_pairs = 10
    emb_dim = 384
    emb1 = np.random.randn(n_pairs, emb_dim).astype(np.float32)
    emb2 = np.random.randn(n_pairs, emb_dim).astype(np.float32)

    # Save to temporary .npz file
    embeddings_path = tmp_path / "test_embeddings.npz"
    np.savez(embeddings_path, text1=emb1, text2=emb2)

    # Create fake eval pairs
    eval_pairs = [
        DatingPair(
            text_1=f"boy: statement {i}",
            text_2=f"girl: statement {i}",
            label=i % 2,  # Alternate compatible/incompatible
            category=f"category_{i % 3}",  # 3 categories
            subcategory=f"subcat_{i}",
            pair_type="CC" if i % 2 == 0 else "II",
        )
        for i in range(n_pairs)
    ]

    # Run evaluation
    bundle = evaluate_from_embeddings(embeddings_path, eval_pairs)

    # --- Verify bundle structure ---
    assert bundle.similarities.shape == (n_pairs,)
    assert bundle.projections.shape == (2 * n_pairs, 2)  # UMAP 2D for text1 + text2
    assert bundle.cluster_labels.shape == (n_pairs,)
    assert len(bundle.labels) == n_pairs
    assert len(bundle.categories) == n_pairs
    assert len(bundle.pair_types) == n_pairs

    # --- Verify metrics are populated ---
    metrics = bundle.metrics
    assert -1.0 <= metrics.spearman_correlation <= 1.0
    assert 0.0 <= metrics.auc_roc <= 1.0
    assert 0.0 <= metrics.cluster_purity <= 1.0
    assert metrics.n_clusters >= 0
    assert 0.0 <= metrics.noise_ratio <= 1.0
    assert 0.0 <= metrics.best_f1 <= 1.0
    assert 0.0 <= metrics.accuracy_at_best_threshold <= 1.0
    assert 0.0 <= metrics.precision_at_best_threshold <= 1.0
    assert 0.0 <= metrics.recall_at_best_threshold <= 1.0

    # --- Verify category and pair_type metrics exist ---
    assert len(metrics.category_metrics) > 0
    assert len(metrics.pair_type_metrics) > 0

    # --- Verify false positive counts is a dict ---
    assert isinstance(metrics.false_positive_counts, dict)
