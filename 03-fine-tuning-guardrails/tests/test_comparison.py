from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.comparison import (
    plot_comparison_cluster_purity,
    plot_comparison_cosine_distributions,
    plot_comparison_roc_curves,
    plot_comparison_umap,
)


@pytest.fixture
def fake_similarity_data() -> tuple[np.ndarray, list[int]]:
    """Generate fake similarity data for testing charts."""
    # WHY fixture: all chart tests need similar test data
    n_pairs = 50
    sims = np.random.uniform(-1, 1, n_pairs).astype(np.float32)
    labels = [i % 2 for i in range(n_pairs)]  # Alternate 0/1
    return sims, labels


@pytest.fixture
def fake_umap_data() -> tuple[np.ndarray, list[int]]:
    """Generate fake UMAP projection data for testing."""
    n_pairs = 50
    # WHY 2N rows: UMAP returns 2 rows per pair (text_1 + text_2)
    projections = np.random.randn(2 * n_pairs, 2).astype(np.float32)
    labels = [i % 2 for i in range(n_pairs)]
    return projections, labels


def test_plot_comparison_cosine_distributions_creates_file(
    tmp_path: Path,
    fake_similarity_data: tuple[np.ndarray, list[int]],
) -> None:
    """Test plot_comparison_cosine_distributions creates a PNG file."""
    sims, labels = fake_similarity_data
    output_path = tmp_path / "distributions.png"

    result = plot_comparison_cosine_distributions(
        baseline_sims=sims,
        baseline_labels=labels,
        standard_sims=sims,
        standard_labels=labels,
        lora_sims=sims,
        lora_labels=labels,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000  # PNG should be >1KB


def test_plot_comparison_umap_creates_file(
    tmp_path: Path,
    fake_umap_data: tuple[np.ndarray, list[int]],
) -> None:
    """Test plot_comparison_umap creates a PNG file with 1x3 panels."""
    projections, labels = fake_umap_data
    output_path = tmp_path / "umap.png"

    result = plot_comparison_umap(
        baseline_projections=projections,
        baseline_labels=labels,
        standard_projections=projections,
        standard_labels=labels,
        lora_projections=projections,
        lora_labels=labels,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_comparison_cluster_purity_creates_file(tmp_path: Path) -> None:
    """Test plot_comparison_cluster_purity creates a 3-bar chart."""
    output_path = tmp_path / "purity.png"

    result = plot_comparison_cluster_purity(
        baseline_purity=0.75,
        standard_purity=0.95,
        lora_purity=0.85,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_comparison_roc_curves_creates_file(
    tmp_path: Path,
    fake_similarity_data: tuple[np.ndarray, list[int]],
) -> None:
    """Test plot_comparison_roc_curves creates a file with 3 ROC curves."""
    sims, labels = fake_similarity_data
    output_path = tmp_path / "roc.png"

    result = plot_comparison_roc_curves(
        baseline_sims=sims,
        baseline_labels=labels,
        standard_sims=sims,
        standard_labels=labels,
        lora_sims=sims,
        lora_labels=labels,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_comparison_cosine_distributions_handles_edge_case_all_same_label(
    tmp_path: Path,
) -> None:
    """Test distributions chart handles edge case where all pairs have same label."""
    # WHY: KDE can fail with single data points or empty groups
    n_pairs = 20
    sims = np.random.uniform(-1, 1, n_pairs).astype(np.float32)
    labels = [1] * n_pairs  # All compatible
    output_path = tmp_path / "distributions_edge.png"

    # Should not crash even with degenerate data
    result = plot_comparison_cosine_distributions(
        baseline_sims=sims,
        baseline_labels=labels,
        standard_sims=sims,
        standard_labels=labels,
        lora_sims=sims,
        lora_labels=labels,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
