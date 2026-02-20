from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.comparison import (
    plot_comparison_category_heatmap,
    plot_comparison_classification_metrics,
    plot_comparison_cluster_purity,
    plot_comparison_cohens_d,
    plot_comparison_cosine_distributions,
    plot_comparison_false_positives,
    plot_comparison_roc_curves,
    plot_comparison_umap,
    write_false_positive_analysis,
)
from src.models import BaselineMetrics, CategoryMetrics


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


def test_plot_comparison_category_heatmap_creates_file(tmp_path: Path) -> None:
    """Test plot_comparison_category_heatmap creates a heatmap PNG."""
    output_path = tmp_path / "category_heatmap.png"

    # Create fake CategoryMetrics (3 categories × 3 models)
    # WHY fields: CategoryMetrics = category, compatible_mean, incompatible_mean, margin, count, cohens_d
    baseline_cat = [
        CategoryMetrics(category="music_taste", compatible_mean=0.25, incompatible_mean=0.15, margin=0.10, count=10, cohens_d=0.3),
        CategoryMetrics(category="travel_style", compatible_mean=0.30, incompatible_mean=0.15, margin=0.15, count=10, cohens_d=0.4),
        CategoryMetrics(category="food_preferences", compatible_mean=0.23, incompatible_mean=0.15, margin=0.08, count=10, cohens_d=0.2),
    ]
    standard_cat = [
        CategoryMetrics(category="music_taste", compatible_mean=0.90, incompatible_mean=0.05, margin=0.85, count=10, cohens_d=2.5),
        CategoryMetrics(category="travel_style", compatible_mean=0.95, incompatible_mean=0.05, margin=0.90, count=10, cohens_d=2.8),
        CategoryMetrics(category="food_preferences", compatible_mean=0.85, incompatible_mean=0.05, margin=0.80, count=10, cohens_d=2.3),
    ]
    lora_cat = [
        CategoryMetrics(category="music_taste", compatible_mean=0.82, incompatible_mean=0.07, margin=0.75, count=10, cohens_d=2.2),
        CategoryMetrics(category="travel_style", compatible_mean=0.85, incompatible_mean=0.07, margin=0.78, count=10, cohens_d=2.4),
        CategoryMetrics(category="food_preferences", compatible_mean=0.79, incompatible_mean=0.07, margin=0.72, count=10, cohens_d=2.1),
    ]

    result = plot_comparison_category_heatmap(
        baseline_category_metrics=baseline_cat,
        standard_category_metrics=standard_cat,
        lora_category_metrics=lora_cat,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_comparison_cohens_d_creates_file(tmp_path: Path) -> None:
    """Test plot_comparison_cohens_d creates a color-coded bar chart."""
    output_path = tmp_path / "cohens_d.png"

    result = plot_comparison_cohens_d(
        baseline_d=0.15,  # small (red)
        standard_d=2.50,  # large (green)
        lora_d=0.50,  # medium (yellow)
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_comparison_classification_metrics_creates_file(tmp_path: Path) -> None:
    """Test plot_comparison_classification_metrics creates a grouped bar chart."""
    output_path = tmp_path / "classification_metrics.png"

    # Create fake BaselineMetrics instances
    baseline = BaselineMetrics(
        compatible_mean_cosine=0.3,
        incompatible_mean_cosine=0.2,
        compatibility_margin=0.1,
        cohens_d=0.3,
        t_statistic=1.5,
        p_value=0.05,
        auc_roc=0.65,
        best_threshold=0.25,
        best_f1=0.55,
        accuracy_at_best_threshold=0.60,
        precision_at_best_threshold=0.58,
        recall_at_best_threshold=0.62,
        cluster_purity=0.50,
        n_clusters=2,
        noise_ratio=0.1,
        spearman_correlation=0.30,
        false_positive_counts={},
        category_metrics=[],
        pair_type_metrics=[],
    )

    standard = BaselineMetrics(
        compatible_mean_cosine=0.85,
        incompatible_mean_cosine=-0.05,
        compatibility_margin=0.90,
        cohens_d=2.5,
        t_statistic=15.0,
        p_value=0.001,
        auc_roc=0.95,
        best_threshold=0.40,
        best_f1=0.92,
        accuracy_at_best_threshold=0.93,
        precision_at_best_threshold=0.91,
        recall_at_best_threshold=0.94,
        cluster_purity=0.88,
        n_clusters=2,
        noise_ratio=0.05,
        spearman_correlation=0.85,
        false_positive_counts={},
        category_metrics=[],
        pair_type_metrics=[],
    )

    lora = BaselineMetrics(
        compatible_mean_cosine=0.80,
        incompatible_mean_cosine=-0.03,
        compatibility_margin=0.83,
        cohens_d=2.3,
        t_statistic=13.5,
        p_value=0.001,
        auc_roc=0.93,
        best_threshold=0.38,
        best_f1=0.89,
        accuracy_at_best_threshold=0.90,
        precision_at_best_threshold=0.88,
        recall_at_best_threshold=0.91,
        cluster_purity=0.85,
        n_clusters=2,
        noise_ratio=0.06,
        spearman_correlation=0.82,
        false_positive_counts={},
        category_metrics=[],
        pair_type_metrics=[],
    )

    result = plot_comparison_classification_metrics(
        baseline_metrics=baseline,
        standard_metrics=standard,
        lora_metrics=lora,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_plot_comparison_false_positives_creates_file(tmp_path: Path) -> None:
    """Test plot_comparison_false_positives creates a horizontal grouped bar chart."""
    output_path = tmp_path / "false_positives.png"

    baseline_fp = {"CC": 5, "II": 8, "CI": 12, "IC": 10}
    standard_fp = {"CC": 1, "II": 2, "CI": 3, "IC": 2}
    lora_fp = {"CC": 2, "II": 3, "CI": 4, "IC": 3}

    result = plot_comparison_false_positives(
        baseline_fp_counts=baseline_fp,
        standard_fp_counts=standard_fp,
        lora_fp_counts=lora_fp,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 1000


def test_write_false_positive_analysis_creates_file(tmp_path: Path) -> None:
    """Test write_false_positive_analysis creates a markdown report."""
    output_path = tmp_path / "fp_analysis.txt"

    # Use same fake metrics as classification_metrics test
    baseline = BaselineMetrics(
        compatible_mean_cosine=0.3,
        incompatible_mean_cosine=0.2,
        compatibility_margin=0.1,
        cohens_d=0.3,
        t_statistic=1.5,
        p_value=0.05,
        auc_roc=0.65,
        best_threshold=0.25,
        best_f1=0.55,
        accuracy_at_best_threshold=0.60,
        precision_at_best_threshold=0.58,
        recall_at_best_threshold=0.62,
        cluster_purity=0.50,
        n_clusters=2,
        noise_ratio=0.1,
        spearman_correlation=0.30,
        false_positive_counts={"CC": 5, "II": 8},
        category_metrics=[],
        pair_type_metrics=[],
    )

    standard = BaselineMetrics(
        compatible_mean_cosine=0.85,
        incompatible_mean_cosine=-0.05,
        compatibility_margin=0.90,
        cohens_d=2.5,
        t_statistic=15.0,
        p_value=0.001,
        auc_roc=0.95,
        best_threshold=0.40,
        best_f1=0.92,
        accuracy_at_best_threshold=0.93,
        precision_at_best_threshold=0.91,
        recall_at_best_threshold=0.94,
        cluster_purity=0.88,
        n_clusters=2,
        noise_ratio=0.05,
        spearman_correlation=0.85,
        false_positive_counts={"CC": 1, "II": 2},
        category_metrics=[],
        pair_type_metrics=[],
    )

    lora = BaselineMetrics(
        compatible_mean_cosine=0.80,
        incompatible_mean_cosine=-0.03,
        compatibility_margin=0.83,
        cohens_d=2.3,
        t_statistic=13.5,
        p_value=0.001,
        auc_roc=0.93,
        best_threshold=0.38,
        best_f1=0.89,
        accuracy_at_best_threshold=0.90,
        precision_at_best_threshold=0.88,
        recall_at_best_threshold=0.91,
        cluster_purity=0.85,
        n_clusters=2,
        noise_ratio=0.06,
        spearman_correlation=0.82,
        false_positive_counts={"CC": 2, "II": 3},
        category_metrics=[],
        pair_type_metrics=[],
    )

    result = write_false_positive_analysis(
        baseline_metrics=baseline,
        standard_metrics=standard,
        lora_metrics=lora,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    content = output_path.read_text()
    # Verify key sections exist
    assert "# False Positive Analysis" in content
    assert "## Summary by Pair Type" in content
    assert "## Overall Totals" in content
    # Verify reduction percentages calculated
    assert "% reduction" in content


def test_generate_all_comparison_charts_creates_8_pngs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_all_comparison_charts creates all 8 chart files."""
    # WHY: This orchestrator function calls all 8 chart functions
    from src.comparison import generate_all_comparison_charts
    from src.models import EvaluationBundle

    # WHY monkeypatch cwd: function hardcodes output_dir = Path("eval/visualizations/comparison")
    monkeypatch.chdir(tmp_path)

    # Create fake bundles (minimal data to pass through)
    n_pairs = 20
    sims = np.random.uniform(-1, 1, n_pairs).astype(np.float32)
    labels = [i % 2 for i in range(n_pairs)]
    projections = np.random.randn(2 * n_pairs, 2).astype(np.float32)
    cluster_labels = np.zeros(n_pairs, dtype=np.int32)
    categories = [f"cat_{i % 3}" for i in range(n_pairs)]
    pair_types = ["CC" if i % 2 == 0 else "II" for i in range(n_pairs)]

    fake_metrics = BaselineMetrics(
        compatible_mean_cosine=0.3,
        incompatible_mean_cosine=0.2,
        compatibility_margin=0.1,
        cohens_d=0.3,
        t_statistic=1.5,
        p_value=0.05,
        auc_roc=0.65,
        best_threshold=0.25,
        best_f1=0.55,
        accuracy_at_best_threshold=0.60,
        precision_at_best_threshold=0.58,
        recall_at_best_threshold=0.62,
        cluster_purity=0.50,
        n_clusters=2,
        noise_ratio=0.1,
        spearman_correlation=0.30,
        false_positive_counts={"CC": 5, "II": 8},
        category_metrics=[
            CategoryMetrics(category="cat_0", compatible_mean=0.3, incompatible_mean=0.2, margin=0.1, count=7, cohens_d=0.3),
            CategoryMetrics(category="cat_1", compatible_mean=0.3, incompatible_mean=0.2, margin=0.1, count=7, cohens_d=0.3),
            CategoryMetrics(category="cat_2", compatible_mean=0.3, incompatible_mean=0.2, margin=0.1, count=6, cohens_d=0.3),
        ],
        pair_type_metrics=[],
    )

    baseline_bundle = EvaluationBundle(
        metrics=fake_metrics,
        similarities=sims,
        projections=projections,
        cluster_labels=cluster_labels,
        labels=labels,
        categories=categories,
        pair_types=pair_types,
    )

    # Call orchestrator (no output_dir parameter — it's hardcoded)
    chart_paths = generate_all_comparison_charts(
        baseline_bundle=baseline_bundle,
        standard_bundle=baseline_bundle,  # Use same data for simplicity
        lora_bundle=baseline_bundle,
    )

    # Verify all 8 chart files created
    assert len(chart_paths) == 8
    # WHY check chart keys match actual function output (line 678-742 in src/comparison.py)
    expected_keys = [
        "distributions",  # Not "cosine_distributions"
        "umap",
        "cluster_purity",
        "roc",  # Not "roc_curves"
        "category_heatmap",
        "cohens_d",
        "classification_metrics",
        "false_positives",
    ]
    for key in expected_keys:
        assert key in chart_paths
        assert chart_paths[key].exists()
        assert chart_paths[key].stat().st_size > 1000


def test_build_comparison_result_computes_improvements(tmp_path: Path) -> None:
    """Test build_comparison_result computes improvement deltas correctly."""
    from src.comparison import build_comparison_result

    baseline = BaselineMetrics(
        compatible_mean_cosine=0.3,
        incompatible_mean_cosine=0.2,
        compatibility_margin=0.1,
        cohens_d=0.3,
        t_statistic=1.5,
        p_value=0.05,
        auc_roc=0.65,
        best_threshold=0.25,
        best_f1=0.55,
        accuracy_at_best_threshold=0.60,
        precision_at_best_threshold=0.58,
        recall_at_best_threshold=0.62,
        cluster_purity=0.50,
        n_clusters=2,
        noise_ratio=0.1,
        spearman_correlation=0.30,
        false_positive_counts={},
        category_metrics=[],
        pair_type_metrics=[],
    )

    standard = BaselineMetrics(
        compatible_mean_cosine=0.85,
        incompatible_mean_cosine=-0.05,
        compatibility_margin=0.90,
        cohens_d=2.5,
        t_statistic=15.0,
        p_value=0.001,
        auc_roc=0.95,
        best_threshold=0.40,
        best_f1=0.92,
        accuracy_at_best_threshold=0.93,
        precision_at_best_threshold=0.91,
        recall_at_best_threshold=0.94,
        cluster_purity=0.88,
        n_clusters=2,
        noise_ratio=0.05,
        spearman_correlation=0.85,
        false_positive_counts={},
        category_metrics=[],
        pair_type_metrics=[],
    )

    lora = BaselineMetrics(
        compatible_mean_cosine=0.80,
        incompatible_mean_cosine=-0.03,
        compatibility_margin=0.83,
        cohens_d=2.3,
        t_statistic=13.5,
        p_value=0.001,
        auc_roc=0.93,
        best_threshold=0.38,
        best_f1=0.89,
        accuracy_at_best_threshold=0.90,
        precision_at_best_threshold=0.88,
        recall_at_best_threshold=0.91,
        cluster_purity=0.85,
        n_clusters=2,
        noise_ratio=0.06,
        spearman_correlation=0.82,
        false_positive_counts={},
        category_metrics=[],
        pair_type_metrics=[],
    )

    result = build_comparison_result(baseline, standard, lora)

    # Verify improvement deltas computed correctly
    # spearman_improvement = standard - baseline = 0.85 - 0.30 = 0.55
    assert abs(result.spearman_improvement - 0.55) < 0.01
    # margin_improvement = standard - baseline = 0.90 - 0.10 = 0.80
    assert abs(result.margin_improvement - 0.80) < 0.01


def test_generate_comparison_report_html_creates_self_contained_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_comparison_report_html creates self-contained HTML with embedded charts."""
    from src.comparison import generate_comparison_report_html

    # WHY monkeypatch cwd: output_path is hardcoded to "eval/comparison_report.html" (line 817)
    monkeypatch.chdir(tmp_path)

    # Create minimal fake metrics
    fake_metrics = BaselineMetrics(
        compatible_mean_cosine=0.3,
        incompatible_mean_cosine=0.2,
        compatibility_margin=0.1,
        cohens_d=0.3,
        t_statistic=1.5,
        p_value=0.05,
        auc_roc=0.65,
        best_threshold=0.25,
        best_f1=0.55,
        accuracy_at_best_threshold=0.60,
        precision_at_best_threshold=0.58,
        recall_at_best_threshold=0.62,
        cluster_purity=0.50,
        n_clusters=2,
        noise_ratio=0.1,
        spearman_correlation=0.30,
        false_positive_counts={},
        category_metrics=[],
        pair_type_metrics=[],
    )

    # Create fake chart PNG files (empty is fine for test)
    chart_dir = tmp_path / "charts"
    chart_dir.mkdir()
    chart_paths = {}
    # WHY chart names match keys expected by generate_comparison_report_html (line 860-869 in src/comparison.py)
    for chart_name in [
        "distributions",  # Not "cosine_distributions"
        "umap",
        "cluster_purity",
        "roc",  # Not "roc_curves"
        "category_heatmap",
        "cohens_d",
        "classification_metrics",
        "false_positives",
    ]:
        chart_path = chart_dir / f"{chart_name}.png"
        # WHY write minimal PNG: base64 encoding needs valid file
        chart_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        chart_paths[chart_name] = chart_path

    # WHY no output_path parameter: function signature doesn't accept it (line 800-802)
    result = generate_comparison_report_html(
        baseline_metrics=fake_metrics,
        standard_metrics=fake_metrics,
        lora_metrics=fake_metrics,
        chart_paths=chart_paths,
    )

    # Verify output created at hardcoded path
    expected_path = Path("eval/comparison_report.html")
    assert result == expected_path
    assert expected_path.exists()

    content = expected_path.read_text()
    # Verify HTML structure
    assert "<!DOCTYPE html>" in content
    assert "<html" in content  # WHY "<html" not "<html>": actual tag is <html lang="en">
    assert "Comparison Report" in content
    # Verify metrics table included
    assert "Spearman" in content
    assert "Margin" in content
    # Verify base64-encoded charts included
    assert "data:image/png;base64," in content
    # Should have 8 embedded charts
    assert content.count("data:image/png;base64,") == 8
    # File should be >1KB (self-contained with 8 minimal charts + HTML structure)
    # WHY 1000 not 10000: test uses minimal PNGs (header + 100 bytes each)
    assert expected_path.stat().st_size > 1000
