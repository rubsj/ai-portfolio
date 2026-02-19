from __future__ import annotations


import numpy as np

from src.models import BaselineMetrics, CategoryMetrics


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _make_category_metrics(category: str, margin: float = 0.1) -> CategoryMetrics:
    return CategoryMetrics(
        category=category,
        compatible_mean=0.6,
        incompatible_mean=0.6 - margin,
        margin=margin,
        count=10,
        cohens_d=0.5,
    )


def _make_baseline_metrics() -> BaselineMetrics:
    return BaselineMetrics(
        compatible_mean_cosine=0.65,
        incompatible_mean_cosine=0.45,
        compatibility_margin=0.20,
        cohens_d=1.2,
        t_statistic=5.0,
        p_value=0.001,
        auc_roc=0.78,
        best_threshold=0.55,
        best_f1=0.80,
        accuracy_at_best_threshold=0.82,
        precision_at_best_threshold=0.79,
        recall_at_best_threshold=0.81,
        cluster_purity=0.85,
        n_clusters=4,
        noise_ratio=0.15,
        spearman_correlation=0.45,
        false_positive_counts={"dealbreaker": 5, "subtle_mismatch": 2},
        category_metrics=[_make_category_metrics("hobbies", 0.25)],
        pair_type_metrics=[_make_category_metrics("compatible_match", 0.30)],
    )


def _make_sims_labels(n: int = 30, rng_seed: int = 0) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(rng_seed)
    # Compatible sims higher, incompatible lower
    compat = rng.uniform(0.5, 1.0, n // 2)
    incompat = rng.uniform(0.0, 0.5, n - n // 2)
    sims = np.concatenate([compat, incompat])
    labels: list[int] = [1] * (n // 2) + [0] * (n - n // 2)
    return sims, labels


# ------------------------------------------------------------------ #
# plot_cosine_distributions
# ------------------------------------------------------------------ #

def test_plot_cosine_distributions_saves_png(tmp_path):
    """plot_cosine_distributions saves a PNG file at the given path."""
    from src.visualizations import plot_cosine_distributions

    sims, labels = _make_sims_labels()
    out = tmp_path / "dist.png"
    result = plot_cosine_distributions(sims, labels, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_cosine_distributions_default_path(tmp_path, monkeypatch):
    """When save_path is None, uses BASELINE_VIZ_DIR / cosine_distributions.png."""
    import src.visualizations as viz_module

    sims, labels = _make_sims_labels()
    fake_dir = tmp_path / "baseline"
    fake_dir.mkdir()
    monkeypatch.setattr(viz_module, "BASELINE_VIZ_DIR", fake_dir)

    result = viz_module.plot_cosine_distributions(sims, labels, save_path=None)
    assert result.name == "cosine_distributions.png"
    assert result.exists()


# ------------------------------------------------------------------ #
# plot_umap_scatter
# ------------------------------------------------------------------ #

def test_plot_umap_scatter_saves_html(tmp_path):
    """plot_umap_scatter saves an HTML file (Plotly interactive chart)."""
    from src.visualizations import plot_umap_scatter

    rng = np.random.default_rng(5)
    n = 20
    proj = rng.standard_normal((2 * n, 2))
    labels = [i % 2 for i in range(n)]
    categories = ["hobbies"] * n

    out = tmp_path / "umap.html"
    result = plot_umap_scatter(proj, labels, categories, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 1000  # Non-trivial HTML


# ------------------------------------------------------------------ #
# plot_hdbscan_clusters
# ------------------------------------------------------------------ #

def test_plot_hdbscan_clusters_saves_png(tmp_path):
    """plot_hdbscan_clusters saves a PNG given projections and cluster labels."""
    from src.visualizations import plot_hdbscan_clusters

    rng = np.random.default_rng(7)
    n = 20
    proj = rng.standard_normal((2 * n, 2))
    cluster_labels = np.array([0] * 8 + [1] * 7 + [-1] * 5)  # 2 clusters + noise

    out = tmp_path / "hdbscan.png"
    result = plot_hdbscan_clusters(proj, cluster_labels, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_hdbscan_clusters_default_path(tmp_path, monkeypatch):
    """When save_path is None, uses BASELINE_VIZ_DIR."""
    import src.visualizations as viz_module

    rng = np.random.default_rng(8)
    n = 15
    proj = rng.standard_normal((2 * n, 2))
    cluster_labels = np.array([0] * n)

    fake_dir = tmp_path / "baseline"
    fake_dir.mkdir()
    monkeypatch.setattr(viz_module, "BASELINE_VIZ_DIR", fake_dir)

    result = viz_module.plot_hdbscan_clusters(proj, cluster_labels, save_path=None)
    assert result.name == "hdbscan_clusters.png"
    assert result.exists()


# ------------------------------------------------------------------ #
# plot_roc_curve
# ------------------------------------------------------------------ #

def test_plot_roc_curve_saves_png(tmp_path):
    """plot_roc_curve saves a PNG with ROC curve."""
    from src.visualizations import plot_roc_curve

    sims, labels = _make_sims_labels()
    out = tmp_path / "roc.png"
    result = plot_roc_curve(sims, labels, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


# ------------------------------------------------------------------ #
# plot_category_margins
# ------------------------------------------------------------------ #

def test_plot_category_margins_saves_png(tmp_path):
    """plot_category_margins saves a horizontal bar chart PNG."""
    from src.visualizations import plot_category_margins

    cat_metrics = [
        _make_category_metrics("hobbies", 0.30),
        _make_category_metrics("values", 0.10),
        _make_category_metrics("lifestyle", -0.05),
    ]
    out = tmp_path / "margins.png"
    result = plot_category_margins(cat_metrics, "Test Margins", out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


# ------------------------------------------------------------------ #
# plot_false_positive_breakdown
# ------------------------------------------------------------------ #

def test_plot_false_positive_breakdown_with_fps(tmp_path):
    """plot_false_positive_breakdown with non-empty counts saves a bar chart."""
    from src.visualizations import plot_false_positive_breakdown

    fp_counts = {"dealbreaker": 10, "subtle_mismatch": 4, "complex": 1}
    out = tmp_path / "fp.png"
    result = plot_false_positive_breakdown(fp_counts, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_false_positive_breakdown_no_fps(tmp_path):
    """plot_false_positive_breakdown with empty dict saves a placeholder chart."""
    from src.visualizations import plot_false_positive_breakdown

    out = tmp_path / "fp_empty.png"
    result = plot_false_positive_breakdown({}, out)

    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


# ------------------------------------------------------------------ #
# generate_baseline_report_html
# ------------------------------------------------------------------ #

def test_generate_baseline_report_html_saves_file(tmp_path):
    """generate_baseline_report_html writes a non-empty HTML file."""
    from src.visualizations import generate_baseline_report_html

    metrics = _make_baseline_metrics()
    # Use empty chart_paths — missing charts produce placeholder text
    result = generate_baseline_report_html(metrics, {}, tmp_path / "report.html")

    assert result.exists()
    assert result.stat().st_size > 500


def test_generate_baseline_report_html_contains_metrics(tmp_path):
    """HTML report contains key metric values from BaselineMetrics."""
    from src.visualizations import generate_baseline_report_html

    metrics = _make_baseline_metrics()
    report_path = tmp_path / "report.html"
    generate_baseline_report_html(metrics, {}, report_path)

    html = report_path.read_text()
    # Should contain the metric names or values
    assert "compatibility_margin" in html.lower() or "Compatibility Margin" in html
    assert "auc_roc" in html.lower() or "Auc Roc" in html


def test_generate_baseline_report_html_embeds_png(tmp_path):
    """PNG charts in chart_paths are embedded as base64 data URIs."""
    from src.visualizations import (
        generate_baseline_report_html,
        plot_cosine_distributions,
    )

    sims, labels = _make_sims_labels()
    png_path = tmp_path / "dist.png"
    plot_cosine_distributions(sims, labels, png_path)

    metrics = _make_baseline_metrics()
    report_path = tmp_path / "report.html"
    generate_baseline_report_html(metrics, {"distributions": png_path}, report_path)

    html = report_path.read_text()
    assert "data:image/png;base64," in html
