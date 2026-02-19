from __future__ import annotations

import numpy as np

from src.metrics import (
    compute_category_metrics,
    compute_cohens_d,
    compute_cosine_similarities,
    compute_false_positive_analysis,
    compute_margin,
    compute_roc_metrics,
    compute_spearman,
    compute_welch_ttest,
)


# ------------------------------------------------------------------ #
# Cosine similarity
# ------------------------------------------------------------------ #

def test_cosine_identical_vectors():
    """Identical vectors → similarity = 1.0."""
    v = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sims = compute_cosine_similarities(v, v)
    np.testing.assert_allclose(sims, [1.0, 1.0], atol=1e-6)


def test_cosine_orthogonal_vectors():
    """Orthogonal vectors → similarity = 0.0."""
    v1 = np.array([[1.0, 0.0, 0.0]])
    v2 = np.array([[0.0, 1.0, 0.0]])
    sims = compute_cosine_similarities(v1, v2)
    np.testing.assert_allclose(sims, [0.0], atol=1e-6)


def test_cosine_opposite_vectors():
    """Negated vectors → similarity = -1.0."""
    v1 = np.array([[1.0, 0.0, 0.0]])
    v2 = np.array([[-1.0, 0.0, 0.0]])
    sims = compute_cosine_similarities(v1, v2)
    np.testing.assert_allclose(sims, [-1.0], atol=1e-6)


def test_cosine_returns_correct_shape():
    """Output shape matches number of pairs."""
    rng = np.random.default_rng(42)
    v1 = rng.standard_normal((10, 4))
    v2 = rng.standard_normal((10, 4))
    sims = compute_cosine_similarities(v1, v2)
    assert sims.shape == (10,)


# ------------------------------------------------------------------ #
# Margin
# ------------------------------------------------------------------ #

def test_margin_perfect_separation():
    """Compatible all 0.9, incompatible all 0.1 → margin = 0.8."""
    sims = np.array([0.9, 0.9, 0.1, 0.1])
    labels = [1, 1, 0, 0]
    compat_mean, incompat_mean, margin = compute_margin(sims, labels)
    assert abs(compat_mean - 0.9) < 1e-6
    assert abs(incompat_mean - 0.1) < 1e-6
    assert abs(margin - 0.8) < 1e-6


def test_margin_no_separation():
    """All similarities identical → margin = 0.0."""
    sims = np.array([0.5, 0.5, 0.5, 0.5])
    labels = [1, 1, 0, 0]
    _, _, margin = compute_margin(sims, labels)
    assert abs(margin) < 1e-6


# ------------------------------------------------------------------ #
# Cohen's d
# ------------------------------------------------------------------ #

def test_cohens_d_large_effect():
    """Well-separated groups (0.9 vs 0.1) → d > 0.8 (large effect)."""
    group1 = np.array([0.85, 0.87, 0.90, 0.92, 0.88])
    group2 = np.array([0.10, 0.12, 0.09, 0.11, 0.13])
    d = compute_cohens_d(group1, group2)
    assert d > 0.8


def test_cohens_d_zero_effect():
    """Same distribution → d ≈ 0.0."""
    rng = np.random.default_rng(0)
    group = rng.normal(0.5, 0.05, 100)
    d = compute_cohens_d(group, group)
    assert abs(d) < 1e-6


def test_cohens_d_zero_variance():
    """Both groups identical constant → pooled_std = 0 → d = 0.0."""
    group1 = np.array([0.5, 0.5, 0.5])
    group2 = np.array([0.5, 0.5, 0.5])
    assert compute_cohens_d(group1, group2) == 0.0


# ------------------------------------------------------------------ #
# Welch t-test
# ------------------------------------------------------------------ #

def test_welch_ttest_significant():
    """Clearly separated groups → p_value < 0.05."""
    group1 = np.array([0.9] * 30)
    group2 = np.array([0.1] * 30)
    t_stat, p_val = compute_welch_ttest(group1, group2)
    assert t_stat > 0
    assert p_val < 0.05


def test_welch_ttest_nonsignificant():
    """Same distribution → p_value close to 1.0."""
    rng = np.random.default_rng(99)
    group = rng.normal(0.5, 0.1, 50)
    # Split same array → t should be near 0, p near 1
    _, p_val = compute_welch_ttest(group[:25], group[25:])
    assert p_val > 0.05


# ------------------------------------------------------------------ #
# Spearman correlation
# ------------------------------------------------------------------ #

def test_spearman_perfect_correlation():
    """Sims increasing with label → high positive correlation."""
    sims = np.array([0.1, 0.2, 0.9, 1.0])
    labels = [0, 0, 1, 1]
    corr = compute_spearman(sims, labels)
    # WHY 0.8 not 0.9: with 4 points and 2-group binary labels, the max achievable
    # Spearman rank correlation is sqrt(4/5) ≈ 0.894 due to tied ranks in labels
    assert corr > 0.8


def test_spearman_negative_correlation():
    """Higher sim → lower label → negative correlation."""
    sims = np.array([0.9, 1.0, 0.1, 0.2])
    labels = [0, 0, 1, 1]
    corr = compute_spearman(sims, labels)
    assert corr < -0.8


# ------------------------------------------------------------------ #
# ROC metrics
# ------------------------------------------------------------------ #

def test_roc_perfect_classifier():
    """Perfect separation → AUC = 1.0."""
    sims = np.array([0.9, 0.95, 0.1, 0.05])
    labels = [1, 1, 0, 0]
    auc_roc, _, best_f1, _, _, _ = compute_roc_metrics(sims, labels)
    assert abs(auc_roc - 1.0) < 1e-6
    assert abs(best_f1 - 1.0) < 1e-6


def test_roc_random_classifier():
    """Random similarities → AUC near 0.5."""
    rng = np.random.default_rng(42)
    sims = rng.uniform(0, 1, 200)
    labels = rng.integers(0, 2, 200).tolist()
    auc_roc, _, _, _, _, _ = compute_roc_metrics(sims, labels)
    assert 0.3 < auc_roc < 0.7  # Wide tolerance for random data


def test_roc_returns_six_values():
    """compute_roc_metrics returns a 6-tuple."""
    sims = np.array([0.8, 0.4, 0.9, 0.2])
    labels = [1, 0, 1, 0]
    result = compute_roc_metrics(sims, labels)
    assert len(result) == 6


# ------------------------------------------------------------------ #
# False positive analysis
# ------------------------------------------------------------------ #

def test_false_positive_analysis_basic():
    """Only label=0 pairs with sim >= threshold count as FP.

    sims=[0.8, 0.3, 0.7, 0.9], labels=[0, 0, 1, 1]
    pair_types=["dealbreaker", "subtle_mismatch", "compatible", "compatible"]
    threshold=0.5

    Pair 0: label=0, sim=0.8 >= 0.5 → FP (dealbreaker)
    Pair 1: label=0, sim=0.3 < 0.5 → not FP
    Pair 2: label=1 → not applicable (TP, not FP)
    Pair 3: label=1 → not applicable (TP, not FP)

    Expected: {"dealbreaker": 1}
    """
    sims = np.array([0.8, 0.3, 0.7, 0.9])
    labels = [0, 0, 1, 1]
    pair_types = ["dealbreaker", "subtle_mismatch", "compatible", "compatible"]
    result = compute_false_positive_analysis(sims, labels, pair_types, threshold=0.5)
    assert result == {"dealbreaker": 1}


def test_false_positive_analysis_no_fps():
    """All incompatible pairs below threshold → empty dict."""
    sims = np.array([0.1, 0.2, 0.9, 0.8])
    labels = [0, 0, 1, 1]
    pair_types = ["dealbreaker", "subtle_mismatch", "compatible", "compatible"]
    result = compute_false_positive_analysis(sims, labels, pair_types, threshold=0.5)
    assert result == {}


def test_false_positive_analysis_multiple_types():
    """Multiple FPs across pair types → correct counts, sorted descending."""
    sims = np.array([0.8, 0.9, 0.7, 0.6, 0.3])
    labels = [0, 0, 0, 0, 0]  # All incompatible
    pair_types = ["dealbreaker", "dealbreaker", "subtle_mismatch", "dealbreaker", "no_fp"]
    result = compute_false_positive_analysis(sims, labels, pair_types, threshold=0.5)
    # All 4 sims >= 0.5 are FPs; "no_fp" has sim=0.3 < 0.5
    assert result["dealbreaker"] == 3
    assert result["subtle_mismatch"] == 1
    assert "no_fp" not in result
    # Verify sorted descending by count
    counts = list(result.values())
    assert counts == sorted(counts, reverse=True)


# ------------------------------------------------------------------ #
# Category metrics
# ------------------------------------------------------------------ #

def test_category_metrics_groups():
    """Two categories with known values → correct per-group metrics."""
    sims = np.array([0.9, 0.8, 0.2, 0.3,   # cat A: compatible high, incompatible low
                     0.6, 0.7, 0.4, 0.5])   # cat B: compatible mid, incompatible mid
    labels = [1, 1, 0, 0,
              1, 1, 0, 0]
    groups = ["A", "A", "A", "A",
              "B", "B", "B", "B"]

    results = compute_category_metrics(sims, labels, groups)
    assert len(results) == 2

    cat_a = next(r for r in results if r.category == "A")
    cat_b = next(r for r in results if r.category == "B")

    # Cat A: compat_mean=0.85, incompat_mean=0.25, margin=0.60
    assert abs(cat_a.compatible_mean - 0.85) < 1e-6
    assert abs(cat_a.incompatible_mean - 0.25) < 1e-6
    assert abs(cat_a.margin - 0.60) < 1e-6
    assert cat_a.count == 4
    assert cat_a.cohens_d is not None

    # Cat B: compat_mean=0.65, incompat_mean=0.45, margin=0.20
    assert abs(cat_b.compatible_mean - 0.65) < 1e-6
    assert abs(cat_b.incompatible_mean - 0.45) < 1e-6
    assert abs(cat_b.margin - 0.20) < 1e-6


def test_category_metrics_small_group_no_cohens_d():
    """Group with <2 members in one class → cohens_d is None."""
    sims = np.array([0.9, 0.1])
    labels = [1, 0]
    groups = ["A", "A"]
    results = compute_category_metrics(sims, labels, groups)
    # Each class has only 1 member → Cohen's d undefined
    assert results[0].cohens_d is None
