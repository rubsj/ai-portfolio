from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.models import CategoryMetrics


def compute_cosine_similarities(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between paired embeddings.

    WHY manual instead of sklearn.cosine_similarity: sklearn computes an NxN
    matrix (all pairs vs all pairs). We only need the diagonal — each text_1[i]
    paired with text_2[i]. Manual normalization + dot product is O(N) not O(N²).

    Returns array of shape (N,) with cosine similarity for each pair.
    """
    # Normalize each row to unit vector: v / ||v||
    # keepdims=True preserves shape for broadcasting: (N,384) / (N,1) = (N,384)
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    # Row-wise dot product of unit vectors = cosine similarity
    return np.sum(emb1_norm * emb2_norm, axis=1)


def compute_margin(
    similarities: np.ndarray,
    labels: list[int],
) -> tuple[float, float, float]:
    """Compute compatible/incompatible means and their margin.

    WHY margin: this is the primary training signal — Day 3's fine-tuning goal
    is to widen this gap. Baseline margin anchors the "before" side of the story.

    Returns: (compatible_mean, incompatible_mean, margin)
    where margin = compatible_mean - incompatible_mean.
    """
    labels_arr = np.array(labels)
    compat_sims = similarities[labels_arr == 1]
    incompat_sims = similarities[labels_arr == 0]
    compat_mean = float(np.mean(compat_sims))
    incompat_mean = float(np.mean(incompat_sims))
    return compat_mean, incompat_mean, compat_mean - incompat_mean


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d effect size between two groups.

    WHY pooled std instead of group1 std: pooled accounts for both groups'
    variance, giving a more robust estimate when group sizes differ.

    d = (mean1 - mean2) / pooled_std
    pooled_std = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))

    Interpretation: 0.2 = small, 0.5 = medium, 0.8 = large effect.
    """
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def compute_welch_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
) -> tuple[float, float]:
    """Welch's t-test for difference in means (does not assume equal variance).

    WHY Welch not Student's t: compatible and incompatible similarity distributions
    likely have different variances — Student's t would be miscalibrated.

    Returns: (t_statistic, p_value)
    """
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
    return float(t_stat), float(p_val)


def compute_spearman(
    similarities: np.ndarray,
    labels: list[int],
) -> float:
    """Spearman rank correlation between cosine similarities and binary labels.

    WHY Spearman not Pearson: rank correlation is more robust to the non-normal
    distribution of cosine similarities and doesn't assume linearity.

    Returns the correlation coefficient (positive = higher sim → compatible).
    """
    corr, _ = stats.spearmanr(similarities, labels)
    return float(corr)


def compute_roc_metrics(
    similarities: np.ndarray,
    labels: list[int],
) -> tuple[float, float, float, float, float, float]:
    """Compute ROC-based metrics with optimal threshold.

    WHY sweep threshold: the default 0.5 is arbitrary for cosine similarities —
    the actual decision boundary shifts based on the model's score distribution.
    Finding best F1 threshold gives a meaningful operating point.

    Returns: (auc_roc, best_threshold, best_f1, accuracy, precision, recall)
    """
    fpr, tpr, _ = roc_curve(labels, similarities)
    auc_roc = float(auc(fpr, tpr))

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.0, 1.01, 0.01):
        preds = (similarities >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    preds = (similarities >= best_thresh).astype(int)
    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))

    return auc_roc, float(best_thresh), float(best_f1), acc, prec, rec


def compute_false_positive_analysis(
    similarities: np.ndarray,
    labels: list[int],
    pair_types: list[str],
    threshold: float,
) -> dict[str, int]:
    """PRD Metric #4: false positive count per pair_type.

    Finds incompatible pairs (label=0) whose cosine sim >= threshold —
    i.e., pairs the model would incorrectly call compatible.

    WHY group by pair_type: identifies which relationship dynamics the
    pre-trained model confuses most. These are the "before" numbers for
    Day 3's false positive reduction comparison chart.

    Returns dict sorted by count descending.
    """
    labels_arr = np.array(labels)
    pair_types_arr = np.array(pair_types)
    # Incompatible pairs above threshold = false positives
    fp_mask = (labels_arr == 0) & (similarities >= threshold)
    fp_pair_types = pair_types_arr[fp_mask]
    counts = Counter(fp_pair_types)
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def compute_category_metrics(
    similarities: np.ndarray,
    labels: list[int],
    groups: list[str],
) -> list[CategoryMetrics]:
    """Per-group breakdown of similarity metrics (works for category and pair_type).

    For each unique group value: compatible mean, incompatible mean, margin, Cohen's d.
    WHY reusable: called twice — once for category breakdown, once for pair_type.
    Cohen's d is None when a group has <2 members in either class (undefined).
    """
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)
    results = []

    for group in sorted(set(groups)):
        mask = groups_arr == group
        group_sims = similarities[mask]
        group_labels = labels_arr[mask]

        compat = group_sims[group_labels == 1]
        incompat = group_sims[group_labels == 0]

        compat_mean = float(np.mean(compat)) if len(compat) > 0 else 0.0
        incompat_mean = float(np.mean(incompat)) if len(incompat) > 0 else 0.0
        margin = compat_mean - incompat_mean

        # WHY >=2 check: Cohen's d requires variance estimate (ddof=1), needs n>=2
        d = compute_cohens_d(compat, incompat) if len(compat) >= 2 and len(incompat) >= 2 else None

        results.append(CategoryMetrics(
            category=group,
            compatible_mean=compat_mean,
            incompatible_mean=incompat_mean,
            margin=margin,
            count=int(mask.sum()),
            cohens_d=d,
        ))

    return results
