from __future__ import annotations

import gc
from pathlib import Path

import hdbscan
import numpy as np
import umap

from src.data_loader import get_categories, get_pair_types, load_pairs, pairs_to_texts
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
from src.models import BaselineMetrics, DatingPair

EMBEDDINGS_DIR = Path("data/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"
# WHY 16: M2 8GB RAM — smaller batches prevent OOM during sentence-transformers encoding
BATCH_SIZE = 16


def generate_all_baseline_embeddings(
    train_pairs: list[DatingPair],
    eval_pairs: list[DatingPair],
) -> None:
    """Generate and save embeddings for both splits in a single model load.

    WHY single load: loading the ~90MB model twice doubles peak RAM. With 8GB M2,
    we encode train + eval sequentially before releasing the model.

    Saves:
        data/embeddings/baseline_train.npz — shape (1195, 384) × 2
        data/embeddings/baseline_eval.npz  — shape (295, 384) × 2
    """
    # Import here to keep the module importable without torch/sentence-transformers
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)

    # --- Train split ---
    train_t1, train_t2, _ = pairs_to_texts(train_pairs)
    print(f"Encoding {len(train_t1)} train text_1 pairs…")
    train_emb1 = model.encode(train_t1, batch_size=BATCH_SIZE, show_progress_bar=True)
    print(f"Encoding {len(train_t2)} train text_2 pairs…")
    train_emb2 = model.encode(train_t2, batch_size=BATCH_SIZE, show_progress_bar=True)
    np.savez(EMBEDDINGS_DIR / "baseline_train.npz", text1=train_emb1, text2=train_emb2)
    print(f"Saved baseline_train.npz — shape {train_emb1.shape}")

    # --- Eval split ---
    eval_t1, eval_t2, _ = pairs_to_texts(eval_pairs)
    print(f"Encoding {len(eval_t1)} eval text_1 pairs…")
    eval_emb1 = model.encode(eval_t1, batch_size=BATCH_SIZE, show_progress_bar=True)
    print(f"Encoding {len(eval_t2)} eval text_2 pairs…")
    eval_emb2 = model.encode(eval_t2, batch_size=BATCH_SIZE, show_progress_bar=True)
    np.savez(EMBEDDINGS_DIR / "baseline_eval.npz", text1=eval_emb1, text2=eval_emb2)
    print(f"Saved baseline_eval.npz — shape {eval_emb1.shape}")

    # WHY explicit del + gc: Python GC is non-deterministic; torch caches weights
    # on MPS/CPU. Explicit cleanup ensures the ~90MB model is released before
    # UMAP/HDBSCAN load their own data structures.
    del model
    gc.collect()


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load saved .npz embeddings. Returns (text1_emb, text2_emb)."""
    data = np.load(path)
    return data["text1"], data["text2"]


def run_umap_projection(
    emb1: np.ndarray,
    emb2: np.ndarray,
    random_state: int = 42,
) -> np.ndarray:
    """Project concatenated paired embeddings from 384d → 2d via UMAP.

    Input:  emb1 shape (N, 384), emb2 shape (N, 384)
    Output: projections shape (2N, 2)
            First N rows = text_1 projections, last N rows = text_2 projections.

    WHY concatenate: UMAP builds a shared manifold for both sides of each pair.
    This lets us visualize whether compatible pairs cluster together regardless
    of which person is text_1 or text_2.

    Config:
        n_neighbors=15: balances local vs global structure for ~590 points (2×295)
        min_dist=0.1: moderate compactness — shows clusters without over-squishing
        metric='cosine': matches the similarity metric used in the baseline
    """
    combined = np.vstack([emb1, emb2])  # shape (2N, 384)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
    )
    projections: np.ndarray = reducer.fit_transform(combined)
    return projections  # shape (2N, 2)


def run_hdbscan_clustering(
    projections: np.ndarray,
    labels: list[int],
) -> tuple[float, int, float, np.ndarray]:
    """Cluster averaged pair projections with HDBSCAN.

    WHY average text_1 + text_2 projections: each pair has two UMAP points
    (one for each person). Averaging gives a single "pair location" in 2D space
    that represents the combined semantic position of the pair.

    Config:
        min_cluster_size=10: prevents many tiny clusters on 295 pairs
        min_samples=5: controls how conservative cluster boundaries are
        cluster_selection_method='eom': excess of mass — better for variable-density data

    Returns: (purity, n_clusters, noise_ratio, cluster_labels)
        purity: weighted average of dominant-label fraction per cluster
    """
    n = len(labels)
    # Average the two 2D points per pair → one point per pair
    pair_projections = (projections[:n] + projections[n:]) / 2  # shape (N, 2)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        cluster_selection_method="eom",
    )
    cluster_labels: np.ndarray = clusterer.fit_predict(pair_projections)

    labels_arr = np.array(labels)
    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)  # -1 = noise in HDBSCAN

    n_clusters = len(unique_clusters)
    noise_count = int(np.sum(cluster_labels == -1))
    noise_ratio = float(noise_count / len(cluster_labels))

    if n_clusters == 0:
        return 0.0, 0, noise_ratio, cluster_labels

    # Purity: for each cluster, fraction belonging to the majority class
    # Weighted by cluster size so large clusters have more influence
    purity_sum = 0
    total_in_clusters = 0
    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_size = int(mask.sum())
        dominant_count = int(max(
            np.sum(labels_arr[mask] == 0),
            np.sum(labels_arr[mask] == 1),
        ))
        purity_sum += dominant_count
        total_in_clusters += cluster_size

    purity = float(purity_sum / total_in_clusters) if total_in_clusters > 0 else 0.0
    return purity, n_clusters, noise_ratio, cluster_labels


def run_full_baseline(
    train_pairs: list[DatingPair],
    eval_pairs: list[DatingPair],
) -> BaselineMetrics:
    """End-to-end Day 1 baseline pipeline.

    Steps:
    1. Generate embeddings for BOTH splits (single model load)
    2. Load EVAL embeddings (WHY eval: measures generalization, not memorization)
    3. Compute cosine similarities on eval pairs
    4. Compute all metrics: margin, Cohen's d, Welch t-test, Spearman, ROC,
       false positive analysis, category breakdown, pair_type breakdown
    5. Run UMAP projection on eval embeddings
    6. Run HDBSCAN clustering on UMAP projections
    7. Generate 7 charts (6 PNG + 1 HTML)
    8. Save eval/baseline_metrics.json
    9. Generate eval/baseline_report.html

    Returns the populated BaselineMetrics instance.
    """
    from src.visualizations import (
        generate_baseline_report_html,
        plot_category_margins,
        plot_cosine_distributions,
        plot_false_positive_breakdown,
        plot_hdbscan_clusters,
        plot_roc_curve,
        plot_umap_scatter,
    )

    # --- Step 1: Generate embeddings ---
    train_emb_path = EMBEDDINGS_DIR / "baseline_train.npz"
    eval_emb_path = EMBEDDINGS_DIR / "baseline_eval.npz"
    if not train_emb_path.exists() or not eval_emb_path.exists():
        print("Embeddings not found — generating now…")
        generate_all_baseline_embeddings(train_pairs, eval_pairs)
    else:
        print("Embeddings already exist — skipping generation.")

    # --- Step 2: Load eval embeddings ---
    eval_emb1, eval_emb2 = load_embeddings(eval_emb_path)
    _, _, eval_labels = pairs_to_texts(eval_pairs)
    eval_categories = get_categories(eval_pairs)
    eval_pair_types = get_pair_types(eval_pairs)

    # --- Step 3: Cosine similarities ---
    print("Computing cosine similarities…")
    similarities = compute_cosine_similarities(eval_emb1, eval_emb2)

    # --- Step 4: Metrics ---
    print("Computing metrics…")
    compat_mean, incompat_mean, margin = compute_margin(similarities, eval_labels)
    compat_sims = similarities[np.array(eval_labels) == 1]
    incompat_sims = similarities[np.array(eval_labels) == 0]
    d = compute_cohens_d(compat_sims, incompat_sims)
    t_stat, p_val = compute_welch_ttest(compat_sims, incompat_sims)
    spearman_corr = compute_spearman(similarities, eval_labels)
    auc_roc, best_thresh, best_f1, acc, prec, rec = compute_roc_metrics(similarities, eval_labels)
    fp_counts = compute_false_positive_analysis(similarities, eval_labels, eval_pair_types, threshold=best_thresh)
    cat_metrics = compute_category_metrics(similarities, eval_labels, eval_categories)
    ptype_metrics = compute_category_metrics(similarities, eval_labels, eval_pair_types)

    # --- Step 5: UMAP ---
    print("Running UMAP projection…")
    projections = run_umap_projection(eval_emb1, eval_emb2)

    # --- Step 6: HDBSCAN ---
    print("Running HDBSCAN clustering…")
    purity, n_clusters, noise_ratio, cluster_labels = run_hdbscan_clustering(projections, eval_labels)

    # --- Assemble metrics ---
    metrics = BaselineMetrics(
        compatible_mean_cosine=compat_mean,
        incompatible_mean_cosine=incompat_mean,
        compatibility_margin=margin,
        cohens_d=d,
        t_statistic=t_stat,
        p_value=p_val,
        auc_roc=auc_roc,
        best_threshold=float(best_thresh),
        best_f1=best_f1,
        accuracy_at_best_threshold=acc,
        precision_at_best_threshold=prec,
        recall_at_best_threshold=rec,
        cluster_purity=purity,
        n_clusters=n_clusters,
        noise_ratio=noise_ratio,
        spearman_correlation=spearman_corr,
        false_positive_counts=fp_counts,
        category_metrics=cat_metrics,
        pair_type_metrics=ptype_metrics,
    )

    # --- Step 7: Charts ---
    print("Generating charts…")
    viz_dir = Path("eval/visualizations/baseline")
    viz_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: dict[str, Path] = {}
    chart_paths["distributions"] = plot_cosine_distributions(
        similarities, eval_labels, viz_dir / "cosine_distributions.png"
    )
    chart_paths["umap"] = plot_umap_scatter(
        projections, eval_labels, eval_categories, viz_dir / "umap_scatter.html"
    )
    chart_paths["hdbscan"] = plot_hdbscan_clusters(
        projections, cluster_labels, viz_dir / "hdbscan_clusters.png"
    )
    chart_paths["roc"] = plot_roc_curve(
        similarities, eval_labels, viz_dir / "roc_curve.png"
    )
    chart_paths["category_margins"] = plot_category_margins(
        cat_metrics, "Margin by Category", viz_dir / "category_margins.png"
    )
    chart_paths["pairtype_margins"] = plot_category_margins(
        ptype_metrics, "Margin by Pair Type", viz_dir / "pairtype_margins.png"
    )
    chart_paths["false_positives"] = plot_false_positive_breakdown(
        fp_counts, viz_dir / "false_positive_breakdown.png"
    )

    # --- Step 8: Save metrics JSON ---
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = eval_dir / "baseline_metrics.json"
    metrics_path.write_text(metrics.model_dump_json(indent=2))
    print(f"Saved {metrics_path}")

    # --- Step 9: HTML report ---
    report_path = generate_baseline_report_html(metrics, chart_paths, eval_dir / "baseline_report.html")
    print(f"Saved {report_path}")

    # --- Print summary ---
    print("\n=== BASELINE METRICS SUMMARY ===")
    print(f"  Compatible mean cosine:   {compat_mean:.4f}")
    print(f"  Incompatible mean cosine: {incompat_mean:.4f}")
    print(f"  Compatibility margin:     {margin:.4f}")
    print(f"  Cohen's d:                {d:.4f}")
    print(f"  Spearman r:               {spearman_corr:.4f}")
    print(f"  AUC-ROC:                  {auc_roc:.4f}")
    print(f"  Best F1:                  {best_f1:.4f} @ threshold {best_thresh:.2f}")
    print(f"  Cluster purity:           {purity:.4f}")
    print(f"  HDBSCAN clusters:         {n_clusters}")
    print(f"  Noise ratio:              {noise_ratio:.4f}")

    return metrics


if __name__ == "__main__":
    data_dir = Path("data/raw")
    train_pairs = load_pairs(data_dir / "dating_pairs.jsonl")
    eval_pairs = load_pairs(data_dir / "eval_pairs.jsonl")
    run_full_baseline(train_pairs, eval_pairs)
