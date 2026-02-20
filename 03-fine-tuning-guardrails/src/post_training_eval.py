from __future__ import annotations

import gc
from pathlib import Path

import numpy as np

# From src/baseline_analysis.py — embedding I/O + dimensionality reduction
from src.baseline_analysis import (
    load_embeddings,
    run_hdbscan_clustering,
    run_umap_projection,
)

# From src/data_loader.py
from src.data_loader import get_categories, get_pair_types, load_pairs, pairs_to_texts

# From src/metrics.py — all 8 metric functions
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
from src.models import BaselineMetrics, DatingPair, EvaluationBundle

EMBEDDINGS_DIR = Path("data/embeddings")
EVAL_DIR = Path("eval")


def resolve_baseline_path(filename: str) -> Path:
    """Resolve baseline file path with fallback logic.

    WHY fallback: baseline_metrics.json might be in eval/ (Day 1 output)
    or data/baseline/ (legacy location). Check both to avoid brittle paths.

    Args:
        filename: Name of the baseline file (e.g., "baseline_metrics.json")

    Returns:
        Path to the found file

    Raises:
        FileNotFoundError: If file not found in either location
    """
    for parent in [Path("eval"), Path("data/baseline")]:
        candidate = parent / filename
        if candidate.exists():
            return candidate
    msg = f"Cannot find {filename} in eval/ or data/baseline/"
    raise FileNotFoundError(msg)


def resolve_baseline_embeddings_path() -> Path:
    """Resolve baseline embeddings path.

    WHY: Baseline embeddings are always at data/embeddings/baseline_eval.npz
    per Day 1 output, but we check existence for defensive programming.
    """
    path = EMBEDDINGS_DIR / "baseline_eval.npz"
    if not path.exists():
        msg = f"Baseline embeddings not found at {path}"
        raise FileNotFoundError(msg)
    return path


def generate_finetuned_embeddings(
    model_path: str,
    eval_pairs: list[DatingPair],
    output_path: Path,
    is_lora: bool = False,
) -> None:
    """Generate embeddings from a fine-tuned model.

    WHY is_lora flag: LoRA models have two possible loading patterns:
    1. Merged model saved as SentenceTransformer (can load directly)
    2. Adapter-only format (need to load base + apply adapter + merge)

    The try/except fallback handles both cases gracefully.

    Args:
        model_path: Path to the fine-tuned model directory
        eval_pairs: List of evaluation pairs to encode
        output_path: Path to save embeddings (.npz format)
        is_lora: Whether this is a LoRA model (enables fallback loading)
    """
    from sentence_transformers import SentenceTransformer

    # WHY import here: keeps module importable without torch/sentence-transformers
    # installed (useful for pure evaluation from cached embeddings)

    print(f"Loading model from {model_path} (is_lora={is_lora})...")

    if is_lora:
        # LoRA loading with try/except fallback (Correction #1)
        try:
            # Try loading as merged SentenceTransformer first
            model = SentenceTransformer(model_path)
            print("  → Loaded as merged SentenceTransformer")
        except Exception as e:
            print(f"  → Merged loading failed ({e}), trying adapter fallback...")
            # Fallback: load base + apply LoRA adapter + merge
            from peft import PeftModel

            base = SentenceTransformer("all-MiniLM-L6-v2")
            base[0].auto_model = PeftModel.from_pretrained(
                base[0].auto_model,
                model_path,
            )
            base[0].auto_model = base[0].auto_model.merge_and_unload()
            model = base
            print("  → Loaded via base + adapter + merge")
    else:
        # Standard model: direct SentenceTransformer load
        model = SentenceTransformer(model_path)
        print("  → Loaded as SentenceTransformer")

    # Generate embeddings
    text1, text2, _ = pairs_to_texts(eval_pairs)
    print(f"Encoding {len(text1)} text_1 embeddings...")
    emb1 = model.encode(text1, batch_size=16, show_progress_bar=True)
    print(f"Encoding {len(text2)} text_2 embeddings...")
    emb2 = model.encode(text2, batch_size=16, show_progress_bar=True)

    # Save embeddings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, text1=emb1, text2=emb2)
    print(f"Saved embeddings to {output_path} — shape {emb1.shape}")

    # WHY explicit cleanup: M2 8GB RAM — release model memory before next step
    del model
    gc.collect()


def evaluate_from_embeddings(
    embeddings_path: Path,
    eval_pairs: list[DatingPair],
) -> EvaluationBundle:
    """Compute all 8 metrics from saved embeddings.

    WHY separate function: enables fast re-evaluation from cached embeddings
    without reloading the model (useful for metric tweaking and comparison).

    This function reuses ALL metric computation logic from existing modules —
    it does NOT re-implement any calculations.

    Args:
        embeddings_path: Path to .npz embeddings file
        eval_pairs: List of evaluation pairs (for labels, categories, pair_types)

    Returns:
        EvaluationBundle with all metrics + intermediate arrays for visualization
    """
    print(f"Loading embeddings from {embeddings_path}...")
    emb1, emb2 = load_embeddings(embeddings_path)

    # Extract metadata from eval_pairs
    _, _, labels = pairs_to_texts(eval_pairs)
    categories = get_categories(eval_pairs)
    pair_types = get_pair_types(eval_pairs)

    # --- Compute cosine similarities (Metric 1) ---
    print("Computing cosine similarities...")
    similarities = compute_cosine_similarities(emb1, emb2)

    # --- Margin metrics (Metric 2) ---
    print("Computing margin...")
    compat_mean, incompat_mean, margin = compute_margin(similarities, labels)

    # --- Effect size (Metric 3) ---
    print("Computing Cohen's d...")
    compat_sims = similarities[np.array(labels) == 1]
    incompat_sims = similarities[np.array(labels) == 0]
    d = compute_cohens_d(compat_sims, incompat_sims)

    # --- Statistical significance (Metric 4) ---
    print("Computing Welch's t-test...")
    t_stat, p_val = compute_welch_ttest(compat_sims, incompat_sims)

    # --- Rank correlation (Metric 5) ---
    print("Computing Spearman correlation...")
    spearman_corr = compute_spearman(similarities, labels)

    # --- ROC metrics (Metric 6) ---
    print("Computing ROC metrics...")
    auc_roc, best_thresh, best_f1, acc, prec, rec = compute_roc_metrics(
        similarities,
        labels,
    )

    # --- False positive analysis (Metric 7) ---
    print("Computing false positive analysis...")
    fp_counts = compute_false_positive_analysis(
        similarities,
        labels,
        pair_types,
        threshold=best_thresh,
    )

    # --- Category breakdown (Metric 8) ---
    print("Computing category metrics...")
    cat_metrics = compute_category_metrics(similarities, labels, categories)
    ptype_metrics = compute_category_metrics(similarities, labels, pair_types)

    # --- UMAP projection ---
    print("Running UMAP projection...")
    projections = run_umap_projection(emb1, emb2)

    # --- HDBSCAN clustering ---
    print("Running HDBSCAN clustering...")
    purity, n_clusters, noise_ratio, cluster_labels = run_hdbscan_clustering(
        projections,
        labels,
    )

    # --- Assemble BaselineMetrics ---
    # WHY same schema: enables apples-to-apples comparison with baseline
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

    # --- Return EvaluationBundle ---
    # WHY bundle: charts need both metrics AND intermediate arrays
    # (similarities, projections, cluster_labels). Bundle packages them together.
    return EvaluationBundle(
        metrics=metrics,
        similarities=similarities,
        projections=projections,
        cluster_labels=cluster_labels,
        labels=labels,
        categories=categories,
        pair_types=pair_types,
    )


def run_post_training_evaluation() -> tuple[EvaluationBundle, EvaluationBundle]:
    """Run full post-training evaluation for both standard and LoRA models.

    WHY orchestrator: coordinates 4 steps for each model (generate embeddings,
    evaluate, save metrics, cleanup) while enforcing the ONE-MODEL-AT-A-TIME
    memory constraint.

    Steps:
    1. Load eval pairs
    2. Generate standard embeddings → evaluate → save
    3. Generate LoRA embeddings → evaluate → save
    4. Print summary table comparing both models

    Returns:
        Tuple of (standard_bundle, lora_bundle)
    """
    # --- Step 1: Load eval pairs ---
    print("=== POST-TRAINING EVALUATION ===\n")
    print("Loading eval pairs...")
    eval_pairs = load_pairs(Path("data/raw/eval_pairs.jsonl"))
    print(f"Loaded {len(eval_pairs)} evaluation pairs\n")

    # --- Step 2: Standard model ---
    print("--- STANDARD MODEL ---")
    standard_emb_path = EMBEDDINGS_DIR / "finetuned_eval.npz"
    if not standard_emb_path.exists():
        generate_finetuned_embeddings(
            model_path="training/model/standard_model",
            eval_pairs=eval_pairs,
            output_path=standard_emb_path,
            is_lora=False,
        )
    else:
        print(f"Embeddings already exist at {standard_emb_path}, skipping generation\n")

    standard_bundle = evaluate_from_embeddings(standard_emb_path, eval_pairs)

    # Save standard metrics
    standard_metrics_path = EVAL_DIR / "finetuned_metrics.json"
    standard_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    standard_metrics_path.write_text(standard_bundle.metrics.model_dump_json(indent=2))
    print(f"Saved {standard_metrics_path}\n")

    # --- Step 3: LoRA model ---
    print("--- LORA MODEL ---")
    lora_emb_path = EMBEDDINGS_DIR / "lora_eval.npz"
    if not lora_emb_path.exists():
        generate_finetuned_embeddings(
            model_path="training/model/lora_model",
            eval_pairs=eval_pairs,
            output_path=lora_emb_path,
            is_lora=True,
        )
    else:
        print(f"Embeddings already exist at {lora_emb_path}, skipping generation\n")

    lora_bundle = evaluate_from_embeddings(lora_emb_path, eval_pairs)

    # Save LoRA metrics
    lora_metrics_path = EVAL_DIR / "lora_metrics.json"
    lora_metrics_path.write_text(lora_bundle.metrics.model_dump_json(indent=2))
    print(f"Saved {lora_metrics_path}\n")

    # --- Step 4: Print summary table ---
    print("\n=== EVALUATION SUMMARY ===")
    print(f"{'Metric':<30} {'Standard':<15} {'LoRA':<15}")
    print("-" * 60)
    print(
        f"{'Spearman Correlation':<30} "
        f"{standard_bundle.metrics.spearman_correlation:>14.4f} "
        f"{lora_bundle.metrics.spearman_correlation:>14.4f}",
    )
    print(
        f"{'Compatibility Margin':<30} "
        f"{standard_bundle.metrics.compatibility_margin:>14.4f} "
        f"{lora_bundle.metrics.compatibility_margin:>14.4f}",
    )
    print(
        f"{'AUC-ROC':<30} "
        f"{standard_bundle.metrics.auc_roc:>14.4f} "
        f"{lora_bundle.metrics.auc_roc:>14.4f}",
    )
    print(
        f"{'Cohen d':<30} "
        f"{standard_bundle.metrics.cohens_d:>14.4f} "
        f"{lora_bundle.metrics.cohens_d:>14.4f}",
    )
    print(
        f"{'Best F1':<30} "
        f"{standard_bundle.metrics.best_f1:>14.4f} "
        f"{lora_bundle.metrics.best_f1:>14.4f}",
    )
    print(
        f"{'Cluster Purity':<30} "
        f"{standard_bundle.metrics.cluster_purity:>14.4f} "
        f"{lora_bundle.metrics.cluster_purity:>14.4f}",
    )

    return standard_bundle, lora_bundle


if __name__ == "__main__":  # pragma: no cover
    run_post_training_evaluation()
