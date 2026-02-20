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


def test_generate_finetuned_embeddings_saves_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_finetuned_embeddings creates .npz file with correct structure."""
    # WHY mock: avoid loading actual SentenceTransformer model (expensive)
    from unittest.mock import MagicMock, patch
    from src.post_training_eval import generate_finetuned_embeddings

    monkeypatch.chdir(tmp_path)

    # Create fake eval pairs
    eval_pairs = [
        DatingPair(
            text_1=f"boy: statement {i}",
            text_2=f"girl: statement {i}",
            label=i % 2,
            category=f"category_{i % 3}",
            subcategory=f"subcat_{i}",
            pair_type="CC" if i % 2 == 0 else "II",
        )
        for i in range(5)
    ]

    output_path = tmp_path / "embeddings.npz"

    # Create mock model that returns fake embeddings
    mock_model = MagicMock()
    # WHY return shape (N, 384): SentenceTransformer.encode() returns (num_texts, embedding_dim)
    # WHY **kwargs: encode() accepts batch_size, show_progress_bar, etc. (line 125-127 in src/post_training_eval.py)
    mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 384).astype(np.float32)

    # WHY patch sentence_transformers.SentenceTransformer: import happens inside function at line 91
    with patch("sentence_transformers.SentenceTransformer", return_value=mock_model):
        generate_finetuned_embeddings(
            model_path="fake/path",
            eval_pairs=eval_pairs,
            output_path=output_path,
            is_lora=False,
        )

    # Verify .npz file created
    assert output_path.exists()

    # Verify structure matches expected format
    with np.load(output_path) as data:
        assert "text1" in data
        assert "text2" in data
        assert data["text1"].shape == (5, 384)
        assert data["text2"].shape == (5, 384)


def test_generate_finetuned_embeddings_handles_lora_with_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test generate_finetuned_embeddings falls back to base+adapter loading for LoRA."""
    # WHY: LoRA models may not be loadable directly as SentenceTransformer
    from unittest.mock import MagicMock, patch
    from src.post_training_eval import generate_finetuned_embeddings

    monkeypatch.chdir(tmp_path)

    eval_pairs = [
        DatingPair(
            text_1="boy: test",
            text_2="girl: test",
            label=1,
            category="test",
            subcategory="test",
            pair_type="CC",
        )
    ]

    output_path = tmp_path / "lora_embeddings.npz"

    # Mock LoRA loading scenario: first SentenceTransformer() fails, then base + PeftModel succeeds
    mock_model = MagicMock()
    # WHY **kwargs: encode() accepts batch_size, show_progress_bar, etc.
    mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 384).astype(np.float32)

    call_count = 0

    def mock_sentence_transformer(model_path):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call (try direct load): raise exception to trigger fallback
            raise Exception("Cannot load merged model")
        else:
            # Second call (base model load): succeed
            return mock_model

    # WHY patch sentence_transformers and peft: imports happen inside function
    with patch("sentence_transformers.SentenceTransformer", side_effect=mock_sentence_transformer):
        with patch("peft.PeftModel") as mock_peft:
            # Mock PeftModel.from_pretrained to return mock that has merge_and_unload
            mock_adapter = MagicMock()
            mock_adapter.merge_and_unload.return_value = MagicMock()
            mock_peft.from_pretrained.return_value = mock_adapter

            generate_finetuned_embeddings(
                model_path="fake/lora/path",
                eval_pairs=eval_pairs,
                output_path=output_path,
                is_lora=True,
            )

    # Verify .npz created even with fallback loading
    assert output_path.exists()


def test_run_post_training_evaluation_creates_metrics_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test run_post_training_evaluation creates finetuned_metrics.json and lora_metrics.json."""
    # WHY: This is the main orchestrator that calls generate + evaluate for both models
    from unittest.mock import MagicMock, patch
    from src.post_training_eval import run_post_training_evaluation

    monkeypatch.chdir(tmp_path)

    # Create fake eval pairs file
    (tmp_path / "data" / "raw").mkdir(parents=True)
    eval_pairs_path = tmp_path / "data" / "raw" / "eval_pairs.jsonl"
    eval_pairs_path.write_text(
        '{"text_1": "boy: test", "text_2": "girl: test", "label": 1, "category": "test", "subcategory": "test"}\n'
    )

    # Create fake embeddings directory structure
    (tmp_path / "data" / "embeddings").mkdir(parents=True)
    (tmp_path / "eval").mkdir()
    (tmp_path / "training" / "model" / "standard_model").mkdir(parents=True)
    (tmp_path / "training" / "model" / "lora_model").mkdir(parents=True)

    # Mock all the expensive operations using proper patch targets
    # WHY: load_pairs is imported at module level (line 6), so patch at src.post_training_eval
    # WHY: generate_finetuned_embeddings uses SentenceTransformer which is imported from sentence_transformers
    with patch("src.post_training_eval.load_pairs") as mock_load_pairs:
        # Return minimal eval pairs
        mock_load_pairs.return_value = [
            DatingPair(
                text_1="boy: test",
                text_2="girl: test",
                label=1,
                category="test",
                subcategory="test",
                pair_type="CC",
            )
        ]

        with patch("sentence_transformers.SentenceTransformer"):
            with patch("src.post_training_eval.evaluate_from_embeddings") as mock_eval:
                # Return mock EvaluationBundle
                from src.models import BaselineMetrics, EvaluationBundle
                fake_metrics = BaselineMetrics(
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
                mock_eval.return_value = EvaluationBundle(
                    metrics=fake_metrics,
                    similarities=np.array([0.9]),
                    projections=np.array([[0.1, 0.2], [0.3, 0.4]]),
                    cluster_labels=np.array([0]),
                    labels=[1],
                    categories=["test"],
                    pair_types=["CC"],
                )

                # Run the orchestrator
                run_post_training_evaluation()

    # Verify metrics JSON files were created
    assert (tmp_path / "eval" / "finetuned_metrics.json").exists()
    assert (tmp_path / "eval" / "lora_metrics.json").exists()
