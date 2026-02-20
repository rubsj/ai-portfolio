"""Tests for StandardTrainer with mocked sentence-transformers."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from src.models import CompatibilityLabel, DatingPair, TrainingResult
from src.trainer import StandardTrainer


def _make_pair(
    label: int = 1,
    category: str = "values_lifestyle",
    subcategory: str = "social_habits",
    pair_type: str = "compatible",
) -> DatingPair:
    """Helper to create DatingPair test fixtures.

    WHY helper: reduces boilerplate in tests, ensures valid gender prefix format.
    """
    return DatingPair(
        text_1="boy: I enjoy quiet evenings at home",
        text_2="girl: I prefer staying in over going out",
        label=CompatibilityLabel(label),
        category=category,
        subcategory=subcategory,
        pair_type=pair_type,
    )


@pytest.fixture
def mock_sentence_transformers(monkeypatch):
    """Mock sentence-transformers to avoid loading torch in tests.

    WHY mock at sys.modules: trainer.py imports inside methods, need to intercept
    before import happens. Mocking at module level doesn't work for lazy imports.
    """
    mock_st = MagicMock()

    # WHY InputExample mock: trainer.prepare_data() creates these
    mock_st.InputExample = Mock(
        side_effect=lambda texts, label: {"texts": texts, "label": label}
    )

    # WHY EmbeddingSimilarityEvaluator mock: trainer.prepare_data() creates evaluator
    mock_evaluator = Mock()
    mock_st.evaluation.EmbeddingSimilarityEvaluator.from_input_examples = Mock(
        return_value=mock_evaluator
    )

    # WHY SentenceTransformer mock: trainer.train() loads model
    mock_model = Mock()
    mock_model.parameters = Mock(
        return_value=[
            Mock(numel=Mock(return_value=1000), requires_grad=True),
            Mock(numel=Mock(return_value=500), requires_grad=True),
        ]
    )
    mock_model.fit = Mock()
    mock_st.SentenceTransformer = Mock(return_value=mock_model)

    # WHY CosineSimilarityLoss mock: trainer.train() creates loss
    mock_loss = Mock()
    mock_st.losses.CosineSimilarityLoss = Mock(return_value=mock_loss)

    # WHY monkeypatch sys.modules: intercepts import before it happens
    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
    monkeypatch.setitem(sys.modules, "sentence_transformers.evaluation", mock_st.evaluation)
    monkeypatch.setitem(sys.modules, "sentence_transformers.losses", mock_st.losses)

    # WHY mock torch DataLoader: trainer.prepare_data() uses it
    mock_torch = MagicMock()
    mock_dataloader = Mock()
    mock_torch.utils.data.DataLoader = Mock(return_value=mock_dataloader)
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)

    return {
        "module": mock_st,
        "model": mock_model,
        "dataloader": mock_dataloader,
        "evaluator": mock_evaluator,
        "loss": mock_loss,
    }


def test_prepare_data_creates_input_examples(mock_sentence_transformers):
    """Verify prepare_data converts DatingPair to InputExample format."""
    train_pairs = [_make_pair(label=1), _make_pair(label=0)]
    eval_pairs = [_make_pair(label=1)]

    trainer = StandardTrainer(train_pairs, eval_pairs)
    dataloader, evaluator = trainer.prepare_data()

    # WHY check call count: should create InputExample for each train + eval pair
    mock_st = mock_sentence_transformers["module"]
    assert mock_st.InputExample.call_count == 3  # 2 train + 1 eval


def test_prepare_data_labels_are_float(mock_sentence_transformers):
    """Verify labels converted to float for CosineSimilarityLoss."""
    train_pairs = [_make_pair(label=0), _make_pair(label=1)]
    eval_pairs = [_make_pair(label=1)]

    trainer = StandardTrainer(train_pairs, eval_pairs)
    trainer.prepare_data()

    mock_st = mock_sentence_transformers["module"]
    calls = mock_st.InputExample.call_args_list

    # WHY float conversion: CosineSimilarityLoss expects float in [0,1], not int
    assert calls[0][1]["label"] == 0.0
    assert calls[1][1]["label"] == 1.0
    assert calls[2][1]["label"] == 1.0


def test_train_hyperparams_match_prd(mock_sentence_transformers, tmp_path):
    """Verify training hyperparameters match PRD specification."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    trainer = StandardTrainer(train_pairs, eval_pairs, output_dir=tmp_path / "model")

    # WHY explicit assertions: PRD specifies exact values, must not drift
    assert trainer.epochs == 4
    assert trainer.batch_size == 16
    assert trainer.learning_rate == 2e-5
    assert trainer.warmup_steps == 100
    assert trainer.evaluation_steps == 500


def test_train_calls_model_fit_with_correct_args(mock_sentence_transformers, tmp_path, monkeypatch):
    """Verify model.fit() called with all required kwargs."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = StandardTrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY mock gc.collect: prevents actual garbage collection in test
    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    trainer.train()

    mock_model = mock_sentence_transformers["model"]
    fit_call = mock_model.fit.call_args

    # WHY check kwargs: all hyperparams must be explicit, not library defaults
    assert fit_call[1]["epochs"] == 4
    assert fit_call[1]["evaluation_steps"] == 500
    assert fit_call[1]["warmup_steps"] == 100
    assert fit_call[1]["optimizer_params"] == {"lr": 2e-5}
    assert fit_call[1]["scheduler"] == "WarmupLinear"
    assert fit_call[1]["save_best_model"] is True
    assert fit_call[1]["show_progress_bar"] is True


def test_train_returns_training_result(mock_sentence_transformers, tmp_path, monkeypatch):
    """Verify train() returns TrainingResult with correct model_type."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = StandardTrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY mock gc.collect: prevents actual garbage collection in test
    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    # WHY create mock CSV: _parse_evaluator_csv() expects this file
    csv_dir = output_dir / "eval"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "similarity_evaluation_dating-eval_results.csv"
    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "cosine_spearman"])
        writer.writeheader()
        writer.writerow({"epoch": "0", "cosine_spearman": "0.75"})

    result = trainer.train()

    # WHY isinstance: ensures Pydantic validation ran
    assert isinstance(result, TrainingResult)
    assert result.model_type == "standard"
    assert result.final_spearman == 0.75
    assert result.trainable_parameters == 1500  # 1000 + 500 from mock


def test_train_saves_training_info_json(mock_sentence_transformers, tmp_path, monkeypatch):
    """Verify training_info.json written with correct fields."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = StandardTrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY override TRAINING_INFO_PATH: avoid writing to real project directory
    from src import trainer as trainer_module

    info_path = tmp_path / "training_info.json"
    monkeypatch.setattr(trainer_module, "TRAINING_INFO_PATH", info_path)

    # WHY mock gc.collect: prevents actual garbage collection in test
    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    trainer.train()

    # WHY check file exists: training info must persist for analysis
    assert info_path.exists()

    data = json.loads(info_path.read_text())
    assert data["model_type"] == "standard"
    assert "training_time_seconds" in data
    assert "final_spearman" in data


def test_train_cleans_up_memory(mock_sentence_transformers, tmp_path, monkeypatch):
    """Verify gc.collect() called to free memory."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = StandardTrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY mock gc.collect: verify memory cleanup happens
    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    trainer.train()

    # WHY gc.collect() critical: prevents OOM on 8GB M2 MacBook Air
    mock_gc.assert_called_once()


def test_parse_evaluator_csv_valid(tmp_path):
    """Verify CSV parsing extracts (step, spearman) tuples."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = StandardTrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY create mock CSV: simulates sentence-transformers evaluator output
    csv_dir = output_dir / "eval"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "similarity_evaluation_dating-eval_results.csv"

    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "cosine_spearman"])
        writer.writeheader()
        writer.writerow({"epoch": "0", "cosine_spearman": "0.5"})
        writer.writerow({"epoch": "1", "cosine_spearman": "0.75"})

    history = trainer._parse_evaluator_csv()

    # WHY check values: must preserve step order and exact Spearman values
    assert history == [(0, 0.5), (1, 0.75)]


def test_parse_evaluator_csv_missing(tmp_path):
    """Verify missing CSV returns empty list without crash."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = StandardTrainer(train_pairs, eval_pairs, output_dir=output_dir)

    history = trainer._parse_evaluator_csv()

    # WHY empty list: graceful degradation if evaluator didn't write CSV
    assert history == []
