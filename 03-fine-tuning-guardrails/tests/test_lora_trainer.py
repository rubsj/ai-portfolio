"""Tests for LoRATrainer with mocked sentence-transformers and PEFT."""

from __future__ import annotations

import csv
import json
import sys
from unittest.mock import MagicMock, Mock

import pytest

from src.lora_trainer import LoRATrainer
from src.models import CompatibilityLabel, DatingPair, TrainingResult


def _make_pair(
    label: int = 1,
    category: str = "values_lifestyle",
    subcategory: str = "social_habits",
    pair_type: str = "compatible",
) -> DatingPair:
    """Helper to create DatingPair test fixtures."""
    return DatingPair(
        text_1="boy: I enjoy quiet evenings at home",
        text_2="girl: I prefer staying in over going out",
        label=CompatibilityLabel(label),
        category=category,
        subcategory=subcategory,
        pair_type=pair_type,
    )


@pytest.fixture
def mock_sentence_transformers_and_peft(monkeypatch):
    """Mock both sentence-transformers and PEFT to avoid loading torch.

    WHY combined fixture: LoRATrainer needs both libraries, simpler to mock together.
    """
    # Mock sentence-transformers (same as test_trainer.py)
    mock_st = MagicMock()

    mock_st.InputExample = Mock(
        side_effect=lambda texts, label: {"texts": texts, "label": label}
    )

    mock_evaluator = Mock()
    mock_st.evaluation.EmbeddingSimilarityEvaluator.from_input_examples = Mock(
        return_value=mock_evaluator
    )

    # WHY mock auto_model: LoRA wraps model[0].auto_model
    mock_base_model = Mock()
    mock_transformer = Mock()
    mock_transformer.auto_model = mock_base_model

    mock_model = Mock()
    mock_model.__getitem__ = Mock(return_value=mock_transformer)
    # WHY parameters: need both total and trainable for LoRA efficiency check
    mock_model.parameters = Mock(
        return_value=[
            Mock(numel=Mock(return_value=10000), requires_grad=False),  # frozen base
            Mock(numel=Mock(return_value=500), requires_grad=True),  # LoRA adapters
        ]
    )
    mock_model.fit = Mock()
    mock_st.SentenceTransformer = Mock(return_value=mock_model)

    mock_loss = Mock()
    mock_st.losses.CosineSimilarityLoss = Mock(return_value=mock_loss)

    monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st)
    monkeypatch.setitem(sys.modules, "sentence_transformers.evaluation", mock_st.evaluation)
    monkeypatch.setitem(sys.modules, "sentence_transformers.losses", mock_st.losses)

    # Mock torch DataLoader
    mock_torch = MagicMock()
    mock_dataloader = Mock()
    mock_torch.utils.data.DataLoader = Mock(return_value=mock_dataloader)
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", mock_torch.utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", mock_torch.utils.data)

    # Mock PEFT
    mock_peft = MagicMock()

    # WHY LoraConfig: trainer._apply_lora() creates this
    mock_lora_config = Mock()
    mock_peft.LoraConfig = Mock(return_value=mock_lora_config)

    # WHY TaskType.FEATURE_EXTRACTION: trainer uses this for embedding models
    mock_peft.TaskType.FEATURE_EXTRACTION = "FEATURE_EXTRACTION"

    # WHY get_peft_model: wraps base model with LoRA adapters
    mock_peft_model = Mock()
    # WHY parameters: PEFT model has same structure as base (some frozen, some trainable)
    mock_peft_model.parameters = Mock(
        return_value=[
            Mock(numel=Mock(return_value=10000), requires_grad=False),
            Mock(numel=Mock(return_value=500), requires_grad=True),
        ]
    )
    mock_peft.get_peft_model = Mock(return_value=mock_peft_model)

    monkeypatch.setitem(sys.modules, "peft", mock_peft)

    return {
        "st_module": mock_st,
        "model": mock_model,
        "transformer": mock_transformer,
        "base_model": mock_base_model,
        "peft_model": mock_peft_model,
        "dataloader": mock_dataloader,
        "evaluator": mock_evaluator,
        "loss": mock_loss,
        "peft_module": mock_peft,
        "lora_config": mock_lora_config,
    }


def test_lora_config_matches_prd(mock_sentence_transformers_and_peft, tmp_path, monkeypatch):
    """Verify LoRA hyperparameters match PRD specification."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = LoRATrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY mock gc.collect: prevents actual garbage collection in test
    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    trainer.train()

    mock_peft = mock_sentence_transformers_and_peft["peft_module"]
    config_call = mock_peft.LoraConfig.call_args

    # WHY check LoRA config: PRD specifies exact values
    assert config_call[1]["r"] == 8
    assert config_call[1]["lora_alpha"] == 16
    assert config_call[1]["lora_dropout"] == 0.1
    assert config_call[1]["target_modules"] == ["query", "value"]
    assert config_call[1]["task_type"] == "FEATURE_EXTRACTION"


def test_apply_lora_calls_get_peft_model(mock_sentence_transformers_and_peft, tmp_path, monkeypatch):
    """Verify get_peft_model() called with correct config."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = LoRATrainer(train_pairs, eval_pairs, output_dir=output_dir)

    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    trainer.train()

    mock_peft = mock_sentence_transformers_and_peft["peft_module"]

    # WHY check called: _apply_lora() must wrap base model with PEFT
    mock_peft.get_peft_model.assert_called_once()

    # WHY check arguments: base model and LoRA config passed
    call_args = mock_peft.get_peft_model.call_args
    base_model_arg = call_args[0][0]
    config_arg = call_args[0][1]

    assert base_model_arg == mock_sentence_transformers_and_peft["base_model"]
    assert config_arg == mock_sentence_transformers_and_peft["lora_config"]


def test_apply_lora_replaces_auto_model(mock_sentence_transformers_and_peft, tmp_path, monkeypatch):
    """Verify model[0].auto_model replaced with PEFT model."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = LoRATrainer(train_pairs, eval_pairs, output_dir=output_dir)

    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    trainer.train()

    mock_transformer = mock_sentence_transformers_and_peft["transformer"]
    mock_peft_model = mock_sentence_transformers_and_peft["peft_model"]

    # WHY check replacement: _apply_lora() must update auto_model in-place
    assert mock_transformer.auto_model == mock_peft_model


def test_train_returns_lora_result(mock_sentence_transformers_and_peft, tmp_path, monkeypatch):
    """Verify train() returns TrainingResult with LoRA fields populated."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = LoRATrainer(train_pairs, eval_pairs, output_dir=output_dir)

    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    # WHY create mock CSV: _parse_evaluator_csv() expects this file
    csv_dir = output_dir / "eval"
    csv_dir.mkdir(parents=True)
    csv_path = csv_dir / "similarity_evaluation_dating-eval-lora_results.csv"
    with csv_path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "cosine_spearman"])
        writer.writeheader()
        writer.writerow({"epoch": "0", "cosine_spearman": "0.80"})

    result = trainer.train()

    # WHY isinstance: ensures Pydantic validation ran
    assert isinstance(result, TrainingResult)
    assert result.model_type == "lora"
    assert result.lora_rank == 8
    assert result.lora_alpha == 16
    assert result.lora_dropout == 0.1
    assert result.lora_target_modules == ["query", "value"]
    assert result.final_spearman == 0.80
    # WHY trainable < total: demonstrates LoRA efficiency
    assert result.trainable_parameters == 500
    assert result.total_parameters == 10500  # 10000 frozen + 500 trainable


def test_hyperparams_same_as_standard(mock_sentence_transformers_and_peft, tmp_path):
    """Verify shared hyperparameters match StandardTrainer (except learning_rate)."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = LoRATrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY epochs/batch/warmup/eval identical: same training schedule as standard
    assert trainer.epochs == 4
    assert trainer.batch_size == 16
    assert trainer.warmup_steps == 100
    assert trainer.evaluation_steps == 500
    # WHY 2e-4 not 2e-5: LoRA adapters need higher LR (only 0.32% of params trainable)
    assert trainer.learning_rate == 2e-4


def test_saves_training_info_json(mock_sentence_transformers_and_peft, tmp_path, monkeypatch):
    """Verify training_info.json written with LoRA fields."""
    train_pairs = [_make_pair()]
    eval_pairs = [_make_pair()]

    output_dir = tmp_path / "model"
    trainer = LoRATrainer(train_pairs, eval_pairs, output_dir=output_dir)

    # WHY override TRAINING_INFO_PATH: avoid writing to real project directory
    from src import lora_trainer as lora_trainer_module

    info_path = tmp_path / "lora_training_info.json"
    monkeypatch.setattr(lora_trainer_module, "TRAINING_INFO_PATH", info_path)

    mock_gc = Mock()
    monkeypatch.setattr("gc.collect", mock_gc)

    trainer.train()

    # WHY check file exists: training info must persist for comparison
    assert info_path.exists()

    data = json.loads(info_path.read_text())
    assert data["model_type"] == "lora"
    assert data["lora_rank"] == 8
    assert data["lora_alpha"] == 16
    assert data["lora_dropout"] == 0.1
    assert data["lora_target_modules"] == ["query", "value"]
