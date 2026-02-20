"""Tests for CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from src.cli import cli


def test_cli_has_train_command():
    """Verify train command registered in CLI."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    # WHY check help text: confirms Click registered command
    assert result.exit_code == 0
    assert "train" in result.output


def test_cli_has_baseline_command():
    """Verify baseline command registered in CLI."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    # WHY check help text: confirms Click registered command
    assert result.exit_code == 0
    assert "baseline" in result.output


def test_train_mode_validation():
    """Verify --mode option validates choices."""
    runner = CliRunner()

    # WHY invalid mode should fail: Click Choice validation
    result = runner.invoke(cli, ["train", "--mode", "invalid"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "invalid choice" in result.output.lower()

    # WHY valid modes accepted: Click Choice includes ["standard", "lora"]
    # Note: these will fail with import errors (expected), but mode validation passes
    result = runner.invoke(cli, ["train", "--mode", "standard"])
    # Exit code will be non-zero due to missing data, but different error
    assert "Invalid value" not in result.output

    result = runner.invoke(cli, ["train", "--mode", "lora"])
    # Exit code will be non-zero due to missing data, but different error
    assert "Invalid value" not in result.output


def test_baseline_command_calls_run_full_baseline():
    """Test baseline command executes run_full_baseline and displays metrics."""
    # WHY mock: avoid loading actual models and generating embeddings
    runner = CliRunner()

    # Create fake metrics to return
    from src.models import BaselineMetrics
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

    with patch("src.cli.run_full_baseline", return_value=fake_metrics) as mock_run:
        result = runner.invoke(cli, ["baseline"])

    # Verify command succeeded
    assert result.exit_code == 0

    # Verify run_full_baseline was called
    mock_run.assert_called_once()

    # Verify metrics displayed in output
    assert "Baseline Spearman" in result.output
    assert "0.30" in result.output  # Spearman value
    assert "Compatibility Margin" in result.output


def test_evaluate_command_calls_run_post_training_evaluation():
    """Test evaluate command executes run_post_training_evaluation."""
    # WHY mock: avoid loading models and generating embeddings (expensive)
    runner = CliRunner()

    # WHY patch src.post_training_eval.run_post_training_evaluation: imported inside evaluate() at line 130
    with patch("src.post_training_eval.run_post_training_evaluation") as mock_run:
        result = runner.invoke(cli, ["evaluate", "--mode", "all"])

    # Verify command succeeded
    assert result.exit_code == 0

    # Verify orchestrator was called
    mock_run.assert_called_once()

    # Verify output shows success messages
    assert "Evaluation complete" in result.output
    assert "finetuned_metrics.json" in result.output
    assert "lora_metrics.json" in result.output


def test_compare_command_calls_run_comparison():
    """Test compare command executes run_comparison."""
    # WHY mock: avoid loading embeddings and generating charts
    runner = CliRunner()

    # WHY patch src.comparison.run_comparison: imported inside compare() at line 153
    with patch("src.comparison.run_comparison") as mock_run:
        result = runner.invoke(cli, ["compare"])

    # Verify command succeeded
    assert result.exit_code == 0

    # Verify orchestrator was called
    mock_run.assert_called_once()

    # Verify output shows success messages
    assert "Comparison complete" in result.output
    assert "comparison_report.html" in result.output
    assert "false_positive_analysis.txt" in result.output
