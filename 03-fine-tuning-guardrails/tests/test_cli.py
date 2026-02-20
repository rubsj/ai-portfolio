"""Tests for CLI commands."""

from __future__ import annotations

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
