"""Tests for src/analysis.py — correction improvement chart and compute_metrics.

Covers:
- plot_correction_improvement: creates PNG when comparison JSON exists,
  raises FileNotFoundError when missing
- compute_metrics: includes correction_pipeline key when comparison JSON exists
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.analysis import (
    FAILURE_MODES,
    compute_metrics,
    plot_correction_improvement,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_comparison_json() -> dict:
    """Build a minimal valid correction_comparison.json structure."""
    return {
        "generated_at": "2026-03-01T00:00:00+00:00",
        "generator_model": "gpt-4o-mini",
        "judge_model": "gpt-4o",
        "pipeline_version": "1.0",
        "v1_original": {
            "total_failures": 36,
            "failure_rate": "20.0%",
            "per_mode": {m: 0 for m in FAILURE_MODES},
        },
        "corrected": {
            "total_failures": 12,
            "failure_rate": "6.7%",
            "per_mode": {m: 0 for m in FAILURE_MODES},
            "improvement_vs_v1": "66.7%",
        },
        "v2_generated": {
            "total_failures": 8,
            "failure_rate": "4.4%",
            "per_mode": {m: 0 for m in FAILURE_MODES},
            "improvement_vs_v1": "77.8%",
        },
        "v2_corrected": {
            "total_failures": 0,
            "failure_rate": "0.0%",
            "per_mode": {m: 0 for m in FAILURE_MODES},
            "improvement_vs_v1": "100.0%",
        },
        "target_met": {
            "corrected_meets_80pct": False,
            "v2_meets_80pct": False,
            "v2_corrected_meets_80pct": True,
        },
    }


def _make_analysis_df() -> pd.DataFrame:
    """Build a minimal DataFrame matching build_analysis_dataframe output."""
    data = {
        "trace_id": ["r1", "r2"],
        "category": ["plumbing_repair", "plumbing_repair"],
        "difficulty": ["beginner", "beginner"],
        "incomplete_answer": [1, 0],
        "safety_violations": [0, 0],
        "unrealistic_tools": [0, 0],
        "overcomplicated_solution": [0, 0],
        "missing_context": [0, 0],
        "poor_quality_tips": [1, 0],
        "total_failures": [2, 0],
        "quality_score": [3, 5],
    }
    return pd.DataFrame(data)


# ===========================================================================
# plot_correction_improvement
# ===========================================================================


class TestPlotCorrectionImprovement:
    """Tests for plot_correction_improvement."""

    def test_plot_correction_improvement_when_json_exists_creates_png(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        charts_dir = results_dir / "charts"
        charts_dir.mkdir()
        (results_dir / "correction_comparison.json").write_text(
            json.dumps(_make_comparison_json(), indent=2)
        )

        with patch("src.analysis._RESULTS_DIR", results_dir), \
             patch("src.analysis._CHARTS_DIR", charts_dir):
            path = plot_correction_improvement()

        assert path.exists()
        assert path.name == "correction_improvement.png"

    def test_plot_correction_improvement_when_json_missing_raises_error(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        with patch("src.analysis._RESULTS_DIR", results_dir):
            with pytest.raises(FileNotFoundError, match="correction_comparison.json"):
                plot_correction_improvement()


# ===========================================================================
# compute_metrics
# ===========================================================================


class TestComputeMetrics:
    """Tests for compute_metrics with correction pipeline data."""

    def test_compute_metrics_when_comparison_exists_includes_pipeline(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "correction_comparison.json").write_text(
            json.dumps(_make_comparison_json(), indent=2)
        )

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        df = _make_analysis_df()

        with patch("src.analysis._RESULTS_DIR", results_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            metrics = compute_metrics(df)

        assert "correction_pipeline" in metrics
        pipeline = metrics["correction_pipeline"]
        assert pipeline["v1_original"]["total_failures"] == 36
        assert pipeline["v2_corrected"]["total_failures"] == 0

    def test_compute_metrics_when_comparison_missing_excludes_pipeline(
        self, tmp_path: Path
    ) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()

        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        df = _make_analysis_df()

        with patch("src.analysis._RESULTS_DIR", results_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            metrics = compute_metrics(df)

        assert "correction_pipeline" not in metrics
