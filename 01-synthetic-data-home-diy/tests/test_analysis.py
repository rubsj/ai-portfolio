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
    build_analysis_dataframe,
    compute_metrics,
    plot_agreement_matrix,
    plot_category_failures,
    plot_correction_improvement,
    plot_difficulty_failures,
    plot_failure_correlation,
    plot_failure_frequency,
    plot_failure_heatmap,
    run_full_analysis,
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


# ===========================================================================
# build_analysis_dataframe
# ===========================================================================


class TestBuildAnalysisDataframe:
    """Tests for build_analysis_dataframe."""

    def _write_batch_json(self, path: Path) -> None:
        """Write a minimal batch_v1.json to the given directory."""
        records = [
            {
                "trace_id": "r1",
                "category": "plumbing_repair",
                "difficulty": "beginner",
                "template_version": "v1",
                "generation_timestamp": "2026-01-01T00:00:00Z",
                "model_used": "gpt-4o-mini",
                "prompt_hash": "abc",
                "record": {
                    "question": "How do I fix a leaky faucet?",
                    "answer": "Turn off the water. Remove the handle. Replace the cartridge.",
                    "equipment_problem": "Leaky faucet",
                    "tools_required": ["wrench"],
                    "steps": ["Turn off water", "Remove handle"],
                    "safety_info": "Turn off water first",
                    "tips": "Take photos first",
                },
            },
            {
                "trace_id": "r2",
                "category": "electrical_repair",
                "difficulty": "advanced",
                "template_version": "v1",
                "generation_timestamp": "2026-01-01T00:00:00Z",
                "model_used": "gpt-4o-mini",
                "prompt_hash": "def",
                "record": {
                    "question": "How do I replace an outlet?",
                    "answer": "Turn off power at the breaker. Remove the cover plate. Replace outlet.",
                    "equipment_problem": "Faulty outlet",
                    "tools_required": ["screwdriver"],
                    "steps": ["Turn off breaker", "Remove cover plate"],
                    "safety_info": "Turn off breaker first",
                    "tips": "Verify power is off with tester",
                },
            },
        ]
        (path / "batch_v1.json").write_text(json.dumps(records))

    def _write_labels_csv(self, path: Path) -> None:
        """Write a minimal llm_labels.csv to the given directory."""
        import csv as _csv
        fieldnames = ["trace_id"] + FAILURE_MODES + ["overall_quality_score"]
        with open(path / "llm_labels.csv", "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({
                "trace_id": "r1",
                **{m: "0" for m in FAILURE_MODES},
                "overall_quality_score": "4",
            })
            writer.writerow({
                "trace_id": "r2",
                **{m: "1" if m == "incomplete_answer" else "0" for m in FAILURE_MODES},
                "overall_quality_score": "3",
            })

    def test_build_analysis_dataframe_returns_dataframe(self, tmp_path: Path) -> None:
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        self._write_batch_json(gen_dir)
        self._write_labels_csv(labels_dir)

        with patch("src.analysis._GENERATED_DIR", gen_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            df = build_analysis_dataframe()

        import pandas as _pd
        assert isinstance(df, _pd.DataFrame)
        assert len(df) == 2

    def test_build_analysis_dataframe_has_required_columns(self, tmp_path: Path) -> None:
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        self._write_batch_json(gen_dir)
        self._write_labels_csv(labels_dir)

        with patch("src.analysis._GENERATED_DIR", gen_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            df = build_analysis_dataframe()

        for col in ["trace_id", "category", "difficulty", "total_failures", "quality_score"]:
            assert col in df.columns

    def test_build_analysis_dataframe_total_failures_computed(self, tmp_path: Path) -> None:
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        self._write_batch_json(gen_dir)
        self._write_labels_csv(labels_dir)

        with patch("src.analysis._GENERATED_DIR", gen_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            df = build_analysis_dataframe()

        # r1: 0 failures, r2: 1 failure (incomplete_answer)
        assert df[df["trace_id"] == "r1"]["total_failures"].iloc[0] == 0
        assert df[df["trace_id"] == "r2"]["total_failures"].iloc[0] == 1

    def test_build_analysis_dataframe_quality_score_from_csv(self, tmp_path: Path) -> None:
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        self._write_batch_json(gen_dir)
        self._write_labels_csv(labels_dir)

        with patch("src.analysis._GENERATED_DIR", gen_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            df = build_analysis_dataframe()

        assert df[df["trace_id"] == "r1"]["quality_score"].iloc[0] == 4

    def test_build_analysis_dataframe_fallback_quality_score_when_no_column(
        self, tmp_path: Path
    ) -> None:
        """When overall_quality_score column is absent, derive quality_score from failures."""
        import csv as _csv
        gen_dir = tmp_path / "generated"
        gen_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        self._write_batch_json(gen_dir)
        # Write CSV WITHOUT overall_quality_score column
        fieldnames = ["trace_id"] + FAILURE_MODES
        with open(labels_dir / "llm_labels.csv", "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"trace_id": "r1", **{m: "0" for m in FAILURE_MODES}})
            writer.writerow({"trace_id": "r2", **{m: "0" for m in FAILURE_MODES}})

        with patch("src.analysis._GENERATED_DIR", gen_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            df = build_analysis_dataframe()

        # Should have quality_score derived from total_failures: (6 - 0).clip(1, 5) = 5
        assert "quality_score" in df.columns
        assert df["quality_score"].iloc[0] == 5


# ===========================================================================
# Chart functions (DataFrame-based)
# ===========================================================================


class TestChartFunctions:
    """Tests for the 5 DataFrame-based chart functions."""

    def _df(self) -> "pd.DataFrame":
        return _make_analysis_df()

    def test_plot_failure_heatmap_creates_png(self, tmp_path: Path) -> None:
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        with patch("src.analysis._CHARTS_DIR", charts_dir):
            path = plot_failure_heatmap(self._df())
        assert path.exists()
        assert path.suffix == ".png"

    def test_plot_failure_frequency_creates_png(self, tmp_path: Path) -> None:
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        with patch("src.analysis._CHARTS_DIR", charts_dir):
            path = plot_failure_frequency(self._df())
        assert path.exists()
        assert path.suffix == ".png"

    def test_plot_failure_correlation_creates_png(self, tmp_path: Path) -> None:
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        with patch("src.analysis._CHARTS_DIR", charts_dir):
            path = plot_failure_correlation(self._df())
        assert path.exists()
        assert path.suffix == ".png"

    def test_plot_category_failures_creates_png(self, tmp_path: Path) -> None:
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        with patch("src.analysis._CHARTS_DIR", charts_dir):
            path = plot_category_failures(self._df())
        assert path.exists()
        assert path.suffix == ".png"

    def test_plot_difficulty_failures_creates_png(self, tmp_path: Path) -> None:
        df = _make_analysis_df()
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        with patch("src.analysis._CHARTS_DIR", charts_dir):
            path = plot_difficulty_failures(df)
        assert path.exists()
        assert path.suffix == ".png"


# ===========================================================================
# plot_agreement_matrix
# ===========================================================================


class TestPlotAgreementMatrix:
    """Tests for plot_agreement_matrix."""

    def _write_label_csvs(self, labels_dir: Path, trace_ids: list[str]) -> None:
        import csv as _csv
        fieldnames = ["trace_id"] + FAILURE_MODES
        for filename in ["manual_labels.csv", "llm_labels.csv"]:
            with open(labels_dir / filename, "w", newline="") as f:
                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for tid in trace_ids:
                    writer.writerow({"trace_id": tid, **{m: "0" for m in FAILURE_MODES}})

    def test_plot_agreement_matrix_creates_png(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        charts_dir = tmp_path / "charts"
        charts_dir.mkdir()
        self._write_label_csvs(labels_dir, ["r1", "r2"])

        with patch("src.analysis._LABELS_DIR", labels_dir), \
             patch("src.analysis._CHARTS_DIR", charts_dir):
            path = plot_agreement_matrix()

        assert path.exists()
        assert path.suffix == ".png"


# ===========================================================================
# compute_metrics — additional coverage
# ===========================================================================


class TestComputeMetricsAdditional:
    """Additional tests for compute_metrics covering per-mode and category data."""

    def test_compute_metrics_returns_dataset_summary(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        df = _make_analysis_df()

        with patch("src.analysis._RESULTS_DIR", results_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            metrics = compute_metrics(df)

        assert "dataset_summary" in metrics
        assert metrics["dataset_summary"]["total_records"] == 2

    def test_compute_metrics_has_per_mode_failures(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        df = _make_analysis_df()

        with patch("src.analysis._RESULTS_DIR", results_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            metrics = compute_metrics(df)

        assert "per_mode_failures" in metrics
        assert "incomplete_answer" in metrics["per_mode_failures"]

    def test_compute_metrics_has_per_category(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        df = _make_analysis_df()

        with patch("src.analysis._RESULTS_DIR", results_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            metrics = compute_metrics(df)

        assert "per_category" in metrics

    def test_compute_metrics_loads_agreement_when_present(self, tmp_path: Path) -> None:
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # Write an agreement_report.json
        (labels_dir / "agreement_report.json").write_text(
            json.dumps({"overall_agreement": "81.7%"})
        )
        df = _make_analysis_df()

        with patch("src.analysis._RESULTS_DIR", results_dir), \
             patch("src.analysis._LABELS_DIR", labels_dir):
            metrics = compute_metrics(df)

        assert metrics["inter_rater_agreement"]["overall_agreement"] == "81.7%"


# ===========================================================================
# run_full_analysis
# ===========================================================================


class TestRunFullAnalysis:
    """Tests for run_full_analysis orchestrator."""

    def test_run_full_analysis_returns_metrics_dict(self, tmp_path: Path) -> None:
        """run_full_analysis returns metrics and saves metrics.json."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        charts_dir = results_dir / "charts"
        charts_dir.mkdir()
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()

        df = _make_analysis_df()
        fake_metrics = {"dataset_summary": {"total_records": 2}}
        fake_path = tmp_path / "chart.png"
        fake_path.write_text("")

        with patch("src.analysis.build_analysis_dataframe", return_value=df), \
             patch("src.analysis.plot_failure_heatmap", return_value=fake_path), \
             patch("src.analysis.plot_failure_frequency", return_value=fake_path), \
             patch("src.analysis.plot_failure_correlation", return_value=fake_path), \
             patch("src.analysis.plot_category_failures", return_value=fake_path), \
             patch("src.analysis.plot_difficulty_failures", return_value=fake_path), \
             patch("src.analysis.plot_agreement_matrix", return_value=fake_path), \
             patch("src.analysis.plot_correction_improvement", return_value=fake_path), \
             patch("src.analysis.compute_metrics", return_value=fake_metrics), \
             patch("src.analysis._RESULTS_DIR", results_dir):
            result = run_full_analysis()

        assert result == fake_metrics

    def test_run_full_analysis_saves_metrics_json(self, tmp_path: Path) -> None:
        """run_full_analysis saves metrics.json to _RESULTS_DIR."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        df = _make_analysis_df()
        fake_metrics = {"dataset_summary": {"total_records": 2}}
        fake_path = tmp_path / "chart.png"
        fake_path.write_text("")

        with patch("src.analysis.build_analysis_dataframe", return_value=df), \
             patch("src.analysis.plot_failure_heatmap", return_value=fake_path), \
             patch("src.analysis.plot_failure_frequency", return_value=fake_path), \
             patch("src.analysis.plot_failure_correlation", return_value=fake_path), \
             patch("src.analysis.plot_category_failures", return_value=fake_path), \
             patch("src.analysis.plot_difficulty_failures", return_value=fake_path), \
             patch("src.analysis.plot_agreement_matrix", return_value=fake_path), \
             patch("src.analysis.plot_correction_improvement", return_value=fake_path), \
             patch("src.analysis.compute_metrics", return_value=fake_metrics), \
             patch("src.analysis._RESULTS_DIR", results_dir):
            run_full_analysis()

        assert (results_dir / "metrics.json").exists()

    def test_run_full_analysis_chart_failure_does_not_abort(self, tmp_path: Path) -> None:
        """A failing chart (raises Exception) should be caught; analysis continues."""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        df = _make_analysis_df()
        fake_metrics = {"dataset_summary": {"total_records": 2}}
        fake_path = tmp_path / "chart.png"
        fake_path.write_text("")

        with patch("src.analysis.build_analysis_dataframe", return_value=df), \
             patch("src.analysis.plot_failure_heatmap", side_effect=RuntimeError("no display")), \
             patch("src.analysis.plot_failure_frequency", return_value=fake_path), \
             patch("src.analysis.plot_failure_correlation", return_value=fake_path), \
             patch("src.analysis.plot_category_failures", return_value=fake_path), \
             patch("src.analysis.plot_difficulty_failures", return_value=fake_path), \
             patch("src.analysis.plot_agreement_matrix", return_value=fake_path), \
             patch("src.analysis.plot_correction_improvement", return_value=fake_path), \
             patch("src.analysis.compute_metrics", return_value=fake_metrics), \
             patch("src.analysis._RESULTS_DIR", results_dir):
            result = run_full_analysis()  # should not raise

        assert result == fake_metrics
