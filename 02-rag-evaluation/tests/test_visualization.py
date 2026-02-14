"""Tests for visualization.py — data loading, DataFrame conversion, chart generation.

Tests the chart functions by calling them with mock ConfigEvaluation data
and verifying they produce PNG files at the expected paths.

Java/TS parallel: like testing a report generator — give it data, assert
the output file exists. No pixel-level assertions — just contract tests.
"""

from __future__ import annotations

import json

import pytest

from src.models import ConfigEvaluation, RetrievalMethod
from src.visualization import (
    _evals_to_dataframe,
    generate_all_charts,
    load_evaluations,
    plot_bm25_comparison,
    plot_config_heatmap,
    plot_metric_bars,
    plot_semantic_vs_fixed,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _make_config_eval(
    config_id: str = "B-minilm",
    chunk_config: str = "B",
    embedding_model: str = "minilm",
    retrieval_method: RetrievalMethod = RetrievalMethod.VECTOR,
    num_chunks: int = 100,
    r1: float = 0.3,
    r3: float = 0.5,
    r5: float = 0.7,
    p1: float = 0.4,
    p3: float = 0.3,
    p5: float = 0.2,
    mrr5: float = 0.5,
) -> ConfigEvaluation:
    """Create a minimal ConfigEvaluation for chart testing."""
    return ConfigEvaluation(
        config_id=config_id,
        chunk_config=chunk_config,
        embedding_model=embedding_model,
        retrieval_method=retrieval_method,
        num_chunks=num_chunks,
        num_questions=10,
        avg_recall_at_1=r1,
        avg_recall_at_3=r3,
        avg_recall_at_5=r5,
        avg_precision_at_1=p1,
        avg_precision_at_3=p3,
        avg_precision_at_5=p5,
        avg_mrr_at_5=mrr5,
        individual_results=[],
        metrics_by_question_type={},
    )


def _make_full_eval_set() -> list[ConfigEvaluation]:
    """Create a minimal set of ConfigEvaluations mimicking the 16-config grid.

    Includes vector configs for B and E across 3 models + 1 BM25.
    """
    evals = []
    configs = [
        ("B", "minilm", 0.48), ("B", "mpnet", 0.46), ("B", "openai", 0.60),
        ("E", "minilm", 0.45), ("E", "mpnet", 0.41), ("E", "openai", 0.62),
        ("A", "minilm", 0.29), ("A", "mpnet", 0.23), ("A", "openai", 0.30),
        ("C", "minilm", 0.43), ("C", "mpnet", 0.37), ("C", "openai", 0.51),
        ("D", "minilm", 0.42), ("D", "mpnet", 0.34), ("D", "openai", 0.39),
    ]
    for chunk, model, r5 in configs:
        evals.append(_make_config_eval(
            config_id=f"{chunk}-{model}",
            chunk_config=chunk,
            embedding_model=model,
            r5=r5,
            r1=r5 * 0.5,
            r3=r5 * 0.8,
            p1=r5 * 0.6,
            p3=r5 * 0.4,
            p5=r5 * 0.3,
            mrr5=r5 * 0.9,
        ))

    # BM25 baseline
    evals.append(_make_config_eval(
        config_id="bm25",
        chunk_config="B",
        embedding_model="bm25",
        retrieval_method=RetrievalMethod.BM25,
        r5=0.38,
        r1=0.19,
        r3=0.30,
        p1=0.25,
        p3=0.18,
        p5=0.13,
        mrr5=0.34,
    ))
    return evals


# ===========================================================================
# load_evaluations Tests
# ===========================================================================

class TestLoadEvaluations:
    """Tests for load_evaluations."""

    def test_loads_from_json_file(self, tmp_path) -> None:
        """Parses JSON file into ConfigEvaluation list."""
        evals = [_make_config_eval()]
        data = [e.model_dump(mode="json") for e in evals]
        path = tmp_path / "results.json"
        path.write_text(json.dumps(data))

        loaded = load_evaluations(path)

        assert len(loaded) == 1
        assert loaded[0].config_id == "B-minilm"

    def test_loads_multiple_evaluations(self, tmp_path) -> None:
        """Handles multiple evaluations in a single file."""
        evals = _make_full_eval_set()
        data = [e.model_dump(mode="json") for e in evals]
        path = tmp_path / "results.json"
        path.write_text(json.dumps(data))

        loaded = load_evaluations(path)
        assert len(loaded) == 16


# ===========================================================================
# _evals_to_dataframe Tests
# ===========================================================================

class TestEvalsToDataframe:
    """Tests for _evals_to_dataframe."""

    def test_sorted_by_recall_at_5_descending(self) -> None:
        """DataFrame rows are sorted by avg_recall_at_5 descending."""
        evals = [
            _make_config_eval(config_id="low", r5=0.3),
            _make_config_eval(config_id="high", r5=0.9),
            _make_config_eval(config_id="mid", r5=0.6),
        ]
        df = _evals_to_dataframe(evals)

        assert list(df["config_id"]) == ["high", "mid", "low"]

    def test_contains_all_metric_columns(self) -> None:
        """DataFrame has all 7 metric columns."""
        evals = [_make_config_eval()]
        df = _evals_to_dataframe(evals)

        expected_cols = [
            "avg_recall_at_1", "avg_recall_at_3", "avg_recall_at_5",
            "avg_precision_at_1", "avg_precision_at_3", "avg_precision_at_5",
            "avg_mrr_at_5",
        ]
        for col in expected_cols:
            assert col in df.columns

    def test_contains_metadata_columns(self) -> None:
        """DataFrame includes config_id, chunk_config, embedding_model."""
        evals = [_make_config_eval()]
        df = _evals_to_dataframe(evals)

        assert "config_id" in df.columns
        assert "chunk_config" in df.columns
        assert "embedding_model" in df.columns
        assert "retrieval_method" in df.columns

    def test_metric_values_match_input(self) -> None:
        """DataFrame metric values match the input ConfigEvaluation."""
        evals = [_make_config_eval(r5=0.75, mrr5=0.60)]
        df = _evals_to_dataframe(evals)

        assert df.iloc[0]["avg_recall_at_5"] == pytest.approx(0.75)
        assert df.iloc[0]["avg_mrr_at_5"] == pytest.approx(0.60)


# ===========================================================================
# Chart Generation Tests
# ===========================================================================

class TestPlotConfigHeatmap:
    """Tests for plot_config_heatmap."""

    def test_creates_png_file(self, tmp_path) -> None:
        """Heatmap saves a PNG file at the specified path."""
        evals = _make_full_eval_set()
        save_path = tmp_path / "heatmap.png"

        result = plot_config_heatmap(evals, save_path=save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0
        assert result == save_path

    def test_creates_parent_directories(self, tmp_path) -> None:
        """Creates parent dirs if they don't exist."""
        evals = _make_full_eval_set()
        save_path = tmp_path / "sub" / "dir" / "heatmap.png"

        plot_config_heatmap(evals, save_path=save_path)
        assert save_path.exists()


class TestPlotMetricBars:
    """Tests for plot_metric_bars."""

    def test_creates_png_file(self, tmp_path) -> None:
        """Bar chart saves a PNG file."""
        evals = _make_full_eval_set()
        save_path = tmp_path / "bars.png"

        result = plot_metric_bars(evals, save_path=save_path)

        assert save_path.exists()
        assert result == save_path


class TestPlotBm25Comparison:
    """Tests for plot_bm25_comparison."""

    def test_creates_png_file(self, tmp_path) -> None:
        """BM25 comparison chart saves a PNG file."""
        evals = _make_full_eval_set()
        save_path = tmp_path / "bm25.png"

        result = plot_bm25_comparison(evals, save_path=save_path)

        assert save_path.exists()
        assert result == save_path


class TestPlotSemanticVsFixed:
    """Tests for plot_semantic_vs_fixed."""

    def test_creates_png_file(self, tmp_path) -> None:
        """Semantic vs fixed chart saves a PNG file."""
        evals = _make_full_eval_set()
        save_path = tmp_path / "semantic.png"

        result = plot_semantic_vs_fixed(evals, save_path=save_path)

        assert save_path.exists()
        assert result == save_path


class TestGenerateAllCharts:
    """Tests for generate_all_charts."""

    def test_generates_four_charts(self, tmp_path, monkeypatch) -> None:
        """Generates all 4 charts in the specified directory."""
        monkeypatch.setattr("src.visualization.CHARTS_DIR", tmp_path)
        evals = _make_full_eval_set()

        paths = generate_all_charts(evals)

        assert len(paths) == 4
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
