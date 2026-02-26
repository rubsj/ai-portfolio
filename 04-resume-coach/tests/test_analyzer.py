"""
test_analyzer.py — Unit tests for src/analyzer.py.

Strategy:
- All chart functions receive a synthetic pandas DataFrame (no file I/O).
- _CHARTS_DIR is monkeypatched to tmp_path so PNGs are written to the test directory.
- _CORRECTION_SUMMARY_FILE is monkeypatched for charts 8 and generate_pipeline_summary.
- _RESULTS_DIR is monkeypatched for generate_pipeline_summary.
- build_analysis_dataframe() is tested with five minimal JSONL fixtures in tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.analyzer as amod
from src.analyzer import (
    _FAILURE_COLS,
    _FIT_ORDER,
    build_analysis_dataframe,
    compute_template_ab_test,
    generate_all_charts,
    generate_pipeline_summary,
    plot_correction_success,
    plot_failure_correlation,
    plot_failure_rates_by_fit,
    plot_failure_rates_by_template,
    plot_hallucination_by_seniority,
    plot_judge_vs_rules_agreement,
    plot_niche_vs_standard,
    plot_skills_overlap_distribution,
    plot_validation_summary,
)


# ---------------------------------------------------------------------------
# Synthetic DataFrame factory
# ---------------------------------------------------------------------------

_TEMPLATES = ["technical", "formal", "casual", "achievement", "career_changer"]


def _make_df(n: int = 20) -> pd.DataFrame:
    """
    Create a synthetic analysis DataFrame with all columns expected by analyzer functions.
    n must be divisible by 5 so each fit level has equal representation.
    """
    fit_levels = [_FIT_ORDER[i % 5] for i in range(n)]
    templates = [_TEMPLATES[i % 5] for i in range(n)]

    data = {
        "pair_id": [f"pair_{i:03d}" for i in range(n)],
        "fit_level": fit_levels,
        "resume_trace_id": [f"r{i}" for i in range(n)],
        "job_trace_id": [f"j{i % 5}" for i in range(n)],
        "template": templates,
        "is_niche": [i % 3 == 0 for i in range(n)],
        "industry": ["Technology"] * n,
        # Jaccard scores — excellent→high, mismatch→near zero
        "skills_overlap": [
            [0.8, 0.65, 0.45, 0.15, 0.02][i % 5] for i in range(n)
        ],
        "experience_mismatch": [i % 4 == 0 for i in range(n)],
        "seniority_mismatch": [i % 5 == 0 for i in range(n)],
        "missing_core_skills": [i % 3 == 0 for i in range(n)],
        "has_hallucinations": [i % 3 == 1 for i in range(n)],
        "has_awkward_language": [i % 2 == 0 for i in range(n)],
        "seniority_level_resume": [i % 5 for i in range(n)],
        "overall_quality_score": [round(0.9 - i * 0.03, 2) for i in range(n)],
        "judge_has_hallucinations": [i % 4 == 0 for i in range(n)],
        "judge_has_awkward_language": [i % 2 == 1 for i in range(n)],
    }
    df = pd.DataFrame(data)
    df["total_flags"] = df[_FAILURE_COLS].sum(axis=1)
    return df


def _make_correction_summary(tmp_path: Path) -> Path:
    """Write a correction_summary.json fixture and return its path."""
    summary = {
        "total_invalid": 8,
        "total_corrected": 8,
        "correction_rate": 1.0,
        "avg_attempts_per_success": 1.0,
        "common_failure_reasons": [],
    }
    p = tmp_path / "correction_summary.json"
    p.write_text(json.dumps(summary))
    return p


# ---------------------------------------------------------------------------
# TestComputeTemplateAbTest
# ---------------------------------------------------------------------------


class TestComputeTemplateAbTest:
    def test_returns_dict_with_required_keys(self) -> None:
        df = _make_df()
        result = compute_template_ab_test(df)
        assert "chi_squared_statistic" in result
        assert "chi_squared_p_value" in result
        assert "degrees_of_freedom" in result
        assert "significant" in result
        assert "best_template" in result
        assert "worst_template" in result
        assert "failure_rates_by_template" in result

    def test_best_template_in_known_templates(self) -> None:
        df = _make_df()
        result = compute_template_ab_test(df)
        assert result["best_template"] in _TEMPLATES

    def test_worst_template_in_known_templates(self) -> None:
        df = _make_df()
        result = compute_template_ab_test(df)
        assert result["worst_template"] in _TEMPLATES

    def test_significant_is_bool(self) -> None:
        df = _make_df()
        result = compute_template_ab_test(df)
        assert isinstance(result["significant"], bool)

    def test_chi2_is_non_negative(self) -> None:
        df = _make_df()
        result = compute_template_ab_test(df)
        assert result["chi_squared_statistic"] >= 0

    def test_failure_rates_between_zero_and_one(self) -> None:
        df = _make_df()
        result = compute_template_ab_test(df)
        for rate in result["failure_rates_by_template"].values():
            assert 0.0 <= rate <= 1.0

    def test_uniform_df_has_no_significant_difference(self) -> None:
        """If every template has the exact same failure rate, chi2 should be ~0."""
        # Each template gets 2 failed + 2 non-failed rows → identical 50% failure rate.
        # WHY not all-False: scipy raises ValueError when expected frequency is zero
        # (which happens when the "failed" column is all-zero in the contingency table).
        rows = []
        for template in _TEMPLATES:
            for failed in [True, True, False, False]:  # 50% failure rate per template
                rows.append({"template": template, **{col: failed for col in _FAILURE_COLS}})
        df = pd.DataFrame(rows)
        result = compute_template_ab_test(df)
        assert result["chi_squared_statistic"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestChartFunctions — each chart returns a Path and creates a PNG
# ---------------------------------------------------------------------------


class TestChartFunctions:
    """Tests for all 9 chart functions. Uses monkeypatched _CHARTS_DIR."""

    def test_plot_failure_correlation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        df = _make_df()
        path = plot_failure_correlation(df)
        assert isinstance(path, Path)
        assert path.exists()
        assert path.suffix == ".png"

    def test_plot_failure_rates_by_fit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        df = _make_df()
        path = plot_failure_rates_by_fit(df)
        assert path.exists()

    def test_plot_failure_rates_by_template(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        df = _make_df()
        path = plot_failure_rates_by_template(df)
        assert path.exists()

    def test_plot_niche_vs_standard(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        df = _make_df()
        path = plot_niche_vs_standard(df)
        assert path.exists()

    def test_plot_validation_summary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        path = plot_validation_summary()
        assert path.exists()
        assert path.name == "validation_summary.png"

    def test_plot_skills_overlap_distribution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        df = _make_df()
        path = plot_skills_overlap_distribution(df)
        assert path.exists()

    def test_plot_hallucination_by_seniority(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        df = _make_df()
        path = plot_hallucination_by_seniority(df)
        assert path.exists()

    def test_plot_correction_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)
        path = plot_correction_success()
        assert path.exists()
        assert path.name == "correction_success.png"

    def test_plot_judge_vs_rules_agreement(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        df = _make_df()
        path = plot_judge_vs_rules_agreement(df)
        assert path.exists()

    def test_all_chart_files_have_png_extension(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)
        df = _make_df()
        paths = generate_all_charts(df)
        assert len(paths) == 9
        for p in paths:
            assert p.suffix == ".png"
            assert p.exists()


# ---------------------------------------------------------------------------
# TestGeneratePipelineSummary
# ---------------------------------------------------------------------------


class TestGeneratePipelineSummary:
    def test_returns_dict_with_all_sections(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)
        monkeypatch.setattr(amod, "_RESULTS_DIR", tmp_path)

        df = _make_df()
        ab = compute_template_ab_test(df)
        result = generate_pipeline_summary(df, ab)

        assert "generation" in result
        assert "labeling" in result
        assert "judge" in result
        assert "correction" in result
        assert "ab_testing" in result
        assert "charts_generated" in result

    def test_saves_json_file_to_results_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)
        monkeypatch.setattr(amod, "_RESULTS_DIR", tmp_path)

        df = _make_df()
        ab = compute_template_ab_test(df)
        generate_pipeline_summary(df, ab)

        output = tmp_path / "pipeline_summary.json"
        assert output.exists()
        loaded = json.loads(output.read_text())
        assert loaded["project"] == "P4 — Resume Coach"

    def test_labeling_section_has_jaccard_per_fit_level(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)
        monkeypatch.setattr(amod, "_RESULTS_DIR", tmp_path)

        df = _make_df()
        ab = compute_template_ab_test(df)
        result = generate_pipeline_summary(df, ab)

        jaccard = result["labeling"]["avg_jaccard_by_fit_level"]
        assert "excellent" in jaccard
        assert "mismatch" in jaccard
        assert jaccard["excellent"] > jaccard["mismatch"]  # gradient preserved

    def test_judge_section_has_quality_metrics(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)
        monkeypatch.setattr(amod, "_RESULTS_DIR", tmp_path)

        df = _make_df()
        ab = compute_template_ab_test(df)
        result = generate_pipeline_summary(df, ab)

        judging = result["judge"]
        assert "avg_quality_score" in judging
        assert "hallucination_rate" in judging
        assert "awkward_language_rate" in judging
        assert 0.0 <= judging["avg_quality_score"] <= 1.0

    def test_correction_section_from_summary_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)
        monkeypatch.setattr(amod, "_RESULTS_DIR", tmp_path)

        df = _make_df()
        ab = compute_template_ab_test(df)
        result = generate_pipeline_summary(df, ab)

        assert result["correction"]["records_corrected"] == 8
        assert result["correction"]["correction_rate"] == 1.0


# ---------------------------------------------------------------------------
# TestBuildAnalysisDataframe
# ---------------------------------------------------------------------------


class TestBuildAnalysisDataframe:
    """Tests build_analysis_dataframe() with monkeypatched file path constants."""

    def _make_jsonl_fixtures(self, tmp_path: Path, n: int = 5) -> tuple[Path, ...]:
        """Write minimal JSONL files for n pairs (1 per fit level)."""
        fit_levels = ["excellent", "good", "partial", "poor", "mismatch"]
        writing_styles = ["technical", "formal", "casual", "achievement", "career_changer"]

        pairs_data, resumes_data, jobs_data, labels_data, judge_data = [], [], [], [], []

        for i in range(n):
            r_id = f"r{i}"
            j_id = f"j{i}"
            p_id = f"p{i}"
            fl = fit_levels[i % 5]
            ws = writing_styles[i % 5]

            pairs_data.append(
                json.dumps({
                    "pair_id": p_id,
                    "fit_level": fl,
                    "resume_trace_id": r_id,
                    "job_trace_id": j_id,
                    "created_at": "2026-02-25T14:00:00Z",
                })
            )
            resumes_data.append(
                json.dumps({
                    "trace_id": r_id,
                    "writing_style": ws,
                    "fit_level": fl,
                    "template_version": "v1",
                    "generated_at": "2026-02-25T14:00:00Z",
                    "prompt_template": f"v1-{ws}",
                    "model_used": "gpt-4o-mini",
                    "resume": {},
                })
            )
            jobs_data.append(
                json.dumps({
                    "trace_id": j_id,
                    "is_niche_role": i % 2 == 0,
                    "generated_at": "2026-02-25T14:00:00Z",
                    "prompt_template": "v1",
                    "model_used": "gpt-4o-mini",
                    "job": {"company": {"industry": "Technology"}},
                })
            )
            overlap = [0.8, 0.65, 0.45, 0.15, 0.02][i % 5]
            labels_data.append(
                json.dumps({
                    "pair_id": p_id,
                    "skills_overlap": overlap,
                    "skills_overlap_raw": 4,
                    "skills_union_raw": 5,
                    "experience_mismatch": fl in ("poor", "mismatch"),
                    "seniority_mismatch": fl == "mismatch",
                    "missing_core_skills": fl in ("partial", "poor", "mismatch"),
                    "has_hallucinations": fl == "mismatch",
                    "has_awkward_language": fl in ("poor", "mismatch"),
                    "experience_years_resume": 5.0,
                    "experience_years_required": 3,
                    "seniority_level_resume": i % 5,
                    "seniority_level_job": 2,
                    "missing_skills": [],
                    "hallucination_reasons": [],
                    "awkward_language_reasons": [],
                    "resume_skills_normalized": ["python"],
                    "job_skills_normalized": ["python"],
                })
            )
            judge_data.append(
                json.dumps({
                    "pair_id": p_id,
                    "overall_quality_score": 0.8,
                    "has_hallucinations": False,
                    "has_awkward_language": False,
                    "hallucination_details": "",
                    "awkward_language_details": "",
                    "fit_assessment": "Good",
                    "recommendations": [],
                    "red_flags": [],
                })
            )

        pairs_file = tmp_path / "pairs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        jobs_file = tmp_path / "jobs.jsonl"
        labels_file = tmp_path / "labels.jsonl"
        judge_file = tmp_path / "judge.jsonl"

        pairs_file.write_text("\n".join(pairs_data) + "\n")
        resumes_file.write_text("\n".join(resumes_data) + "\n")
        jobs_file.write_text("\n".join(jobs_data) + "\n")
        labels_file.write_text("\n".join(labels_data) + "\n")
        judge_file.write_text("\n".join(judge_data) + "\n")

        return pairs_file, resumes_file, jobs_file, labels_file, judge_file

    def test_returns_dataframe_with_correct_row_count(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pf, rf, jf, lf, judg = self._make_jsonl_fixtures(tmp_path, n=5)
        monkeypatch.setattr(amod, "_PAIRS_FILE", pf)
        monkeypatch.setattr(amod, "_RESUMES_FILE", rf)
        monkeypatch.setattr(amod, "_JOBS_FILE", jf)
        monkeypatch.setattr(amod, "_FAILURE_LABELS_FILE", lf)
        monkeypatch.setattr(amod, "_JUDGE_RESULTS_FILE", judg)

        df = build_analysis_dataframe()
        assert len(df) == 5

    def test_dataframe_has_required_columns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pf, rf, jf, lf, judg = self._make_jsonl_fixtures(tmp_path, n=5)
        monkeypatch.setattr(amod, "_PAIRS_FILE", pf)
        monkeypatch.setattr(amod, "_RESUMES_FILE", rf)
        monkeypatch.setattr(amod, "_JOBS_FILE", jf)
        monkeypatch.setattr(amod, "_FAILURE_LABELS_FILE", lf)
        monkeypatch.setattr(amod, "_JUDGE_RESULTS_FILE", judg)

        df = build_analysis_dataframe()
        for col in ["pair_id", "fit_level", "template", "is_niche", "skills_overlap",
                    "overall_quality_score", "total_flags"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_total_flags_computed_correctly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pf, rf, jf, lf, judg = self._make_jsonl_fixtures(tmp_path, n=5)
        monkeypatch.setattr(amod, "_PAIRS_FILE", pf)
        monkeypatch.setattr(amod, "_RESUMES_FILE", rf)
        monkeypatch.setattr(amod, "_JOBS_FILE", jf)
        monkeypatch.setattr(amod, "_FAILURE_LABELS_FILE", lf)
        monkeypatch.setattr(amod, "_JUDGE_RESULTS_FILE", judg)

        df = build_analysis_dataframe()
        # total_flags = sum of 5 boolean failure columns
        expected = df[_FAILURE_COLS].sum(axis=1)
        pd.testing.assert_series_equal(df["total_flags"], expected, check_names=False)


# ---------------------------------------------------------------------------
# TestGenerateAllCharts
# ---------------------------------------------------------------------------


class TestGenerateAllCharts:
    def test_returns_nine_paths(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)

        df = _make_df()
        paths = generate_all_charts(df)
        assert len(paths) == 9

    def test_all_paths_are_existing_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        summary_file = _make_correction_summary(tmp_path)
        monkeypatch.setattr(amod, "_CHARTS_DIR", tmp_path)
        monkeypatch.setattr(amod, "_CORRECTION_SUMMARY_FILE", summary_file)

        df = _make_df()
        paths = generate_all_charts(df)
        for p in paths:
            assert p.is_file(), f"Chart not written: {p}"
