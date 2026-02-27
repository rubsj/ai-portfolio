"""Tests for src/pipeline.py — orchestrator coverage.

Strategy:
- Every step function (_step_generate, _step_label, etc.) is tested by patching
  its heavyweight dependencies (subprocess.run, lazy imports, etc.) so no real
  subprocesses or API calls are made.
- run_pipeline() is tested by patching all step functions and asserting which
  ones are called / skipped based on the skip flags.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# _run_module
# ---------------------------------------------------------------------------


class TestRunModule:
    def test_uses_sys_executable(self) -> None:
        with patch("src.pipeline.subprocess.run") as mock_run:
            import src.pipeline as pm

            pm._run_module("src.corrector")
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == sys.executable

    def test_uses_dash_m_flag(self) -> None:
        with patch("src.pipeline.subprocess.run") as mock_run:
            import src.pipeline as pm

            pm._run_module("src.corrector")
        cmd = mock_run.call_args[0][0]
        assert cmd[1] == "-m"

    def test_includes_module_name(self) -> None:
        with patch("src.pipeline.subprocess.run") as mock_run:
            import src.pipeline as pm

            pm._run_module("src.corrector")
        cmd = mock_run.call_args[0][0]
        assert "src.corrector" in cmd

    def test_appends_extra_args(self) -> None:
        with patch("src.pipeline.subprocess.run") as mock_run:
            import src.pipeline as pm

            pm._run_module("src.run_generation", extra_args=["--dry-run"])
        cmd = mock_run.call_args[0][0]
        assert "--dry-run" in cmd

    def test_check_true(self) -> None:
        with patch("src.pipeline.subprocess.run") as mock_run:
            import src.pipeline as pm

            pm._run_module("src.corrector")
        assert mock_run.call_args[1]["check"] is True

    def test_no_extra_args_produces_three_element_cmd(self) -> None:
        with patch("src.pipeline.subprocess.run") as mock_run:
            import src.pipeline as pm

            pm._run_module("src.corrector")
        cmd = mock_run.call_args[0][0]
        # [sys.executable, "-m", "src.corrector"] — exactly 3 parts
        assert len(cmd) == 3


# ---------------------------------------------------------------------------
# _step_generate
# ---------------------------------------------------------------------------


class TestStepGenerate:
    def test_dry_run_true_passes_dry_run_arg(self) -> None:
        with patch("src.pipeline._run_module") as mock_run:
            import src.pipeline as pm

            pm._step_generate(dry_run=True)
        mock_run.assert_called_once_with("src.run_generation", extra_args=["--dry-run"])

    def test_dry_run_false_passes_empty_list(self) -> None:
        with patch("src.pipeline._run_module") as mock_run:
            import src.pipeline as pm

            pm._step_generate(dry_run=False)
        mock_run.assert_called_once_with("src.run_generation", extra_args=[])


# ---------------------------------------------------------------------------
# _step_label
# ---------------------------------------------------------------------------


class TestStepLabel:
    def test_calls_run_labeling_run(self) -> None:
        mock_run = MagicMock()
        with patch("src.run_labeling.run", mock_run):
            import src.pipeline as pm

            pm._step_label()
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# _step_judge
# ---------------------------------------------------------------------------


class TestStepJudge:
    def test_calls_judge_batch_when_pairs_present(self) -> None:
        mock_store = MagicMock()
        mock_store.pairs = ["pair1"]

        with patch("src.data_paths.DataStore", return_value=mock_store):
            with patch("src.judge.judge_batch") as mock_jb:
                import src.pipeline as pm

                pm._step_judge()
        mock_jb.assert_called_once_with(mock_store.pairs, mock_store.jobs, mock_store.resumes)

    def test_skips_judge_batch_when_no_pairs(self) -> None:
        mock_store = MagicMock()
        mock_store.pairs = []

        with patch("src.data_paths.DataStore", return_value=mock_store):
            with patch("src.judge.judge_batch") as mock_jb:
                import src.pipeline as pm

                pm._step_judge()
        mock_jb.assert_not_called()


# ---------------------------------------------------------------------------
# _step_correct / _step_analyze
# ---------------------------------------------------------------------------


class TestStepCorrect:
    def test_calls_run_module_corrector(self) -> None:
        with patch("src.pipeline._run_module") as mock_run:
            import src.pipeline as pm

            pm._step_correct()
        mock_run.assert_called_once_with("src.corrector")


class TestStepAnalyze:
    def test_calls_run_module_analyzer(self) -> None:
        with patch("src.pipeline._run_module") as mock_run:
            import src.pipeline as pm

            pm._step_analyze()
        mock_run.assert_called_once_with("src.analyzer")


# ---------------------------------------------------------------------------
# _step_multi_hop
# ---------------------------------------------------------------------------


class TestStepMultiHop:
    def test_calls_multihop_run(self) -> None:
        mock_run = MagicMock()
        with patch("src.multi_hop.run", mock_run):
            import src.pipeline as pm

            pm._step_multi_hop()
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# _step_vector_store
# ---------------------------------------------------------------------------


class TestStepVectorStore:
    def test_calls_build_resume_index(self) -> None:
        mock_build = MagicMock()
        with patch("src.vector_store.build_resume_index", mock_build):
            import src.pipeline as pm

            pm._step_vector_store()
        mock_build.assert_called_once()


# ---------------------------------------------------------------------------
# run_pipeline — flag combinations
# ---------------------------------------------------------------------------


class TestRunPipeline:
    """Test orchestrator skip-flag logic without running any real stages."""

    def _patch_all(self):
        return (
            patch("src.pipeline._step_generate"),
            patch("src.pipeline._step_label"),
            patch("src.pipeline._step_judge"),
            patch("src.pipeline._step_correct"),
            patch("src.pipeline._step_analyze"),
            patch("src.pipeline._step_multi_hop"),
            patch("src.pipeline._step_vector_store"),
        )

    def test_all_steps_called_by_default(self) -> None:
        with (
            patch("src.pipeline._step_generate") as mg,
            patch("src.pipeline._step_label") as ml,
            patch("src.pipeline._step_judge") as mj,
            patch("src.pipeline._step_correct") as mc,
            patch("src.pipeline._step_analyze") as ma,
            patch("src.pipeline._step_multi_hop") as mh,
            patch("src.pipeline._step_vector_store") as mv,
        ):
            import src.pipeline as pm

            pm.run_pipeline()

        mg.assert_called_once()
        ml.assert_called_once()
        mj.assert_called_once()
        mc.assert_called_once()
        ma.assert_called_once()
        mh.assert_called_once()
        mv.assert_called_once()

    def test_skip_generation_omits_generate_step(self) -> None:
        with (
            patch("src.pipeline._step_generate") as mg,
            patch("src.pipeline._step_label"),
            patch("src.pipeline._step_judge"),
            patch("src.pipeline._step_correct"),
            patch("src.pipeline._step_analyze"),
            patch("src.pipeline._step_multi_hop"),
            patch("src.pipeline._step_vector_store"),
        ):
            import src.pipeline as pm

            pm.run_pipeline(skip_generation=True)
        mg.assert_not_called()

    def test_skip_judge_omits_judge_step(self) -> None:
        with (
            patch("src.pipeline._step_generate"),
            patch("src.pipeline._step_label"),
            patch("src.pipeline._step_judge") as mj,
            patch("src.pipeline._step_correct"),
            patch("src.pipeline._step_analyze"),
            patch("src.pipeline._step_multi_hop"),
            patch("src.pipeline._step_vector_store"),
        ):
            import src.pipeline as pm

            pm.run_pipeline(skip_judge=True)
        mj.assert_not_called()

    def test_skip_vector_store_omits_vector_store_step(self) -> None:
        with (
            patch("src.pipeline._step_generate"),
            patch("src.pipeline._step_label"),
            patch("src.pipeline._step_judge"),
            patch("src.pipeline._step_correct"),
            patch("src.pipeline._step_analyze"),
            patch("src.pipeline._step_multi_hop"),
            patch("src.pipeline._step_vector_store") as mv,
        ):
            import src.pipeline as pm

            pm.run_pipeline(skip_vector_store=True)
        mv.assert_not_called()

    def test_dry_run_forwarded_to_generate(self) -> None:
        with (
            patch("src.pipeline._step_generate") as mg,
            patch("src.pipeline._step_label"),
            patch("src.pipeline._step_judge"),
            patch("src.pipeline._step_correct"),
            patch("src.pipeline._step_analyze"),
            patch("src.pipeline._step_multi_hop"),
            patch("src.pipeline._step_vector_store"),
        ):
            import src.pipeline as pm

            pm.run_pipeline(dry_run=True)
        mg.assert_called_once_with(dry_run=True)

    def test_all_skips_still_runs_label_correct_analyze_hop(self) -> None:
        with (
            patch("src.pipeline._step_generate") as mg,
            patch("src.pipeline._step_label") as ml,
            patch("src.pipeline._step_judge") as mj,
            patch("src.pipeline._step_correct") as mc,
            patch("src.pipeline._step_analyze") as ma,
            patch("src.pipeline._step_multi_hop") as mh,
            patch("src.pipeline._step_vector_store") as mv,
        ):
            import src.pipeline as pm

            pm.run_pipeline(skip_generation=True, skip_judge=True, skip_vector_store=True)

        # These steps are always executed regardless of flags
        ml.assert_called_once()
        mc.assert_called_once()
        ma.assert_called_once()
        mh.assert_called_once()
        # These are skipped
        mg.assert_not_called()
        mj.assert_not_called()
        mv.assert_not_called()
