from __future__ import annotations

import json
from pathlib import Path

from src.schemas import (
    CorrectionResult,
    FeedbackRequest,
    FailureLabels,
    GeneratedJob,
    GeneratedResume,
    JudgeResult,
    ResumeJobPair,
)

# ---------------------------------------------------------------------------
# Path constants — all relative to the project root (04-resume-coach/)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent  # 04-resume-coach/

DATA_DIR = _ROOT / "data"
GENERATED_DIR = DATA_DIR / "generated"
LABELED_DIR = DATA_DIR / "labeled"
CORRECTED_DIR = DATA_DIR / "corrected"
FEEDBACK_DIR = DATA_DIR / "feedback"
ANALYSIS_DIR = DATA_DIR / "analysis"
CHROMADB_DIR = DATA_DIR / "chromadb"
RESULTS_DIR = _ROOT / "results"


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def find_latest(directory: Path, prefix: str) -> Path | None:
    """Return the largest `{prefix}_*.jsonl` file in *directory*, or None.

    WHY max by file size: partial dry-run files share the same glob pattern
    but have fewer records — the biggest file is the canonical full-run artifact.
    Same pattern as run_generation.py:44.
    """
    candidates = list(directory.glob(f"{prefix}_*.jsonl"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


# ---------------------------------------------------------------------------
# Individual loaders — each reads one file and returns a typed collection
# ---------------------------------------------------------------------------


def load_jobs(path: Path) -> dict[str, GeneratedJob]:
    """Load jobs JSONL keyed by trace_id."""
    result: dict[str, GeneratedJob] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            gj = GeneratedJob.model_validate_json(line)
            result[gj.trace_id] = gj
    return result


def load_resumes(path: Path) -> dict[str, GeneratedResume]:
    """Load resumes JSONL keyed by trace_id."""
    result: dict[str, GeneratedResume] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            gr = GeneratedResume.model_validate_json(line)
            result[gr.trace_id] = gr
    return result


def load_pairs(path: Path) -> list[ResumeJobPair]:
    """Load pairs JSONL as a list (order preserved for pagination)."""
    result: list[ResumeJobPair] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            result.append(ResumeJobPair.model_validate_json(line))
    return result


def load_failure_labels(path: Path) -> dict[str, FailureLabels]:
    """Load failure labels JSONL keyed by pair_id."""
    result: dict[str, FailureLabels] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            fl = FailureLabels.model_validate_json(line)
            result[fl.pair_id] = fl
    return result


def load_judge_results(path: Path) -> dict[str, JudgeResult]:
    """Load judge results JSONL keyed by pair_id."""
    result: dict[str, JudgeResult] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            jr = JudgeResult.model_validate_json(line)
            result[jr.pair_id] = jr
    return result


def load_correction_results(path: Path) -> dict[str, list[CorrectionResult]]:
    """Load correction results JSONL keyed by pair_id (multiple attempts per pair)."""
    result: dict[str, list[CorrectionResult]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            cr = CorrectionResult.model_validate_json(line)
            result.setdefault(cr.pair_id, []).append(cr)
    return result


def load_feedback(path: Path) -> dict[str, list[FeedbackRequest]]:
    """Load feedback JSONL keyed by pair_id (multiple feedback entries per pair)."""
    result: dict[str, list[FeedbackRequest]] = {}
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            fb = FeedbackRequest.model_validate_json(line)
            result.setdefault(fb.pair_id, []).append(fb)
    return result


def load_pipeline_summary() -> dict:
    """Load pipeline_summary.json from results/. Returns {} if missing."""
    summary_path = RESULTS_DIR / "pipeline_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


# ---------------------------------------------------------------------------
# DataStore — singleton loaded once at API startup (like a Spring @Bean)
# ---------------------------------------------------------------------------


class DataStore:
    """In-memory store of all pipeline artifacts.

    WHY: API endpoints need O(1) lookup by pair_id / trace_id. Loading once at
    startup and keeping ~1MB in memory is far cheaper than re-reading JSONL on
    every request. Same pattern as Spring's @Bean / @Singleton scope.
    """

    def __init__(self) -> None:
        jobs_path = find_latest(GENERATED_DIR, "jobs")
        resumes_path = find_latest(GENERATED_DIR, "resumes")
        pairs_path = find_latest(GENERATED_DIR, "pairs")

        self.jobs: dict[str, GeneratedJob] = load_jobs(jobs_path) if jobs_path else {}
        self.resumes: dict[str, GeneratedResume] = (
            load_resumes(resumes_path) if resumes_path else {}
        )
        self.pairs: list[ResumeJobPair] = load_pairs(pairs_path) if pairs_path else []

        labels_path = LABELED_DIR / "failure_labels.jsonl"
        self.failure_labels: dict[str, FailureLabels] = (
            load_failure_labels(labels_path) if labels_path.exists() else {}
        )

        judge_path = LABELED_DIR / "judge_results.jsonl"
        self.judge_results: dict[str, JudgeResult] = (
            load_judge_results(judge_path) if judge_path.exists() else {}
        )

        correction_path = CORRECTED_DIR / "correction_results.jsonl"
        self.correction_results: dict[str, list[CorrectionResult]] = (
            load_correction_results(correction_path) if correction_path.exists() else {}
        )

        feedback_path = FEEDBACK_DIR / "feedback.jsonl"
        self.feedback: dict[str, list[FeedbackRequest]] = load_feedback(feedback_path)

        self.pipeline_summary: dict = load_pipeline_summary()

    # Convenience accessors used by API endpoints

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    @property
    def pair_count(self) -> int:
        return len(self.pairs)

    @property
    def resume_count(self) -> int:
        return len(self.resumes)
