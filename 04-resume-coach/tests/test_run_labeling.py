"""
test_run_labeling.py — Unit tests for src/run_labeling.py.

Strategy:
- Loader functions (load_jobs, load_resumes, load_pairs) are tested with
  minimal in-memory JSONL fixtures written to tmp_path.
- run() is tested by monkeypatching the 5 module-level file Path constants
  (JOBS_FILE, RESUMES_FILE, PAIRS_FILE, OUTPUT_FILE, LABELED_DIR) to tmp_path
  files so no real data files are needed and no filesystem side effects occur.
- label_pair() is called for real (no mock) — run_labeling.py has no LLM calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


import src.run_labeling as rl
from src.run_labeling import load_jobs, load_pairs, load_resumes
from src.schemas import (
    CompanyInfo,
    ContactInfo,
    Education,
    ExperienceLevel,
    FitLevel,
    GeneratedJob,
    GeneratedResume,
    JobDescription,
    JobRequirements,
    ProficiencyLevel,
    Resume,
    ResumeJobPair,
    Skill,
    WritingStyle,
)
from src.schemas import Experience as ExpSchema


# ---------------------------------------------------------------------------
# Builder helpers — mirrors run_labeling test data shape
# ---------------------------------------------------------------------------


def _make_contact() -> ContactInfo:
    return ContactInfo(
        name="Alice Smith",
        email="alice@example.com",
        phone="555-111-2222",
        location="Seattle, WA",
    )


def _make_resume() -> Resume:
    return Resume(
        contact_info=_make_contact(),
        education=[
            Education(
                degree="BS Computer Science",
                institution="State University",
                graduation_date="2019-05",
            )
        ],
        experience=[
            ExpSchema(
                company="TechCorp",
                title="Software Engineer",
                start_date="2019-06",
                end_date="2023-05",
                responsibilities=["Built microservices", "Led code reviews"],
            )
        ],
        skills=[
            Skill(name="Python", proficiency_level=ProficiencyLevel.ADVANCED, years=4),
            Skill(name="SQL", proficiency_level=ProficiencyLevel.INTERMEDIATE, years=3),
        ],
        summary="Experienced engineer with Python and SQL expertise.",
    )


def _make_job() -> JobDescription:
    return JobDescription(
        title="Senior Software Engineer",
        company=CompanyInfo(
            name="Acme Inc",
            industry="Technology",
            size="Enterprise (500+)",
            location="San Francisco, CA",
        ),
        description="Build scalable backend systems",
        requirements=JobRequirements(
            required_skills=["Python", "SQL"],
            education="BS degree required",
            experience_years=3,
            experience_level=ExperienceLevel.MID,
        ),
    )


def _make_generated_job(trace_id: str = "j1") -> GeneratedJob:
    return GeneratedJob(
        trace_id=trace_id,
        job=_make_job(),
        is_niche_role=False,
        generated_at="2026-02-25T14:00:00Z",
        prompt_template="v1-technical",
        model_used="gpt-4o-mini",
    )


def _make_generated_resume(trace_id: str = "r1") -> GeneratedResume:
    return GeneratedResume(
        trace_id=trace_id,
        resume=_make_resume(),
        fit_level=FitLevel.EXCELLENT,
        writing_style=WritingStyle.TECHNICAL,
        template_version="v1",
        generated_at="2026-02-25T14:00:00Z",
        prompt_template="v1-technical",
        model_used="gpt-4o-mini",
    )


def _make_pair(
    pair_id: str = "p1",
    resume_trace_id: str = "r1",
    job_trace_id: str = "j1",
    fit_level: FitLevel = FitLevel.EXCELLENT,
) -> ResumeJobPair:
    return ResumeJobPair(
        pair_id=pair_id,
        resume_trace_id=resume_trace_id,
        job_trace_id=job_trace_id,
        fit_level=fit_level,
        created_at="2026-02-25T14:00:00Z",
    )


# ---------------------------------------------------------------------------
# TestLoadFunctions — unit tests for the three loader functions
# ---------------------------------------------------------------------------


class TestLoadJobs:
    def test_returns_dict_keyed_by_trace_id(self, tmp_path: Path) -> None:
        gj = _make_generated_job("job-abc")
        jsonl = tmp_path / "jobs.jsonl"
        jsonl.write_text(gj.model_dump_json() + "\n")

        result = load_jobs(jsonl)

        assert "job-abc" in result
        assert result["job-abc"].trace_id == "job-abc"

    def test_loads_multiple_jobs(self, tmp_path: Path) -> None:
        gj1 = _make_generated_job("j1")
        gj2 = _make_generated_job("j2")
        jsonl = tmp_path / "jobs.jsonl"
        jsonl.write_text(gj1.model_dump_json() + "\n" + gj2.model_dump_json() + "\n")

        result = load_jobs(jsonl)

        assert len(result) == 2
        assert "j1" in result
        assert "j2" in result

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "jobs.jsonl"
        jsonl.write_text("")

        result = load_jobs(jsonl)

        assert result == {}

    def test_job_object_is_valid_generated_job(self, tmp_path: Path) -> None:
        gj = _make_generated_job("j99")
        jsonl = tmp_path / "jobs.jsonl"
        jsonl.write_text(gj.model_dump_json() + "\n")

        result = load_jobs(jsonl)

        loaded = result["j99"]
        assert loaded.job.company.name == "Acme Inc"
        assert loaded.job.requirements.experience_years == 3


class TestLoadResumes:
    def test_returns_dict_keyed_by_trace_id(self, tmp_path: Path) -> None:
        gr = _make_generated_resume("resume-xyz")
        jsonl = tmp_path / "resumes.jsonl"
        jsonl.write_text(gr.model_dump_json() + "\n")

        result = load_resumes(jsonl)

        assert "resume-xyz" in result
        assert result["resume-xyz"].trace_id == "resume-xyz"

    def test_loads_multiple_resumes(self, tmp_path: Path) -> None:
        gr1 = _make_generated_resume("r1")
        gr2 = _make_generated_resume("r2")
        jsonl = tmp_path / "resumes.jsonl"
        jsonl.write_text(gr1.model_dump_json() + "\n" + gr2.model_dump_json() + "\n")

        result = load_resumes(jsonl)

        assert len(result) == 2

    def test_empty_file_returns_empty_dict(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "resumes.jsonl"
        jsonl.write_text("")

        result = load_resumes(jsonl)

        assert result == {}

    def test_resume_skills_accessible(self, tmp_path: Path) -> None:
        gr = _make_generated_resume("r-skills")
        jsonl = tmp_path / "resumes.jsonl"
        jsonl.write_text(gr.model_dump_json() + "\n")

        result = load_resumes(jsonl)

        skills = [s.name for s in result["r-skills"].resume.skills]
        assert "Python" in skills


class TestLoadPairs:
    def test_returns_list_of_pairs(self, tmp_path: Path) -> None:
        pair = _make_pair("p1")
        jsonl = tmp_path / "pairs.jsonl"
        jsonl.write_text(pair.model_dump_json() + "\n")

        result = load_pairs(jsonl)

        assert len(result) == 1
        assert result[0].pair_id == "p1"

    def test_loads_multiple_pairs(self, tmp_path: Path) -> None:
        pair1 = _make_pair("p1")
        pair2 = _make_pair("p2", fit_level=FitLevel.POOR)
        jsonl = tmp_path / "pairs.jsonl"
        jsonl.write_text(pair1.model_dump_json() + "\n" + pair2.model_dump_json() + "\n")

        result = load_pairs(jsonl)

        assert len(result) == 2
        assert result[1].fit_level == FitLevel.POOR

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "pairs.jsonl"
        jsonl.write_text("")

        result = load_pairs(jsonl)

        assert result == []

    def test_pair_fields_are_correct(self, tmp_path: Path) -> None:
        pair = _make_pair("p-check", "r99", "j99", FitLevel.GOOD)
        jsonl = tmp_path / "pairs.jsonl"
        jsonl.write_text(pair.model_dump_json() + "\n")

        result = load_pairs(jsonl)

        assert result[0].resume_trace_id == "r99"
        assert result[0].job_trace_id == "j99"
        assert result[0].fit_level == FitLevel.GOOD


# ---------------------------------------------------------------------------
# TestRunPipeline — tests for run() with monkeypatched file paths
# ---------------------------------------------------------------------------


def _write_fixtures(tmp_path: Path, fit_level: FitLevel = FitLevel.EXCELLENT) -> dict[str, Path]:
    """Write minimal JSONL fixtures; return dict of path constants to monkeypatch."""
    gj = _make_generated_job("j1")
    gr = _make_generated_resume("r1")
    pair = _make_pair("p1", "r1", "j1", fit_level)

    jobs_file = tmp_path / "jobs.jsonl"
    resumes_file = tmp_path / "resumes.jsonl"
    pairs_file = tmp_path / "pairs.jsonl"
    output_file = tmp_path / "failure_labels.jsonl"
    labeled_dir = tmp_path / "labeled"
    labeled_dir.mkdir(parents=True, exist_ok=True)

    jobs_file.write_text(gj.model_dump_json() + "\n")
    resumes_file.write_text(gr.model_dump_json() + "\n")
    pairs_file.write_text(pair.model_dump_json() + "\n")

    return {
        "JOBS_FILE": jobs_file,
        "RESUMES_FILE": resumes_file,
        "PAIRS_FILE": pairs_file,
        "OUTPUT_FILE": output_file,
        "LABELED_DIR": labeled_dir,
    }


class TestRunPipeline:
    def test_run_creates_output_file(self, tmp_path: Path) -> None:
        """Successful run with 1 pair produces failure_labels.jsonl."""
        paths = _write_fixtures(tmp_path)
        with (
            patch.object(rl, "JOBS_FILE", paths["JOBS_FILE"]),
            patch.object(rl, "RESUMES_FILE", paths["RESUMES_FILE"]),
            patch.object(rl, "PAIRS_FILE", paths["PAIRS_FILE"]),
            patch.object(rl, "OUTPUT_FILE", paths["OUTPUT_FILE"]),
            patch.object(rl, "LABELED_DIR", paths["LABELED_DIR"]),
        ):
            rl.run()

        assert paths["OUTPUT_FILE"].exists()

    def test_run_writes_one_label_per_pair(self, tmp_path: Path) -> None:
        paths = _write_fixtures(tmp_path)
        with (
            patch.object(rl, "JOBS_FILE", paths["JOBS_FILE"]),
            patch.object(rl, "RESUMES_FILE", paths["RESUMES_FILE"]),
            patch.object(rl, "PAIRS_FILE", paths["PAIRS_FILE"]),
            patch.object(rl, "OUTPUT_FILE", paths["OUTPUT_FILE"]),
            patch.object(rl, "LABELED_DIR", paths["LABELED_DIR"]),
        ):
            rl.run()

        lines = paths["OUTPUT_FILE"].read_text().strip().splitlines()
        assert len(lines) == 1  # 1 pair → 1 label

    def test_run_output_is_valid_json(self, tmp_path: Path) -> None:
        import json

        paths = _write_fixtures(tmp_path)
        with (
            patch.object(rl, "JOBS_FILE", paths["JOBS_FILE"]),
            patch.object(rl, "RESUMES_FILE", paths["RESUMES_FILE"]),
            patch.object(rl, "PAIRS_FILE", paths["PAIRS_FILE"]),
            patch.object(rl, "OUTPUT_FILE", paths["OUTPUT_FILE"]),
            patch.object(rl, "LABELED_DIR", paths["LABELED_DIR"]),
        ):
            rl.run()

        line = paths["OUTPUT_FILE"].read_text().strip()
        record = json.loads(line)
        # FailureLabels fields must be present
        assert "pair_id" in record
        assert "skills_overlap" in record
        assert "experience_mismatch" in record

    def test_run_handles_missing_trace_id_gracefully(self, tmp_path: Path) -> None:
        """Pair references a missing resume trace_id — KeyError is caught, pipeline continues.

        WHY include p-ok: run_labeling.run() divides by total=len(all_labels) when building
        the overall stats table. If all pairs fail, total=0 → ZeroDivisionError in source.
        Including one valid pair ensures total>=1 and exercises the error-logging branch.
        """
        gj = _make_generated_job("j1")
        gr = _make_generated_resume("r1")  # only r1 exists — r-MISSING will KeyError
        pair_ok = _make_pair("p-ok", resume_trace_id="r1", job_trace_id="j1")
        pair_bad = _make_pair("p-bad", resume_trace_id="r-MISSING", job_trace_id="j1")

        jobs_file = tmp_path / "jobs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        pairs_file = tmp_path / "pairs.jsonl"
        output_file = tmp_path / "failure_labels.jsonl"
        labeled_dir = tmp_path / "labeled"
        labeled_dir.mkdir(parents=True, exist_ok=True)

        jobs_file.write_text(gj.model_dump_json() + "\n")
        resumes_file.write_text(gr.model_dump_json() + "\n")  # r1 only, not r-MISSING
        pairs_file.write_text(pair_ok.model_dump_json() + "\n" + pair_bad.model_dump_json() + "\n")

        with (
            patch.object(rl, "JOBS_FILE", jobs_file),
            patch.object(rl, "RESUMES_FILE", resumes_file),
            patch.object(rl, "PAIRS_FILE", pairs_file),
            patch.object(rl, "OUTPUT_FILE", output_file),
            patch.object(rl, "LABELED_DIR", labeled_dir),
        ):
            rl.run()  # must not raise

        # p-ok succeeded (1 label), p-bad errored (KeyError) — output has 1 line
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_run_with_multiple_fit_levels(self, tmp_path: Path) -> None:
        """Multiple pairs across fit levels — summary table is rendered without error."""
        gj = _make_generated_job("j1")
        gr = _make_generated_resume("r1")

        # Two pairs: excellent + poor
        pair_exc = _make_pair("p1", "r1", "j1", FitLevel.EXCELLENT)
        pair_poor = _make_pair("p2", "r1", "j1", FitLevel.POOR)

        jobs_file = tmp_path / "jobs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        pairs_file = tmp_path / "pairs.jsonl"
        output_file = tmp_path / "failure_labels.jsonl"
        labeled_dir = tmp_path / "labeled"
        labeled_dir.mkdir(parents=True, exist_ok=True)

        jobs_file.write_text(gj.model_dump_json() + "\n")
        resumes_file.write_text(gr.model_dump_json() + "\n")
        pairs_file.write_text(
            pair_exc.model_dump_json() + "\n" + pair_poor.model_dump_json() + "\n"
        )

        with (
            patch.object(rl, "JOBS_FILE", jobs_file),
            patch.object(rl, "RESUMES_FILE", resumes_file),
            patch.object(rl, "PAIRS_FILE", pairs_file),
            patch.object(rl, "OUTPUT_FILE", output_file),
            patch.object(rl, "LABELED_DIR", labeled_dir),
        ):
            rl.run()

        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_run_increments_failure_flag_counters(self, tmp_path: Path) -> None:
        """Pair with experience_mismatch=True causes stats counter (line 160) to increment.

        WHY experience_years=10: resume has ~4 years (2019-2023). The mismatch condition
        is resume_years < job_years * 0.5 → 4 < 5 → True, triggering s[col] += 1 at line 160.
        """
        import json

        mismatched_job = JobDescription(
            title="Principal Software Engineer",
            company=CompanyInfo(
                name="MegaCorp",
                industry="Technology",
                size="Enterprise (500+)",
                location="New York, NY",
            ),
            description="Senior leadership role",
            requirements=JobRequirements(
                required_skills=["Rust", "Kubernetes"],
                experience_years=10,
                experience_level=ExperienceLevel.SENIOR,
                education="MS required",
            ),
        )
        gj = GeneratedJob(
            trace_id="j-mismatch",
            job=mismatched_job,
            is_niche_role=False,
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )
        gr = _make_generated_resume("r1")
        pair = _make_pair("p1", "r1", "j-mismatch", FitLevel.MISMATCH)

        jobs_file = tmp_path / "jobs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        pairs_file = tmp_path / "pairs.jsonl"
        output_file = tmp_path / "failure_labels.jsonl"
        labeled_dir = tmp_path / "labeled"
        labeled_dir.mkdir(parents=True, exist_ok=True)

        jobs_file.write_text(gj.model_dump_json() + "\n")
        resumes_file.write_text(gr.model_dump_json() + "\n")
        pairs_file.write_text(pair.model_dump_json() + "\n")

        with (
            patch.object(rl, "JOBS_FILE", jobs_file),
            patch.object(rl, "RESUMES_FILE", resumes_file),
            patch.object(rl, "PAIRS_FILE", pairs_file),
            patch.object(rl, "OUTPUT_FILE", output_file),
            patch.object(rl, "LABELED_DIR", labeled_dir),
        ):
            rl.run()

        # Verify label was written and experience_mismatch was True (triggered line 160)
        record = json.loads(output_file.read_text().strip())
        assert record["experience_mismatch"] is True

    def test_run_handles_general_exception_gracefully(self, tmp_path: Path) -> None:
        """Non-KeyError exception from label_pair is caught; pipeline continues.

        WHY use two pairs + selective failure: label_pair failure on ALL pairs causes
        ZeroDivisionError in run_labeling.run() stats table (total=0). We include p-ok
        (which calls real label_pair) + p-fail (which triggers RuntimeError), so total=1.
        """
        from src.labeler import label_pair as real_label_pair

        gj = _make_generated_job("j1")
        gr1 = _make_generated_resume("r1")
        gr2 = _make_generated_resume("r2")
        pair_ok = _make_pair("p-ok", "r1", "j1")
        pair_fail = _make_pair("p-fail", "r2", "j1")

        jobs_file = tmp_path / "jobs.jsonl"
        resumes_file = tmp_path / "resumes.jsonl"
        pairs_file = tmp_path / "pairs.jsonl"
        output_file = tmp_path / "failure_labels.jsonl"
        labeled_dir = tmp_path / "labeled"
        labeled_dir.mkdir(parents=True, exist_ok=True)

        jobs_file.write_text(gj.model_dump_json() + "\n")
        resumes_file.write_text(gr1.model_dump_json() + "\n" + gr2.model_dump_json() + "\n")
        pairs_file.write_text(pair_ok.model_dump_json() + "\n" + pair_fail.model_dump_json() + "\n")

        def selective_label_pair(resume, job, pair_id, normalizer=None):
            """Fail for p-fail only; delegate to real label_pair for all others."""
            if pair_id == "p-fail":
                raise RuntimeError("simulated labeling crash")
            return real_label_pair(resume, job, pair_id, normalizer)

        with (
            patch.object(rl, "JOBS_FILE", jobs_file),
            patch.object(rl, "RESUMES_FILE", resumes_file),
            patch.object(rl, "PAIRS_FILE", pairs_file),
            patch.object(rl, "OUTPUT_FILE", output_file),
            patch.object(rl, "LABELED_DIR", labeled_dir),
            patch("src.run_labeling.label_pair", side_effect=selective_label_pair),
        ):
            rl.run()  # must not raise

        # p-ok succeeded (1 label written), p-fail errored (caught gracefully)
        lines = output_file.read_text().strip().splitlines()
        assert len(lines) == 1
