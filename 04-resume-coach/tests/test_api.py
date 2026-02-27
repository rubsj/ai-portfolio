"""
Tests for src/api.py — FastAPI endpoint coverage.

Strategy:
- FastAPI TestClient (httpx-based) drives all HTTP interactions.
- Module-level singletons (_store, _normalizer, _collection) are patched via
  unittest.mock.patch before the TestClient is constructed, so the real DataStore
  (which needs JSONL files on disk) is never loaded.
- Judge and multi_hop are imported lazily inside endpoints and patched via their
  full dotted path (src.api.judge_pair, etc.).
- All fixtures are built inline — no conftest.py required.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from src.schemas import (
    CompanyInfo,
    ContactInfo,
    Education,
    Experience,
    ExperienceLevel,
    FailureLabels,
    FitLevel,
    GeneratedJob,
    GeneratedResume,
    JobDescription,
    JobRequirements,
    JudgeResult,
    MultiHopQuestion,
    MultiHopResponse,
    ProficiencyLevel,
    Resume,
    ResumeJobPair,
    Skill,
    WritingStyle,
)


# ---------------------------------------------------------------------------
# Fixture builders — inline helpers for lean, readable tests
# ---------------------------------------------------------------------------


def _make_skill(name: str = "Python") -> Skill:
    return Skill(name=name, proficiency_level=ProficiencyLevel.ADVANCED, years=3)


def _make_experience(title: str = "Software Engineer") -> Experience:
    return Experience(
        company="Acme Corp",
        title=title,
        start_date="2020-01",
        end_date="2023-01",
        responsibilities=["Developed APIs", "Led code reviews"],
        achievements=["Reduced latency by 30%"],
    )


def _make_resume() -> Resume:
    return Resume(
        contact_info=ContactInfo(
            name="Jane Doe",
            email="jane@example.com",
            phone="555-555-5555",
            location="New York, NY",
            linkedin=None,
            portfolio=None,
        ),
        education=[
            Education(
                degree="B.S. Computer Science",
                institution="State University",
                graduation_date="2019-05",
                gpa=None,
                coursework=[],
            )
        ],
        experience=[_make_experience()],
        skills=[_make_skill("Python"), _make_skill("FastAPI")],
        summary="Experienced software engineer.",
    )


def _make_job_description() -> JobDescription:
    return JobDescription(
        title="Senior Software Engineer",
        company=CompanyInfo(
            name="TechCorp",
            industry="Technology",
            size="500-1000",
            location="San Francisco, CA",
        ),
        description="Build scalable APIs using Python and FastAPI.",
        requirements=JobRequirements(
            required_skills=["Python", "FastAPI"],
            preferred_skills=["Docker"],
            education="B.S. Computer Science",
            experience_years=3,
            experience_level=ExperienceLevel.SENIOR,
        ),
    )


def _make_generated_job(trace_id: str = "job-001") -> GeneratedJob:
    return GeneratedJob(
        trace_id=trace_id,
        job=_make_job_description(),
        is_niche_role=False,
        generated_at="2026-02-20T00:00:00",
        prompt_template="standard",
        model_used="gpt-4o-mini",
    )


def _make_generated_resume(trace_id: str = "resume-001") -> GeneratedResume:
    return GeneratedResume(
        trace_id=trace_id,
        resume=_make_resume(),
        fit_level=FitLevel.GOOD,
        writing_style=WritingStyle.FORMAL,
        template_version="v1",
        generated_at="2026-02-20T00:00:00",
        prompt_template="formal",
        model_used="gpt-4o-mini",
    )


def _make_pair(
    pair_id: str = "pair-001",
    resume_trace_id: str = "resume-001",
    job_trace_id: str = "job-001",
) -> ResumeJobPair:
    return ResumeJobPair(
        pair_id=pair_id,
        resume_trace_id=resume_trace_id,
        job_trace_id=job_trace_id,
        fit_level=FitLevel.GOOD,
        created_at="2026-02-20T00:00:00",
    )


def _make_failure_labels(pair_id: str = "pair-001") -> FailureLabels:
    return FailureLabels(
        pair_id=pair_id,
        skills_overlap=0.6,
        skills_overlap_raw=2,
        skills_union_raw=2,
        experience_mismatch=False,
        seniority_mismatch=False,
        missing_core_skills=False,
        has_hallucinations=False,
        has_awkward_language=False,
        experience_years_resume=3.0,
        experience_years_required=3,
        seniority_level_resume=3,
        seniority_level_job=3,
        missing_skills=[],
        hallucination_reasons=[],
        awkward_language_reasons=[],
        resume_skills_normalized=["python", "fastapi"],
        job_skills_normalized=["python", "fastapi"],
    )


def _make_judge_result(pair_id: str = "pair-001") -> JudgeResult:
    return JudgeResult(
        pair_id=pair_id,
        has_hallucinations=False,
        hallucination_details="",
        has_awkward_language=False,
        awkward_language_details="",
        overall_quality_score=0.8,
        fit_assessment="Good fit",
        recommendations=["Highlight FastAPI experience"],
        red_flags=[],
    )


def _make_pipeline_summary() -> dict:
    return {
        "day": "Day 2",
        "generation": {
            "pairs_generated": 250,
            "validation_rate": 1.0,
        },
        "labeling": {
            "failure_mode_rates": {"missing_core_skills": 0.5},
            "avg_jaccard_by_fit_level": {"excellent": 0.67, "mismatch": 0.005},
        },
        "correction": {"correction_rate": 1.0},
        "ab_testing": {
            "chi_squared_statistic": 32.74,
            "chi_squared_p_value": 1.35e-06,
            "significant": True,
            "best_template": "casual",
            "worst_template": "career_changer",
            "failure_rates_by_template": {
                "casual": 0.66,
                "career_changer": 1.0,
                "formal": 0.82,
                "achievement": 0.98,
                "technical": 0.74,
            },
        },
    }


# ---------------------------------------------------------------------------
# Mock DataStore factory — patch the module-level _store singleton
# ---------------------------------------------------------------------------


def _make_mock_store(
    *,
    pair_id: str = "pair-001",
    resume_trace_id: str = "resume-001",
    job_trace_id: str = "job-001",
) -> MagicMock:
    store = MagicMock()
    gr = _make_generated_resume(resume_trace_id)
    gj = _make_generated_job(job_trace_id)
    pair = _make_pair(pair_id, resume_trace_id, job_trace_id)
    labels = _make_failure_labels(pair_id)

    store.job_count = 1
    store.resume_count = 1
    store.pair_count = 1
    store.jobs = {job_trace_id: gj}
    store.resumes = {resume_trace_id: gr}
    store.pairs = [pair]
    store.failure_labels = {pair_id: labels}
    store.judge_results = {}
    store.correction_results = {}
    store.feedback = {}
    store.pipeline_summary = _make_pipeline_summary()
    return store


# ---------------------------------------------------------------------------
# TestClient factory — patches all module-level singletons then builds client
# ---------------------------------------------------------------------------


def _make_client(store=None, collection=None) -> TestClient:
    """
    WHY: Patching must happen before TestClient() is constructed, because the
    app is imported at module level. We patch src.api._store (the singleton
    already constructed), not DataStore.__init__, to keep it simple.
    """
    from src.api import app

    store = store or _make_mock_store()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Endpoint 1: GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_ok(self) -> None:
        store = _make_mock_store()
        with patch("src.api._store", store), patch("src.api._collection", MagicMock()):
            client = TestClient(__import__("src.api", fromlist=["app"]).app)
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert body["jobs"] == 1
        assert body["resumes"] == 1
        assert body["pairs"] == 1
        assert body["vector_store_ready"] is True

    def test_health_vector_store_not_ready_when_collection_none(self) -> None:
        store = _make_mock_store()
        with patch("src.api._store", store), patch("src.api._collection", None):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["vector_store_ready"] is False


# ---------------------------------------------------------------------------
# Endpoint 2: POST /review-resume
# ---------------------------------------------------------------------------


class TestReviewResume:
    def _payload(self) -> dict:
        return {
            "resume": _make_resume().model_dump(),
            "job_description": _make_job_description().model_dump(),
        }

    def test_review_resume_happy_path_no_judge(self) -> None:
        store = _make_mock_store()
        labels = _make_failure_labels()
        with (
            patch("src.api._store", store),
            patch("src.api._normalizer", MagicMock()),
            patch("src.api.label_pair", return_value=labels),
        ):
            from src.api import app

            client = TestClient(app)
            resp = client.post("/review-resume", json=self._payload())
        assert resp.status_code == 200
        body = resp.json()
        assert "pair_id" in body
        assert body["judge_result"] is None
        assert body["processing_time_seconds"] >= 0

    def test_review_resume_with_judge(self) -> None:
        store = _make_mock_store()
        labels = _make_failure_labels()
        judge_result = _make_judge_result()
        mock_client = MagicMock()
        with (
            patch("src.api._store", store),
            patch("src.api._normalizer", MagicMock()),
            patch("src.api.label_pair", return_value=labels),
            patch("src.judge._create_judge_client", return_value=mock_client),
            patch("src.judge.judge_pair", return_value=judge_result),
        ):
            from src.api import app

            client = TestClient(app)
            resp = client.post("/review-resume?use_judge=true", json=self._payload())
        assert resp.status_code == 200

    def test_review_resume_missing_fields_returns_422(self) -> None:
        from src.api import app

        client = TestClient(app)
        resp = client.post("/review-resume", json={"resume": {}})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint 3: GET /analysis/failure-rates
# ---------------------------------------------------------------------------


class TestFailureRates:
    def test_failure_rates_returns_data(self) -> None:
        store = _make_mock_store()
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/analysis/failure-rates")
        assert resp.status_code == 200
        body = resp.json()
        assert "total_pairs" in body
        assert "failure_mode_rates" in body
        assert body["total_pairs"] == 250

    def test_failure_rates_503_when_no_pipeline(self) -> None:
        store = _make_mock_store()
        store.pipeline_summary = {}
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/analysis/failure-rates")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Endpoint 4: GET /analysis/template-comparison
# ---------------------------------------------------------------------------


class TestTemplateComparison:
    def test_template_comparison_returns_data(self) -> None:
        store = _make_mock_store()
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/analysis/template-comparison")
        assert resp.status_code == 200
        body = resp.json()
        assert body["significant"] is True
        assert body["best_template"] == "casual"
        assert body["worst_template"] == "career_changer"
        assert len(body["template_results"]) == 5

    def test_template_comparison_503_when_no_ab_data(self) -> None:
        store = _make_mock_store()
        store.pipeline_summary = {}
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/analysis/template-comparison")
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Endpoint 5: POST /evaluate/multi-hop
# ---------------------------------------------------------------------------


class TestMultiHop:
    def _payload(self) -> dict:
        return {
            "resume": _make_resume().model_dump(),
            "job_description": _make_job_description().model_dump(),
        }

    def test_multi_hop_returns_questions(self) -> None:
        store = _make_mock_store()
        labels = _make_failure_labels()
        questions = [
            MultiHopQuestion(
                question=f"Q{i}",
                requires_sections=["experience"],
                answer=f"A{i}",
                assessment="correct",
            )
            for i in range(4)
        ]
        mock_result = MultiHopResponse(
            pair_id="pair-001",
            questions=questions,
            processing_time_seconds=0.1,
        )
        with (
            patch("src.api._store", store),
            patch("src.api._normalizer", MagicMock()),
            patch("src.api.label_pair", return_value=labels),
            patch("src.multi_hop.generate_multi_hop_questions", return_value=mock_result),
        ):
            from src.api import app

            client = TestClient(app)
            resp = client.post("/evaluate/multi-hop", json=self._payload())
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["questions"]) == 4

    def test_multi_hop_invalid_payload_422(self) -> None:
        from src.api import app

        client = TestClient(app)
        resp = client.post("/evaluate/multi-hop", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint 6: GET /search/similar-candidates
# ---------------------------------------------------------------------------


class TestSimilarCandidates:
    def _mock_collection(self) -> MagicMock:
        coll = MagicMock()
        coll.count.return_value = 250
        return coll

    def _mock_hits(self, trace_id: str = "resume-001") -> list[dict]:
        return [
            {
                "trace_id": trace_id,
                "score": 0.92,
                "metadata": {"fit_level": "good"},
            }
        ]

    def test_similar_candidates_returns_results(self) -> None:
        store = _make_mock_store()
        collection = self._mock_collection()
        hits = self._mock_hits()
        with (
            patch("src.api._store", store),
            patch("src.api._collection", collection),
            patch("src.api.search_similar", return_value=hits),
        ):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/search/similar-candidates?query=Python+engineer&top_k=3")
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "Python engineer"
        assert len(body["results"]) == 1
        assert body["results"][0]["similarity_score"] == 0.92
        assert body["total_in_index"] == 250

    def test_similar_candidates_with_fit_level_filter(self) -> None:
        store = _make_mock_store()
        collection = self._mock_collection()
        hits = self._mock_hits()
        with (
            patch("src.api._store", store),
            patch("src.api._collection", collection),
            patch("src.api.search_similar", return_value=hits) as mock_search,
        ):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/search/similar-candidates?query=senior+dev&fit_level=excellent")
        assert resp.status_code == 200
        # Verify fit_level was passed through
        _, kwargs = mock_search.call_args
        assert kwargs.get("fit_level") == "excellent" or mock_search.call_args[0][3] == "excellent"

    def test_similar_candidates_503_when_no_vector_store(self) -> None:
        with patch("src.api._collection", None):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/search/similar-candidates?query=test")
        assert resp.status_code == 503

    def test_similar_candidates_top_k_over_50_returns_422(self) -> None:
        collection = self._mock_collection()
        with patch("src.api._collection", collection):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/search/similar-candidates?query=test&top_k=99")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint 7: POST /feedback
# ---------------------------------------------------------------------------


class TestFeedback:
    def test_feedback_logs_and_returns_uuid(self, tmp_path: Any) -> None:
        store = _make_mock_store()
        store.feedback = {}
        with (
            patch("src.api._store", store),
            patch("src.api.FEEDBACK_DIR", tmp_path),
        ):
            from src.api import app

            client = TestClient(app)
            resp = client.post(
                "/feedback",
                json={"pair_id": "pair-001", "rating": "5", "comment": "Great!"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert "feedback_id" in body
        # feedback_id must be a valid UUID
        uuid.UUID(body["feedback_id"])
        assert "disk" in body["logged_to"]
        assert "memory" in body["logged_to"]
        # File was written
        feedback_file = tmp_path / "feedback.jsonl"
        assert feedback_file.exists()

    def test_feedback_missing_pair_id_returns_422(self) -> None:
        from src.api import app

        client = TestClient(app)
        resp = client.post("/feedback", json={"rating": 5})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint 8: GET /jobs
# ---------------------------------------------------------------------------


class TestListJobs:
    def _multi_job_store(self, count: int = 5) -> MagicMock:
        store = MagicMock()
        jobs = {}
        for i in range(count):
            gj = _make_generated_job(f"job-{i:03d}")
            gj.job.company.industry = "Technology" if i % 2 == 0 else "Finance"
            gj.is_niche_role = i % 3 == 0
            jobs[f"job-{i:03d}"] = gj
        store.jobs = jobs
        store.job_count = count
        store.resume_count = count
        store.pair_count = count
        store.pipeline_summary = _make_pipeline_summary()
        return store

    def test_jobs_returns_paginated_results(self) -> None:
        store = self._multi_job_store(5)
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/jobs?page=1&page_size=3")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["jobs"]) == 3
        assert body["total"] == 5
        assert body["total_pages"] == 2
        assert body["page"] == 1

    def test_jobs_page_2_returns_remaining(self) -> None:
        store = self._multi_job_store(5)
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/jobs?page=2&page_size=3")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["jobs"]) == 2

    def test_jobs_filter_by_industry(self) -> None:
        store = self._multi_job_store(5)
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/jobs?industry=Technology")
        assert resp.status_code == 200
        body = resp.json()
        for job in body["jobs"]:
            assert job["industry"].lower() == "technology"

    def test_jobs_filter_is_niche(self) -> None:
        store = self._multi_job_store(6)
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/jobs?is_niche=true")
        assert resp.status_code == 200
        body = resp.json()
        for job in body["jobs"]:
            assert job["is_niche"] is True

    def test_jobs_page_size_over_50_returns_422(self) -> None:
        from src.api import app

        client = TestClient(app)
        resp = client.get("/jobs?page_size=51")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Endpoint 9: GET /pairs/{pair_id}
# ---------------------------------------------------------------------------


class TestPairDetail:
    def test_pair_detail_returns_full_response(self) -> None:
        store = _make_mock_store()
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/pairs/pair-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["pair_id"] == "pair-001"
        assert body["resume"] is not None
        assert body["job"] is not None

    def test_pair_detail_unknown_id_returns_404(self) -> None:
        store = _make_mock_store()
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/pairs/does-not-exist")
        assert resp.status_code == 404

    def test_pair_detail_includes_failure_labels(self) -> None:
        store = _make_mock_store()
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/pairs/pair-001")
        assert resp.status_code == 200
        body = resp.json()
        assert body["failure_labels"] is not None
        assert "skills_overlap" in body["failure_labels"]

    def test_pair_detail_judge_result_none_when_not_run(self) -> None:
        store = _make_mock_store()
        store.judge_results = {}
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/pairs/pair-001")
        assert resp.status_code == 200
        assert resp.json()["judge_result"] is None

    def test_pair_detail_404_when_resume_missing_from_store(self) -> None:
        """Covers line 336: resume is None or job is None → 404."""
        store = _make_mock_store()
        # Pair exists but its resume_trace_id is not in store.resumes
        store.resumes = {}  # Empty — resume lookup returns None
        with patch("src.api._store", store):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/pairs/pair-001")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# search/similar-candidates — trace_id not in store (line 218)
# ---------------------------------------------------------------------------


class TestSimilarCandidatesOrphanHit:
    """Covers line 218: gr is None → continue (trace_id in vector store but not in DataStore)."""

    def test_orphan_trace_id_is_silently_skipped(self) -> None:
        store = _make_mock_store()
        collection = MagicMock()
        collection.count.return_value = 250
        # Return a trace_id that does NOT exist in store.resumes
        hits = [{"trace_id": "nonexistent-trace-id", "score": 0.95, "metadata": {"fit_level": "excellent"}}]
        with (
            patch("src.api._store", store),
            patch("src.api._collection", collection),
            patch("src.api.search_similar", return_value=hits),
        ):
            from src.api import app

            client = TestClient(app)
            resp = client.get("/search/similar-candidates?query=python")
        assert resp.status_code == 200
        # Orphan hit is skipped → results list is empty
        assert resp.json()["results"] == []
