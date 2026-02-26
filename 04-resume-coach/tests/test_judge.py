"""
test_judge.py — Unit tests for src/judge.py.

All LLM API calls are mocked using MagicMock. No API keys or real network calls needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from src.judge import (
    _JudgeLLMOutput,
    _build_judge_prompt,
    _load_cache,
    _prompt_hash,
    _save_cache,
    judge_batch,
    judge_pair,
    save_judge_results,
)
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
    JudgeResult,
    ProficiencyLevel,
    Resume,
    ResumeJobPair,
    Skill,
    WritingStyle,
)
from src.schemas import Experience as ExpSchema


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _make_contact() -> ContactInfo:
    return ContactInfo(
        name="Test User",
        email="test@example.com",
        phone="555-000-1111",
        location="Austin, TX",
    )


def _make_resume() -> Resume:
    return Resume(
        contact_info=_make_contact(),
        education=[Education(degree="BS CS", institution="UT Austin", graduation_date="2019-05")],
        experience=[
            ExpSchema(
                company="ACME",
                title="Software Engineer",
                start_date="2019-06",
                end_date="2024-01",
                responsibilities=["Developed APIs"],
            )
        ],
        skills=[Skill(name="Python", proficiency_level=ProficiencyLevel.ADVANCED, years=5)],
        summary="Software engineer with 5 years experience",
    )


def _make_job() -> JobDescription:
    return JobDescription(
        title="Senior Software Engineer",
        company=CompanyInfo(
            name="Globex",
            industry="Technology",
            size="Enterprise (500+)",
            location="Austin, TX",
        ),
        description="Senior engineer role",
        requirements=JobRequirements(
            required_skills=["Python", "Django"],
            education="BS degree",
            experience_years=4,
            experience_level=ExperienceLevel.SENIOR,
        ),
    )


def _make_llm_output() -> _JudgeLLMOutput:
    return _JudgeLLMOutput(
        has_hallucinations=False,
        hallucination_details="No hallucinations detected",
        has_awkward_language=False,
        awkward_language_details="Language is natural",
        overall_quality_score=0.75,
        fit_assessment="Good fit for the role",
        recommendations=["Highlight leadership experience"],
        red_flags=[],
    )


def _make_mock_client(return_value: _JudgeLLMOutput | None = None) -> MagicMock:
    """Return a mock Instructor client that yields `return_value` on create()."""
    client = MagicMock()
    if return_value is not None:
        client.chat.completions.create.return_value = return_value
    return client


# ---------------------------------------------------------------------------
# TestJudgeLLMOutput
# ---------------------------------------------------------------------------


class TestJudgeLLMOutput:
    def test_valid_score_accepted(self) -> None:
        out = _JudgeLLMOutput(
            has_hallucinations=True,
            hallucination_details="Some hallucination",
            has_awkward_language=False,
            awkward_language_details="",
            overall_quality_score=0.5,
            fit_assessment="Partial fit",
            recommendations=[],
            red_flags=["Gap in timeline"],
        )
        assert out.overall_quality_score == 0.5

    def test_score_above_one_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="Quality score"):
            _JudgeLLMOutput(
                has_hallucinations=False,
                hallucination_details="",
                has_awkward_language=False,
                awkward_language_details="",
                overall_quality_score=1.5,  # invalid
                fit_assessment="",
                recommendations=[],
                red_flags=[],
            )

    def test_score_below_zero_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="Quality score"):
            _JudgeLLMOutput(
                has_hallucinations=False,
                hallucination_details="",
                has_awkward_language=False,
                awkward_language_details="",
                overall_quality_score=-0.1,  # invalid
                fit_assessment="",
                recommendations=[],
                red_flags=[],
            )

    def test_boundary_score_zero_accepted(self) -> None:
        out = _JudgeLLMOutput(
            has_hallucinations=False,
            hallucination_details="",
            has_awkward_language=False,
            awkward_language_details="",
            overall_quality_score=0.0,
            fit_assessment="Not suitable",
            recommendations=[],
            red_flags=[],
        )
        assert out.overall_quality_score == 0.0

    def test_boundary_score_one_accepted(self) -> None:
        out = _make_llm_output()
        out_max = out.model_copy(update={"overall_quality_score": 1.0})
        assert out_max.overall_quality_score == 1.0


# ---------------------------------------------------------------------------
# TestPromptHash
# ---------------------------------------------------------------------------


class TestPromptHash:
    def test_same_inputs_produce_same_hash(self) -> None:
        h1 = _prompt_hash("system", "user")
        h2 = _prompt_hash("system", "user")
        assert h1 == h2

    def test_different_system_prompt_different_hash(self) -> None:
        h1 = _prompt_hash("system_A", "user")
        h2 = _prompt_hash("system_B", "user")
        assert h1 != h2

    def test_different_user_prompt_different_hash(self) -> None:
        h1 = _prompt_hash("system", "user_A")
        h2 = _prompt_hash("system", "user_B")
        assert h1 != h2

    def test_returns_32_char_hex_string(self) -> None:
        h = _prompt_hash("sys", "usr")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# TestJudgeCache
# ---------------------------------------------------------------------------


class TestJudgeCache:
    def test_load_cache_miss_returns_none(self, tmp_path: Path) -> None:
        import src.judge as jmod

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            result = _load_cache("nonexistent_key")
        assert result is None

    def test_load_cache_hit_returns_output(self, tmp_path: Path) -> None:
        import src.judge as jmod

        llm_out = _make_llm_output()
        cache_file = tmp_path / "abc123.json"
        payload = {
            "cache_key": "abc123",
            "model": "gpt-4o",
            "timestamp": "2026-02-25T00:00:00Z",
            "response": llm_out.model_dump(),
        }
        cache_file.write_text(json.dumps(payload))

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            result = _load_cache("abc123")

        assert result is not None
        assert result.overall_quality_score == 0.75

    def test_load_cache_bad_json_returns_none(self, tmp_path: Path) -> None:
        import src.judge as jmod

        cache_file = tmp_path / "badkey.json"
        cache_file.write_text("{not valid json")

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            result = _load_cache("badkey")
        assert result is None

    def test_load_cache_missing_response_key_returns_none(self, tmp_path: Path) -> None:
        import src.judge as jmod

        cache_file = tmp_path / "noresponse.json"
        cache_file.write_text(json.dumps({"cache_key": "noresponse"}))  # no "response" key

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            result = _load_cache("noresponse")
        assert result is None

    def test_save_cache_creates_file(self, tmp_path: Path) -> None:
        import src.judge as jmod

        llm_out = _make_llm_output()

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            _save_cache("testkey", llm_out)

        cache_file = tmp_path / "testkey.json"
        assert cache_file.exists()

    def test_save_cache_file_has_required_keys(self, tmp_path: Path) -> None:
        import src.judge as jmod

        llm_out = _make_llm_output()

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            _save_cache("mykey", llm_out)

        payload = json.loads((tmp_path / "mykey.json").read_text())
        assert "cache_key" in payload
        assert "model" in payload
        assert "timestamp" in payload
        assert "response" in payload
        assert payload["response"]["overall_quality_score"] == 0.75

    def test_save_cache_creates_directory_if_missing(self, tmp_path: Path) -> None:
        import src.judge as jmod

        subdir = tmp_path / "deep" / "cache"  # does not exist yet
        llm_out = _make_llm_output()

        with patch.object(jmod, "_CACHE_DIR", subdir):
            _save_cache("key99", llm_out)

        assert (subdir / "key99.json").exists()


# ---------------------------------------------------------------------------
# TestBuildJudgePrompt
# ---------------------------------------------------------------------------


class TestBuildJudgePrompt:
    def test_returns_two_strings(self) -> None:
        job = _make_job()
        resume = _make_resume()
        system, user = _build_judge_prompt(job, resume)
        assert isinstance(system, str) and isinstance(user, str)

    def test_system_prompt_contains_hallucination(self) -> None:
        job = _make_job()
        resume = _make_resume()
        system, _ = _build_judge_prompt(job, resume)
        assert "HALLUCINATION" in system.upper() or "hallucination" in system

    def test_user_prompt_contains_job_and_resume_sections(self) -> None:
        job = _make_job()
        resume = _make_resume()
        _, user = _build_judge_prompt(job, resume)
        assert "JOB DESCRIPTION" in user
        assert "RESUME" in user

    def test_user_prompt_includes_job_title(self) -> None:
        job = _make_job()
        resume = _make_resume()
        _, user = _build_judge_prompt(job, resume)
        assert "Senior Software Engineer" in user

    def test_user_prompt_includes_resume_name(self) -> None:
        job = _make_job()
        resume = _make_resume()
        _, user = _build_judge_prompt(job, resume)
        assert "Test User" in user  # from contact_info.name


# ---------------------------------------------------------------------------
# TestJudgePair
# ---------------------------------------------------------------------------


class TestJudgePair:
    def test_cache_hit_returns_judge_result_without_api(self, tmp_path: Path) -> None:
        """If cache hit, LLM client should NOT be called."""
        import src.judge as jmod

        job = _make_job()
        resume = _make_resume()
        llm_out = _make_llm_output()

        # Pre-populate cache
        system, user = _build_judge_prompt(job, resume)
        key = _prompt_hash(system, user)

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            _save_cache(key, llm_out)

        client = _make_mock_client()

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            result = judge_pair(client, job, resume, "pair-001", use_cache=True)

        client.chat.completions.create.assert_not_called()
        assert result.pair_id == "pair-001"
        assert result.overall_quality_score == 0.75

    def test_cache_miss_calls_api(self, tmp_path: Path) -> None:
        """On cache miss, the API client should be called once."""
        import src.judge as jmod

        job = _make_job()
        resume = _make_resume()
        llm_out = _make_llm_output()
        client = _make_mock_client(return_value=llm_out)

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            result = judge_pair(client, job, resume, "pair-002", use_cache=True)

        client.chat.completions.create.assert_called_once()
        assert result.pair_id == "pair-002"

    def test_cache_miss_saves_to_cache(self, tmp_path: Path) -> None:
        import src.judge as jmod

        job = _make_job()
        resume = _make_resume()
        llm_out = _make_llm_output()
        client = _make_mock_client(return_value=llm_out)

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            judge_pair(client, job, resume, "pair-003", use_cache=True)

        assert len(list(tmp_path.glob("*.json"))) == 1

    def test_no_cache_skips_read_and_write(self, tmp_path: Path) -> None:
        """use_cache=False should not read from or write to cache."""
        import src.judge as jmod

        job = _make_job()
        resume = _make_resume()
        llm_out = _make_llm_output()
        client = _make_mock_client(return_value=llm_out)

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            judge_pair(client, job, resume, "pair-nocache", use_cache=False)

        assert len(list(tmp_path.glob("*.json"))) == 0

    def test_pair_id_injected_correctly(self, tmp_path: Path) -> None:
        import src.judge as jmod

        job = _make_job()
        resume = _make_resume()
        client = _make_mock_client(return_value=_make_llm_output())

        with patch.object(jmod, "_CACHE_DIR", tmp_path):
            result = judge_pair(client, job, resume, "my-special-id", use_cache=False)

        assert result.pair_id == "my-special-id"
        # Confirm pair_id is not in the LLM call arguments
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages", call_kwargs.args[0] if call_kwargs.args else [])
        # pair_id should NOT appear in the prompts (it's injected post-call)
        all_content = " ".join(m.get("content", "") for m in messages)
        assert "my-special-id" not in all_content


# ---------------------------------------------------------------------------
# TestJudgeBatch
# ---------------------------------------------------------------------------


class TestJudgeBatch:
    def _make_pair(self, pair_id: str, resume_trace: str, job_trace: str) -> ResumeJobPair:
        return ResumeJobPair(
            pair_id=pair_id,
            resume_trace_id=resume_trace,
            job_trace_id=job_trace,
            fit_level=FitLevel.GOOD,
            created_at="2026-02-25T14:00:00Z",
        )

    def test_successful_batch_returns_all_results(self, tmp_path: Path) -> None:
        import src.judge as jmod

        job = _make_job()
        resume = _make_resume()
        gj = GeneratedJob(
            trace_id="j1",
            job=job,
            is_niche_role=False,
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1",
            model_used="gpt-4o-mini",
        )
        gr = GeneratedResume(
            trace_id="r1",
            resume=resume,
            fit_level=FitLevel.GOOD,
            writing_style=WritingStyle.FORMAL,
            template_version="v1",
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-formal",
            model_used="gpt-4o-mini",
        )
        pairs = [self._make_pair("p1", "r1", "j1")]
        jobs = {"j1": gj}
        resumes = {"r1": gr}

        llm_out = _make_llm_output()

        with (
            patch.object(jmod, "_CACHE_DIR", tmp_path),
            patch.object(jmod, "_create_judge_client", return_value=_make_mock_client(return_value=llm_out)),
        ):
            results = judge_batch(pairs, jobs, resumes, use_cache=False)

        assert len(results) == 1
        assert results[0].pair_id == "p1"

    def test_missing_trace_id_excluded_from_results(self, tmp_path: Path) -> None:
        """If a pair references an unknown trace_id, it should be excluded (not crash)."""
        import src.judge as jmod

        pairs = [self._make_pair("p-bad", "r-MISSING", "j-MISSING")]

        with (
            patch.object(jmod, "_CACHE_DIR", tmp_path),
            patch.object(jmod, "_create_judge_client", return_value=_make_mock_client()),
        ):
            results = judge_batch(pairs, {}, {}, use_cache=False)

        assert results == []

    def test_api_exception_excluded_from_results(self, tmp_path: Path) -> None:
        """If the LLM call raises, the pair should be excluded (not crash)."""
        import src.judge as jmod

        job = _make_job()
        resume = _make_resume()
        gj = GeneratedJob(
            trace_id="j1",
            job=job,
            is_niche_role=False,
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1",
            model_used="gpt-4o-mini",
        )
        gr = GeneratedResume(
            trace_id="r1",
            resume=resume,
            fit_level=FitLevel.GOOD,
            writing_style=WritingStyle.FORMAL,
            template_version="v1",
            generated_at="2026-02-25T14:00:00Z",
            prompt_template="v1-formal",
            model_used="gpt-4o-mini",
        )

        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")

        pairs = [self._make_pair("p-err", "r1", "j1")]

        with (
            patch.object(jmod, "_CACHE_DIR", tmp_path),
            patch.object(jmod, "_create_judge_client", return_value=client),
        ):
            results = judge_batch(pairs, {"j1": gj}, {"r1": gr}, use_cache=False)

        assert results == []


# ---------------------------------------------------------------------------
# TestSaveJudgeResults
# ---------------------------------------------------------------------------


class TestSaveJudgeResults:
    def test_saves_all_results_to_jsonl(self, tmp_path: Path) -> None:
        import src.judge as jmod

        results = [
            JudgeResult(
                pair_id=f"p{i}",
                has_hallucinations=False,
                hallucination_details="",
                has_awkward_language=False,
                awkward_language_details="",
                overall_quality_score=0.8,
                fit_assessment="Good fit",
                recommendations=[],
                red_flags=[],
            )
            for i in range(3)
        ]

        with patch.object(jmod, "_LABELED_DIR", tmp_path):
            output_path = save_judge_results(results)

        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_returns_path_to_output_file(self, tmp_path: Path) -> None:
        import src.judge as jmod

        with patch.object(jmod, "_LABELED_DIR", tmp_path):
            path = save_judge_results([])

        assert path.name == "judge_results.jsonl"


# ---------------------------------------------------------------------------
# TestJudgeLoaders
# ---------------------------------------------------------------------------


class TestJudgeLoaders:
    def test_load_pairs(self, tmp_path: Path) -> None:
        from src.judge import _load_pairs

        pair = ResumeJobPair(
            pair_id="p1",
            resume_trace_id="r1",
            job_trace_id="j1",
            fit_level=FitLevel.PARTIAL,
            created_at="2026-02-25T00:00:00Z",
        )
        f = tmp_path / "pairs.jsonl"
        f.write_text(pair.model_dump_json() + "\n")
        result = _load_pairs(f)
        assert len(result) == 1
        assert result[0].pair_id == "p1"

    def test_load_jobs(self, tmp_path: Path) -> None:
        from src.judge import _load_jobs

        gj = GeneratedJob(
            trace_id="j1",
            job=_make_job(),
            is_niche_role=True,
            generated_at="2026-02-25T00:00:00Z",
            prompt_template="v1-technical",
            model_used="gpt-4o-mini",
        )
        f = tmp_path / "jobs.jsonl"
        f.write_text(gj.model_dump_json() + "\n")
        result = _load_jobs(f)
        assert "j1" in result
        assert result["j1"].is_niche_role is True

    def test_load_resumes(self, tmp_path: Path) -> None:
        from src.judge import _load_resumes

        gr = GeneratedResume(
            trace_id="r1",
            resume=_make_resume(),
            fit_level=FitLevel.EXCELLENT,
            writing_style=WritingStyle.CASUAL,
            template_version="v1",
            generated_at="2026-02-25T00:00:00Z",
            prompt_template="v1-casual",
            model_used="gpt-4o-mini",
        )
        f = tmp_path / "resumes.jsonl"
        f.write_text(gr.model_dump_json() + "\n")
        result = _load_resumes(f)
        assert "r1" in result
        assert result["r1"].writing_style == WritingStyle.CASUAL


# ---------------------------------------------------------------------------
# TestCreateJudgeClient — covers _create_judge_client (lines 97-104)
# ---------------------------------------------------------------------------


class TestCreateJudgeClient:
    def test_raises_without_api_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_create_judge_client raises RuntimeError when OPENAI_API_KEY is not set."""
        import src.judge as jmod

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Point _PROJECT_ROOT to tmp_path — no .env file there → load_dotenv is no-op
        monkeypatch.setattr(jmod, "_PROJECT_ROOT", tmp_path)

        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            jmod._create_judge_client()

    def test_returns_instructor_client_with_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_create_judge_client returns an Instructor client when OPENAI_API_KEY is set."""
        import src.judge as jmod

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")
        monkeypatch.setattr(jmod, "_PROJECT_ROOT", tmp_path)

        mock_instructor = MagicMock()
        with (
            patch("instructor.from_openai", return_value=mock_instructor),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            client = jmod._create_judge_client()

        assert client is mock_instructor
