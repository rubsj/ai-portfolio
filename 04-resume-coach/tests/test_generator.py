from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.generator import (
    _append_jsonl,
    _load_cache,
    _prompt_hash,
    _save_cache,
    generate_all_jobs,
    generate_all_resumes,
    generate_job,
    generate_resume,
    generate_with_cache,
)
from src.schemas import (
    CompanyInfo,
    ContactInfo,
    Education,
    ExperienceLevel,
    Experience,
    FitLevel,
    GeneratedJob,
    GeneratedResume,
    JobDescription,
    JobRequirements,
    ProficiencyLevel,
    Resume,
    Skill,
    WritingStyle,
)
from src.templates import PromptTemplateLibrary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contact(idx: int = 0) -> ContactInfo:
    return ContactInfo(
        name=f"Person{idx}",
        email=f"person{idx}@example.com",
        phone="+15551234567",
        location="Austin, TX",
    )


def _make_job_description() -> JobDescription:
    return JobDescription(
        title="Software Engineer",
        company=CompanyInfo(
            name="Acme", industry="Technology", size="Mid-size", location="Austin, TX"
        ),
        description="Build software",
        requirements=JobRequirements(
            required_skills=["Python", "AWS"],
            education="BS CS",
            experience_years=5,
            experience_level=ExperienceLevel.SENIOR,
        ),
    )


def _make_generated_job(trace_id: str | None = None) -> GeneratedJob:
    return GeneratedJob(
        trace_id=trace_id or str(uuid.uuid4()),
        job=_make_job_description(),
        is_niche_role=False,
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="job-standard",
        model_used="gpt-4o-mini",
    )


def _make_resume() -> Resume:
    return Resume(
        contact_info=_make_contact(),
        education=[
            Education(degree="BS CS", institution="MIT", graduation_date="2020-05")
        ],
        experience=[
            Experience(
                company="Acme",
                title="Dev",
                start_date="2020-06",
                responsibilities=["Built things"],
            )
        ],
        skills=[Skill(name="Python", proficiency_level=ProficiencyLevel.ADVANCED)],
    )


def _make_progress_mock() -> MagicMock:
    """Returns a MagicMock that acts as a context manager for rich.progress.Progress."""
    mock = MagicMock()
    mock.__enter__ = MagicMock(return_value=mock)
    mock.__exit__ = MagicMock(return_value=False)
    return mock


# ---------------------------------------------------------------------------
# _prompt_hash
# ---------------------------------------------------------------------------


class TestPromptHash:
    def test_same_inputs_produce_same_hash(self):
        assert _prompt_hash("sys", "usr", "model") == _prompt_hash("sys", "usr", "model")

    def test_different_system_prompt_different_hash(self):
        assert _prompt_hash("sys1", "usr", "model") != _prompt_hash("sys2", "usr", "model")

    def test_different_user_prompt_different_hash(self):
        assert _prompt_hash("sys", "usr1", "model") != _prompt_hash("sys", "usr2", "model")

    def test_different_model_different_hash(self):
        # WHY: Including model prevents stale mini-responses being served for 4o requests
        assert _prompt_hash("sys", "usr", "gpt-4o-mini") != _prompt_hash("sys", "usr", "gpt-4o")

    def test_returns_32_char_hex_string(self):
        h = _prompt_hash("a", "b", "c")
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# _load_cache
# ---------------------------------------------------------------------------


class TestLoadCache:
    def test_miss_returns_none_when_file_absent(self, tmp_path):
        with patch("src.generator._CACHE_DIR", tmp_path):
            assert _load_cache("nonexistent", ContactInfo) is None

    def test_hit_returns_validated_model(self, tmp_path):
        model = _make_contact()
        cache_file = tmp_path / "mykey.json"
        cache_file.write_text(json.dumps({"response": model.model_dump()}))
        with patch("src.generator._CACHE_DIR", tmp_path):
            result = _load_cache("mykey", ContactInfo)
        assert result is not None
        assert result.name == "Person0"

    def test_bad_json_returns_none(self, tmp_path):
        (tmp_path / "badkey.json").write_text("{ not valid json }")
        with patch("src.generator._CACHE_DIR", tmp_path):
            assert _load_cache("badkey", ContactInfo) is None

    def test_missing_response_key_returns_none(self, tmp_path):
        (tmp_path / "noresponse.json").write_text(json.dumps({"wrong": {}}))
        with patch("src.generator._CACHE_DIR", tmp_path):
            assert _load_cache("noresponse", ContactInfo) is None

    def test_invalid_model_data_returns_none(self, tmp_path):
        # Valid JSON + valid key but data fails ContactInfo validation
        (tmp_path / "badmodel.json").write_text(
            json.dumps({"response": {"name": "X", "email": "not-an-email", "phone": "123"}})
        )
        with patch("src.generator._CACHE_DIR", tmp_path):
            assert _load_cache("badmodel", ContactInfo) is None


# ---------------------------------------------------------------------------
# _save_cache
# ---------------------------------------------------------------------------


class TestSaveCache:
    def test_creates_json_file(self, tmp_path):
        with patch("src.generator._CACHE_DIR", tmp_path):
            _save_cache("key001", "prompt_key", _make_contact())
        assert (tmp_path / "key001.json").exists()

    def test_saved_file_structure(self, tmp_path):
        with patch("src.generator._CACHE_DIR", tmp_path):
            _save_cache("key002", "test_prompt", _make_contact())
        data = json.loads((tmp_path / "key002.json").read_text())
        assert data["cache_key"] == "key002"
        assert data["prompt_key"] == "test_prompt"
        assert "response" in data
        assert "timestamp" in data
        assert "model" in data

    def test_creates_cache_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "new_cache"
        with patch("src.generator._CACHE_DIR", new_dir):
            _save_cache("key003", "pk", _make_contact())
        assert (new_dir / "key003.json").exists()


# ---------------------------------------------------------------------------
# _append_jsonl
# ---------------------------------------------------------------------------


class TestAppendJsonl:
    def test_creates_file_with_one_record(self, tmp_path):
        filepath = tmp_path / "out.jsonl"
        _append_jsonl(_make_contact(), filepath)
        lines = filepath.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["name"] == "Person0"

    def test_appends_multiple_records(self, tmp_path):
        filepath = tmp_path / "out.jsonl"
        for i in range(3):
            _append_jsonl(_make_contact(i), filepath)
        assert len(filepath.read_text().splitlines()) == 3

    def test_creates_parent_dirs_if_missing(self, tmp_path):
        filepath = tmp_path / "nested" / "dir" / "out.jsonl"
        _append_jsonl(_make_contact(), filepath)
        assert filepath.exists()


# ---------------------------------------------------------------------------
# generate_with_cache
# ---------------------------------------------------------------------------


class TestGenerateWithCache:
    def test_cache_hit_returns_cached_result_and_true(self, tmp_path):
        cached = _make_contact()
        with patch("src.generator._load_cache", return_value=cached):
            client = MagicMock()
            result, from_cache = generate_with_cache(client, "key", "sys", "usr", ContactInfo)
        assert from_cache is True
        assert result.name == "Person0"
        client.chat.completions.create.assert_not_called()

    def test_cache_miss_calls_api_returns_false(self, tmp_path):
        api_result = _make_contact()
        client = MagicMock()
        client.chat.completions.create.return_value = api_result

        with patch("src.generator._load_cache", return_value=None):
            with patch("src.generator._save_cache") as mock_save:
                result, from_cache = generate_with_cache(
                    client, "key", "sys", "usr", ContactInfo
                )
        assert from_cache is False
        assert result.name == "Person0"
        mock_save.assert_called_once()

    def test_cache_miss_saves_result(self, tmp_path):
        api_result = _make_contact()
        client = MagicMock()
        client.chat.completions.create.return_value = api_result

        with patch("src.generator._load_cache", return_value=None):
            with patch("src.generator._save_cache") as mock_save:
                generate_with_cache(client, "my_key", "sys", "usr", ContactInfo)
        # Verify save was called with the prompt_key
        args = mock_save.call_args[0]
        assert args[1] == "my_key"


# ---------------------------------------------------------------------------
# _create_client
# ---------------------------------------------------------------------------


class TestCreateClient:
    def test_missing_api_key_raises_runtime_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("dotenv.load_dotenv"):
                from src.generator import _create_client

                with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
                    _create_client()

    def test_returns_instructor_client_with_api_key(self, tmp_path, monkeypatch):
        """_create_client returns an Instructor client when OPENAI_API_KEY is set.

        Covers generator.py line 49: return instructor.from_openai(OpenAI(), ...).
        """
        import src.generator as gmod

        monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-unit-tests")
        monkeypatch.setattr(gmod, "_PROJECT_ROOT", tmp_path)

        mock_instructor = MagicMock()
        with (
            patch("instructor.from_openai", return_value=mock_instructor),
            patch("openai.OpenAI", return_value=MagicMock()),
        ):
            client = gmod._create_client()

        assert client is mock_instructor


# ---------------------------------------------------------------------------
# generate_job
# ---------------------------------------------------------------------------


class TestGenerateJob:
    def test_returns_generated_job_with_correct_metadata(self):
        job_desc = _make_job_description()
        client = MagicMock()
        templates = PromptTemplateLibrary()

        with patch("src.generator.generate_with_cache", return_value=(job_desc, False)):
            result, from_cache = generate_job(
                client, "Technology", False, ExperienceLevel.MID, templates
            )

        assert from_cache is False
        assert result.job.title == "Software Engineer"
        assert result.model_used == "gpt-4o-mini"
        assert result.is_niche_role is False
        assert result.prompt_template == "job-standard"

    def test_niche_role_sets_flag_true(self):
        job_desc = _make_job_description()
        client = MagicMock()
        templates = PromptTemplateLibrary()

        with patch("src.generator.generate_with_cache", return_value=(job_desc, True)):
            result, from_cache = generate_job(
                client, "Healthcare", True, ExperienceLevel.SENIOR, templates
            )

        assert result.is_niche_role is True
        assert from_cache is True

    def test_trace_id_is_valid_uuid(self):
        job_desc = _make_job_description()
        templates = PromptTemplateLibrary()

        with patch("src.generator.generate_with_cache", return_value=(job_desc, False)):
            result, _ = generate_job(MagicMock(), "Finance", False, ExperienceLevel.ENTRY, templates)

        uuid.UUID(result.trace_id)  # Raises ValueError if not a valid UUID


# ---------------------------------------------------------------------------
# generate_resume
# ---------------------------------------------------------------------------


class TestGenerateResume:
    def test_returns_generated_resume_with_correct_metadata(self):
        resume = _make_resume()
        gen_job = _make_generated_job()
        templates = PromptTemplateLibrary()

        with patch("src.generator.generate_with_cache", return_value=(resume, False)):
            result, from_cache = generate_resume(
                MagicMock(), gen_job, FitLevel.EXCELLENT, WritingStyle.FORMAL, templates
            )

        assert from_cache is False
        assert result.fit_level == FitLevel.EXCELLENT
        assert result.writing_style == WritingStyle.FORMAL
        assert result.model_used == "gpt-4o-mini"
        assert result.prompt_template == "v1-formal"
        assert result.template_version == "v1"

    def test_from_cache_true_propagated(self):
        resume = _make_resume()
        gen_job = _make_generated_job()
        templates = PromptTemplateLibrary()

        with patch("src.generator.generate_with_cache", return_value=(resume, True)):
            _, from_cache = generate_resume(
                MagicMock(), gen_job, FitLevel.GOOD, WritingStyle.CASUAL, templates
            )

        assert from_cache is True


# ---------------------------------------------------------------------------
# generate_all_jobs
# ---------------------------------------------------------------------------


class TestGenerateAllJobs:
    def test_generates_correct_job_count(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_job", return_value=(gen_job, False)):
                    jobs, api_calls = generate_all_jobs(
                        MagicMock(), count_per_industry=2, max_industries=3
                    )

        assert len(jobs) == 6  # 3 industries × 2 jobs
        assert api_calls == 6

    def test_cache_hits_not_counted_as_api_calls(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                # from_cache=True → no API call counted
                with patch("src.generator.generate_job", return_value=(gen_job, True)):
                    jobs, api_calls = generate_all_jobs(
                        MagicMock(), count_per_industry=1, max_industries=2
                    )

        assert len(jobs) == 2
        assert api_calls == 0

    def test_failed_jobs_excluded_from_results(self, tmp_path):
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_job", side_effect=Exception("API error")):
                    jobs, api_calls = generate_all_jobs(
                        MagicMock(), count_per_industry=1, max_industries=2
                    )

        assert len(jobs) == 0
        assert api_calls == 2  # Counted even on failure

    def test_rate_limiting_triggered_at_10_api_calls(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_job", return_value=(gen_job, False)):
                    with patch("src.generator.time.sleep") as mock_sleep:
                        # 10 industries × 1 job = 10 API calls → sleep triggered once
                        generate_all_jobs(MagicMock(), count_per_industry=1, max_industries=10)

        mock_sleep.assert_called_once_with(2)

    def test_writes_jsonl_file(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_job", return_value=(gen_job, False)):
                    generate_all_jobs(MagicMock(), count_per_industry=1, max_industries=1)

        jsonl_files = list(tmp_path.glob("jobs_*.jsonl"))
        assert len(jsonl_files) == 1
        assert len(jsonl_files[0].read_text().splitlines()) == 1


# ---------------------------------------------------------------------------
# generate_all_resumes
# ---------------------------------------------------------------------------


class TestGenerateAllResumes:
    def _make_gen_resume(
        self, client, job, fit_level, writing_style, templates
    ) -> tuple[GeneratedResume, bool]:
        return (
            GeneratedResume(
                trace_id=str(uuid.uuid4()),
                resume=_make_resume(),
                fit_level=fit_level,
                writing_style=writing_style,
                template_version="v1",
                generated_at=datetime.now(timezone.utc).isoformat(),
                prompt_template=f"v1-{writing_style.value}",
                model_used="gpt-4o-mini",
            ),
            False,
        )

    def test_generates_5_resumes_and_pairs_per_job(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_resume", side_effect=self._make_gen_resume):
                    resumes, pairs, api_calls = generate_all_resumes(MagicMock(), [gen_job])

        assert len(resumes) == 5  # One per FitLevel
        assert len(pairs) == 5
        assert api_calls == 5

    def test_multiple_jobs_multiply_resume_count(self, tmp_path):
        jobs = [_make_generated_job() for _ in range(3)]
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_resume", side_effect=self._make_gen_resume):
                    resumes, pairs, _ = generate_all_resumes(MagicMock(), jobs)

        assert len(resumes) == 15  # 3 jobs × 5 fit levels
        assert len(pairs) == 15

    def test_cache_hits_not_counted_as_api_calls(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        def cached_resume(client, job, fit_level, writing_style, templates):
            gen = GeneratedResume(
                trace_id=str(uuid.uuid4()),
                resume=_make_resume(),
                fit_level=fit_level,
                writing_style=writing_style,
                template_version="v1",
                generated_at=datetime.now(timezone.utc).isoformat(),
                prompt_template="v1-formal",
                model_used="gpt-4o-mini",
            )
            return gen, True  # from_cache=True

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_resume", side_effect=cached_resume):
                    resumes, pairs, api_calls = generate_all_resumes(MagicMock(), [gen_job])

        assert api_calls == 0

    def test_failed_resumes_excluded_from_results(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_resume", side_effect=Exception("fail")):
                    resumes, pairs, api_calls = generate_all_resumes(MagicMock(), [gen_job])

        assert len(resumes) == 0
        assert len(pairs) == 0
        assert api_calls == 5  # 5 FitLevels tried, all counted

    def test_rate_limiting_triggered_at_10_api_calls(self, tmp_path):
        # 2 jobs × 5 FitLevels = 10 API calls → sleep triggered once
        jobs = [_make_generated_job() for _ in range(2)]
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_resume", side_effect=self._make_gen_resume):
                    with patch("src.generator.time.sleep") as mock_sleep:
                        generate_all_resumes(MagicMock(), jobs)

        mock_sleep.assert_called_once_with(2)

    def test_writes_resume_and_pair_jsonl_files(self, tmp_path):
        gen_job = _make_generated_job()
        mock_progress = _make_progress_mock()

        with patch("src.generator._GENERATED_DIR", tmp_path):
            with patch("src.generator.Progress", return_value=mock_progress):
                with patch("src.generator.generate_resume", side_effect=self._make_gen_resume):
                    generate_all_resumes(MagicMock(), [gen_job])

        assert len(list(tmp_path.glob("resumes_*.jsonl"))) == 1
        assert len(list(tmp_path.glob("pairs_*.jsonl"))) == 1
