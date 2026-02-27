"""Tests for src/vector_store.py — embedding, indexing, and search coverage.

Strategy:
- SentenceTransformer is always patched — no real model loading.
- chromadb.PersistentClient is patched — no real disk I/O.
- The module-level _ef singleton is manually reset before each test that needs
  to control its state to avoid cross-test contamination.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from src.schemas import (
    CompanyInfo,
    ContactInfo,
    Education,
    Experience,
    ExperienceLevel,
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


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _make_resume(
    *,
    summary: str | None = "Experienced Python developer",
    include_skills: bool = True,
    include_experience: bool = True,
    include_education: bool = True,
) -> Resume:
    skills = [Skill(name="Python", proficiency_level=ProficiencyLevel.ADVANCED, years=3)] if include_skills else []
    experience = (
        [
            Experience(
                company="Acme",
                title="Engineer",
                start_date="2020-01",
                end_date="2023-01",
                responsibilities=["Built APIs", "Led code reviews"],
            )
        ]
        if include_experience
        else []
    )
    education = (
        [Education(degree="BS CS", institution="MIT", graduation_date="2020-05")]
        if include_education
        else []
    )
    return Resume(
        contact_info=ContactInfo(
            name="Jane",
            email="jane@example.com",
            phone="+15551234567",
            location="Austin, TX",
        ),
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
    )


def _make_generated_resume(trace_id: str | None = None) -> GeneratedResume:
    return GeneratedResume(
        trace_id=trace_id or str(uuid.uuid4()),
        resume=_make_resume(),
        fit_level=FitLevel.EXCELLENT,
        writing_style=WritingStyle.FORMAL,
        template_version="v1",
        generated_at=datetime.now(timezone.utc).isoformat(),
        prompt_template="v1-formal",
        model_used="gpt-4o-mini",
    )


# ---------------------------------------------------------------------------
# resume_to_text
# ---------------------------------------------------------------------------


class TestResumeToText:
    def test_includes_summary(self) -> None:
        from src.vector_store import resume_to_text

        resume = _make_resume(summary="Python expert with 5 years")
        text = resume_to_text(resume)
        assert "Python expert with 5 years" in text

    def test_includes_skills(self) -> None:
        from src.vector_store import resume_to_text

        text = resume_to_text(_make_resume())
        assert "Python" in text

    def test_skills_prefixed_with_skills_label(self) -> None:
        from src.vector_store import resume_to_text

        text = resume_to_text(_make_resume())
        assert "Skills:" in text

    def test_includes_experience_title_and_company(self) -> None:
        from src.vector_store import resume_to_text

        text = resume_to_text(_make_resume())
        assert "Engineer at Acme" in text

    def test_includes_responsibilities(self) -> None:
        from src.vector_store import resume_to_text

        text = resume_to_text(_make_resume())
        assert "Built APIs" in text

    def test_includes_education_degree_and_institution(self) -> None:
        from src.vector_store import resume_to_text

        text = resume_to_text(_make_resume())
        assert "BS CS from MIT" in text

    def test_no_summary_skips_it(self) -> None:
        from src.vector_store import resume_to_text

        resume = _make_resume(summary=None)
        text = resume_to_text(resume)
        # Summary is None — should not crash and should not add an empty string
        assert text  # still has skills, experience, education

    def test_all_parts_joined_by_space(self) -> None:
        from src.vector_store import resume_to_text

        text = resume_to_text(_make_resume())
        # Space-joined means no newlines or double-spaces at boundaries
        assert "\n" not in text


# ---------------------------------------------------------------------------
# _SentenceTransformerEF.__call__
# ---------------------------------------------------------------------------


class TestSentenceTransformerEF:
    def _make_ef_with_mock_model(self, encode_return: np.ndarray) -> tuple:
        """Create _SentenceTransformerEF bypassing __init__ to avoid real model loading.

        WHY object.__new__: The __init__ does a lazy SentenceTransformer load —
        patching sentence_transformers.SentenceTransformer is unreliable when the
        package is already cached. Setting _model directly tests __call__ cleanly.
        """
        from src.vector_store import _SentenceTransformerEF

        ef = object.__new__(_SentenceTransformerEF)
        mock_model = MagicMock()
        mock_model.encode.return_value = encode_return
        ef._model = mock_model
        return ef, mock_model

    def test_call_encodes_input_and_returns_list_with_one_embedding(self) -> None:
        ef, _ = self._make_ef_with_mock_model(np.array([[0.1, 0.2, 0.3]]))
        result = ef(["test text"])
        # ChromaDB's EmbeddingFunction.__init_subclass__ wraps __call__ with
        # normalize_embeddings() which returns list[np.ndarray].
        # We verify the outer container is a list and has one embedding per input.
        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]) == 3  # three-dimensional embedding

    def test_call_passes_show_progress_bar_false(self) -> None:
        ef, mock_model = self._make_ef_with_mock_model(np.array([[0.1]]))
        ef(["hello"])
        mock_model.encode.assert_called_once_with(["hello"], show_progress_bar=False)

    def test_call_handles_batch_input(self) -> None:
        ef, _ = self._make_ef_with_mock_model(np.array([[0.1, 0.2], [0.3, 0.4]]))
        result = ef(["text1", "text2"])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# build_resume_index
# ---------------------------------------------------------------------------


class TestBuildResumeIndex:
    def test_returns_collection(self, tmp_path: Path) -> None:
        resumes = {"r1": _make_generated_resume("r1")}

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])

            with patch("chromadb.PersistentClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_collection = MagicMock()
                mock_client.create_collection.return_value = mock_collection

                from src.vector_store import build_resume_index

                result = build_resume_index(resumes=resumes, persist_dir=tmp_path / "chromadb")

        assert result is mock_collection

    def test_calls_collection_add_with_ids(self, tmp_path: Path) -> None:
        resumes = {"r1": _make_generated_resume("r1"), "r2": _make_generated_resume("r2")}

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.encode.return_value = np.array([0.1, 0.2])

            with patch("chromadb.PersistentClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_collection = MagicMock()
                mock_client.create_collection.return_value = mock_collection

                from src.vector_store import build_resume_index

                build_resume_index(resumes=resumes, persist_dir=tmp_path / "chromadb")

        add_call = mock_collection.add.call_args
        assert set(add_call[1]["ids"]) == {"r1", "r2"}

    def test_deletes_existing_collection_before_creating(self, tmp_path: Path) -> None:
        resumes = {"r1": _make_generated_resume("r1")}

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.encode.return_value = np.array([0.1, 0.2])

            with patch("chromadb.PersistentClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.create_collection.return_value = MagicMock()

                from src.vector_store import build_resume_index

                build_resume_index(resumes=resumes, persist_dir=tmp_path / "chromadb")

        mock_client.delete_collection.assert_called_once_with("resumes")

    def test_delete_collection_exception_is_swallowed(self, tmp_path: Path) -> None:
        resumes = {"r1": _make_generated_resume("r1")}

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.encode.return_value = np.array([0.1, 0.2])

            with patch("chromadb.PersistentClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.delete_collection.side_effect = Exception("collection not found")
                mock_client.create_collection.return_value = MagicMock()

                from src.vector_store import build_resume_index

                # Must not raise even though delete_collection throws
                result = build_resume_index(resumes=resumes, persist_dir=tmp_path / "chromadb")

        assert result is not None

    def test_fetches_resumes_from_file_when_none_provided(self, tmp_path: Path) -> None:
        mock_resumes = {"r1": _make_generated_resume("r1")}

        with patch("src.vector_store.find_latest", return_value=tmp_path / "resumes.jsonl") as mock_find:
            with patch("src.vector_store.load_resumes", return_value=mock_resumes):
                with patch("sentence_transformers.SentenceTransformer") as mock_st:
                    mock_model = MagicMock()
                    mock_st.return_value = mock_model
                    mock_model.encode.return_value = np.array([0.1, 0.2])

                    with patch("chromadb.PersistentClient") as mock_client_cls:
                        mock_client = MagicMock()
                        mock_client_cls.return_value = mock_client
                        mock_client.create_collection.return_value = MagicMock()

                        from src.vector_store import build_resume_index

                        build_resume_index(persist_dir=tmp_path / "chromadb")

        mock_find.assert_called_once()

    def test_raises_file_not_found_when_no_resumes_file(self, tmp_path: Path) -> None:
        with patch("src.vector_store.find_latest", return_value=None):
            from src.vector_store import build_resume_index

            with pytest.raises(FileNotFoundError, match="No resumes JSONL"):
                build_resume_index(persist_dir=tmp_path / "chromadb")

    def test_metadata_includes_fit_level_and_skills(self, tmp_path: Path) -> None:
        resumes = {"r1": _make_generated_resume("r1")}

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.encode.return_value = np.array([0.1, 0.2])

            with patch("chromadb.PersistentClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.create_collection.return_value = MagicMock()

                from src.vector_store import build_resume_index

                build_resume_index(resumes=resumes, persist_dir=tmp_path / "chromadb")

        add_kwargs = mock_client.create_collection.return_value.add.call_args[1]
        meta = add_kwargs["metadatas"][0]
        assert "fit_level" in meta
        assert "skills" in meta
        assert "name" in meta

    def test_model_unloaded_after_encoding(self, tmp_path: Path) -> None:
        """del model + gc.collect() keeps RAM low — verify model is loaded then deleted."""
        resumes = {"r1": _make_generated_resume("r1")}

        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model
            mock_model.encode.return_value = np.array([0.1, 0.2])

            with patch("chromadb.PersistentClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.create_collection.return_value = MagicMock()

                with patch("gc.collect") as mock_gc:
                    from src.vector_store import build_resume_index

                    build_resume_index(resumes=resumes, persist_dir=tmp_path / "chromadb")

        # gc.collect() is called to free the model RAM
        mock_gc.assert_called_once()


# ---------------------------------------------------------------------------
# get_collection
# ---------------------------------------------------------------------------


class TestGetCollection:
    def test_initializes_ef_when_none(self, tmp_path: Path) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            vs._ef = None  # Force fresh initialization

            with patch.object(vs, "_SentenceTransformerEF") as mock_ef_cls:
                mock_ef = MagicMock()
                mock_ef_cls.return_value = mock_ef

                with patch("chromadb.PersistentClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client_cls.return_value = mock_client
                    mock_collection = MagicMock()
                    mock_client.get_collection.return_value = mock_collection

                    result = vs.get_collection(persist_dir=tmp_path)

            mock_ef_cls.assert_called_once()
            assert vs._ef is mock_ef
            assert result is mock_collection
        finally:
            vs._ef = original_ef

    def test_reuses_existing_ef_singleton(self, tmp_path: Path) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            existing_ef = MagicMock()
            vs._ef = existing_ef  # Already initialized

            with patch.object(vs, "_SentenceTransformerEF") as mock_ef_cls:
                with patch("chromadb.PersistentClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client_cls.return_value = mock_client
                    mock_client.get_collection.return_value = MagicMock()

                    vs.get_collection(persist_dir=tmp_path)

            # Should NOT create a new EF since one already exists
            mock_ef_cls.assert_not_called()
        finally:
            vs._ef = original_ef

    def test_opens_resumes_collection(self, tmp_path: Path) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            vs._ef = MagicMock()  # Pre-populate to skip init

            with patch("chromadb.PersistentClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client_cls.return_value = mock_client
                mock_client.get_collection.return_value = MagicMock()

                vs.get_collection(persist_dir=tmp_path)

            mock_client.get_collection.assert_called_once_with(name="resumes")
        finally:
            vs._ef = original_ef


# ---------------------------------------------------------------------------
# search_similar
# ---------------------------------------------------------------------------


class TestSearchSimilar:
    def _mock_query_result(self, trace_ids: list[str], distances: list[float]) -> dict:
        return {
            "ids": [trace_ids],
            "distances": [distances],
            "metadatas": [[{"fit_level": "excellent"} for _ in trace_ids]],
        }

    def test_returns_hits_with_trace_id_and_score(self, tmp_path: Path) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            mock_ef = MagicMock()
            mock_ef.return_value = [[0.1, 0.2, 0.3]]
            vs._ef = mock_ef

            mock_collection = MagicMock()
            mock_collection.query.return_value = self._mock_query_result(
                ["r1", "r2"], [0.1, 0.3]
            )

            from src.vector_store import search_similar

            results = search_similar(mock_collection, "python developer", top_k=2)

            assert len(results) == 2
            assert results[0]["trace_id"] == "r1"
            assert results[0]["score"] == round(1.0 - 0.1, 4)
            assert results[1]["trace_id"] == "r2"
            assert results[1]["score"] == round(1.0 - 0.3, 4)
        finally:
            vs._ef = original_ef

    def test_distance_converted_to_similarity(self, tmp_path: Path) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            mock_ef = MagicMock()
            mock_ef.return_value = [[0.0]]
            vs._ef = mock_ef

            mock_collection = MagicMock()
            mock_collection.query.return_value = self._mock_query_result(["r1"], [0.2])

            from src.vector_store import search_similar

            results = search_similar(mock_collection, "query", top_k=1)

            assert results[0]["score"] == round(1.0 - 0.2, 4)
        finally:
            vs._ef = original_ef

    def test_fit_level_filter_passed_as_where_clause(self) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            mock_ef = MagicMock()
            mock_ef.return_value = [[0.0]]
            vs._ef = mock_ef

            mock_collection = MagicMock()
            mock_collection.query.return_value = self._mock_query_result(["r1"], [0.1])

            from src.vector_store import search_similar

            search_similar(mock_collection, "python dev", fit_level="excellent")

            _, kwargs = mock_collection.query.call_args
            assert kwargs["where"] == {"fit_level": "excellent"}
        finally:
            vs._ef = original_ef

    def test_no_fit_level_passes_none_where(self) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            mock_ef = MagicMock()
            mock_ef.return_value = [[0.0]]
            vs._ef = mock_ef

            mock_collection = MagicMock()
            mock_collection.query.return_value = self._mock_query_result(["r1"], [0.1])

            from src.vector_store import search_similar

            search_similar(mock_collection, "python dev", fit_level=None)

            _, kwargs = mock_collection.query.call_args
            assert kwargs["where"] is None
        finally:
            vs._ef = original_ef

    def test_initializes_ef_when_none_in_search(self) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            vs._ef = None  # Force initialization path in search_similar

            with patch.object(vs, "_SentenceTransformerEF") as mock_ef_cls:
                mock_ef = MagicMock()
                mock_ef.return_value = [[0.1, 0.2]]
                mock_ef_cls.return_value = mock_ef

                mock_collection = MagicMock()
                mock_collection.query.return_value = self._mock_query_result(["r1"], [0.1])

                from src.vector_store import search_similar

                search_similar(mock_collection, "query")

            mock_ef_cls.assert_called_once()
        finally:
            vs._ef = original_ef

    def test_result_includes_metadata(self) -> None:
        import src.vector_store as vs

        original_ef = vs._ef
        try:
            mock_ef = MagicMock()
            mock_ef.return_value = [[0.1]]
            vs._ef = mock_ef

            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["r1"]],
                "distances": [[0.1]],
                "metadatas": [[{"fit_level": "good", "name": "Jane"}]],
            }

            from src.vector_store import search_similar

            results = search_similar(mock_collection, "query")

            assert results[0]["metadata"]["fit_level"] == "good"
            assert results[0]["metadata"]["name"] == "Jane"
        finally:
            vs._ef = original_ef
