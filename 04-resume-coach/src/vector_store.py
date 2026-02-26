from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import chromadb
from chromadb import EmbeddingFunction, Embeddings

from src.data_paths import CHROMADB_DIR, GENERATED_DIR, find_latest, load_resumes
from src.schemas import GeneratedResume, Resume

_COLLECTION_NAME = "resumes"
_MODEL_NAME = "all-MiniLM-L6-v2"

# WHY module-level singleton: The EF is loaded once when get_collection() is first
# called and stays alive for the process lifetime. search_similar() uses it to encode
# query strings without reloading the model — ~50ms per query vs ~2s if we loaded
# per call. ChromaDB 1.5 rejects passing a custom EF to get_collection() when the
# persisted collection was built without one, so we manage the model ourselves.
_ef: _SentenceTransformerEF | None = None


class _SentenceTransformerEF(EmbeddingFunction):
    """Wraps SentenceTransformer as a callable that returns list[list[float]].

    WHY: Implements ChromaDB's EmbeddingFunction interface as a typed wrapper,
    giving us a reusable, cacheable encoder. Loaded once; encodes any batch of
    strings on demand without reloading weights.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        # WHY: Import inside __init__ so the module can be imported without
        # triggering a heavy model load at import time (matches P2/P3 pattern).
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def __call__(self, input: list[str]) -> Embeddings:  # noqa: A002
        return self._model.encode(input, show_progress_bar=False).tolist()


# ---------------------------------------------------------------------------
# Text conversion
# ---------------------------------------------------------------------------


def resume_to_text(resume: Resume) -> str:
    """Flatten a Resume into a single searchable text string.

    WHY: ChromaDB stores and retrieves by vector similarity, so we need all
    semantically meaningful fields concatenated. Summary, skills, experience
    titles/responsibilities, and education degree cover the key match signals.
    """
    parts: list[str] = []

    if resume.summary:
        parts.append(resume.summary)

    skill_names = [s.name for s in resume.skills]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    for exp in resume.experience:
        parts.append(f"{exp.title} at {exp.company}")
        parts.extend(exp.responsibilities)

    for edu in resume.education:
        parts.append(f"{edu.degree} from {edu.institution}")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Index builder — manual load → encode → del → gc (batch path)
# ---------------------------------------------------------------------------


def build_resume_index(
    resumes: dict[str, GeneratedResume] | None = None,
    persist_dir: Path = CHROMADB_DIR,
) -> chromadb.Collection:
    """Build (or rebuild) the ChromaDB resume index.

    WHY manual encode + del: Batch indexing is a one-shot operation. We load
    the model, encode all 250 resumes, then immediately delete the model and
    gc.collect() to free ~90MB of RAM — critical on the 8GB M2 Air. This is
    the same pattern used in P2/P3 for local embeddings.

    Returns the populated collection.
    """
    if resumes is None:
        resumes_path = find_latest(GENERATED_DIR, "resumes")
        if resumes_path is None:
            raise FileNotFoundError("No resumes JSONL found in data/generated/")
        resumes = load_resumes(resumes_path)

    # --- Encode ---
    from sentence_transformers import SentenceTransformer

    print(f"Loading {_MODEL_NAME} for batch encoding...")
    model = SentenceTransformer(_MODEL_NAME)

    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for trace_id, gr in resumes.items():
        text = resume_to_text(gr.resume)
        ids.append(trace_id)
        embeddings.append(model.encode(text, show_progress_bar=False).tolist())
        documents.append(text)
        metadatas.append(
            {
                "fit_level": gr.fit_level.value,
                "writing_style": gr.writing_style.value,
                "name": gr.resume.contact_info.name,
                # WHY comma-joined string: ChromaDB metadata values must be str/int/float/bool
                "skills": ", ".join(s.name for s in gr.resume.skills),
            }
        )

    # WHY del + gc: Free ~90MB SentenceTransformer RAM immediately after encoding.
    del model
    gc.collect()
    print(f"Encoded {len(ids)} resumes. Model unloaded.")

    # --- Store in ChromaDB ---
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_dir))

    # WHY delete + recreate: Avoids duplicate IDs on re-index runs.
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass  # Collection didn't exist yet — fine.

    collection = client.create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    print(f"ChromaDB index built: {len(ids)} resumes at {persist_dir}")
    return collection


# ---------------------------------------------------------------------------
# Read-only collection accessor — used by API at startup
# ---------------------------------------------------------------------------


def get_collection(persist_dir: Path = CHROMADB_DIR) -> chromadb.Collection:
    """Open the existing ChromaDB collection and warm up the query-time EF.

    WHY no EF passed to get_collection: ChromaDB 1.5 raises ValueError if a
    custom EF is passed but the collection was persisted without one (conflict:
    new vs persisted "default"). Instead we manage the EF ourselves as a
    module-level singleton (_ef) and pass query_embeddings to collection.query().

    The model is loaded here (first call) and stays alive for the process
    lifetime — no 2s reload per query. Raises if index doesn't exist yet.
    """
    global _ef
    if _ef is None:
        _ef = _SentenceTransformerEF()
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_collection(name=_COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------------


def search_similar(
    collection: chromadb.Collection,
    query_text: str,
    top_k: int = 5,
    fit_level: str | None = None,
) -> list[dict]:
    """Search for resumes similar to *query_text*.

    WHY query_embeddings instead of query_texts: We encode the query ourselves
    using the cached module-level _ef (loaded once at get_collection() time).
    This avoids ChromaDB's EF conflict on get_collection and keeps query
    latency ~50ms since the model is already in memory.

    Returns list of dicts: trace_id, score (cosine similarity 0–1), metadata.
    """
    global _ef
    if _ef is None:
        _ef = _SentenceTransformerEF()

    query_embedding = _ef([query_text])[0]
    where = {"fit_level": fit_level} if fit_level else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["distances", "metadatas"],
    )

    hits: list[dict] = []
    for trace_id, distance, metadata in zip(
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        hits.append(
            {
                "trace_id": trace_id,
                # WHY 1 - distance: ChromaDB cosine space returns distance (lower=closer).
                # Convert to similarity score (higher=closer) for intuitive API output.
                "score": round(1.0 - distance, 4),
                "metadata": metadata,
            }
        )

    return hits
