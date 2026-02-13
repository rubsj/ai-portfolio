"""BM25 lexical baseline — wraps rank_bm25.BM25Okapi for keyword search.

Provides the same search interface as FAISSVectorStore (returns list of
(chunk_id, score) tuples) so the retrieval evaluator can treat both backends
identically.

WHY BM25 as a baseline: BM25 is the standard lexical retrieval algorithm
(used by Elasticsearch, Lucene). Comparing vector search against BM25
shows whether embeddings actually add value over keyword matching. If BM25
beats a vector config, that config's embeddings aren't capturing semantics
well enough to justify the compute cost.

WHY BM25Okapi (not BM25L or BM25Plus): Okapi BM25 is the most widely used
variant and the default in Elasticsearch. It's the industry-standard baseline.

WHY pickle for persistence: BM25Okapi doesn't have a native save/load.
Pickle serializes the entire fitted object (including IDF weights and
corpus statistics). JSON would require manually extracting internal state.
Pickle is standard for scikit-learn models and similar fitted objects.

Java/TS parallel: like a Lucene IndexSearcher — builds an inverted index
from tokenized documents and scores queries using TF-IDF-like weighting.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.models import Chunk

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Tokenize text by lowercasing and splitting on whitespace.

    WHY .lower().split() (not NLTK/spaCy tokenizer): PRD specifies this
    simple tokenization. BM25 is a baseline — fancy tokenization would
    blur the comparison between lexical and semantic retrieval.
    """
    return text.lower().split()


class BM25Retriever:
    """BM25 lexical retriever wrapping rank_bm25.BM25Okapi.

    Built from a list of Chunk objects. Tokenizes each chunk's text with
    .lower().split() and fits BM25Okapi on the corpus.

    Java/TS parallel: like a DAO backed by a Lucene index instead of a
    vector database. Same search() signature as FAISSVectorStore.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks.

        WHY store chunk_ids separately: BM25Okapi only knows about tokenized
        documents (list of token lists). We maintain a parallel list of chunk
        IDs so we can map BM25 result indices back to chunk IDs.
        """
        self._chunk_ids = [chunk.id for chunk in chunks]

        # WHY tokenize here (not in search): BM25Okapi needs the full corpus
        # tokenized at init time to compute IDF weights. Each document is a
        # list of tokens; the corpus is a list of documents.
        tokenized_corpus = [_tokenize(chunk.text) for chunk in chunks]

        # WHY guard: BM25Okapi raises ZeroDivisionError on empty corpus
        # (avgdl = num_doc / corpus_size). Store None and return early in search().
        self._bm25: BM25Okapi | None = (
            BM25Okapi(tokenized_corpus) if tokenized_corpus else None
        )

        logger.info("Built BM25 index from %d chunks", len(chunks))

    @property
    def size(self) -> int:
        """Number of documents in the BM25 index."""
        return len(self._chunk_ids)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Search for chunks most relevant to the query.

        Returns up to k (chunk_id, score) pairs sorted by descending BM25 score.
        Same return signature as FAISSVectorStore.search() for uniform handling.

        WHY argsort (not get_top_n): BM25Okapi.get_top_n returns document texts,
        not indices. We need indices to map back to chunk IDs. get_scores()
        returns all scores, then we sort and take top-k ourselves.
        """
        if self.size == 0:
            return []

        tokenized_query = _tokenize(query)
        # WHY get_scores: returns np.ndarray of scores for ALL documents,
        # one score per document in corpus order. We then pick the top-k.
        scores = self._bm25.get_scores(tokenized_query)

        # WHY argsort descending: np.argsort is ascending; negate scores to
        # sort descending. Take top-k indices.
        effective_k = min(k, self.size)
        top_indices = scores.argsort()[::-1][:effective_k]

        results: list[tuple[str, float]] = [
            (self._chunk_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        return results

    def save(self, path: Path) -> None:
        """Save BM25 model and chunk IDs to disk.

        Creates two files:
        - {path}.pkl — pickled BM25Okapi object
        - {path}.json — chunk ID list

        WHY two files (matching vector_store pattern): keeps the interface
        consistent. The .json sidecar is human-readable for debugging.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        pkl_path = path.with_suffix(".pkl")
        json_path = path.with_suffix(".json")

        with open(pkl_path, "wb") as f:
            pickle.dump(self._bm25, f)
        json_path.write_text(json.dumps(self._chunk_ids, indent=2))

        logger.info("Saved BM25 index (%d docs) to %s", self.size, pkl_path)

    @classmethod
    def load(cls, path: Path) -> BM25Retriever:
        """Load a BM25 model and chunk IDs from disk.

        WHY classmethod: same pattern as FAISSVectorStore.load() — static
        factory that returns a fully initialized instance.
        """
        pkl_path = path.with_suffix(".pkl")
        json_path = path.with_suffix(".json")

        with open(pkl_path, "rb") as f:
            bm25 = pickle.load(f)  # noqa: S301 — trusted local cache file
        chunk_ids = json.loads(json_path.read_text())

        # Construct without calling __init__ (we already have the fitted model)
        retriever = cls.__new__(cls)
        retriever._chunk_ids = chunk_ids
        retriever._bm25 = bm25

        logger.info("Loaded BM25 index (%d docs) from %s", len(chunk_ids), pkl_path)
        return retriever
