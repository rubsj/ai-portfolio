"""BM25 retriever — sparse lexical retrieval using rank-bm25.

BM25Okapi takes tokenized corpus (list of token lists). Scores are unbounded
[0, ∞) — MUST be normalized before combining with cosine scores in hybrid retrieval.

Used as both a standalone baseline (5 BM25-only configs in the experiment grid)
and as the sparse component inside HybridRetriever.
"""
