"""Cross-encoder reranker — local sentence-transformers CrossEncoder.

Takes top-20 candidates, scores each (query, chunk) pair through a
cross-encoder model (ms-marco-MiniLM-L-6-v2). More accurate than bi-encoder
because it attends to both query and document jointly.

Load → rerank → del → gc.collect() pattern to avoid holding model in memory
between reranking calls (memory budget constraint on M2 8GB).

Why cross-encoder after bi-encoder: bi-encoder retrieval is O(1) at query time
(pre-computed embeddings). Cross-encoder is O(n) — only feasible on top-20.
"""
