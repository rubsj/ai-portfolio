"""Reranker implementations — all extend BaseReranker.

Rerankers: CohereReranker (API), CrossEncoderReranker (local sentence-transformers).
Both rerank top-20 candidates only — bi-encoder retrieves, cross-encoder refines.
"""
