"""Cohere reranker — uses Cohere's rerank API to re-score retrieved chunks.

Takes top-20 candidates from retrieval, calls Cohere rerank-english-v3.0,
returns top-k re-ordered by relevance score.

Why Cohere API: production reranking without local model memory overhead.
Limited to top-20 candidates to stay within Cohere free tier limits.
"""
