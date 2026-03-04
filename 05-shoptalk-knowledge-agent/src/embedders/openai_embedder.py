"""OpenAI embedder — text-embedding-3-small, 1536-dimensional embeddings.

API-based embedder: zero local RAM, batched requests (100 texts/call).
Provides build-vs-buy comparison: local model quality vs managed API.

Why text-embedding-3-small over large: cost (5× cheaper) with competitive
quality for retrieval. Upgrade to text-embedding-3-large in extended experiments
if quality gap is significant.

Runs LAST in experiment grid — after all local model experiments complete.
"""
