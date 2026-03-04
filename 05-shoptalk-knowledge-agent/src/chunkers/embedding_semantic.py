"""Embedding-semantic chunker — splits where consecutive sentence similarity drops.

Embeds every sentence, computes cosine similarity between consecutive sentences,
and splits where similarity falls below a threshold (semantic breakpoint).

ALWAYS uses MiniLM (all-MiniLM-L6-v2) for boundary detection regardless of the
indexing embedder — MiniLM is smallest (22.7M params), boundary detection only
needs relative similarity comparison, not absolute quality.

After chunking: del model + gc.collect() before indexing phase begins.

Config params: breakpoint_threshold (default 0.85), min_chunk_size (default 100).
"""
