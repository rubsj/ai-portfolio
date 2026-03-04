"""mpnet embedder — all-mpnet-base-v2, 768-dimensional embeddings.

Larger model (109M params, ~420MB) with higher quality than MiniLM.
Used for mid-tier retrieval experiments. Experiment hypothesis: higher-dimensional
embeddings improve retrieval quality at the cost of index size and query latency.

Memory constraint: load AFTER MiniLM experiments complete and MiniLM is unloaded.
"""
