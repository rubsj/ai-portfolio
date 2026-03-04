"""Experiment result visualization — 10+ charts answering the 4 required questions.

Charts:
  1.  Config × Metric heatmap (overview of all configs vs all retrieval metrics)
  2.  Chunking strategy comparison (answers Q1)
  3.  Embedding model comparison (answers Q4)
  4.  Dense vs BM25 vs Hybrid (answers Q2)
  5.  Hybrid alpha sweep (optimal α identification)
  6.  Reranking before/after (answers Q3)
  7.  NDCG@5 distribution per config family
  8.  LLM Judge 5-axis radar (generation quality for top configs)
  9.  Latency vs quality scatter (performance trade-off)
  10. Per-query difficulty analysis (where the system struggles)

All charts saved as PNGs to results/charts/.
"""
