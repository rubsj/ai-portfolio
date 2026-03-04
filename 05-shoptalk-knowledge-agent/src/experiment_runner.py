"""Experiment grid runner — executes 35+ configurations systematically.

Pools experiments by embedding model to minimize model load/unload cycles:
  1. MiniLM configs (10) → del + gc.collect()
  2. mpnet configs (10) → del + gc.collect()
  3. BM25-only configs (5, no model)
  4. OpenAI configs (API calls, zero local RAM)

Why pooling: loading a SentenceTransformer takes 2-5s. Loading it 10 times = waste.
Same batch-by-resource pattern used in P2/P3.

Memory monitoring: psutil.virtual_memory().percent — logs warning if >85%.
"""
