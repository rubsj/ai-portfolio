"""Experiment grid runner — runs 35+ configs.

Pools by embedding model to minimize load/unload cycles:
  1. MiniLM configs (10) → del + gc.collect()
  2. mpnet configs (10) → del + gc.collect()
  3. BM25-only configs (5, no model)
  4. OpenAI configs (API calls, zero local RAM)

Loading a SentenceTransformer takes 2-5s. Loading it 10 times = waste.
Same batch-by-resource pattern from P2/P3.

Memory: psutil.virtual_memory().percent — warns if >85%.
"""