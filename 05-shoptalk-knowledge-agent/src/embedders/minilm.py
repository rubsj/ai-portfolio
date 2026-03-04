"""MiniLM embedder — all-MiniLM-L6-v2, 384-dimensional embeddings.

Smallest and fastest local model (22.7M params, ~90MB). Used for:
  - Dense retrieval experiments requiring 384d vectors
  - Embedding-semantic chunker boundary detection (always MiniLM, per PRD Decision 1)

Vectors are L2-normalized by SentenceTransformers by default. Verify with
np.linalg.norm(emb) ≈ 1.0 before FAISS IndexFlatIP add().
"""
