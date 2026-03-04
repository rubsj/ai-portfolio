"""FAISS vector store with explicit persistence.

FAISSVectorStore implements BaseVectorStore using IndexFlatIP (inner product).
Vectors must be L2-normalized before add() and search() — IndexFlatIP equals
cosine similarity only on unit vectors.

Why FAISS over ChromaDB: P4 used ChromaDB. P5 needs low-level control for
35+ experiment configs — explicit index type selection, direct save/load,
no metadata overhead. Different tools for different jobs.
"""
