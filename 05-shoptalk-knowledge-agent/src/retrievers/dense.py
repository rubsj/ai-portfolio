"""Dense retriever — cosine similarity search over FAISS index.

Embeds query with the same model used during ingestion, searches FAISS
IndexFlatIP, returns top-k RetrievalResults with cosine similarity scores.

Requires L2-normalized query vector (faiss.normalize_L2) to match index vectors.
"""
