"""Hybrid retriever — weighted combination of dense and BM25 retrieval.

Internal composition: owns both DenseRetriever and BM25Retriever (PRD Decision 2).
Score fusion:
  1. Retrieve top-k from dense (cosine [0,1])
  2. Retrieve top-k from BM25 (unbounded [0,∞))
  3. Min-max normalize BM25 scores to [0,1]
  4. Merge result sets: combined_score = α * dense_score + (1-α) * bm25_score
  5. Re-rank by combined score, return top-k

Why min-max normalize: naive combination without normalization lets BM25 dominate
because its scores can be 10-100× larger than cosine scores in [0,1].

Config param: alpha (default 0.7) — higher α = more weight on dense retrieval.
"""
