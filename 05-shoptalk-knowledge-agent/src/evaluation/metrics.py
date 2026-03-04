"""Retrieval evaluation metrics — implemented from scratch (not sklearn).

Metrics:
    precision_at_k(retrieved_ids, relevant_ids, k) → float
    recall_at_k(retrieved_ids, relevant_ids, k) → float
    mrr(retrieved_ids, relevant_ids) → float          # Mean Reciprocal Rank
    ndcg_at_k(retrieved_ids, graded_relevance, k) → float

NDCG graded relevance scale (PRD Decision 3):
    3 = Gold chunk (directly answers the question)
    2 = Same section (contextually relevant)
    1 = Same document (topically related)
    0 = Irrelevant

Why from scratch: understanding NDCG's log₂ discount function is an interview
talking point. DCG@K = Σ(rel_i / log₂(i+1)). NDCG = DCG / ideal_DCG.
"""
