"""Ground truth management — loading, validation, and generation helpers.

Ground truth schema:
    queries: list of {query_id, question, relevant_chunks: [{chunk_id, relevance_grade}]}
    relevance_grade: int 0-3 (PRD Decision 3 scale)

Generation workflow:
    1. LLM generates 30 candidate QA pairs with chunk mappings
    2. Developer curates 15, assigning relevance grades manually
    3. Result saved to data/ground_truth.json

Hybrid approach avoids circular evaluation pitfall: LLM-generated questions
tested against the same LLM's retrieval would inflate quality metrics.
"""
