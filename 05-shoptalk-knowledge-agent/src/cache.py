"""JSON file cache for LLM responses and embeddings.

Cache key: MD5 hash of (model + system_prompt + user_prompt).
Cache location: data/cache/ as JSON files.

Why cache everything: 35+ experiment configs × 15 queries = 500+ LLM calls.
Caching enables re-runs without API cost. Same pattern proven in P1/P2/P4.
"""
