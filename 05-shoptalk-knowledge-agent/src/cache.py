"""JSON file cache for LLM responses and embeddings.

Cache key: MD5 hash of (model + system_prompt + user_prompt).
Cache location: data/cache/ as JSON files.

35+ configs x 15 queries = 500+ LLM calls per full run.
Cache avoids re-spend. Same pattern from P1/P2/P4.
"""