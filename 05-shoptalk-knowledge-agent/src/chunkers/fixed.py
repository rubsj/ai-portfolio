"""Fixed-size chunker — splits document into chunks of exactly N characters.

Simplest strategy. Ignores sentence/paragraph boundaries.
Useful as a baseline: any retrieval improvement over fixed-size is meaningful.

Config params: chunk_size (default 512), chunk_overlap (default 50).
"""
