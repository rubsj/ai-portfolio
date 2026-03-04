"""Sliding window chunker — overlapping windows of N tokens.

Every chunk overlaps with the previous by overlap_size tokens. Ensures no
information is lost at chunk boundaries — a fact split across a boundary
will appear in at least one complete chunk.

Why useful: dense retrieval can miss facts that land on chunk boundaries.
Sliding window trades index size for coverage completeness.

Config params: window_size (tokens), step_size (tokens, controls overlap).
"""
