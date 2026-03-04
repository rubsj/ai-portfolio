"""Recursive character text splitter — tries separators in order of preference.

Splits on ["\n\n", "\n", ". ", " ", ""] in sequence, falling back to the next
separator when a chunk exceeds chunk_size. Preserves paragraph structure when
possible. Most practical general-purpose strategy.

Config params: chunk_size (default 512), chunk_overlap (default 50).
"""
