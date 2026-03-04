"""Heading-semantic chunker — splits on document section headings.

Detects section boundaries using regex patterns for common heading formats
(ALL CAPS lines, numbered sections like "3.1", lines followed by dashes).
Each section becomes one chunk, preserving semantic coherence.

Requires no embeddings during chunking (unlike EmbeddingSemanticChunker).
Best for well-structured academic papers with clear section hierarchy.
"""
