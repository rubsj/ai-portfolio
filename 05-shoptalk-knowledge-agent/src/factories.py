"""Factory: YAML config strings → class instances.

Maps config values to concrete implementations:
  {"chunker": "recursive"} → RecursiveChunker(config)
  {"embedder": "minilm"}   → MiniLMEmbedder(config)
  {"retriever": "hybrid"}  → HybridRetriever(dense, bm25, alpha)

Like Spring's @Component + @Qualifier. Callers never import concrete classes.
"""