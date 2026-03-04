"""Factory pattern: YAML config strings → class instances.

Maps config dimension values to concrete implementations:
  {"chunker": "recursive"} → RecursiveChunker(config)
  {"embedder": "minilm"}   → MiniLMEmbedder(config)
  {"retriever": "hybrid"}  → HybridRetriever(dense, bm25, alpha)

Same pattern as Java's @Component + @Qualifier or Spring's bean registry.
Enforces the strategy pattern: callers never import concrete classes directly.
"""
