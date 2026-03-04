"""Abstract base classes (strategy pattern) for all swappable pipeline components.

Every component implements an ABC so the factory can swap implementations
via YAML config without code changes. Same pattern as Java's interface + @Component.

ABCs:
    BaseChunker     — chunk(document) → list[Chunk]
    BaseEmbedder    — embed(texts) → np.ndarray, embed_query(query) → np.ndarray
    BaseVectorStore — add(), search(), save(), load()
    BaseRetriever   — retrieve(query, top_k) → list[RetrievalResult]
    BaseReranker    — rerank(query, results, top_k) → list[RetrievalResult]
    BaseLLM         — generate(prompt, system_prompt, temperature) → str
"""
