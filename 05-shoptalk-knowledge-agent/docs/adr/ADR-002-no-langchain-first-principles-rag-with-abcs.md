# ADR-002: No LangChain — First-Principles RAG

**Project:** P5: ShopTalk Knowledge Agent
**Category:** Architecture
**Status:** Accepted
**Date:** 2026-03-03

---

## Context

LangChain provides `RecursiveCharacterTextSplitter`, `RetrievalQA`, `HuggingFaceEmbeddings`, and dozens of pre-built chains for RAG pipelines.

I didn't use it. P5 is an experiment platform that compares chunking strategies and retriever types. The experiment variables *are* the implementation details LangChain hides. I need to control chunk boundaries, measure where splits happen, and swap components without fighting a framework.

The portfolio angle matters too. "I configured LangChain" is a different statement than "I built the chunking pipeline and measured the tradeoffs." This project needs to show the second one.

---

## Decision

Build all RAG components from scratch behind Abstract Base Classes. No LangChain dependency.

Six interfaces in `src/interfaces.py`:

```python
# Python ABCs (src/interfaces.py)
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...

class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...
    @property
    @abstractmethod
    def dimensions(self) -> int: ...

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...
    @abstractmethod
    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[Chunk, float]]: ...
```

Five chunking strategies, all extending `BaseChunker`:
- `FixedSizeChunker`: character sliding window baseline
- `RecursiveChunker`: separator hierarchy `["\n\n", "\n", ". ", " ", ""]`
- `SlidingWindowChunker`: tiktoken-based token windows
- `HeadingSemanticChunker`: regex section boundary detection
- `EmbeddingSemanticChunker`: cosine similarity breakpoints via MiniLM

53 unit tests cover all five chunkers. None of them load real PDFs or real models. `EmbeddingSemanticChunker` is tested with a mocked `SentenceTransformer` that crafts orthogonal embeddings to place boundaries exactly where the test expects them. `RecursiveChunker._split_text()` takes any text + separator list and returns segments directly, no splitter object needed. LangChain's equivalent is a class method coupled to the full splitter.

---

## Alternatives

| Option | Trade-off | Why rejected |
|--------|-----------|--------------|
| **LangChain** | Pre-built components, fast scaffolding. Hides internals so you can't inspect why a boundary was placed. Version churn. Hard to unit-test without mocking LangChain's own guts. | Abstracts away the exact things I need to measure. |
| **Haystack** | RAG-focused pipeline abstractions. | Same problem: abstracts the experiment variables. |
| **LlamaIndex** | Good document indexing. Wraps chunking and retrieval into node/pipeline abstractions. | Same problem. |

---

## Consequences

Every component is independently benchmarkable. Swapping a chunker in an experiment is one line: the `ExperimentConfig.chunking_strategy` field. Test coverage is high because there are no third-party abstractions to stub out.

The cost is more boilerplate per component. I also have to handle retry and error logic myself, since there's no `Runnable` infrastructure.

The `BaseChunker` / `BaseEmbedder` / `BaseRetriever` pattern is reusable across P6 through P9. If I ever migrate to LangChain for a production project, it's an adapter implementation, not a rewrite.

The ABC pattern maps to Java interfaces directly (like defining your own `Splitter` interface instead of depending on Spring's `TextSplitter` bean):

```java
// Java interface
public interface Chunker {
    List<Chunk> chunk(Document document);
}

public class FixedSizeChunker implements Chunker {
    @Override
    public List<Chunk> chunk(Document document) { ... }
}
```

```python
# Python ABC equivalent
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]: ...

class FixedSizeChunker(BaseChunker):
    def chunk(self, document: Document) -> list[Chunk]: ...
```
