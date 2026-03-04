# PRD: P5 — ShopTalk Knowledge Management Agent

> **This is the product requirements and architecture contract.**
> Claude Code: read this + both CLAUDE.md files before starting. Use this for WHAT to build and WHY.
> Claude Code Opus handles implementation planning. Do NOT re-debate architecture decisions — they are final.

**Project:** P5 — ShopTalk Knowledge Management Agent: Production RAG System
**Timeline:** 5 sessions (~22h total), flexible pacing prioritizing learning depth
**Owner:** Developer (Java/TS background, completed P1 + P2 + P3 + P4)
**Source of Truth:** [Notion Requirements](https://www.notion.so/Mini_Project_5_requirements-2ffdb630640a81668269f59ead053417)
**Concepts Primer:** `p5-concepts-primer.html` in project root
**PRD Version:** v3 (all design decisions finalized, ADRs distributed, test coverage 95%+)

---

## 1. Objective

Build a **production-grade RAG system** that ingests arbitrary PDF documents, retrieves relevant chunks through multiple swappable strategies, generates LLM answers with traceable citations, and serves the system through both CLI tools and an interactive Streamlit web interface.

**The Portfolio Narrative:** P2 was the quality inspector — it tested production lines and found the retrieval-generation gap (0.747 Recall@5 vs 0.511 faithfulness). P5 is the entire factory. It builds the system that P2 measured, tackling that gap with hybrid retrieval, reranking, and citation extraction. Together: "I measured, then I built — and I improved on what I measured."

**The output is a COMPLETE SYSTEM with EXPERIMENTAL EVIDENCE.** Every component must be swappable through configuration. The system must prove — with data — which configuration works best.

---

## 2. Success Criteria

> These are the hard targets from the requirements document. The project is not complete until all are met.

### 2a. Retrieval Performance Targets

| Metric | Description | Target |
|--------|-------------|--------|
| Recall@5 | % of relevant chunks in top-5 results | **> 0.80** |
| Precision@5 | % of top-5 results that are relevant | **> 0.60** |
| MRR | Mean Reciprocal Rank of first relevant result | **> 0.70** |
| NDCG@5 | Normalized Discounted Cumulative Gain | **> 0.75** |

These targets must be met by **at least one configuration** in the experiment grid. Document which config achieves them and why.

### 2b. Generation Quality Targets (LLM-as-Judge)

| Axis | Description | Scale |
|------|-------------|-------|
| Relevance | Answer addresses the question | 1–5 |
| Accuracy | Information is factually correct per source context | 1–5 |
| Completeness | Answer is thorough | 1–5 |
| Conciseness | Answer is not unnecessarily verbose | 1–5 |
| Citation Quality | Proper source attribution | 1–5 |

**Target:** Average score **> 4.0** across all 5 axes for the best configuration.

### 2c. Experiment Coverage

| Dimension | Minimum | Our Target |
|-----------|---------|------------|
| Chunking strategies | 3 | 5 (Fixed, Recursive, Sliding Window, Heading-Semantic, Embedding-Semantic) |
| Embedding models | 2 | 3 (MiniLM, mpnet, OpenAI) |
| Retrieval methods | 2 | 3 (Dense, BM25, Hybrid) |
| Configurations tested | 12 | 24+ core grid |
| Test queries | 10 | 15 curated |

### 2d. System Functionality

| Requirement | How to Verify |
|-------------|---------------|
| PDF ingestion creates searchable FAISS indices | CLI ingest command produces index files on disk |
| All components swappable through configuration files | YAML config change = different pipeline, no code changes |
| CLI tools work for ingestion, querying, and evaluation | All 3 commands execute without errors |
| Streamlit UI provides interactive QA experience | Upload PDF → configure → ask → get cited answer |
| Experiment results saved with full configuration metadata | JSON files with config + metrics per experiment |
| Results show clear winner with explanation | Comparison report answers the 4 required questions |

### 2e. System Performance Metrics (must be tracked)

| Metric | What to Measure |
|--------|----------------|
| Ingestion time | Seconds to process N PDF pages through full pipeline |
| Query latency | Time from question to answer (including retrieval + generation) |
| Index size | Disk storage for FAISS index + metadata per configuration |
| Memory usage | Peak RAM during ingestion and query operations |

### 2f. Test Coverage

| Target | Metric |
|--------|--------|
| **≥95% coverage on core modules** | pytest --cov report |
| Unit tests for each component | Chunkers, embedders, retrievers, metrics, generator |
| Integration test | Full pipeline PDF → answer |
| Metric verification | Known inputs → known outputs for all IR metrics |
| Edge case tests | Empty documents, malformed PDFs, very long queries, no results found |

---

## 3. Architecture Decisions (FINAL)

### 3a. Core Technology Stack (Required by Spec)

| Component | Technology | Why (Spec Alignment) |
|-----------|-----------|---------------------|
| Language | Python 3.12+ | Spec requires 3.10+. We use 3.12 for native generics. |
| PDF extraction | PyMuPDF | Spec-required. `import fitz`. |
| Embeddings | SentenceTransformers | Spec-required. Local models. |
| Vector store | FAISS | Spec-required. P4 demonstrated ChromaDB — P5 uses FAISS for low-level experiment control. |
| Sparse retrieval | rank-bm25 | Spec-required. BM25Okapi implementation. |
| LLM generation | LiteLLM | Spec says "LiteLLM or OpenAI." LiteLLM wraps multiple providers behind one interface — satisfies spec AND teaches multi-provider pattern. Default to OpenAI models. |
| Data validation | Pydantic v2 | Spec-required. All data models. |
| Web UI | Streamlit | Spec-required. Interactive QA. |
| Reranking | Cohere API + local CrossEncoder | Spec-optional but recommended. Both implemented for comparison. |
| Testing | pytest | Spec-optional. We treat as required (consistent with P1–P4). ≥95% coverage. |
| CLI | Click | Root CLAUDE.md standard. Consistent with P2. |
| Experiment configs | YAML + Pydantic | Production ML pattern. Human-editable, machine-validated. |

### 3b. Strategic Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **No LangChain** | Build all components from scratch behind ABCs | P5's purpose is first-principles RAG engineering. Custom components behind abstract interfaces teach what LangChain hides. P2 already used LangChain text splitters. README explains the deliberate choice. |
| **No RAGAS** | Custom 5-axis LLM-as-Judge | P2 already used RAGAS. Custom judge gives per-axis diagnostics and demonstrates understanding of what RAGAS does internally. Stronger interview story. |
| **Braintrust** | Optional stretch — JSON baseline, Braintrust if time permits | Spec lists as optional. JSON experiment results with Pydantic validation as baseline. Add Braintrust enrichment on Day 5 if time allows. |
| **No FastAPI** | CLI + Streamlit only | P4 proved FastAPI skills. P5's value is the RAG pipeline, not another REST wrapper. Streamlit + CLI cover all spec requirements. |
| **FAISS over ChromaDB** | FAISS with explicit persistence | P4 demonstrated ChromaDB. P5 needs low-level control for 24+ experiment configs (index types, explicit save/load). Different tools for different jobs — this IS the interview signal. |
| **LiteLLM over raw OpenAI** | Multi-provider wrapper behind BaseLLM | Satisfies spec requirement. Teaches adapter pattern. Default to OpenAI models but infrastructure supports any provider. Minimal code difference from raw OpenAI SDK. |
| **5 chunking strategies** | Fixed, Recursive, Sliding Window, Heading-Semantic, Embedding-Semantic | Spec requires minimum 3, lists 4 explicitly (incl. Sliding Window). We add Embedding-Semantic for deeper learning. All behind BaseChunker. |
| **3 embedding models** | MiniLM (384d), mpnet (768d), OpenAI text-embedding-3-small (1536d) | Spec requires min 2. OpenAI added Day 3–4 for build-vs-buy comparison story. |
| **Hybrid ground truth** | LLM generates 30, developer curates 15 | Avoids circular evaluation pitfall. Manual curation adds trustworthiness. Spec recommends this approach. |
| **Index-reference citations** | [N] notation, parsed back to chunks | LLM more reliable at producing [1] than full source paths. Fewer tokens = fewer hallucination chances. |

---

## 4. Component Architecture

### 4a. Abstract Base Classes (Strategy Pattern)

All components must implement abstract interfaces. This is the core architectural pattern — config-driven component selection without code changes.

| Interface | Method | Implementations |
|-----------|--------|----------------|
| `BaseChunker` | `chunk(document) → list[Chunk]` | FixedSizeChunker, RecursiveChunker, SlidingWindowChunker, HeadingSemanticChunker, EmbeddingSemanticChunker |
| `BaseEmbedder` | `embed(texts) → np.ndarray`, `embed_query(query) → np.ndarray` | MiniLMEmbedder, MpnetEmbedder, OpenAIEmbedder |
| `BaseVectorStore` | `add()`, `search()`, `save()`, `load()` | FAISSVectorStore |
| `BaseRetriever` | `retrieve(query, top_k) → list[RetrievalResult]` | DenseRetriever, BM25Retriever, HybridRetriever |
| `BaseReranker` | `rerank(query, results, top_k) → list[RetrievalResult]` | CohereReranker, CrossEncoderReranker |
| `BaseLLM` | `generate(prompt, system_prompt, temperature) → str` | LiteLLMClient (wraps any provider) |

**Factory pattern:** YAML config strings map to class instances. `{"chunker": "recursive"}` → `RecursiveChunker(config)`. Same pattern as Java's `@Component` + `@Qualifier`.

### 4b. Data Models (Pydantic)

All models defined from the requirements spec. Key models:

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `Document` | Extracted PDF with metadata | `id`, `content`, `metadata` (source, title, author, page_count), `pages` |
| `Chunk` | Atomic retrieval unit | `id`, `content`, `metadata` (document_id, source, page_number, start_char, end_char, chunk_index), `embedding` (optional) |
| `RetrievalResult` | Single retrieved chunk + score | `chunk`, `score`, `retriever_type` (dense/bm25/hybrid), `rank` |
| `QAResponse` | Complete query-answer output | `query`, `answer`, `citations`, `chunks_used`, `confidence` (optional), latency |
| `Citation` | Traceable source reference | `chunk_id`, `source`, `page_number`, `text_snippet`, `relevance_score` |
| `ExperimentConfig` | One experiment specification | All config dimensions + validators |
| `ExperimentResult` | Results for one config | Config + per-query retrieval metrics + judge scores + performance metrics |

Claude Code Opus designs the full schema details during implementation planning.

### 4c. Pipeline Architecture

**Phase 1: Ingestion (Offline, once per document+config)**
```
PDF → Load → Preprocess/Clean → Chunk (strategy) → Embed (model) → Index (FAISS) → Save to Disk
```

**Phase 2: Query (Per question, real-time)**
```
Question → Embed Query → Retrieve Top-K → [Rerank] → Generate Answer + Citations → Return QAResponse
```

**Phase 3: Evaluation (Offline, experiment grid)**
```
Ground Truth Queries → Run Each Config → Calculate Metrics → Save Results → Compare → Report
```

---

## 5. Finalized Design Decisions

> These were engineering problems that required careful design. Each decision below is **FINAL**. Claude Code must implement as specified — do not re-evaluate or propose alternatives.

### Decision 1: Embedding-Semantic Chunker Model — ALWAYS MiniLM

**Problem:** The embedding-semantic chunker needs an embedding model during the chunking phase (before the indexing phase begins). This creates a dependency: chunking strategy depends on embedder.

**Decision:** Always use MiniLM (all-MiniLM-L6-v2) for chunk boundary detection, regardless of the indexing embedder.

**Rationale:**
- MiniLM is the smallest model (22.7M params, ~90MB) — minimizes RAM during chunking phase
- Boundary detection doesn't need high-quality embeddings; it only needs to measure relative similarity *between consecutive sentences* to find breakpoints
- Decouples chunking strategy from the indexing embedder, keeping the experiment grid clean
- The same chunks will be re-embedded by the indexing model (mpnet, OpenAI, etc.) for retrieval — only the boundaries are set by MiniLM

**Implementation:** `EmbeddingSemanticChunker.__init__` always loads MiniLM internally. The factory does NOT pass the indexing embedder to this chunker. After chunking completes, `del model` + `gc.collect()` before the indexing phase begins.

### Decision 2: HybridRetriever Uses Internal Composition

**Problem:** Should the HybridRetriever compose a DenseRetriever and BM25Retriever internally, or receive their results externally?

**Decision:** Internal composition — HybridRetriever creates and manages both sub-retrievers internally.

**Rationale:**
- HybridRetriever owns the full retrieval lifecycle: it calls both sub-retrievers, normalizes BM25 scores, applies α weighting, and merges results
- Caching happens at the embedding/BM25 level (the embedder caches vectors, BM25 caches the tokenized corpus), NOT at the retriever output level
- External composition would require the caller to manage two retriever instances and pass intermediate results — this leaks implementation details and complicates the factory
- This mirrors how Java's `CompositeService` pattern works: the composite owns its delegates

**Implementation:** `HybridRetriever.__init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever, alpha: float)`. The factory creates all three retrievers and injects dense + bm25 into hybrid. Alpha is a required config parameter when `retriever_type=hybrid`.

### Decision 3: Ground Truth — 1-3 Gold Chunks per Query, 4-Level NDCG Grading

**Problem:** How many gold chunks per query? What grading scale for NDCG?

**Decision:** 1-3 gold chunks per query with this NDCG grading scale:

| Grade | Meaning | Example |
|-------|---------|---------|
| 3 | Directly answers the question | The chunk contains the specific fact/explanation |
| 2 | Contextually relevant (same section) | The chunk is in the same section and provides supporting context |
| 1 | Topically related (same document) | The chunk discusses the same topic but doesn't answer this specific question |
| 0 | Irrelevant | No meaningful connection to the query |

**Rationale:**
- Single-chunk ground truth makes Recall binary (0 or 1) — insufficient granularity for comparing configs
- 1-3 chunks reflects reality: some questions have one precise answer, others span sections
- The 4-level grading scale gives NDCG discriminative power: a system that ranks grade-3 chunks first scores higher than one that returns grade-2 chunks first
- This is the standard TREC-style graded relevance approach

**Implementation:** The `ground_truth.json` schema includes `relevant_chunks: list[dict]` where each dict has `chunk_id: str` and `relevance_grade: int (0-3)`. The LLM generates 30 candidate QA pairs with chunk mappings; the developer curates 15, adding relevance grades manually.

### Decision 4: Citation Validation — Parse-Only for Grid, Judge Handles Semantic Validation

**Problem:** Should we validate citations post-generation (checking if cited chunk supports the claim) or just verify they parse correctly?

**Decision:** Parse-only validation for the experiment grid. The LLM-as-Judge's "Citation Quality" axis handles semantic validation.

**Rationale:**
- Full citation validation requires an additional LLM call per answer — at 24+ configs × 15 queries = 360+ extra LLM calls. This adds cost and time with marginal benefit
- The judge already receives the full context (question, answer, source chunks). Its Citation Quality axis (1-5) implicitly checks whether citations actually support claims
- Parse-only validation catches the mechanical failures: out-of-range [N] references, missing citations, malformed notation
- Semantic validation via judge is MORE thorough than a simple "does chunk contain the claim" heuristic

**Implementation:** `citation_extractor.py` uses regex to extract `[N]` references, validates N is within the range of provided chunks, and returns `list[Citation]`. Out-of-range or unparseable citations are logged but don't fail the pipeline. The judge separately scores citation quality on a 1-5 scale.

### Decision 5: Experiment Runner — Pool Experiments by Embedding Model

**Problem:** Running 24+ configs on 8GB M2 means cycling through 3 embedding models. Should each experiment manage its own model lifecycle, or should experiments be pooled?

**Decision:** Pool experiments by embedding model. Run ALL configs for one model before switching.

**Execution order:**
1. Load MiniLM → run all 10 MiniLM configs → del MiniLM → gc.collect()
2. Load mpnet → run all 10 mpnet configs → del mpnet → gc.collect()
3. Run all 5 BM25-only configs (no model needed)
4. OpenAI configs last (API calls, zero local RAM)

**Rationale:**
- Loading a SentenceTransformer model takes 2-5 seconds. Loading it 10 times = 20-50s wasted
- Peak memory is controlled: only one model at a time
- This is the standard batch processing optimization: group by the expensive resource

**Implementation:** `experiment_runner.py` groups ExperimentConfigs by `embedding_model`, sorts groups by model size (smallest first), and iterates. Each group: load model → index all chunked docs for that model → run retrieval/evaluation for all configs → unload model. The runner tracks `psutil.virtual_memory().percent` after each model swap and logs a warning if >85%.

---

## 6. Experiment Grid

### 6a. Core Grid (minimum 24 configurations)

```
Chunking (5):     fixed, recursive, sliding_window, heading_semantic, embedding_semantic
Embeddings (3):   minilm, mpnet, openai
Retrieval (2):    dense, hybrid (α=0.7)
```

5 × 3 × 2 = **30 core configurations** (exceeds 24 minimum)

**Note:** BM25 is retrieval-only (no embeddings needed). Add 5 BM25-only configs (one per chunker) as baselines = **35 total**.

### 6b. Extended Experiments (stretch)

- Hybrid α sweep: {0.3, 0.5, 0.7} for best chunker × best embedder = 3 more
- Reranking: best 3 configs × 2 rerankers = 6 more
- Total potential: **35 core + 9 extended = 44 configurations**

### 6c. Experiment Output Format (from requirements)

Every experiment must save results as JSON matching this structure:

```json
{
  "experiment_id": "exp_20260303_abc123",
  "config": {
    "chunking_strategy": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "embedding_model": "all-mpnet-base-v2",
    "retriever_type": "hybrid",
    "hybrid_alpha": 0.7,
    "top_k": 5,
    "use_reranking": false,
    "reranker": null
  },
  "metrics": {
    "recall_at_5": 0.85,
    "precision_at_5": 0.72,
    "mrr": 0.78,
    "ndcg_at_5": 0.81
  },
  "performance": {
    "ingestion_time_seconds": 45.2,
    "avg_query_latency_ms": 2350,
    "index_size_bytes": 15400000,
    "peak_memory_mb": 2100
  },
  "judge_scores": {
    "avg_relevance": 4.2,
    "avg_accuracy": 4.0,
    "avg_completeness": 3.8,
    "avg_conciseness": 4.1,
    "avg_citation_quality": 3.9,
    "overall_average": 4.0
  },
  "query_results": [...]
}
```

### 6d. Test Corpus
- **Primary corpus:** 4 AI/ML foundational papers from arXiv
  - Attention Is All You Need (Vaswani et al., 2017) — 15 pages
  - RAG for Knowledge-Intensive NLP Tasks (Lewis et al., 2020) — 15 pages  
  - Sentence-BERT (Reimers & Gurevych, 2019) — 11 pages
  - BERT (Devlin et al., 2019) — 16 pages
- **Total pages:** ~57 pages
- **Expected chunks:** 500-800 across all chunking strategies
- **Selection criteria:** Single-column layout, well-structured headings, content developer can grade accurately on 4-level NDCG scale
- **Download on Day 0:** Pre-download to `data/pdfs/`, see `data/pdfs/README.md` for URLs and verification script
- **Git policy:** PDFs are .gitignored — only the README is committed
---

## 7. Required Deliverables

> Cross-referenced against the requirements document deliverables section. Every item must be produced.

### 7a. Functional Components

| # | Deliverable | Verification |
|---|-------------|-------------|
| D1 | PDF ingestion pipeline (CLI command) | `python scripts/ingest.py` produces FAISS index on disk |
| D2 | Vector index creation + persistence | Index files loadable across sessions |
| D3 | Query pipeline with retrieval + generation | Question → cited answer in < 5 seconds |
| D4 | Citation extraction from generated answers | [N] references parsed to source chunks |
| D5 | Web UI for interactive QA (Streamlit) | 5 panels: upload, config sidebar, question input, answer+citations, source viewer |
| D6 | All components swappable through config | YAML change = different pipeline behavior |

### 7b. Experiment Results

| # | Deliverable | Verification |
|---|-------------|-------------|
| D7 | 24+ experiment configurations tested | JSON result files per config |
| D8 | 15+ diverse test queries with gold chunk IDs | `ground_truth.json` with curated queries |
| D9 | All results saved with full configuration metadata | Config + metrics in every result file |
| D10 | At least one config meets all 4 retrieval targets | Recall>0.80, Precision>0.60, MRR>0.70, NDCG>0.75 |

### 7c. Comparison Analysis (4 Required Questions)

The experiment report **must** explicitly answer these 4 questions with data:

| # | Question | Expected Output |
|---|----------|----------------|
| Q1 | Which chunking strategy performed best? | Bar chart + explanation of why |
| Q2 | Did hybrid retrieval beat dense-only? | Side-by-side comparison with α analysis |
| Q3 | Was reranking worth the latency cost? | Before/after metrics + latency delta |
| Q4 | Which embedding model gave best quality/speed trade-off? | Scatter plot (quality vs latency) + recommendation |

### 7d. CLI Tools

| Command | Purpose |
|---------|---------|
| `python scripts/ingest.py` | PDF → chunk → embed → index → save |
| `python scripts/serve.py` | Load index → interactive QA REPL |
| `python scripts/evaluate.py` | Run experiment grid → metrics → results |

### 7e. Visualizations (minimum 10 charts)

| # | Chart | Purpose |
|---|-------|---------|
| 1 | Config × Metric heatmap | Overview of all configs vs all retrieval metrics |
| 2 | Chunking strategy comparison | Answers Q1 |
| 3 | Embedding model comparison | Answers Q4 |
| 4 | Dense vs BM25 vs Hybrid | Answers Q2 |
| 5 | Hybrid alpha sweep | Optimal α identification |
| 6 | Reranking before/after | Answers Q3 |
| 7 | NDCG@5 distribution per config family | Quality variation analysis |
| 8 | LLM Judge 5-axis radar | Generation quality profile for top configs |
| 9 | Latency vs quality scatter | Performance trade-off visualization |
| 10 | Per-query difficulty analysis | Where the system struggles |

### 7f. Documentation

| # | Deliverable | Content | Day |
|---|-------------|---------|-----|
| D11 | README.md | Problem, architecture (Mermaid diagram), results, experiment findings, demo link, setup | Day 5 |
| D12 | ADR-001: FAISS over ChromaDB | Low-level control for experiments vs P4's ChromaDB | Day 1 |
| D13 | ADR-002: No LangChain — first-principles RAG | Building from ABCs teaches what LangChain hides | Day 1 |
| D14 | ADR-003: Hybrid retrieval with min-max score fusion | Why naive combination fails. α optimization | Day 2 |
| D15 | ADR-004: LiteLLM over raw OpenAI SDK | Multi-provider pattern. Adapter architecture | Day 2 |
| D16 | ADR-005: YAML + Pydantic for experiment configs | Human-editable with validation. Production ML pattern | Day 3 |
| D17 | Loom recording | 2-min walkthrough | Day 5 |

---

## 8. Evaluation Framework

### 8a. Retrieval Evaluation

Implement from scratch (not sklearn): Precision@K, Recall@K, MRR, NDCG@K.

**Why from scratch:** Understanding NDCG's log₂ discount function is an interview talking point. The implementation IS the learning.

**NDCG graded relevance scale (FINAL — from Decision 3):**
- 3 = Gold chunk (directly answers the question)
- 2 = Same section (contextually relevant)
- 1 = Same document (topically related)
- 0 = Irrelevant

### 8b. Generation Evaluation (5-Axis LLM-as-Judge)

| Axis | Score Anchors |
|------|--------------|
| Relevance | 1=off-topic, 3=partially addresses, 5=directly answers |
| Accuracy | 1=major errors, 3=minor errors, 5=every claim verifiable in context |
| Completeness | 1=fragment only, 3=covers basics, 5=comprehensive |
| Conciseness | 1=extremely verbose/repetitive, 3=some unnecessary content, 5=concise and focused |
| Citation Quality | 1=no citations, 3=some correct, 5=every claim properly cited |

**Calibration:** Manually score 5 answers yourself, compare to LLM judge. Document the calibration offset.

**Aggregation:** Per-query average (find hard queries) AND per-axis average (find systematic weaknesses).

### 8c. Self-Evaluation Questions (from requirements)

The project is not complete until you can answer:

1. Can you explain why configuration X outperformed configuration Y?
2. Do your metrics align with qualitative assessment of answers?
3. Does reranking improve top-3 results even if top-5 metrics are similar?
4. Are citations accurate and helpful for verification?
5. Can the system handle edge cases (very short/long documents, ambiguous questions)?

---

## 9. Directory Structure

```
05-shoptalk-knowledge-agent/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── .env
├── src/
│   ├── schemas.py                     ← All Pydantic models
│   ├── interfaces.py                  ← All ABCs
│   ├── extraction.py                  ← PDF extraction + cleaning
│   ├── chunkers/                      ← 5 strategies, all BaseChunker
│   ├── embedders/                     ← 3 models, all BaseEmbedder
│   ├── vector_store.py                ← FAISSVectorStore
│   ├── retrievers/                    ← Dense, BM25, Hybrid
│   ├── rerankers/                     ← Cohere, CrossEncoder
│   ├── generator.py                   ← LiteLLM generation + citation extraction
│   ├── pipeline.py                    ← End-to-end orchestrator
│   ├── evaluation/                    ← Metrics, judge, ground truth
│   ├── experiment_runner.py           ← Grid search orchestrator
│   ├── visualization.py               ← 10+ charts
│   ├── factories.py                   ← Config → class mapping
│   ├── cache.py                       ← LLM + embedding cache
│   └── streamlit_app.py              ← Web UI
├── scripts/
│   ├── ingest.py                      ← CLI: PDF → index
│   ├── serve.py                       ← CLI: interactive QA
│   └── evaluate.py                    ← CLI: run experiments
├── experiments/configs/               ← YAML experiment configs
├── data/
│   ├── pdfs/                          ← Source PDFs
│   ├── indices/                       ← Saved FAISS indices per config
│   ├── ground_truth.json              ← Curated evaluation queries
│   └── cache/                         ← Response cache
├── results/
│   ├── experiments/                   ← Per-experiment JSON
│   ├── comparison/                    ← Cross-experiment analysis
│   └── charts/                        ← Generated PNGs
├── tests/                             ← pytest suite (≥95% coverage)
└── docs/adr/                          ← Architecture Decision Records
```

---

## 10. ADRs (Distributed Across Days)

> ADRs are written on the day the decision becomes architecturally relevant — not batched at the end.

| ADR | Title | Key Point | Written On |
|-----|-------|-----------|------------|
| ADR-001 | FAISS over ChromaDB for P5 | Low-level control for experiments vs P4's ChromaDB. Different tools for different jobs. | **Day 1** (when FAISS is first implemented) |
| ADR-002 | No LangChain — first-principles RAG | Building from ABCs teaches what LangChain hides. Portfolio signal: can build, not just configure. | **Day 1** (when ABCs are defined) |
| ADR-003 | Hybrid retrieval with min-max score fusion | Why naive combination fails. α optimization. Ensemble principle. | **Day 2** (when hybrid retriever is built) |
| ADR-004 | LiteLLM over raw OpenAI SDK | Multi-provider pattern. Adapter architecture. Spec alignment. | **Day 2** (when LiteLLM generator is built) |
| ADR-005 | YAML + Pydantic for experiment configs | Human-editable with validation. Production ML pattern. | **Day 3** (when experiment configs are defined) |

---

## 11. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| M2 RAM pressure (8GB) | OOM during embedding/indexing | One model at a time. Pool experiments by embedder. del + gc.collect(). Same P2/P3 pattern. |
| PyMuPDF extracts garbage from complex PDFs | Useless chunks | Choose well-structured technical docs. Inspect raw extraction. Add cleaning for headers/footers/ligatures. |
| Embedding-semantic chunker too slow | Blocks Day 1 | Implement heading-semantic first (no embeddings). Embedding-semantic as Day 1 stretch. |
| OpenAI rate limits during 30+ configs | Stalls experiment run | Cache everything. Local models first. OpenAI last. |
| Cohere free tier limits | Not enough rerank calls | Limit reranking to top-5 configs. Local cross-encoder as fallback. |
| LLM judge leniency bias | Inflated scores, false sense of quality | Manual calibration against 5 self-scored answers. Document offset. |
| No config meets retrieval targets | Success criteria unmet | Iterate: try larger chunk overlap, better embedder, reranking. The debugging story itself is valuable. |
| 22h timeline too tight | Incomplete experiments | Days 1-3 = core pipeline. Days 4-5 = experiments + polish. Minimum viable: reduce to 12 configs. |

---

## 12. Session Plan

| Session | Focus | ADRs | Exit Criteria |
|---------|-------|------|---------------|
| Day 1 (~4-5h) | **Foundation:** Project setup, schemas, interfaces, PDF extraction, 5 chunking strategies | ADR-001, ADR-002 | All chunkers produce valid chunks from test PDFs. Tests pass. ≥95% coverage on schemas + chunkers. |
| Day 2 (~6-8h) | **Retrieval Pipeline:** Embedders, FAISS vector store, 3 retrievers, 2 rerankers, LiteLLM generator with citations | ADR-003, ADR-004 | End-to-end query produces cited answer. Smoke test passes. ≥95% coverage on all new components. |
| Day 3 (~4-5h) | **Evaluation:** Retrieval metrics (from scratch), ground truth generation + curation, 5-axis judge, OpenAI embedder, YAML configs, experiment runner | ADR-005 | Evaluation framework complete. 30+ configs defined. Ready for big run. ≥95% coverage on metrics + judge. |
| Day 4 (~4-5h) | **Experiments + Analysis:** Run full grid, generate 10+ charts, comparison report answering 4 required questions, extended experiments | — | All experiments complete. Best configs identified with evidence. |
| Day 5 (~4-5h) | **Polish:** CLI tools, Streamlit demo, README, Loom recording | — | All deliverables complete. P5 DONE. |

---

## 13. Interview Talking Points

1. **"Have you built a production RAG system?"** → P5: full pipeline with 5 chunking strategies, 3 embedding models, dense/BM25/hybrid retrieval, two-stage reranking, LLM generation with citations. All behind abstract interfaces, config-driven experimentation.

2. **"How did you decide which configuration to use?"** → Ran 30+ experiments, measured 4 retrieval metrics + 5 generation quality axes. Config [X] won. Hybrid retrieval at α=0.7 outperformed dense by [Y]%. Reranking added [Z]ms but improved precision by [W]%.

3. **"Why from scratch instead of LangChain?"** → Deliberate first-principles choice. I can use LangChain but wanted to prove I understand the internals. The ABC design means swapping in LangChain components is trivial.

4. **"What surprised you?"** → [Fill after experiments]

5. **"How does this connect to P2?"** → P2 measured RAG. P5 builds it. P2 found the retrieval-generation gap. P5 tackles it with hybrid retrieval + reranking + citations.

6. **"Why FAISS for P5, ChromaDB for P4?"** → Different requirements. P4 needed persistence + metadata filtering for a REST API. P5 needed low-level control for 30+ experiments. Right tool for the job.

7. **"Why LiteLLM?"** → Multi-provider abstraction. Same interface whether I'm calling OpenAI, Anthropic, or Cohere. Production systems need provider flexibility — LiteLLM gives that behind BaseLLM.
