# PRD: P2 — Evaluating RAG for Any PDF

> **This is the implementation contract.** Claude Code: read this + CLAUDE.md before starting each session.
> Do NOT re-debate architecture decisions. They are final. If something is ambiguous, ask the user.

**Project:** P2 — RAG Evaluation Benchmarking Framework
**Timeline:** Feb 12–17, 2026 (Thu–Mon, 5 sessions)
**Owner:** Developer (Java/TS background, learning Python — completed P1)
**Source of Truth:** [Notion Requirements](https://www.notion.so/Mini_Project_2_Requirements-2ffdb630640a81228d16c6e28b48366b)
**Concepts Primer:** `p2-concepts-primer.html` (read for RAG, embeddings, chunking, FAISS, BM25, RAGAS theory)
**PRD Version:** v3 (gap analysis from Notion requirements applied)

---

## 1. Objective

Build a **systematic benchmarking framework** that evaluates which RAG configuration produces the best retrieval and generation quality for a given document.

The pipeline:

1. **Parses** PDF or Markdown documents
2. **Chunks** using multiple configurable strategies — 4 fixed-size configs + 1 semantic (structure-aware) config
3. **Embeds** with multiple models (local + API)
4. **Indexes** into FAISS vector stores (one per config)
5. **Generates** synthetic QA with gold chunk IDs (≥50 questions, 5 strategies, 5 question types)
6. **Evaluates retrieval** — Recall, Precision, MRR at K=1,3,5 across all configs
7. **Compares** against BM25 lexical baseline
8. **Reranks** top configs with Cohere cross-encoder, measures improvement
9. **Evaluates generation** — RAGAS (faithfulness, relevancy, context recall/precision)
10. **Judges** via `judges` library — correctness, hallucination, Bloom taxonomy
11. **Tracks** all experiments in Braintrust with feedback classification
12. **Traces** full pipeline with Logfire for observability
13. **Reports** QA dataset quality metrics (coverage, type distribution, density)
14. **Produces** a data-driven comparison report with charts and heatmaps

**The output is DATA, not a chatbot.** The deliverable is the sentence: _"Config X achieved Y Recall@5 and Z faithfulness, outperforming all other configurations by N%."_ — backed by evidence.

**Success Criterion:** At least one vector search config outperforms BM25 on Recall@5, with the best config identified by data.

---

## 2. Architecture Decisions (FINAL — Do Not Re-Debate)

| Decision                         | Choice                                                                                                          | Rationale                                                                                                                                                                               |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset**                      | Kaggle Enterprise RAG Markdown (1-2 selected reports)                                                           | Realistic enterprise financial data, pre-parsed Markdown. Stronger interview story than single PDF. Keep PDF parser too for extensibility.                                              |
| **Parser**                       | Factory pattern — PyMuPDF for PDF, custom Markdown parser for `.md`                                             | Project is "RAG for Any PDF" — should handle both. Markdown parser preserves header hierarchy for Config E semantic chunking.                                                           |
| **Chunker (fixed-size)**         | `RecursiveCharacterTextSplitter` from `langchain-text-splitters`                                                | Respects sentence boundaries. Configurable. Token-based sizing via `tiktoken`. Used for Configs A–D.                                                                                    |
| **Chunker (semantic)**           | Custom Markdown header splitter — split on `##`/`###` boundaries                                                | Config E: structure-aware chunking. Tests the spec's emphasis on layout-aware chunking without the `unstructured` dependency. Splits on document structure, not arbitrary token counts. |
| **Token counter**                | `tiktoken` as `length_function` for chunker                                                                     | Tokens, not characters — aligns with LLM context windows.                                                                                                                               |
| **Vector store**                 | `faiss-cpu` with `IndexFlatIP`                                                                                  | Low-level control for benchmarking. Brute-force index is correct for <1000 vectors. Normalized embeddings + inner product = cosine similarity.                                          |
| **Local embeddings**             | `sentence-transformers` (MiniLM + mpnet)                                                                        | Free, runs on M2 Neural Engine. Two quality tiers for comparison. Sequential processing for RAM safety.                                                                                 |
| **API embeddings**               | `litellm` for `text-embedding-3-small`                                                                          | Unified interface. One function signature, swap providers by changing a string. Parallelized with `ThreadPoolExecutor` (I/O-bound, safe for RAM).                                       |
| **BM25**                         | `rank-bm25`                                                                                                     | Lexical baseline. The "floor" vector search must beat.                                                                                                                                  |
| **Reranker**                     | `cohere` Rerank API                                                                                             | Cross-encoder reranking. Free tier sufficient for P2 volume.                                                                                                                            |
| **Synthetic QA generation**      | `instructor` + OpenAI GPT-4o-mini                                                                               | Instructor provides auto-retry on Pydantic validation failures — critical for structured QA output. Consistent with P1 pattern.                                                         |
| **LLM generation (RAG answers)** | OpenAI GPT-4o-mini via `openai` SDK                                                                             | Cheap generation for RAG pipeline answers. Raw SDK since RAGAS manages its own LLM calls.                                                                                               |
| **LLM judge**                    | OpenAI GPT-4o (for RAGAS + judges)                                                                              | Higher quality for evaluation tasks.                                                                                                                                                    |
| **RAG evaluation**               | `ragas`                                                                                                         | Faithfulness, Answer Relevancy, Context Recall, Context Precision.                                                                                                                      |
| **LLM-as-Judge**                 | `judges` library (quotient-ai/judges) — NOT `judgy`                                                             | Research-backed judges: RAFTCorrectness, HaluEval, ReliableCIRelevance. Custom BloomTaxonomyClassifier.                                                                                 |
| **Bloom taxonomy**               | Custom judge class extending `judges.base.BaseJudge`                                                            | Neither `judges` nor `judgy` has built-in Bloom classifier. We build one.                                                                                                               |
| **Judge fallback**               | Manual LLM-as-Judge with `openai` structured outputs + Pydantic                                                 | If `judges` library has dependency conflicts with RAGAS/LangChain.                                                                                                                      |
| **`judgy` (ai-evals-course)**    | **Stretch goal only** — statistical bias correction                                                             | Requires human labels on ~30 questions. Adds confidence intervals to judge accuracy. Nice but not core.                                                                                 |
| **Experiment tracking**          | `braintrust`                                                                                                    | Dashboard for comparing configs. Feedback classification (thumbs up/down).                                                                                                              |
| **Observability**                | `logfire`                                                                                                       | OpenTelemetry-based tracing. Built by Pydantic team. Pipeline debugging.                                                                                                                |
| **Caching**                      | JSON file cache keyed on MD5 of (model + prompt)                                                                | Cache ALL LLM responses. Re-run experiments without re-calling APIs. Same pattern as P1.                                                                                                |
| **CLI**                          | `click` + `rich`                                                                                                | `click` for argument parsing, `rich` for tables/progress bars.                                                                                                                          |
| **Visualization**                | `matplotlib` + `seaborn` (+ `plotly` for Streamlit)                                                             | Heatmaps, comparison charts, bar plots.                                                                                                                                                 |
| **Grid size**                    | 5 chunk configs × 3 embedding models = 15 + BM25 = 16 configs                                                   | Configs A–D: fixed-size (controlled experiments). Config E: semantic (structure-aware). B vs D isolates overlap impact. E vs B tests structure-aware vs fixed-size.                     |
| **Memory strategy**              | Local embeddings: sequential. API embeddings: `ThreadPoolExecutor`. Unload models between runs. `gc.collect()`. | 8GB M2 constraint. Local models are RAM-heavy (sequential). API calls are I/O-bound (parallel is safe).                                                                                 |

---

## 3. Grid Search Configuration

### 3a. Chunking Configs

Each config maps to a specific **chunking goal** from the Notion requirements spec. This mapping drives the experimental design — we're testing which chunking philosophy works best for different question types.

| Config |      chunk_size (tokens) | overlap (tokens) | Overlap % | Chunking Goal (from spec)                                   | Why This Config Exists                                                                                                                                                    |
| ------ | -----------------------: | ---------------: | --------: | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A**  |                      128 |               32 |       25% | **Preserve semantic units** (spec: 100–300 tokens)          | Max retrieval granularity — tests if small chunks help factual questions                                                                                                  |
| **B**  |                      256 |               64 |       25% | **Enable dense search** (spec: 256–512 tokens)              | Industry baseline — balanced for most use cases                                                                                                                           |
| **C**  |                      512 |              128 |       25% | **Support long reasoning** (spec: 512–1024 tokens)          | Long-context — tests if bigger chunks help analytical questions                                                                                                           |
| **D**  |                      256 |              128 |       50% | **Maximize retrievability** (spec: high overlap ≈50%)       | **Control experiment** — same size as B, double overlap. Isolates overlap impact.                                                                                         |
| **E**  | Variable (section-based) |                0 |       N/A | **Layout-aware chunking** (spec: "chunk by text structure") | **Semantic chunking** — splits on Markdown headers (`##`/`###`). Tests if document structure outperforms fixed-size. No overlap because splits are at natural boundaries. |

**Config E Implementation Details:**

- Split Markdown on `##` (H2) and `###` (H3) headers
- Each chunk = one section between headers (including the header text)
- If a section exceeds 512 tokens, subdivide with `RecursiveCharacterTextSplitter` at 256/64 (Config B params) to prevent oversized chunks
- If a section is below 32 tokens, merge with the next section
- Chunk IDs: `E_{section_index}` (e.g., `E_0`, `E_1`)
- For PDF inputs where no headers exist: fall back to Config B (logged as a degradation)

**Interview talking point:** "Does structure-aware chunking beat fixed-size? I tested this directly — Config E split on document headers while Configs A-D used fixed token counts. The results showed [X], which means [Y] for production RAG systems."

### 3b. Embedding Models

| Model                    | Dimensions | Where                  | Cost             | Role                             | Parallelization                  |
| ------------------------ | ---------: | ---------------------- | ---------------- | -------------------------------- | -------------------------------- |
| `all-MiniLM-L6-v2`       |        384 | Local (M2)             | Free             | Fast baseline                    | Sequential (RAM-bound)           |
| `all-mpnet-base-v2`      |        768 | Local (M2)             | Free             | Higher quality local             | Sequential (RAM-bound)           |
| `text-embedding-3-small` |       1536 | OpenAI API via LiteLLM | ~$0.02/1M tokens | "Is API quality worth the cost?" | `ThreadPoolExecutor` (I/O-bound) |

**ThreadPoolExecutor for API Embeddings:**

```python
# WHY ThreadPoolExecutor for API calls but NOT local models:
# - Local SentenceTransformers models load ~500MB into RAM. Running multiple
#   in parallel on 8GB would OOM. Sequential is mandatory.
# - OpenAI API calls are I/O-bound (network latency). ThreadPoolExecutor
#   lets us embed multiple chunks concurrently without RAM pressure.
#   The API sends text, receives float arrays — minimal local memory.
#
# Java parallel: ExecutorService + Callable<List<Float>>
# Python parallel: ThreadPoolExecutor + as_completed
# WHY not asyncio: LiteLLM's .embedding() is synchronous. Threads work for I/O.

from concurrent.futures import ThreadPoolExecutor, as_completed

def embed_batch_api(chunks: list[str], model: str, max_workers: int = 8) -> list[list[float]]:
    """Embed chunks via API in parallel. I/O-bound — threads are efficient."""
    embeddings = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(litellm.embedding, model=model, input=[chunk]): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            embeddings[idx] = result.data[0]["embedding"]
    return embeddings
```

### 3c. Full Grid (16 Configs)

Configs 1–15: A/B/C/D/E × MiniLM/mpnet/OpenAI (e.g., `A-minilm`, `B-mpnet`, `C-openai`, `E-minilm`)
Config 16: BM25 baseline using Config B chunks (no embeddings)

### 3d. Evaluation Volume

- 16 configs × 50+ questions × metrics at K=1,3,5 = ~800 retrieval evaluations
- Top-3 configs + reranking = ~300 additional evaluations
- Best config: RAGAS + `judges` evaluation = ~100 generation evaluations
- **All logged to Braintrust**

---

## 4. Data Models (Pydantic Specifications)

> Claude Code: implement these as Pydantic `BaseModel` classes in `src/models.py`. Add appropriate validators.

### Configuration Models

**ChunkConfig:** `chunk_size` (int, >0, ≤2048), `overlap` (int, ≥0, must be < chunk_size), `name` (str), `chunking_goal` (str — maps to spec's chunking purpose), `is_semantic` (bool, default False)

**EmbeddingModel:** Enum with values `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `text-embedding-3-small`

**RetrievalMethod:** Enum — `vector`, `bm25`

### Document Models

**Chunk:** `id` (str, format: `{config_name}_{index}`), `text` (str, min_length=1, not whitespace-only), `token_count` (int, >0), `start_char` (int, ≥0), `end_char` (int, >0), `page_numbers` (list[int]), `config_name` (str), `section_header` (str | None — populated for Config E chunks, None for fixed-size)

### Synthetic QA Models

**QuestionType:** Enum — `factual`, `comparative`, `analytical`, `summarization`, `multi_hop`

**QuestionHierarchy:** Enum — `paragraph` (single chunk), `section` (few chunks), `page` (many chunks)

**SyntheticQAPair:** `id` (str), `question` (str, min_length=10), `question_type` (QuestionType), `hierarchy` (QuestionHierarchy), `gold_chunk_ids` (list[str], min_length=1), `expected_answer` (str), `source_chunk_text` (str), `is_overlap_region` (bool, default False), `generation_strategy` (str — which of the 5 strategies produced this)

### QA Dataset Quality Model

**QADatasetReport:** `total_questions` (int), `questions_per_strategy` (dict[str, int] — count by generation strategy), `questions_per_type` (dict[QuestionType, int] — distribution across 5 types), `questions_per_hierarchy` (dict[QuestionHierarchy, int] — distribution across 3 levels), `chunk_coverage_percent` (float — % of chunks that have at least 1 question), `overlap_question_count` (int), `avg_questions_per_chunk` (float — density metric)

### Retrieval Evaluation Models

**RetrievalResult:** `query_id`, `question`, `question_type`, `gold_chunk_ids`, `retrieved_chunk_ids`, `retrieved_scores`, recall/precision/mrr at 1/3/5 (all float 0–1)

**ConfigEvaluation:** `config_id`, `chunk_config`, `embedding_model`, `retrieval_method`, `num_chunks`, `num_questions`, averaged metrics across all questions, `metrics_by_question_type` (dict breakdown), `individual_results` (list of RetrievalResult)

**RerankingComparison:** before/after metrics for Precision@5, Recall@5, MRR@5, improvement percentages

### Generation Evaluation Models

**RAGASResult:** `config_id`, faithfulness, answer_relevancy, context_recall, context_precision (all float 0–1)

**JudgeResult:** `question_id`, `question`, `generated_answer`, `expected_answer`, `correctness_score` (bool), `has_hallucination` (bool), `relevance_grade` (str: Irrelevant/Related/Highly/Perfectly), `bloom_level` (Bloom enum), reasoning fields for each

**BloomLevel:** Enum — `Remember`, `Understand`, `Apply`, `Analyze`, `Evaluate`, `Create`

### Report Model

**GridSearchReport:** Top-level model rolling up all results — `pdf_name`, `total_configs`, all config evaluations, BM25 baseline, reranking comparisons, RAGAS results, judge results, `best_retrieval_config`, `best_generation_config`, `qa_dataset_report` (QADatasetReport), timestamp, runtime, API cost estimate

---

## 5. Synthetic QA Generation

Generate ≥50 questions from document chunks using GPT-4o-mini **via Instructor** (auto-retry on Pydantic validation failures, consistent with P1 pattern). RAGAS and judges manage their own LLM calls via raw OpenAI SDK.

### 5a. Five Strategies

| #   | Strategy                       | Description                                                                                                                                                                     | Gold Chunk                    | Target Count                 |
| --- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------- |
| 1   | **Per-Chunk Question Chains**  | For each sampled chunk, generate a **3-question chain**: (1) basic factual, (2) deeper analytical, (3) question connecting to other concepts. Ensures type diversity per chunk. | The source chunk              | ~18 questions (6 chunks × 3) |
| 2   | **Multi-Chunk Questions**      | Find semantically similar chunks via embedding cosine similarity, generate questions requiring info from multiple chunks                                                        | Multiple chunk IDs            | ~10 questions                |
| 3   | **Overlap Region Questions**   | Target content at chunk boundaries — the text that exists in the overlap zone                                                                                                   | Chunks sharing the overlap    | ~8 questions                 |
| 4   | **Hierarchical Questions**     | Vary scope: paragraph (1 chunk), section (2-4 chunks), page (5+ chunks)                                                                                                         | Chunk(s) at appropriate scope | ~8 questions                 |
| 5   | **Academic Pattern Questions** | Use real-world question templates (define, compare/contrast, list components, explain process)                                                                                  | Varies                        | ~6 questions                 |

### 5b. Strategy 1 — Per-Chunk Question Chains (Core Strategy)

Based on the spec's "LLM-Generated Question Chains" approach:

```python
# For each sampled chunk, generate 3 progressive questions
chain_prompt = """
Based on this content, generate exactly 3 questions of increasing depth:

1. A basic FACTUAL question (who/what/when/where — directly answerable from the text)
2. A deeper ANALYTICAL question (why/how — requires reasoning about the content)
3. A CONNECTIVE question that relates this content to broader concepts or other topics

Content:
{chunk_text}

For each question, also provide the expected answer based ONLY on the content above.
"""

# Instructor response model — auto-validates structure
class QuestionChain(BaseModel):
    factual: SyntheticQAPair
    analytical: SyntheticQAPair
    connective: SyntheticQAPair

# WHY Instructor here: LLMs sometimes return 2 questions instead of 3,
# or return questions without expected answers. Instructor's auto-retry
# feeds the ValidationError back to the LLM, which self-corrects.
# In P1, this reduced generation failures by ~60%.
```

**Chunk sampling for Strategy 1:** Don't generate for every chunk — sample 6-8 representative chunks spread across the document (beginning, middle, end, varying section topics). This produces 18-24 questions with guaranteed type diversity.

### 5c. Strategy 2 — Multi-Chunk Questions (Implementation Detail)

The spec provides a code hint for finding related chunks. Here's the implementation approach:

```python
def find_semantically_similar_chunks(
    source_chunk: Chunk,
    all_chunks: list[Chunk],
    embeddings: np.ndarray,    # Pre-computed embeddings for all chunks
    source_idx: int,
    top_k: int = 3
) -> list[Chunk]:
    """
    Find chunks semantically similar to source_chunk.

    WHY cosine similarity over Euclidean: Embeddings are normalized (unit vectors).
    For unit vectors, cosine similarity and dot product are equivalent.
    We already normalized for FAISS IndexFlatIP — reuse the same embeddings.

    Java parallel: This is like computing distances in a KNN graph.
    """
    source_vec = embeddings[source_idx].reshape(1, -1)
    # Dot product with all other embeddings (= cosine sim for unit vectors)
    similarities = (embeddings @ source_vec.T).flatten()
    # Exclude self (similarity = 1.0)
    similarities[source_idx] = -1.0
    # Top-k most similar
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [all_chunks[i] for i in top_indices]

# Then generate: "Explain the relationship between {topic_A} and {topic_B}"
# Gold chunk IDs = [source_chunk.id] + [related.id for related in similar_chunks]
```

### 5d. Strategy 5 — Academic Pattern Templates

From the spec's "Real-World Question Patterns":

```python
ACADEMIC_PATTERNS: dict[QuestionType, list[str]] = {
    QuestionType.factual: [
        "What is {concept} as described in the document?",
        "What are the key figures or statistics mentioned about {topic}?",
    ],
    QuestionType.comparative: [
        "How does {concept_a} compare to {concept_b} based on the content?",
        "What are the differences between {approach_a} and {approach_b}?",
    ],
    QuestionType.analytical: [
        "Why does {phenomenon} occur according to the document?",
        "What factors contribute to {outcome} as discussed in the text?",
    ],
    QuestionType.summarization: [
        "What are the main components of {system} as described?",
        "Summarize the key points about {topic} from the document.",
    ],
    QuestionType.multi_hop: [
        "How does {process} described in one section relate to {outcome} in another?",
        "Given {fact_a} and {fact_b}, what conclusion can be drawn?",
    ],
}
```

The LLM fills in `{concept}`, `{topic}`, etc. from the chunk content. These templates ensure questions match patterns that real users ask — not just what an LLM naturally generates (which tends toward overly formal phrasing).

### 5e. QA Dataset Quality Report

After generating all questions, compute and save a `QADatasetReport`:

```python
def compute_qa_quality(questions: list[SyntheticQAPair], total_chunks: int) -> QADatasetReport:
    """
    Measures QA dataset quality. The spec requires 'QA diversity and density.'

    Key metrics:
    - chunk_coverage_percent: % of chunks referenced by at least 1 question.
      Low coverage means blind spots — chunks the retriever is never tested on.
    - Type distribution: Should be roughly even across 5 types.
      Heavily skewed = biased evaluation (e.g., all factual questions favors small chunks).
    - Density: avg questions per chunk. Target: 0.5-1.5 (not too sparse, not redundant).
    """
    # Implementation: group by strategy, type, hierarchy. Count coverage.
```

This report is saved to `results/reports/qa_dataset_report.json` and displayed in the Streamlit app.

### 5f. Requirements

- Each question must have a `gold_chunk_id` (or list of IDs for multi-chunk)
- Each question must have an `expected_answer` (generated from the source chunk text)
- Each question must be classified by `QuestionType` and `QuestionHierarchy`
- Each question must record which `generation_strategy` produced it
- Use Pydantic validation on all generated QA pairs via Instructor (reject malformed ones with auto-retry)
- Cache all LLM calls
- Produce `QADatasetReport` after generation

---

## 6. Evaluation Layers

### Layer 1: Retrieval Metrics (all 16 configs)

For each config, for each question:

- **Recall@K**: Was the gold chunk in the top-K results? (K=1,3,5)
- **Precision@K**: What fraction of top-K were relevant? (K=1,3,5)
- **MRR@K**: Reciprocal rank of first relevant result (K=1,3,5)

Aggregate: average across all questions, breakdown by `QuestionType`.

### Layer 2: BM25 Baseline Comparison

BM25 uses Config B chunks (256/64). Tokenize chunks with `.lower().split()`. Compare BM25 metrics against all 15 vector configs. **At least one vector config must beat BM25 on Recall@5.**

### Layer 3: Reranking (top-3 vector configs)

Take top-3 configs by Recall@5. For each:

- Retrieve top-20 results
- Rerank with Cohere Rerank API → new top-5
- Compute before/after metrics
- Report improvement percentages

### Layer 4: Generation Evaluation (best config only)

Attach LLM (GPT-4o-mini) to best retrieval config. For each question:

- Retrieve top-5 chunks
- Build prompt: "Given this context: {chunks}, answer: {question}"
- Generate answer
- Evaluate with **RAGAS**: faithfulness, answer relevancy, context recall, context precision

### Layer 5: LLM-as-Judge (best config only)

Using `judges` library (quotient-ai/judges):

| Judge                        | Type              | What It Evaluates                      | Import                                                                  |
| ---------------------------- | ----------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| **RAFTCorrectness**          | Classifier (bool) | Is answer correct vs gold answer?      | `from judges.classifiers.correctness import RAFTCorrectness`            |
| **HaluEvalAnswerNonFactual** | Classifier (bool) | Does answer contain hallucinations?    | `from judges.classifiers.hallucination import HaluEvalAnswerNonFactual` |
| **ReliableCIRelevance**      | Grader (4-point)  | How relevant is retrieved context?     | `from judges.graders.relevance import ReliableCIRelevance`              |
| **BloomTaxonomyClassifier**  | Custom classifier | What cognitive level is this question? | Custom class extending `BaseJudge`                                      |
| **Jury**                     | Ensemble          | Average of multiple judges             | `from judges import Jury`                                               |

**Fallback if `judges` has dependency conflicts:** Implement manually with `openai` SDK structured outputs (`client.beta.chat.completions.parse()`) + Pydantic response models. The judge prompts are well-documented in the RAFT and HaluEval papers.

### Layer 6: Experiment Tracking (Braintrust)

- One Braintrust project: `p2-rag-evaluation`
- One experiment per config: `A-minilm`, `B-mpnet`, `E-openai`, etc.
- Log: input (question), output (retrieved chunks or answer), expected (gold chunk IDs), scores (all metrics)
- Add feedback classification: thumbs up/down on QA results

### Layer 7: Observability (Logfire) — Optional

- Instrument key functions with `@logfire.instrument()` or `logfire.span()`
- Track: parsing time, chunking time per config, embedding time per model, search latency, LLM generation time
- Web UI at `logfire.pydantic.dev` for trace inspection

---

## 7. Visualizations to Produce

| Chart                          | Description                                                                                               | File                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **Config × Metric Heatmap**    | Rows: 16 configs (incl. E-\* and BM25). Columns: Recall@1,3,5, Precision@1,3,5, MRR@5. Color: scores      | `results/charts/config_heatmap.png`          |
| **Metric Bar Chart**           | Grouped bars comparing all configs on Recall@5, MRR@5, Precision@5                                        | `results/charts/metric_comparison.png`       |
| **BM25 vs Vector**             | Side-by-side: BM25 scores vs best vector config scores                                                    | `results/charts/bm25_comparison.png`         |
| **Chunk Size Effect**          | Line chart: chunk_size on X, Recall@5 on Y (Configs A/B/C at constant 25% overlap)                        | `results/charts/chunk_size_effect.png`       |
| **Overlap Effect**             | Line chart: Config B (25%) vs Config D (50%) across all metrics                                           | `results/charts/overlap_effect.png`          |
| **Semantic vs Fixed-Size**     | Grouped bar: Config E vs Config B (same embedding model) across all metrics                               | `results/charts/semantic_vs_fixed.png`       |
| **Embedding Model Comparison** | Grouped bar: same chunk config, different embeddings                                                      | `results/charts/embedding_comparison.png`    |
| **Question Type Breakdown**    | Stacked bar: performance by QuestionType per config                                                       | `results/charts/question_type_breakdown.png` |
| **Reranking Before/After**     | Paired bar chart: Precision@5 before vs after reranking                                                   | `results/charts/reranking_impact.png`        |
| **RAGAS Spider Chart**         | Radar chart: 4 RAGAS metrics for best config                                                              | `results/charts/ragas_radar.png`             |
| **Bloom Distribution**         | Pie/bar chart: question distribution across Bloom levels                                                  | `results/charts/bloom_distribution.png`      |
| **QA Dataset Quality**         | Multi-panel: type distribution bar, hierarchy distribution bar, strategy distribution bar, coverage gauge | `results/charts/qa_quality.png`              |

---

## 8. File Structure

```
02-rag-evaluation/
├── CLAUDE.md                          # Project-specific Claude Code context
├── PRD.md                             # THIS FILE — implementation contract
├── pyproject.toml                     # Dependencies
├── src/
│   ├── __init__.py
│   ├── config.py                      # Constants, grid params (5 configs), env vars, paths
│   ├── models.py                      # ALL Pydantic schemas (Section 4) incl. QADatasetReport
│   ├── parser.py                      # Factory: PDF (PyMuPDF) + Markdown parser (preserves headers)
│   ├── chunker.py                     # Fixed-size (RecursiveCharacterTextSplitter × 4) + semantic (header-based Config E)
│   ├── embedder.py                    # Factory: SentenceTransformers (sequential) + LiteLLM (ThreadPoolExecutor)
│   ├── vector_store.py                # FAISS index create/save/load/search + chunk ID mapping
│   ├── bm25_baseline.py               # BM25 retrieval via rank-bm25
│   ├── synthetic_qa.py                # Instructor-based QA generation — 5 strategies + QA quality report
│   ├── retrieval_evaluator.py         # Recall@K, Precision@K, MRR@K computation
│   ├── reranker.py                    # Cohere reranking integration
│   ├── generation_evaluator.py        # RAGAS evaluation wrapper
│   ├── judge.py                       # judges library integration + custom BloomTaxonomy
│   ├── braintrust_logger.py           # Braintrust experiment tracking + feedback
│   ├── observability.py               # Logfire instrumentation (optional)
│   ├── grid_search.py                 # Orchestrator — runs full evaluation matrix
│   ├── cache.py                       # LLM response caching (prompt hash → response)
│   ├── visualization.py               # All charts from Section 7
│   └── cli.py                         # Rich CLI interface (Click + Rich)
├── tests/
│   ├── test_models.py                 # Pydantic validation tests (incl. QADatasetReport)
│   ├── test_chunker.py                # Chunk count, overlap correctness, Config E header splitting
│   ├── test_embedder.py               # Embedding shape, normalization, ThreadPoolExecutor for API
│   ├── test_retrieval_evaluator.py    # Metric computation with known inputs
│   └── test_synthetic_qa.py           # QA validation, strategy distribution, quality report
├── data/
│   ├── input/                         # Source documents (Kaggle MD files or PDFs)
│   ├── cache/                         # LLM response cache (JSON files)
│   └── output/                        # Chunks, QA pairs, FAISS indices
├── results/
│   ├── charts/                        # All PNG visualizations (12 charts)
│   ├── metrics/                       # JSON metric files per config
│   └── reports/                       # GridSearchReport JSON + QADatasetReport JSON
├── docs/
│   └── adr/                           # Architecture Decision Records (5 ADRs)
├── streamlit_app.py                   # Demo app (deployed to Streamlit Cloud)
└── README.md                          # Problem, architecture, results, demo link
```

---

## 9. Dependencies

```
# PDF & Text Processing
PyMuPDF                    # PDF parsing
tiktoken                   # Token counting
langchain-text-splitters   # RecursiveCharacterTextSplitter (NOT full LangChain)

# Embeddings & Vector Search
sentence-transformers      # Local embeddings (MiniLM, mpnet) — sequential processing
litellm                    # Unified API embeddings (OpenAI) — parallelized via ThreadPoolExecutor
faiss-cpu                  # Vector similarity search
rank-bm25                  # BM25 lexical retrieval baseline
numpy                      # Array operations

# Reranking
cohere                     # Cohere Rerank API

# LLM
openai                     # GPT-4o-mini (RAG generation) + GPT-4o (eval)
instructor                 # Structured LLM output for synthetic QA generation (Pydantic auto-retry)

# Evaluation
ragas                      # RAG evaluation framework
judges                     # LLM-as-Judge (quotient-ai/judges) — correctness, hallucination, relevance
braintrust                 # Experiment tracking dashboard

# Observability (optional)
logfire                    # Pipeline tracing (Pydantic team)

# Data
pydantic                   # Runtime validation on all structures

# CLI & Visualization
click                      # CLI argument parsing
rich                       # Beautiful terminal output (tables, progress)
matplotlib                 # Static charts
seaborn                    # Heatmaps
plotly                     # Interactive charts (Streamlit)

# Demo
streamlit                  # Dashboard UI

# Dev
pytest                     # Tests
ruff                       # Linting
python-dotenv              # Environment variables
```

---

## 10. Day-by-Day Implementation Plan

> **Schedule:** P2 runs Feb 12–17. Five sessions: 3 weeknights (4h each) + 1 Sunday deep work (6-8h) + 1 Monday evening (4h).
> **Total budget:** ~22-24h
> **Key constraint:** Process local embedding models sequentially (8GB RAM). API embeddings use ThreadPoolExecutor. Close Chrome during heavy embedding runs.

---

### Day 1 — Foundation: Parse, Chunk, Embed (Thu Feb 12 · 9PM–1AM · 4h)

**Pre-session:** Read concepts primer sections 1-4 (RAG, Embeddings, FAISS, Chunking).

**Claude Code Tasks:**

| #   | Task                          | Module                              | Acceptance Criteria                                                                                                                                                                                                                                                                 |
| --- | ----------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Install all dependencies      | `pyproject.toml`                    | `uv sync` succeeds. All imports work. Including `instructor`.                                                                                                                                                                                                                       |
| 2   | Implement all Pydantic models | `models.py`                         | All models from Section 4 including `QADatasetReport`. Validators reject invalid data. Tests pass.                                                                                                                                                                                  |
| 3   | Implement configuration       | `config.py`                         | 5 ChunkConfigs (A-E with `chunking_goal` annotations), 3 EmbeddingModels, file paths, env vars loaded from `.env`                                                                                                                                                                   |
| 4   | Implement document parser     | `parser.py`                         | Factory pattern. PDF via PyMuPDF extracts text with page numbers. Markdown parser preserves header hierarchy (returns headers list for Config E). Both return `(full_text, page_map, headers)`.                                                                                     |
| 5   | Implement chunker             | `chunker.py`                        | Two strategies: (a) Fixed-size via `RecursiveCharacterTextSplitter` with `tiktoken` length function for Configs A-D. (b) Semantic header-based splitter for Config E — splits on `##`/`###`, merges small sections, subdivides oversized ones. Both return list of `Chunk` objects. |
| 6   | Write tests                   | `test_models.py`, `test_chunker.py` | Valid/invalid Pydantic data. Chunk count sanity (A > D > B, C smallest). Config E produces fewer, larger chunks than Config A. Overlap verified for fixed-size. Header boundaries verified for Config E.                                                                            |

**Checkpoint:** Parse a Kaggle Markdown file. Print chunk counts per config (expect: A most, C fewest, E variable). Verify Config E chunks align with document section headers. Save chunks to `data/output/`.

**Journal entry topics:** What is RAG? How does chunk size affect retrieval? Config E semantic chunking — how does splitting on headers differ from fixed-size? What surprised me about token counts vs character counts?

---

### Day 2 — Embeddings, FAISS, BM25 (Fri Feb 13 · 9PM–1AM · 4h)

**Pre-session:** Read concepts primer sections 5-6 (BM25, Reranking). Understand cosine similarity and `IndexFlatIP`.

**Claude Code Tasks:**

| #   | Task                              | Module             | Acceptance Criteria                                                                                                                                                                                                                                                                                                      |
| --- | --------------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 7   | Implement embedder factory        | `embedder.py`      | `BaseEmbedder` ABC with `.embed()` and `.dimensions`. `SentenceTransformerEmbedder` for local (sequential, batch_size=32). `LiteLLMEmbedder` for API (uses `ThreadPoolExecutor` with max_workers=8). Factory method creates correct embedder from `EmbeddingModel` enum. Normalize all embeddings for cosine similarity. |
| 8   | Implement FAISS vector store      | `vector_store.py`  | Create `IndexFlatIP` index. Add vectors with chunk ID mapping. Search returns top-K chunk IDs + scores. Save/load index to/from disk.                                                                                                                                                                                    |
| 9   | Implement BM25 baseline           | `bm25_baseline.py` | `BM25Okapi` from `rank-bm25`. Tokenize chunks with `.lower().split()`. Search returns top-K chunk IDs + BM25 scores.                                                                                                                                                                                                     |
| 10  | Implement LLM cache               | `cache.py`         | JSON file cache keyed on MD5(model + prompt). Check before API call, save after. Same pattern as P1.                                                                                                                                                                                                                     |
| 11  | Build all 15 FAISS indices + BM25 | Script/CLI         | Process sequentially per local model: load model → embed all 5 chunk configs → save indices → unload → `gc.collect()` → next model. OpenAI embeddings: use ThreadPoolExecutor per chunk config. BM25 on Config B chunks. Total: 15 FAISS indices + 1 BM25.                                                               |
| 12  | Write embedder tests              | `test_embedder.py` | Embedding shape matches expected dimensions. Normalized vectors have unit length. ThreadPoolExecutor produces same results as sequential (order-independent).                                                                                                                                                            |

**Checkpoint:** 16 searchable retrieval backends saved to disk. Manual test: search "What is the total revenue?" → verify retrieved chunks are sensible. Compare Config E retrieval vs Config B on same query — do they return different chunks?

**Journal entry topics:** Embedding dimensions tradeoff. How does cosine similarity work? ThreadPoolExecutor for I/O-bound vs CPU-bound work (Java ExecutorService parallel). First impression of FAISS API.

---

### Day 3 — Synthetic QA + Full Grid Search (Sun Feb 15 · 6-8h deep work)

**Pre-session:** Read concepts primer sections 7-8 (Synthetic QA, Retrieval Metrics).

**Claude Code Tasks:**

| #   | Task                               | Module                                                | Acceptance Criteria                                                                                                                                                                                                                                                                                                                                                                |
| --- | ---------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 13  | Implement synthetic QA generation  | `synthetic_qa.py`                                     | All 5 strategies from Section 5. Strategy 1: 3-question chains via Instructor. Strategy 2: `find_semantically_similar_chunks()` using pre-computed embeddings. Strategy 5: academic pattern templates. Each question has gold_chunk_ids, expected_answer, question_type, hierarchy, generation_strategy. Pydantic validated via Instructor. ≥50 questions total. LLM calls cached. |
| 14  | Implement QA quality report        | `synthetic_qa.py`                                     | Compute `QADatasetReport` after generation: coverage %, type distribution, hierarchy distribution, strategy distribution, density. Save to `results/reports/qa_dataset_report.json`.                                                                                                                                                                                               |
| 15  | Implement retrieval evaluator      | `retrieval_evaluator.py`                              | Recall@K, Precision@K, MRR@K for K=1,3,5. Takes gold_chunk_ids + retrieved_chunk_ids → scores. Aggregate across questions. Breakdown by QuestionType.                                                                                                                                                                                                                              |
| 16  | Implement grid search orchestrator | `grid_search.py`                                      | Loads each of 16 configs. Runs all questions against each. Collects ConfigEvaluation per config. Saves individual results + aggregates.                                                                                                                                                                                                                                            |
| 17  | Implement visualization            | `visualization.py`                                    | Generate heatmap (config × metric), bar charts, semantic vs fixed-size comparison. At minimum: config_heatmap, metric_comparison, bm25_comparison, semantic_vs_fixed.                                                                                                                                                                                                              |
| 18  | Run full grid search               | —                                                     | Execute grid_search.py. Generate first heatmap. Identify top-3 configs. Note where Config E ranks.                                                                                                                                                                                                                                                                                 |
| 19  | Write QA + evaluator tests         | `test_synthetic_qa.py`, `test_retrieval_evaluator.py` | QA passes Pydantic validation. Question chain produces exactly 3 questions per chunk. Known retrieval results produce correct metric values. QA quality report has correct counts.                                                                                                                                                                                                 |

**Checkpoint:** Complete retrieval evaluation for all 16 configs. First heatmap generated. You can identify: best config, worst config, where BM25 wins/loses, **how Config E compares to fixed-size configs**.

**Experiment questions to explore:**

- Does chunk size affect factual vs analytical questions differently?
- Does overlap region fail more?
- Which embedding model wins?
- **Does Config E (semantic) beat Config B (fixed-size) on any metric? On specific question types?**

**Journal entry topics:** Most surprising finding from the grid search. Which config won and why I think so. How BM25 compared. Config E results — did structure-aware chunking matter?

---

### Day 4 — Reranking + RAGAS + Judges + Braintrust (Mon Feb 16 · 9PM–1AM · 4h)

**Pre-session:** Read concepts primer section 9 (RAGAS). Skim `judges` library README.

**Claude Code Tasks:**

| #   | Task                           | Module                    | Acceptance Criteria                                                                                                                                                                                                                                                                                                                                    |
| --- | ------------------------------ | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 20  | Implement reranker             | `reranker.py`             | Cohere Rerank API. Takes query + top-20 chunks → re-scored top-5. Before/after metrics stored as `RerankingComparison`.                                                                                                                                                                                                                                |
| 21  | Implement generation evaluator | `generation_evaluator.py` | RAGAS evaluation on best config. For each question: retrieve → generate answer → score faithfulness, answer_relevancy, context_recall, context_precision. Returns `RAGASResult`.                                                                                                                                                                       |
| 22  | Implement LLM-as-Judge         | `judge.py`                | Integrate `judges` library: RAFTCorrectness, HaluEvalAnswerNonFactual, ReliableCIRelevance. Build custom `BloomTaxonomyClassifier` extending `BaseJudge`. Optionally build `Jury` ensemble. Returns `JudgeResult` per question. **If `judges` has import/dep issues**: implement fallback with `openai` structured outputs + Pydantic response models. |
| 23  | Implement Braintrust logging   | `braintrust_logger.py`    | Create project `p2-rag-evaluation`. Log each config as separate experiment. Log scores, inputs, outputs. Add feedback classification capability.                                                                                                                                                                                                       |
| 24  | Run reranking + evaluation     | —                         | Rerank top-3 configs. Run RAGAS on best. Run judges on best. Log everything to Braintrust.                                                                                                                                                                                                                                                             |

**Checkpoint:** Complete evaluation pipeline. Braintrust dashboard shows all experiments. Reranking improvement quantified. RAGAS scores available. Judge verdicts with reasoning.

**Optional (if time):** Logfire instrumentation (`observability.py`), `judgy` meta-evaluation.

**Journal entry topics:** Reranking impact. Faithfulness score and what it means. What the judges found that RAGAS metrics missed.

---

### Day 5 — CLI, Streamlit, Docs, Deploy (Tue Feb 17 · 9PM–1AM · 4h)

**Claude Code Tasks:**

| #   | Task                 | Module             | Acceptance Criteria                                                                                                                                                             |
| --- | -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 25  | Implement CLI        | `cli.py`           | Click commands: `rag-eval run` (full grid search), `rag-eval report` (show results incl. QA quality report), `rag-eval compare` (specific configs). Rich tables, progress bars. |
| 26  | Build Streamlit app  | `streamlit_app.py` | Upload PDF → show chunking (incl. Config E sections) → run evaluation → display heatmap + charts + RAGAS scores + QA quality report. Interactive config selection.              |
| 27  | Generate all charts  | `visualization.py` | All 12 charts from Section 7 (incl. semantic_vs_fixed and qa_quality). Saved to `results/charts/`.                                                                              |
| 28  | Compile final report | `grid_search.py`   | Build `GridSearchReport` model (incl. `qa_dataset_report`). Save to `results/reports/grid_search_report.json`.                                                                  |
| 29  | Write README         | `README.md`        | Problem, architecture (Mermaid diagram showing 5 chunk configs + 3 embeddings), how to run, results summary with charts, demo link, Loom link.                                  |
| 30  | Write ADRs           | `docs/adr/`        | 5 ADRs (see Section 12).                                                                                                                                                        |

**Deploy:** Push to GitHub. Deploy Streamlit to Streamlit Cloud. Record 2-min Loom walkthrough.

**Journal entry:** Final entry with key findings, Python patterns learned (ThreadPoolExecutor, Instructor for QA, generator functions), what I'd do differently.

---

## 11. Claude Code Session Handoff Protocol

Each session starts by telling Claude Code:

```
Read CLAUDE.md and PRD.md. Today is Day [N].
Here's where I left off: [checkpoint from yesterday]
Focus on tasks [#X through #Y] from the PRD.
```

Each session ends:

1. Git commit and push all work
2. Tell Claude Code to write a journal entry to Notion via MCP
3. Update PRD with any deviations or decisions made during the session

---

## 12. ADRs to Write

| ADR | Title                                                                                         | When                                                                    |
| --- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| 001 | FAISS over ChromaDB/LanceDB for benchmarking                                                  | Day 2 (after building FAISS)                                            |
| 002 | Chunk size selection, controlled overlap experiment, and semantic chunking rationale          | Day 3 (after grid search results — include Config E findings)           |
| 003 | Embedding model comparison — local vs API, cost vs quality, sequential vs parallel            | Day 3 (after grid search results — include ThreadPoolExecutor decision) |
| 004 | Synthetic QA generation — 5 strategies, Instructor for validation, question chain approach    | Day 3 (after QA generation)                                             |
| 005 | Semantic (Config E) vs fixed-size chunking — experimental results and production implications | Day 3 or 5 (after full results available)                               |

---

## 13. What NOT to Build

- No full LangChain framework — only `langchain-text-splitters` for fixed-size chunking
- No LanceDB or Turbopuffer — FAISS only (low-level control for benchmarking)
- No local LLM inference — API-based only (GPT-4o-mini, GPT-4o)
- No parallel LOCAL embedding processing — sequential to manage 8GB RAM (API embeddings use ThreadPoolExecutor)
- No FastAPI endpoint — CLI + Streamlit only (save FastAPI for P5)
- No database (SQLite, Postgres) — JSON/FAISS files only
- No `judgy` library as core dependency — stretch goal only
- No `unstructured` library for layout-aware chunking — Config E uses custom Markdown header splitting instead (lighter, sufficient for our dataset)

---

## 14. Risk Register

| Risk                                 | Impact                                          | Likelihood | Mitigation                                                                                                                                                                   |
| ------------------------------------ | ----------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M2 RAM pressure (8GB)                | OOM loading models + indices                    | High       | Local embeddings sequential. API embeddings via ThreadPoolExecutor (I/O-bound, no RAM risk). Unload between batches. `gc.collect()`. batch_size=32 for SentenceTransformers. |
| OpenAI rate limits                   | QA gen + embedding throttled                    | Medium     | Exponential backoff. Cache all responses. Generate QA once, reuse. ThreadPoolExecutor max_workers=8 (not 50).                                                                |
| RAGAS dependency conflicts           | Version conflicts with LangChain text-splitters | Medium     | Pin versions. Isolated venv if needed.                                                                                                                                       |
| Cohere free tier limits              | Not enough rerank calls                         | Medium     | Limit reranking to top-3 configs. Use open-source cross-encoder as fallback.                                                                                                 |
| `judges` library incompatibility     | Dep conflicts with instructor/ragas             | Medium     | Fallback: manual LLM-as-Judge with `openai` structured outputs + Pydantic.                                                                                                   |
| Config E chunks too variable in size | Some sections might be very long or very short  | Low        | Subdivide >512 tokens (Config B params). Merge <32 tokens. Log distribution in QA quality report.                                                                            |
| Scope creep (24h budget)             | Pushes into P3 timeline                         | Medium     | Priority: Days 1-3 (core retrieval incl. Config E) → Day 4 (evaluation) → Day 5 (polish). Logfire + judgy are stretch goals.                                                 |

---

## 15. Deliverables Checklist

| #   | Deliverable                                                                                | Priority    | Portfolio Signal         |
| --- | ------------------------------------------------------------------------------------------ | ----------- | ------------------------ |
| D1  | RAG pipeline with 15 configurable vector stores (incl. Config E semantic) + BM25           | **Core**    | Systems engineering      |
| D2  | Synthetic QA dataset (≥50 questions, 5 strategies, 5 types, gold chunk IDs) via Instructor | **Core**    | Data engineering         |
| D3  | QA dataset quality report (coverage, type distribution, density)                           | **Core**    | Data quality discipline  |
| D4  | Retrieval metrics: R/P/MRR @1,3,5 for 16 configs                                           | **Core**    | Scientific rigor         |
| D5  | Config × metric heatmap + comparison charts (12 charts)                                    | **Core**    | Data visualization       |
| D6  | BM25 vs vector search comparison                                                           | **Core**    | Baseline discipline      |
| D7  | Semantic vs fixed-size chunking comparison (Config E vs B)                                 | **Core**    | Production RAG insight   |
| D8  | Reranking before/after comparison                                                          | **Core**    | Production optimization  |
| D9  | RAGAS generation evaluation                                                                | **Core**    | End-to-end evaluation    |
| D10 | LLM-as-Judge (correctness, hallucination, Bloom taxonomy)                                  | **Core**    | AI evaluation depth      |
| D11 | Braintrust experiment dashboard                                                            | **Core**    | MLOps maturity           |
| D12 | CLI with Rich output                                                                       | **Core**    | Developer experience     |
| D13 | Streamlit demo on Streamlit Cloud                                                          | **Core**    | Ship mentality           |
| D14 | 5 ADRs (incl. semantic chunking and ThreadPoolExecutor decisions)                          | **Core**    | Technical leadership     |
| D15 | README with Mermaid arch diagram + results                                                 | **Core**    | Communication            |
| D16 | 2-min Loom walkthrough                                                                     | **Core**    | Presentation skills      |
| D17 | Logfire pipeline traces                                                                    | **Stretch** | Production observability |
| D18 | `judgy` bias correction with human labels                                                  | **Stretch** | Statistical rigor        |

---

## 16. Expected Outcomes (Hypotheses to Validate)

| Metric       | BM25 Baseline | Worst Vector | Best Fixed-Size | Config E (Semantic) | Best + Reranking |
| ------------ | :-----------: | :----------: | :-------------: | :-----------------: | :--------------: |
| Recall@5     |   0.55–0.65   |  0.60–0.70   |    0.80–0.90    |    0.75–0.90 ⚡     |    0.85–0.95     |
| Precision@5  |   0.15–0.25   |  0.20–0.30   |    0.30–0.45    |    0.35–0.50 ⚡     |    0.40–0.55     |
| MRR@5        |   0.40–0.55   |  0.50–0.60   |    0.70–0.85    |    0.65–0.85 ⚡     |    0.75–0.90     |
| Faithfulness |      N/A      |  0.60–0.70   |    0.80–0.90    |    0.80–0.92 ⚡     |    0.85–0.95     |

⚡ **Config E hypothesis:** Semantic chunking should have higher Precision (each chunk is a coherent section — less noise) and potentially higher Faithfulness (less fragmented context for the LLM). However, Recall might be lower than small-chunk configs (fewer total chunks = fewer chances to match). This tradeoff is the key finding.

**If your best vector config doesn't beat BM25, that's an interesting finding — document WHY. Negative results are valid in a benchmarking project.**

**If Config E doesn't beat fixed-size configs, that's equally interesting — document what type of questions it helped/hurt and why. This directly informs production RAG decisions.**

---

## 17. Interview Talking Points

**"Tell me about a data-driven technical decision"**
→ "I ran a grid search over 16 RAG configurations — 5 chunking strategies × 3 embedding models + BM25 baseline — including a structure-aware semantic chunking strategy. I found that [config] outperformed all others by X% on Recall@5. Adding Cohere reranking improved Precision@5 by another Y%. Interestingly, semantic chunking [did/didn't] beat fixed-size, which taught me [Z] about production RAG."

**"How do you evaluate AI system quality?"**
→ "Multi-layered: retrieval metrics (Recall, Precision, MRR at K=1,3,5), generation metrics (RAGAS faithfulness, relevancy), LLM-as-Judge with Bloom taxonomy classification, BM25 baseline comparison, experiment tracking in Braintrust. I also measured QA dataset quality — coverage, type distribution, density — to ensure my evaluation wasn't biased."

**"How do you handle tradeoffs?"**
→ "Chunk size vs context quality, overlap vs storage cost, embedding quality vs API dependency, fixed-size vs semantic chunking. I designed controlled experiments — Config B vs D holds chunk size constant and doubles overlap to isolate overlap impact. Config E vs B compares structure-aware vs fixed-size to test if document structure matters more than token count."

**"How did you manage resource constraints?"**
→ "Building on 8GB RAM MacBook Air M2. Local embedding models are RAM-heavy — sequential processing is mandatory. But OpenAI API calls are I/O-bound, so I used ThreadPoolExecutor for parallelism there. It's the same principle as Java's ExecutorService — match the threading model to the bottleneck."
