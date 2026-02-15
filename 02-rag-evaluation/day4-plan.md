# P2 Day 4 Execution Plan — Reranking + RAGAS + Judges + Braintrust (Tasks 20-24)

## Context

Day 3 completed the grid search: 16 configs evaluated, 384 tests passing, 95% coverage. Top-3 by Recall@5:
1. **E-openai** (0.625) — semantic chunking + OpenAI embeddings
2. **B-openai** (0.607) — fixed 256/64 + OpenAI embeddings
3. **D-openai** (0.529) — fixed 256/128 + OpenAI embeddings

Day 4 adds evaluation layers on top: reranking (Cohere), generation evaluation (RAGAS), LLM-as-Judge, and Braintrust experiment tracking. Critical path: **20 → 21 → 24**, then 22 → 23.

Out of scope: Logfire, judgy meta-evaluation, jury ensemble.

---

## Implementation Order

| Step | File(s) | Task | Depends On |
|------|---------|------|------------|
| 0 | `src/config.py` | Add `RERANK_RETRIEVAL_TOP_N = 20`, `COHERE_RERANK_MODEL` | — |
| 1 | `src/reranker.py` + `tests/test_reranker.py` | Task 20: Cohere reranker | Step 0 |
| 2 | `src/generation_evaluator.py` + `tests/test_generation_evaluator.py` | Task 21: RAGAS eval | — |
| 3 | `src/judge.py` + `tests/test_judge.py` | Task 22: LLM-as-Judge | — |
| 4 | `src/braintrust_logger.py` + `tests/test_braintrust_logger.py` | Task 23: Experiment tracking | — |
| 5 | `src/reranker.py` (`if __name__`) | Task 24: Orchestration (follows existing runner pattern) | Steps 1-4 |
| 6 | `docs/adr/ADR-005-semantic-vs-fixed-chunking-results.md` | ADR-005 | Step 5 (needs results) |

---

## Step 0: Config Changes (`src/config.py`)

Add after line 159 (`RERANK_TOP_N = 5`):

```python
RERANK_RETRIEVAL_TOP_N = 20   # wider candidate pool for Cohere cross-encoder
COHERE_RERANK_MODEL = "rerank-v3.5"
```

---

## Step 1: Reranker (`src/reranker.py`) — Task 20

### Key Functions

```python
def rerank_chunks(
    query: str,
    chunk_ids: list[str],
    chunk_texts: list[str],
    *, top_n: int = RERANK_TOP_N,
) -> list[tuple[str, float]]:
    """Call Cohere Rerank API. Returns (chunk_id, relevance_score) sorted desc."""

def rerank_config(
    config_id: str,
    qa_pairs: list[SyntheticQAPair],
    store: FAISSVectorStore,
    chunk_lookup: dict[str, Chunk],
    gold_ids_per_question: list[list[str]],
    query_embeddings: np.ndarray,
) -> RerankingComparison:
    """For one config: FAISS top-20 → Cohere rerank → top-5. Compute before/after metrics."""

def run_reranking(
    top_config_ids: list[str],
    qa_pairs: list[SyntheticQAPair],
    chunks_by_config: dict[str, list[Chunk]],
    b_chunks_lookup: dict[str, Chunk],
) -> list[RerankingComparison]:
    """Entry point: rerank top-3 configs. Embeds queries once (all 3 use OpenAI)."""
```

### RerankingComparison Model Fields (from `src/models.py`)
Exact fields to populate: `config_id`, `recall_at_5_before`, `recall_at_5_after`, `precision_at_5_before`, `precision_at_5_after`, `mrr_at_5_before`, `mrr_at_5_after`, `recall_improvement_pct`, `precision_improvement_pct`, `mrr_improvement_pct`.

### Design Details
- FAISS returns top-20 (`RERANK_RETRIEVAL_TOP_N`), Cohere narrows to top-5 (`RERANK_TOP_N`)
- Before-metrics: explicitly compute on `retrieved_ids[:5]` from the FAISS top-20 results (i.e., the first 5 FAISS results, NOT the full top-20). Code comment: `# Before: metrics on FAISS top-5 (retrieved_ids[:5] of the 20 candidates)`
- After-metrics: computed on Cohere's reranked top-5
- Chunk text for Cohere: looked up via `chunk_lookup[chunk_id].text` (FAISS only stores vectors)
- All 3 top configs use OpenAI embeddings → embed queries once, reuse across configs
- Reuse `compute_recall_at_k`, `compute_precision_at_k`, `compute_mrr_at_k` from `src/retrieval_evaluator.py`
- Cache Cohere responses via `src/cache.py`
- Cohere free tier: 168 calls (56 Qs × 3 configs) fits within 1000/month limit

### Tests (`tests/test_reranker.py`)
- Mock `cohere.ClientV2` — verify correct reranking order, top_n respected, empty input handled
- Mock `rerank_chunks` in `rerank_config` — verify before/after metric computation, improvement percentages, division-by-zero safety
- Reuse `_make_chunk` pattern from `tests/test_grid_search.py`

---

## Step 2: Generation Evaluator (`src/generation_evaluator.py`) — Task 21

### Key Functions

```python
def generate_answer(
    question: str, context_chunks: list[str],
    *, model: str = GENERATION_MODEL,
) -> str:
    """Build RAG prompt, call GPT-4o-mini, cache response. Returns answer text."""

def evaluate_with_ragas(
    qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str],
    retrieved_contexts: list[list[str]],
) -> RAGASResult:
    """Run RAGAS evaluate() on all samples. Returns averaged metrics."""

def run_generation_evaluation(
    config_id: str,
    qa_pairs: list[SyntheticQAPair],
    store: FAISSVectorStore,
    chunk_lookup: dict[str, Chunk],
    query_embeddings: np.ndarray,
) -> tuple[RAGASResult, list[str], list[list[str]]]:
    """Full pipeline: retrieve top-5 → generate answers → RAGAS eval.
    Returns (RAGASResult, generated_answers, retrieved_contexts)."""
```

### RAGAS Dependency Conflict Strategy

1. **Primary**: Use RAGAS directly — `ragas>=0.4.3` is in pyproject.toml and imports work
   ```python
   from ragas import evaluate, EvaluationDataset, SingleTurnSample
   from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
   ```
   - Build `SingleTurnSample(user_input, response, retrieved_contexts, reference)` per question
   - Explicitly pass `llm=ChatOpenAI(model="gpt-4o-mini")` to `evaluate()` to control cost (default is gpt-4o — too expensive)
   - Call `evaluate(dataset, metrics, llm=llm, raise_exceptions=False)`

2. **Fallback** (if RAGAS `evaluate()` fails at runtime due to langchain Pydantic V1 issues):
   - Wrap `evaluate()` in try/except
   - Fall back to manual GPT-4o-mini-based evaluation:
     - Faithfulness: extract claims → check support against context
     - Answer relevancy: score 0-1 for how well answer addresses question
     - Context recall: overlap between gold and retrieved chunks (already computed by retrieval metrics)
     - Context precision: GPT-4o-mini scores whether retrieved chunks are relevant
   - **Do NOT downgrade any existing deps** (langchain-text-splitters, instructor)

3. **Implementation pattern**:
   ```python
   try:
       return _evaluate_with_ragas_library(samples)
   except Exception as exc:
       logger.warning("RAGAS failed: %s — using manual fallback", exc)
       return _evaluate_manually(qa_pairs, answers, contexts)
   ```

### RAG Prompt
```
Answer the following question based ONLY on the provided context.
If the context does not contain enough information, say "I don't have enough context."

Context:
{chunk_texts joined by newlines}

Question: {question}

Answer:
```

### Tests (`tests/test_generation_evaluator.py`)
- Mock `OpenAI` for `generate_answer` — verify prompt construction, caching
- Mock `ragas.evaluate` — verify RAGASResult construction from DataFrame
- Test fallback: mock `evaluate` to raise → verify manual path executes
- Test full pipeline with mocked store + OpenAI

---

## Step 3: LLM-as-Judge (`src/judge.py`) — Task 22

### Key Functions

```python
class BloomTaxonomyClassifier(BaseJudge):
    """Custom judge: classifies questions by Bloom's taxonomy level."""
    def judge(self, input, output=None, expected=None) -> Judgment: ...

def evaluate_single_with_judges(
    question: str, generated_answer: str, expected_answer: str, context: str,
) -> JudgeResult:
    """Run all 4 judges on one QA pair. Each judge wrapped in try/except."""

def run_judge_evaluation(
    qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str],
    retrieved_contexts: list[list[str]],
) -> list[JudgeResult]:
    """Run judges on all QA pairs for the best config."""
```

### Judges Library Usage
```python
from judges.classifiers.correctness import RAFTCorrectness
from judges.classifiers.hallucination import HaluEvalAnswerNonFactual
from judges.graders.relevance import ReliableCIRelevance
from judges import BaseJudge

# Model must use provider/model format for instructor.from_provider()
judge = RAFTCorrectness(model="openai/gpt-4o")
judgment = judge.judge(input=question, output=generated_answer, expected=expected_answer)
# judgment.score (bool), judgment.reasoning (str)
```

### Fallback Strategy
If `judges` has import/runtime issues, fall back to `openai` structured outputs:
```python
client.beta.chat.completions.parse(
    model=JUDGE_MODEL,
    messages=[...],
    response_format=_CorrectnessResponse,  # Pydantic model
)
```
Each judge is individually wrapped — one failing doesn't block others.

### Mapping
| Judge | `JudgeResult` field | Score type |
|-------|---------------------|------------|
| RAFTCorrectness | `correctness_score`, `correctness_reasoning` | bool |
| HaluEvalAnswerNonFactual | `has_hallucination`, `hallucination_reasoning` | bool (True = HAS hallucination) |
| ReliableCIRelevance | `relevance_grade`, `relevance_reasoning` | int → string (0=Irrelevant, 1=Related, 2=Highly, 3=Perfectly) |
| BloomTaxonomyClassifier | `bloom_level`, `bloom_reasoning` | BloomLevel enum |

### Tests (`tests/test_judge.py`)
- Mock each judge class — verify JudgeResult construction
- Test Bloom classifier with mocked `_judge`
- Test fallback triggers when judge raises
- Parametrize relevance score → grade mapping

---

## Step 4: Braintrust Logger (`src/braintrust_logger.py`) — Task 23

### Key Functions

```python
def log_retrieval_experiment(config_eval: ConfigEvaluation) -> None:
    """Log one config's retrieval results. Experiment name = config_id."""

def log_generation_experiment(
    config_id: str, qa_pairs: list[SyntheticQAPair],
    generated_answers: list[str], ragas_result: RAGASResult,
    judge_results: list[JudgeResult],
) -> None:
    """Log generation + judge results. Experiment = '{config_id}-generation'."""

def log_reranking_experiment(comparisons: list[RerankingComparison]) -> None:
    """Log reranking before/after. Experiment = 'reranking-comparison'."""
```

### API Pattern
```python
import braintrust
experiment = braintrust.init(project="p2-rag-evaluation", experiment=config_id)
experiment.log(input=question, output=retrieved_ids[:5], expected=gold_ids,
               scores={"recall_at_5": 0.625, ...}, metadata={"question_type": "factual"})
experiment.flush()
```

### Feedback Classification (PRD Layer 6 — required)

Per PRD Section 6a Layer 6 and Task 23 acceptance criteria: "Add feedback classification capability."

```python
def log_feedback(
    experiment_name: str,
    question_id: str,
    thumbs_up: bool,
    *,
    comment: str = "",
) -> None:
    """Log thumbs up/down feedback on a QA result to Braintrust.

    Uses Braintrust's experiment.log() with a 'feedback' score key.
    This enables filtering experiments by human feedback in the dashboard.
    """
```

- Each QA result logged with a `scores={"feedback": 1.0}` (thumbs up) or `scores={"feedback": 0.0}` (thumbs down) field
- Auto-classify during Day 4 run: thumbs up if `correctness_score=True` AND `has_hallucination=False`, thumbs down otherwise
- The function is also callable standalone for manual feedback post-run (e.g., from Streamlit on Day 5)

### Safety
- All Braintrust calls wrapped in try/except — non-fatal
- If `BRAINTRUST_API_KEY` is empty, skip with warning at start
- Results always saved locally to JSON regardless of Braintrust

### Tests (`tests/test_braintrust_logger.py`)
- Mock `braintrust.init` — verify log calls, experiment names
- Test graceful skip when API key is empty
- Test error handling when API is unreachable
- Test `log_feedback` — verify feedback score logged as 1.0/0.0, verify auto-classification logic

---

## Step 5: Orchestration — Task 24

### Runner Pattern (matches existing codebase)

No separate `run_day4.py` — follows the same `if __name__ == "__main__":` pattern used by `grid_search.py`, `index_builder.py`, `synthetic_qa.py`, and `visualization.py`. The orchestration block lives at the bottom of `src/reranker.py` (the primary Day 4 module).

Add this comment directly above the `if __name__` block:
```python
# TODO(Day 5): Extract orchestration into cli.py when building Click commands
```

Run via: `uv run python -m src.reranker`

### Pipeline Flow (inside `if __name__ == "__main__":`)

```python
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Step 1: Load grid_search_results.json → identify top-3 configs
    # Step 2: Load qa_pairs.json (56 QA pairs)
    # Step 3: Load chunks_{A-E}.json → chunk_lookup dicts

    # Step 4: RERANKING (top-3 configs)
    #   - Embed queries once with OpenAI embedder (all 3 use OpenAI)
    #   - For each config: FAISS top-20 → Cohere rerank → top-5
    #   - Save results/metrics/reranking_results.json

    # Step 5: GENERATION + RAGAS (best config = E-openai)
    #   - Retrieve top-5 → generate with GPT-4o-mini → evaluate with RAGAS
    #   - Save results/metrics/ragas_results.json

    # Step 6: JUDGES (best config = E-openai)
    #   - Run 4 judges on each generated answer
    #   - Save results/metrics/judge_results.json

    # Step 7: BRAINTRUST LOGGING
    #   - Log top-3 retrieval experiments
    #   - Log reranking comparison
    #   - Log generation + judge results
    #   - Auto-classify feedback (thumbs up/down) per QA result

    # Step 8: Print summary
```

### Memory Note
All top-3 configs use OpenAI embeddings (API-based, I/O-bound). No local SentenceTransformer model loading needed. Safe for 8GB RAM.

---

## Step 6: ADR-005

**File**: `docs/adr/ADR-005-semantic-vs-fixed-chunking-results.md`

Documents Config E vs Config B experimental results:
- E-openai #1 at 0.625 R@5, beating B-openai (0.607) by 2.9%
- Include reranking impact (did Cohere widen or narrow the gap?)
- Per-question-type breakdown (where does semantic chunking help most?)
- Production recommendation: use semantic chunking when docs have clear structure

---

## Git Workflow

**Branch**: `feat/p2-day4-reranking-ragas-judges`

**Commits** (one per logical unit):
1. `feat(p2): add reranking constants to config`
2. `feat(p2): implement Cohere reranker with before/after metrics`
3. `feat(p2): implement RAGAS generation evaluator with fallback`
4. `feat(p2): implement LLM-as-Judge with Bloom taxonomy classifier`
5. `feat(p2): implement Braintrust experiment logging`
6. `feat(p2): add Day 4 orchestration runner to reranker`
7. `docs(p2): add ADR-005 semantic vs fixed-size chunking results`

**PR**: Single PR to main — "feat(p2): Day 4 reranking, RAGAS, judges, Braintrust (Tasks 20-24)"

---

## Verification

1. **Unit tests**: `uv run pytest tests/test_reranker.py tests/test_generation_evaluator.py tests/test_judge.py tests/test_braintrust_logger.py -v`
2. **Full test suite**: `uv run pytest --cov=src --cov-report=term-missing` — maintain ≥95% coverage
3. **Pipeline run**: `uv run python -m src.reranker` — produces:
   - `results/metrics/reranking_results.json` (3 RerankingComparison)
   - `results/metrics/ragas_results.json` (1 RAGASResult)
   - `results/metrics/judge_results.json` (56 JudgeResult)
4. **Verify outputs**: reranking improvement quantified, RAGAS faithfulness score available, judge verdicts with reasoning present
5. **Braintrust**: Check dashboard at braintrust.com — project `p2-rag-evaluation` shows experiments

---

## Cost Estimate

| Component | Calls | Model | Est. Cost |
|-----------|-------|-------|-----------|
| Cohere rerank | 168 (56×3) | rerank-v3.5 | Free tier |
| OpenAI embed queries | 56 | text-embedding-3-small | ~$0.01 |
| RAG generation | 56 | gpt-4o-mini | ~$0.02 |
| RAGAS evaluation | ~200 internal | gpt-4o-mini (explicit override) | ~$0.03 |
| Judge evaluation | 224 (56×4) | gpt-4o | ~$0.50 |
| **Total** | | | **~$0.57** |

---

## Session-End: Journal Entry (Notion)

Post to Learning Journal DB (`749b828b-4c71-4e62-8761-e69ad1dcda9d`) at session end. Cover:

1. **Reranking impact** — Did Cohere reranking improve Recall@5/Precision@5 for E-openai? By how much? Did it widen or narrow the gap between E-openai and B-openai?
2. **Faithfulness score interpretation** — What does the RAGAS faithfulness number actually mean for our generated answers? Are they grounded in the retrieved context, or hallucinating?
3. **What judges found that RAGAS missed** — RAGAS gives aggregate scores; judges give per-question verdicts. Did the judges flag specific failure modes (hallucinations, incorrect answers, irrelevant responses) that the aggregate RAGAS metrics hid?

Properties: `Project=P2`, `Session Type=Build`, `Phase=Day 4`, `Hours=<actual>`.

---

## Day 4 → Day 5 Handoff Notes

- `# TODO(Day 5): Extract orchestration into cli.py when building Click commands` — the `if __name__` runner in `reranker.py` is a temporary home. Day 5 should consolidate all runners (`grid_search`, `reranker`, `visualization`, etc.) into a Click CLI.
