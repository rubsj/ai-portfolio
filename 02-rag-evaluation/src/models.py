"""Pydantic models for the RAG evaluation benchmarking framework.

Defines ALL data contracts for the pipeline: configuration, document chunks,
synthetic QA, retrieval evaluation, generation evaluation, and final report.

Java/TS parallel: Think of these as TypeScript interfaces with built-in runtime
validation — like zod schemas that enforce types AND business rules at parse time.

Model ordering: enums → leaf models → composite models (bottom-up dependency).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


# ===========================================================================
# Enums — StrEnum for clean JSON serialization (no .value needed)
# Java/TS parallel: TypeScript string enums (`enum Direction { Up = "up" }`)
# WHY StrEnum over Literal: proper enum methods (.name, iteration, membership
# checks) while still serializing to plain strings in JSON output.
# ===========================================================================

class EmbeddingModel(StrEnum):
    """The 3 embedding models in the grid search."""
    MINILM = "all-MiniLM-L6-v2"
    MPNET = "all-mpnet-base-v2"
    OPENAI = "text-embedding-3-small"


class RetrievalMethod(StrEnum):
    """How chunks are retrieved — vector similarity or lexical matching."""
    VECTOR = "vector"
    BM25 = "bm25"


class QuestionType(StrEnum):
    """5 question types for synthetic QA — tests different retrieval strengths."""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"
    MULTI_HOP = "multi_hop"


class QuestionHierarchy(StrEnum):
    """Scope of context needed to answer — from one chunk to many."""
    PARAGRAPH = "paragraph"   # Single chunk sufficient
    SECTION = "section"       # 2-4 chunks needed
    PAGE = "page"             # 5+ chunks needed


class BloomLevel(StrEnum):
    """Bloom's taxonomy cognitive levels — classifies question complexity."""
    REMEMBER = "Remember"
    UNDERSTAND = "Understand"
    APPLY = "Apply"
    ANALYZE = "Analyze"
    EVALUATE = "Evaluate"
    CREATE = "Create"


# ===========================================================================
# Configuration Models
# ===========================================================================

class ChunkConfig(BaseModel):
    """Defines one chunking strategy in the grid search (A through E).

    5 configs test different hypotheses: small vs large chunks, low vs high
    overlap, and fixed-size vs semantic (structure-aware) chunking.
    """

    name: str = Field(
        description="Config identifier (A, B, C, D, or E)",
    )
    chunk_size: int = Field(
        gt=0,
        le=2048,
        description="Target chunk size in tokens",
    )
    overlap: int = Field(
        ge=0,
        description="Token overlap between consecutive chunks",
    )
    chunking_goal: str = Field(
        description="What chunking hypothesis this config tests (from PRD Section 3a)",
    )
    is_semantic: bool = Field(
        default=False,
        description="True for Config E (header-based splitting), False for fixed-size A-D",
    )

    # WHY model_validator instead of field_validator: overlap depends on chunk_size,
    # so we need access to both fields. mode="after" runs after all fields are set.
    # Java/TS parallel: like a class-level @AssertTrue in Bean Validation.
    @model_validator(mode="after")
    def overlap_must_be_less_than_chunk_size(self) -> ChunkConfig:
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"overlap ({self.overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self


# ===========================================================================
# Document Models
# ===========================================================================

class Chunk(BaseModel):
    """A single text chunk produced by the chunker.

    Every chunk gets a unique ID, token count, and character offsets back into
    the original document. Config E chunks also carry their section header.
    """

    id: str = Field(
        description="Unique ID: {config_name}_{index} (e.g., 'B_42', 'E_7')",
    )
    text: str = Field(
        min_length=1,
        description="The chunk's text content",
    )
    token_count: int = Field(
        gt=0,
        description="Number of tokens (tiktoken cl100k_base)",
    )
    start_char: int = Field(
        ge=0,
        description="Character offset where this chunk starts in the source document",
    )
    end_char: int = Field(
        gt=0,
        description="Character offset where this chunk ends in the source document",
    )
    page_numbers: list[int] = Field(
        description="Which page(s) this chunk spans in the source document",
    )
    config_name: str = Field(
        description="Which chunk config produced this (A, B, C, D, or E)",
    )
    section_header: str | None = Field(
        default=None,
        description="Section header text for Config E chunks; None for fixed-size A-D",
    )

    # WHY: LLMs sometimes return whitespace-only strings that pass min_length
    # but carry no semantic content. Catch it here so downstream embedding/search
    # doesn't operate on empty vectors.
    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace_only(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text must contain non-whitespace characters")
        return v

    @model_validator(mode="after")
    def end_must_be_after_start(self) -> Chunk:
        if self.end_char <= self.start_char:
            raise ValueError(
                f"end_char ({self.end_char}) must be greater than start_char ({self.start_char})"
            )
        return self


# ===========================================================================
# Synthetic QA Models
# ===========================================================================

class SyntheticQAPair(BaseModel):
    """One synthetic question-answer pair generated from document chunks.

    Each question is tied to gold chunk IDs (ground truth for retrieval eval),
    classified by type and hierarchy, and tagged with its generation strategy.
    """

    id: str = Field(
        description="Unique question identifier",
    )
    question: str = Field(
        min_length=10,
        description="The generated question",
    )
    question_type: QuestionType = Field(
        description="Factual, comparative, analytical, summarization, or multi_hop",
    )
    hierarchy: QuestionHierarchy = Field(
        description="Scope: paragraph (1 chunk), section (few), page (many)",
    )
    gold_chunk_ids: list[str] = Field(
        min_length=1,
        description="Ground-truth chunk IDs that contain the answer",
    )
    expected_answer: str = Field(
        description="Expected answer derived from the source chunk text",
    )
    source_chunk_text: str = Field(
        description="The source chunk text used to generate this question",
    )
    is_overlap_region: bool = Field(
        default=False,
        description="True if this question targets content in a chunk overlap zone",
    )
    generation_strategy: str = Field(
        description="Which of the 5 strategies produced this (e.g., 'per_chunk_chain')",
    )


class QADatasetReport(BaseModel):
    """Quality metrics for the synthetic QA dataset.

    Measures diversity, coverage, and density to ensure the evaluation
    isn't biased toward certain question types or chunks.
    """

    total_questions: int = Field(
        description="Total number of generated questions",
    )
    questions_per_strategy: dict[str, int] = Field(
        description="Count of questions by generation strategy",
    )
    questions_per_type: dict[QuestionType, int] = Field(
        description="Distribution across the 5 question types",
    )
    questions_per_hierarchy: dict[QuestionHierarchy, int] = Field(
        description="Distribution across paragraph/section/page scopes",
    )
    chunk_coverage_percent: float = Field(
        ge=0.0,
        le=100.0,
        description="Percentage of chunks referenced by at least 1 question",
    )
    overlap_question_count: int = Field(
        description="Number of questions targeting overlap regions",
    )
    avg_questions_per_chunk: float = Field(
        ge=0.0,
        description="Average number of questions per chunk (density metric)",
    )


# ===========================================================================
# Retrieval Evaluation Models
# ===========================================================================

class RetrievalResult(BaseModel):
    """Retrieval evaluation for a single question against one config.

    Compares retrieved chunk IDs against gold chunk IDs to compute
    Recall, Precision, and MRR at K=1,3,5.
    """

    query_id: str = Field(
        description="ID of the SyntheticQAPair being evaluated",
    )
    question: str = Field(
        description="The question text",
    )
    question_type: QuestionType = Field(
        description="Type of question (for per-type breakdown)",
    )
    gold_chunk_ids: list[str] = Field(
        description="Ground-truth chunk IDs",
    )
    retrieved_chunk_ids: list[str] = Field(
        description="Chunk IDs returned by the retriever, in ranked order",
    )
    retrieved_scores: list[float] = Field(
        description="Similarity/relevance scores for each retrieved chunk",
    )
    recall_at_1: float = Field(ge=0.0, le=1.0)
    recall_at_3: float = Field(ge=0.0, le=1.0)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    precision_at_1: float = Field(ge=0.0, le=1.0)
    precision_at_3: float = Field(ge=0.0, le=1.0)
    precision_at_5: float = Field(ge=0.0, le=1.0)
    mrr_at_1: float = Field(ge=0.0, le=1.0)
    mrr_at_3: float = Field(ge=0.0, le=1.0)
    mrr_at_5: float = Field(ge=0.0, le=1.0)


class ConfigEvaluation(BaseModel):
    """Aggregated retrieval metrics for one config across all questions.

    This is the primary unit of comparison in the grid search — each of the
    16 configs gets one ConfigEvaluation.
    """

    config_id: str = Field(
        description="Config identifier (e.g., 'B-mpnet', 'E-openai', 'bm25')",
    )
    chunk_config: str = Field(
        description="Chunk config name (A, B, C, D, E)",
    )
    embedding_model: str = Field(
        description="Embedding model name or 'bm25' for lexical baseline",
    )
    retrieval_method: RetrievalMethod = Field(
        description="Vector or BM25",
    )
    num_chunks: int = Field(
        description="Total number of chunks in this config's index",
    )
    num_questions: int = Field(
        description="Number of questions evaluated",
    )
    # WHY averaged metrics as top-level fields: enables direct sorting/comparison
    # without digging into individual_results. Dashboard-friendly.
    avg_recall_at_1: float = Field(ge=0.0, le=1.0)
    avg_recall_at_3: float = Field(ge=0.0, le=1.0)
    avg_recall_at_5: float = Field(ge=0.0, le=1.0)
    avg_precision_at_1: float = Field(ge=0.0, le=1.0)
    avg_precision_at_3: float = Field(ge=0.0, le=1.0)
    avg_precision_at_5: float = Field(ge=0.0, le=1.0)
    avg_mrr_at_5: float = Field(ge=0.0, le=1.0)
    metrics_by_question_type: dict[QuestionType, dict[str, float]] = Field(
        description="Per-question-type metric breakdown for analysis",
    )
    individual_results: list[RetrievalResult] = Field(
        description="Per-question retrieval results for drill-down",
    )


class RerankingComparison(BaseModel):
    """Before/after metrics for Cohere reranking on one config.

    Measures whether cross-encoder reranking improves retrieval quality
    over the initial vector search results.
    """

    config_id: str = Field(
        description="Which config was reranked",
    )
    precision_at_5_before: float = Field(ge=0.0, le=1.0)
    precision_at_5_after: float = Field(ge=0.0, le=1.0)
    recall_at_5_before: float = Field(ge=0.0, le=1.0)
    recall_at_5_after: float = Field(ge=0.0, le=1.0)
    mrr_at_5_before: float = Field(ge=0.0, le=1.0)
    mrr_at_5_after: float = Field(ge=0.0, le=1.0)
    precision_improvement_pct: float = Field(
        description="Percentage improvement in Precision@5",
    )
    recall_improvement_pct: float = Field(
        description="Percentage improvement in Recall@5",
    )
    mrr_improvement_pct: float = Field(
        description="Percentage improvement in MRR@5",
    )


# ===========================================================================
# Generation Evaluation Models
# ===========================================================================

class RAGASResult(BaseModel):
    """RAGAS evaluation scores for one config's RAG pipeline.

    Measures end-to-end generation quality: does the LLM produce faithful,
    relevant answers given the retrieved context?
    """

    config_id: str = Field(
        description="Which config was evaluated",
    )
    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="Are generated claims supported by the context? (no hallucination)",
    )
    answer_relevancy: float = Field(
        ge=0.0,
        le=1.0,
        description="Is the answer relevant to the question?",
    )
    context_recall: float = Field(
        ge=0.0,
        le=1.0,
        description="Did retrieval find the relevant context?",
    )
    context_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="Is the retrieved context focused (not noisy)?",
    )


class JudgeResult(BaseModel):
    """LLM-as-Judge evaluation for a single question-answer pair.

    Uses judges library (RAFTCorrectness, HaluEval, ReliableCIRelevance)
    plus custom BloomTaxonomyClassifier.
    """

    question_id: str = Field(
        description="ID of the SyntheticQAPair being judged",
    )
    question: str = Field(
        description="The question text",
    )
    generated_answer: str = Field(
        description="The RAG pipeline's generated answer",
    )
    expected_answer: str = Field(
        description="The gold-standard expected answer",
    )
    correctness_score: bool = Field(
        description="RAFTCorrectness: is the answer correct vs gold answer?",
    )
    correctness_reasoning: str = Field(
        description="Why the answer was judged correct or incorrect",
    )
    has_hallucination: bool = Field(
        description="HaluEval: does the answer contain hallucinations?",
    )
    hallucination_reasoning: str = Field(
        description="Explanation of hallucination judgment",
    )
    relevance_grade: str = Field(
        description="ReliableCIRelevance: Irrelevant/Related/Highly/Perfectly",
    )
    relevance_reasoning: str = Field(
        description="Explanation of relevance grade",
    )
    bloom_level: BloomLevel = Field(
        description="Bloom taxonomy classification of the question",
    )
    bloom_reasoning: str = Field(
        description="Why this Bloom level was assigned",
    )


# ===========================================================================
# Report Model — top-level rollup of all results
# ===========================================================================

class GridSearchReport(BaseModel):
    """Top-level report aggregating the entire grid search evaluation.

    This is the final deliverable — a data-driven comparison of all 16 configs
    with retrieval metrics, reranking improvements, RAGAS scores, and judge verdicts.
    """

    pdf_name: str = Field(
        description="Name of the evaluated document",
    )
    total_configs: int = Field(
        description="Number of configs in the grid search (expected: 16)",
    )
    config_evaluations: list[ConfigEvaluation] = Field(
        description="Retrieval evaluation for each of the 16 configs",
    )
    bm25_baseline: ConfigEvaluation | None = Field(
        default=None,
        description="BM25 baseline evaluation (also in config_evaluations)",
    )
    reranking_comparisons: list[RerankingComparison] = Field(
        default_factory=list,
        description="Reranking before/after for top-3 configs",
    )
    ragas_results: list[RAGASResult] = Field(
        default_factory=list,
        description="RAGAS evaluation for best config(s)",
    )
    judge_results: list[JudgeResult] = Field(
        default_factory=list,
        description="LLM-as-Judge results for best config",
    )
    best_retrieval_config: str = Field(
        description="Config ID with highest Recall@5",
    )
    best_generation_config: str | None = Field(
        default=None,
        description="Config ID with highest RAGAS faithfulness (if evaluated)",
    )
    qa_dataset_report: QADatasetReport | None = Field(
        default=None,
        description="Quality metrics for the synthetic QA dataset",
    )
    timestamp: datetime = Field(
        description="When this report was generated",
    )
    total_runtime_seconds: float = Field(
        description="Total wall-clock time for the grid search",
    )
    estimated_api_cost_usd: float = Field(
        ge=0.0,
        description="Estimated total API cost in USD",
    )
