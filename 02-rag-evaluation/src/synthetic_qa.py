"""Synthetic QA generation — 5 strategies to produce ≥50 diverse questions.

Generates question-answer pairs from Config B chunks using GPT-4o-mini via
Instructor. Each pair has gold_chunk_ids for retrieval evaluation, a question
type/hierarchy classification, and the source chunk text for RAGAS faithfulness.

Java/TS parallel: like a service class with multiple @Strategy beans — each
strategy is a function that samples chunks differently and prompts the LLM with
a different template. Instructor is the validation layer (like Zod + auto-retry).

5 strategies:
  1. Per-Chunk Question Chains — 3 progressive questions per chunk (~24 total)
  2. Multi-Chunk Questions — semantically similar chunks (~10 total)
  3. Overlap Region Questions — chunk boundary content (~8 total)
  4. Hierarchical Questions — varying scope (~8 total)
  5. Academic Pattern Questions — real-world templates (~6 total)
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import Counter
from pathlib import Path

import faiss
import instructor
import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field

from src.cache import compute_cache_key, load_cached, save_cached
from src.config import (
    GENERATION_MODEL,
    INDICES_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
)
from src.models import (
    Chunk,
    QADatasetReport,
    QuestionHierarchy,
    QuestionType,
    SyntheticQAPair,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Instructor Response Models (internal — not in models.py)
# WHY separate: these are LLM response schemas for Instructor validation,
# not domain models. They exist only to parse/validate LLM output before
# converting to SyntheticQAPair.
# ===========================================================================

class _QAPairResponse(BaseModel):
    """Single Q/A pair from the LLM."""
    question: str = Field(min_length=10, description="The question")
    answer: str = Field(min_length=10, description="Expected answer from the text")


class _QuestionChainResponse(BaseModel):
    """3 progressive questions from Strategy 1."""
    factual: _QAPairResponse = Field(description="Basic factual question (who/what/when/where)")
    analytical: _QAPairResponse = Field(description="Deeper analytical question (why/how)")
    connective: _QAPairResponse = Field(description="Connects to broader concepts")


class _MultiChunkQuestionResponse(BaseModel):
    """Single Q/A requiring info from multiple chunks (Strategy 2)."""
    question: str = Field(min_length=10, description="Question requiring info from ALL chunks")
    answer: str = Field(min_length=10, description="Expected answer synthesizing all chunks")


class _SingleQuestionResponse(BaseModel):
    """Reusable single Q/A response for Strategies 3, 4, 5."""
    question: str = Field(min_length=10, description="The question")
    answer: str = Field(min_length=10, description="Expected answer from the text")


# ===========================================================================
# Instructor Client + Caching
# ===========================================================================

def _create_client() -> instructor.Instructor:
    """Create an Instructor-wrapped OpenAI client.

    WHY instructor.from_openai: wraps the OpenAI client to auto-validate
    responses against Pydantic models. On validation failure, feeds the error
    back to the LLM for self-correction (up to max_retries).
    Same pattern as P1 generator.py.
    """
    return instructor.from_openai(OpenAI())


def _cached_instructor_call(
    client: instructor.Instructor,
    prompt: str,
    response_model: type[BaseModel],
    *,
    use_cache: bool = True,
) -> BaseModel:
    """Call Instructor with caching — check cache first, save after.

    WHY cache: each LLM call costs ~$0.001 and takes ~2s. Over 56 questions,
    caching saves ~$0.05 and ~2 minutes on re-runs. More importantly, it
    makes development iteration fast — change downstream code without
    re-generating all QA pairs.
    """
    cache_key = compute_cache_key(GENERATION_MODEL, prompt)

    if use_cache:
        cached = load_cached(cache_key)
        if cached is not None:
            return response_model.model_validate(cached)

    result = client.chat.completions.create(
        model=GENERATION_MODEL,
        response_model=response_model,
        messages=[{"role": "user", "content": prompt}],
        max_retries=3,
    )

    # WHY model_dump(mode="json"): Pydantic v2 serialization to JSON-compatible
    # dict. mode="json" ensures enums become strings, datetimes become ISO strings.
    if use_cache:
        save_cached(
            cache_key,
            result.model_dump(mode="json"),
            model=GENERATION_MODEL,
        )

    return result


# ===========================================================================
# Chunk Sampling Helpers
# ===========================================================================

def _sample_diverse_chunks(chunks: list[Chunk], n: int = 8) -> list[Chunk]:
    """Sample n chunks spread across documents and positions.

    WHY spread sampling: uniform random might cluster all chunks in one doc
    or one section. We want beginning/middle/end coverage across all 3 docs
    so questions test retrieval from different parts of the corpus.

    Distribution: 3 + 3 + 2 chunks across 3 docs.
    """
    # Group chunks by document (doc_idx is the middle segment of "B_0_42")
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        parts = chunk.id.split("_")
        doc_idx = parts[1]
        by_doc.setdefault(doc_idx, []).append(chunk)

    doc_ids = sorted(by_doc.keys())
    # WHY per_doc allocation: [3, 3, 2] distributes evenly across 3 docs.
    # If fewer docs, allocate more per doc.
    per_doc = [n // len(doc_ids)] * len(doc_ids)
    for i in range(n % len(doc_ids)):
        per_doc[i] += 1

    sampled: list[Chunk] = []
    for doc_id, count in zip(doc_ids, per_doc):
        doc_chunks = by_doc[doc_id]
        if count >= len(doc_chunks):
            sampled.extend(doc_chunks[:count])
            continue

        # WHY linspace: picks indices at evenly spaced positions (beginning,
        # middle, end) instead of random clustering in one section.
        indices = np.linspace(0, len(doc_chunks) - 1, count, dtype=int)
        for idx in indices:
            sampled.append(doc_chunks[idx])

    return sampled[:n]


def _find_overlap_pairs(chunks: list[Chunk]) -> list[tuple[Chunk, Chunk]]:
    """Find consecutive chunk pairs from the same doc with overlapping text.

    WHY overlap detection: Config B has 64-token overlap. Questions about
    overlap zones test whether the retriever can find content that exists
    in multiple chunks — a known failure mode for naive retrieval.
    """
    pairs: list[tuple[Chunk, Chunk, int]] = []

    for i in range(len(chunks) - 1):
        c1, c2 = chunks[i], chunks[i + 1]

        # WHY same-doc check: chunks from different docs can't have meaningful overlap
        doc1 = c1.id.split("_")[1]
        doc2 = c2.id.split("_")[1]
        if doc1 != doc2:
            continue

        overlap = max(0, min(c1.end_char, c2.end_char) - max(c1.start_char, c2.start_char))
        if overlap > 0:
            pairs.append((c1, c2, overlap))

    # WHY sort by overlap descending: largest overlaps have the most shared
    # content, making better targets for questions about boundary content.
    pairs.sort(key=lambda x: x[2], reverse=True)
    return [(c1, c2) for c1, c2, _ in pairs]


def _load_precomputed_embeddings() -> tuple[np.ndarray, list[str]]:
    """Load pre-computed embeddings from minilm_B FAISS index.

    WHY reconstruct from FAISS: IndexFlatIP stores full vectors (no
    compression). reconstruct_n extracts exact embeddings — no information
    loss. Avoids loading a SentenceTransformer model just for Strategy 2.
    """
    index_path = INDICES_DIR / "minilm_B.faiss"
    index = faiss.read_index(str(index_path))
    embeddings = np.zeros((index.ntotal, index.d), dtype=np.float32)
    index.reconstruct_n(0, index.ntotal, embeddings)

    ids_path = INDICES_DIR / "minilm_B.json"
    chunk_ids: list[str] = json.loads(ids_path.read_text())

    logger.info("Loaded %d embeddings (%dd) from minilm_B", len(chunk_ids), index.d)
    return embeddings, chunk_ids


def _find_semantically_similar_chunks(
    source_idx: int,
    embeddings: np.ndarray,
    chunk_ids: list[str],
    top_k: int = 3,
) -> list[int]:
    """Find top_k semantically similar chunk indices via dot product.

    WHY dot product: embeddings are L2-normalized (FAISS IndexFlatIP requirement).
    For unit vectors, dot product = cosine similarity. No need for separate
    normalization step.
    """
    source_vec = embeddings[source_idx].reshape(1, -1)
    similarities = (embeddings @ source_vec.T).flatten()
    # WHY exclude self: self-similarity is always 1.0, not useful
    similarities[source_idx] = -1.0
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return list(top_indices)


# ===========================================================================
# Strategy 1 — Per-Chunk Question Chains (~24 questions)
# ===========================================================================

def _strategy_per_chunk_chain(
    chunks: list[Chunk],
    client: instructor.Instructor,
    *,
    use_cache: bool = True,
) -> list[SyntheticQAPair]:
    """Generate 3 progressive questions per sampled chunk.

    Sample 8 diverse chunks → 3 questions each = 24 questions.
    Types: factual → analytical → connective (multi_hop).
    """
    sampled = _sample_diverse_chunks(chunks, n=8)
    pairs: list[SyntheticQAPair] = []

    for chunk in sampled:
        prompt = f"""Based on this content, generate exactly 3 questions of increasing depth:

1. A basic FACTUAL question (who/what/when/where — directly answerable from the text)
2. A deeper ANALYTICAL question (why/how — requires reasoning about the content)
3. A CONNECTIVE question that relates this content to broader concepts or other topics mentioned

Content:
{chunk.text}

For each question, also provide the expected answer based ONLY on the content above.
Do NOT use generic questions. Make them specific to the actual facts and details in the content."""

        response = _cached_instructor_call(
            client, prompt, _QuestionChainResponse, use_cache=use_cache,
        )

        # WHY 3 separate SyntheticQAPair: each has different question_type and
        # hierarchy, enabling per-type metric breakdowns in the evaluator.
        type_map = [
            (response.factual, QuestionType.FACTUAL, QuestionHierarchy.PARAGRAPH),
            (response.analytical, QuestionType.ANALYTICAL, QuestionHierarchy.PARAGRAPH),
            (response.connective, QuestionType.MULTI_HOP, QuestionHierarchy.SECTION),
        ]
        for qa, q_type, hierarchy in type_map:
            pairs.append(SyntheticQAPair(
                id=f"s1_{uuid.uuid4().hex[:8]}",
                question=qa.question,
                question_type=q_type,
                hierarchy=hierarchy,
                gold_chunk_ids=[chunk.id],
                expected_answer=qa.answer,
                source_chunk_text=chunk.text,
                generation_strategy="per_chunk_chain",
            ))

    logger.info("Strategy 1 (per_chunk_chain): generated %d questions", len(pairs))
    return pairs


# ===========================================================================
# Strategy 2 — Multi-Chunk Questions (~10 questions)
# ===========================================================================

def _strategy_multi_chunk(
    chunks: list[Chunk],
    client: instructor.Instructor,
    *,
    use_cache: bool = True,
) -> list[SyntheticQAPair]:
    """Generate questions requiring info from multiple semantically similar chunks.

    Load pre-computed embeddings from FAISS, find similar chunks, prompt LLM
    to generate questions that require synthesizing info from all chunks.
    """
    embeddings, chunk_ids = _load_precomputed_embeddings()
    chunk_lookup = {c.id: c for c in chunks}

    # WHY sample 10: each produces 1 multi-chunk question = 10 total
    source_indices = np.linspace(0, len(chunk_ids) - 1, 10, dtype=int)
    pairs: list[SyntheticQAPair] = []

    for src_idx in source_indices:
        src_id = chunk_ids[int(src_idx)]
        src_chunk = chunk_lookup.get(src_id)
        if src_chunk is None:
            continue

        similar_indices = _find_semantically_similar_chunks(
            int(src_idx), embeddings, chunk_ids, top_k=3,
        )
        similar_chunks = [
            chunk_lookup[chunk_ids[i]]
            for i in similar_indices
            if chunk_ids[i] in chunk_lookup
        ]

        if not similar_chunks:
            continue

        all_texts = [src_chunk.text] + [c.text for c in similar_chunks]
        combined = "\n---\n".join(all_texts)

        prompt = f"""You are given multiple related text chunks from a document.
Generate ONE question that requires information from ALL of the chunks below to answer.
The question should not be answerable from any single chunk alone.

Chunks:
{combined}

Provide the question and an expected answer that synthesizes information from all chunks."""

        response = _cached_instructor_call(
            client, prompt, _MultiChunkQuestionResponse, use_cache=use_cache,
        )

        gold_ids = [src_id] + [c.id for c in similar_chunks]
        pairs.append(SyntheticQAPair(
            id=f"s2_{uuid.uuid4().hex[:8]}",
            question=response.question,
            question_type=QuestionType.MULTI_HOP,
            hierarchy=QuestionHierarchy.SECTION,
            gold_chunk_ids=gold_ids,
            expected_answer=response.answer,
            source_chunk_text=combined,
            generation_strategy="multi_chunk",
        ))

    logger.info("Strategy 2 (multi_chunk): generated %d questions", len(pairs))
    return pairs


# ===========================================================================
# Strategy 3 — Overlap Region Questions (~8 questions)
# ===========================================================================

def _strategy_overlap_region(
    chunks: list[Chunk],
    client: instructor.Instructor,
    *,
    use_cache: bool = True,
) -> list[SyntheticQAPair]:
    """Generate questions about content in chunk overlap zones.

    WHY test overlaps: overlapping text exists in two chunks. If the retriever
    returns the wrong one (or neither), it reveals boundary handling issues.
    """
    overlap_pairs = _find_overlap_pairs(chunks)[:8]
    pairs: list[SyntheticQAPair] = []

    for c1, c2 in overlap_pairs:
        # WHY extract overlap text: the shared region is the most interesting
        # part to question — it tests whether retrieval finds boundary content.
        overlap_start = max(c1.start_char, c2.start_char)
        overlap_end = min(c1.end_char, c2.end_char)

        # WHY use chunk text: we need the overlap text from the chunk content,
        # not character offsets into the original doc. Compute relative positions.
        c1_relative_start = overlap_start - c1.start_char
        c1_relative_end = overlap_end - c1.start_char
        overlap_text = c1.text[c1_relative_start:c1_relative_end]

        if len(overlap_text.strip()) < 20:
            continue

        prompt = f"""Generate a specific question about the following text content.
This text appears at the boundary between two sections of a document.

Content:
{overlap_text}

Generate a factual question that can be answered using ONLY this content.
Also provide the expected answer."""

        response = _cached_instructor_call(
            client, prompt, _SingleQuestionResponse, use_cache=use_cache,
        )

        pairs.append(SyntheticQAPair(
            id=f"s3_{uuid.uuid4().hex[:8]}",
            question=response.question,
            question_type=QuestionType.FACTUAL,
            hierarchy=QuestionHierarchy.PARAGRAPH,
            gold_chunk_ids=[c1.id, c2.id],
            expected_answer=response.answer,
            source_chunk_text=overlap_text,
            is_overlap_region=True,
            generation_strategy="overlap_region",
        ))

    logger.info("Strategy 3 (overlap_region): generated %d questions", len(pairs))
    return pairs


# ===========================================================================
# Strategy 4 — Hierarchical Questions (~8 questions)
# ===========================================================================

def _strategy_hierarchical(
    chunks: list[Chunk],
    client: instructor.Instructor,
    *,
    use_cache: bool = True,
) -> list[SyntheticQAPair]:
    """Generate questions at paragraph, section, and page scope.

    WHY vary scope: small-chunk configs (A) should excel at paragraph questions,
    large-chunk configs (C) at page questions. This tests whether the chunking
    strategy matches the question granularity.
    """
    pairs: list[SyntheticQAPair] = []

    # Group chunks by document
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        doc_idx = chunk.id.split("_")[1]
        by_doc.setdefault(doc_idx, []).append(chunk)

    doc_ids = sorted(by_doc.keys())

    # --- Paragraph scope (3 questions, 1 chunk each) ---
    for doc_id in doc_ids:
        doc_chunks = by_doc[doc_id]
        mid = len(doc_chunks) // 2
        chunk = doc_chunks[mid]

        prompt = f"""Generate a detailed factual question about the following content.
The question should be answerable from this single paragraph of text.

Content:
{chunk.text}

Provide the question and expected answer based ONLY on this content."""

        response = _cached_instructor_call(
            client, prompt, _SingleQuestionResponse, use_cache=use_cache,
        )
        pairs.append(SyntheticQAPair(
            id=f"s4_para_{uuid.uuid4().hex[:8]}",
            question=response.question,
            question_type=QuestionType.FACTUAL,
            hierarchy=QuestionHierarchy.PARAGRAPH,
            gold_chunk_ids=[chunk.id],
            expected_answer=response.answer,
            source_chunk_text=chunk.text,
            generation_strategy="hierarchical",
        ))

    # --- Section scope (3 questions, 2-4 consecutive chunks) ---
    for doc_id in doc_ids:
        doc_chunks = by_doc[doc_id]
        start = len(doc_chunks) // 3
        section = doc_chunks[start:start + 3]
        combined = "\n---\n".join(c.text for c in section)

        prompt = f"""Generate an analytical question about the following section of text.
The question should require understanding of ALL the paragraphs below, not just one.

Section:
{combined}

Provide the question and expected answer that synthesizes information across these paragraphs."""

        response = _cached_instructor_call(
            client, prompt, _SingleQuestionResponse, use_cache=use_cache,
        )
        pairs.append(SyntheticQAPair(
            id=f"s4_sect_{uuid.uuid4().hex[:8]}",
            question=response.question,
            question_type=QuestionType.ANALYTICAL,
            hierarchy=QuestionHierarchy.SECTION,
            gold_chunk_ids=[c.id for c in section],
            expected_answer=response.answer,
            source_chunk_text=combined,
            generation_strategy="hierarchical",
        ))

    # --- Page scope (2 questions, 5+ consecutive chunks) ---
    for doc_id in doc_ids[:2]:
        doc_chunks = by_doc[doc_id]
        page = doc_chunks[:6]
        combined = "\n---\n".join(c.text for c in page)

        prompt = f"""Generate a broad summarization question about the following page of content.
The question should require understanding the overall themes and key points across ALL sections.

Page content:
{combined}

Provide the question and a comprehensive expected answer."""

        response = _cached_instructor_call(
            client, prompt, _SingleQuestionResponse, use_cache=use_cache,
        )
        pairs.append(SyntheticQAPair(
            id=f"s4_page_{uuid.uuid4().hex[:8]}",
            question=response.question,
            question_type=QuestionType.SUMMARIZATION,
            hierarchy=QuestionHierarchy.PAGE,
            gold_chunk_ids=[c.id for c in page],
            expected_answer=response.answer,
            source_chunk_text=combined,
            generation_strategy="hierarchical",
        ))

    logger.info("Strategy 4 (hierarchical): generated %d questions", len(pairs))
    return pairs


# ===========================================================================
# Strategy 5 — Academic Pattern Questions (~6 questions)
# ===========================================================================

# WHY template patterns: ensures questions match real-world patterns that
# users actually ask (define, compare, explain, summarize) rather than
# overly formal LLM-generated phrasing.
ACADEMIC_PATTERNS: dict[QuestionType, list[str]] = {
    QuestionType.FACTUAL: [
        "What is {concept} as described in the document?",
        "What are the key figures or statistics mentioned about {topic}?",
    ],
    QuestionType.COMPARATIVE: [
        "How does {concept_a} compare to {concept_b} based on the content?",
    ],
    QuestionType.ANALYTICAL: [
        "Why does {phenomenon} occur according to the document?",
    ],
    QuestionType.SUMMARIZATION: [
        "What are the main components of {system} as described?",
    ],
    QuestionType.MULTI_HOP: [
        "How does {process} described in one section relate to {outcome} in another?",
    ],
}


def _strategy_academic_pattern(
    chunks: list[Chunk],
    client: instructor.Instructor,
    *,
    use_cache: bool = True,
) -> list[SyntheticQAPair]:
    """Generate questions using academic pattern templates.

    The LLM fills in {concept}, {topic} slots from chunk content.
    One question per pattern = 6 questions.
    """
    sampled = _sample_diverse_chunks(chunks, n=6)
    pairs: list[SyntheticQAPair] = []

    # WHY flatten patterns: iterate through all patterns in order, pair each
    # with one sampled chunk. Ensures 1 question per pattern type.
    flat_patterns: list[tuple[QuestionType, str]] = []
    for q_type, templates in ACADEMIC_PATTERNS.items():
        for template in templates:
            flat_patterns.append((q_type, template))

    for chunk, (q_type, template) in zip(sampled, flat_patterns[:len(sampled)]):
        prompt = f"""You are given a text chunk and a question template.
Fill in the template placeholders ({{concept}}, {{topic}}, etc.) using specific
details from the text, then provide the expected answer.

Text:
{chunk.text}

Question template: {template}

Return the filled-in question and expected answer based on the text."""

        response = _cached_instructor_call(
            client, prompt, _SingleQuestionResponse, use_cache=use_cache,
        )

        hierarchy = (
            QuestionHierarchy.SECTION if q_type == QuestionType.MULTI_HOP
            else QuestionHierarchy.PARAGRAPH
        )
        pairs.append(SyntheticQAPair(
            id=f"s5_{uuid.uuid4().hex[:8]}",
            question=response.question,
            question_type=q_type,
            hierarchy=hierarchy,
            gold_chunk_ids=[chunk.id],
            expected_answer=response.answer,
            source_chunk_text=chunk.text,
            generation_strategy="academic_pattern",
        ))

    logger.info("Strategy 5 (academic_pattern): generated %d questions", len(pairs))
    return pairs


# ===========================================================================
# Public API
# ===========================================================================

def generate_synthetic_qa(
    chunks: list[Chunk],
    *,
    use_cache: bool = True,
) -> list[SyntheticQAPair]:
    """Generate synthetic QA pairs from Config B chunks using all 5 strategies.

    Args:
        chunks: Config B chunk list (loaded from data/output/chunks_B.json).
        use_cache: Whether to use LLM response caching.

    Returns:
        List of SyntheticQAPair objects (target: ≥50 questions).
    """
    client = _create_client()

    all_pairs: list[SyntheticQAPair] = []

    logger.info("Starting synthetic QA generation from %d chunks", len(chunks))

    # Strategy 1: Per-Chunk Question Chains (~24 questions)
    all_pairs.extend(
        _strategy_per_chunk_chain(chunks, client, use_cache=use_cache)
    )

    # Strategy 2: Multi-Chunk Questions (~10 questions)
    all_pairs.extend(
        _strategy_multi_chunk(chunks, client, use_cache=use_cache)
    )

    # Strategy 3: Overlap Region Questions (~8 questions)
    all_pairs.extend(
        _strategy_overlap_region(chunks, client, use_cache=use_cache)
    )

    # Strategy 4: Hierarchical Questions (~8 questions)
    all_pairs.extend(
        _strategy_hierarchical(chunks, client, use_cache=use_cache)
    )

    # Strategy 5: Academic Pattern Questions (~6 questions)
    all_pairs.extend(
        _strategy_academic_pattern(chunks, client, use_cache=use_cache)
    )

    logger.info("Total QA pairs generated: %d", len(all_pairs))
    return all_pairs


def compute_qa_quality(
    questions: list[SyntheticQAPair],
    total_chunks: int,
) -> QADatasetReport:
    """Compute quality metrics for the synthetic QA dataset.

    WHY quality report: ensures the evaluation isn't biased toward certain
    question types, strategies, or chunks. Low coverage means blind spots.
    """
    # WHY Counter: clean Pythonic way to group and count — like Java's
    # Collectors.groupingBy(x -> x, Collectors.counting()).
    by_strategy = Counter(q.generation_strategy for q in questions)
    by_type = Counter(q.question_type for q in questions)
    by_hierarchy = Counter(q.hierarchy for q in questions)

    # WHY set comprehension: unique gold chunk IDs across all questions.
    # A chunk referenced by 3 questions is counted once for coverage.
    unique_gold_ids: set[str] = set()
    for q in questions:
        unique_gold_ids.update(q.gold_chunk_ids)

    coverage = (len(unique_gold_ids) / total_chunks * 100) if total_chunks > 0 else 0.0
    overlap_count = sum(1 for q in questions if q.is_overlap_region)
    avg_per_chunk = len(questions) / total_chunks if total_chunks > 0 else 0.0

    return QADatasetReport(
        total_questions=len(questions),
        questions_per_strategy=dict(by_strategy),
        questions_per_type=dict(by_type),
        questions_per_hierarchy=dict(by_hierarchy),
        chunk_coverage_percent=round(coverage, 1),
        overlap_question_count=overlap_count,
        avg_questions_per_chunk=round(avg_per_chunk, 3),
    )


def save_qa_pairs(
    pairs: list[SyntheticQAPair],
    path: Path | None = None,
) -> Path:
    """Save QA pairs to JSON.

    WHY save as JSON: other modules (grid_search, streamlit) need to load
    QA pairs without re-running the LLM. JSON is human-readable for debugging.
    """
    if path is None:
        path = OUTPUT_DIR / "qa_pairs.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [pair.model_dump(mode="json") for pair in pairs]
    path.write_text(json.dumps(data, indent=2))
    logger.info("Saved %d QA pairs to %s", len(pairs), path)
    return path


def load_qa_pairs(path: Path | None = None) -> list[SyntheticQAPair]:
    """Load QA pairs from JSON."""
    if path is None:
        path = OUTPUT_DIR / "qa_pairs.json"

    data = json.loads(path.read_text())
    return [SyntheticQAPair.model_validate(item) for item in data]


def save_qa_report(report: QADatasetReport, path: Path | None = None) -> Path:
    """Save QA quality report to JSON."""
    if path is None:
        path = REPORTS_DIR / "qa_dataset_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(report.model_dump_json(indent=2))
    logger.info("Saved QA report to %s", path)
    return path


# ===========================================================================
# Main Entry Point
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load Config B chunks
    chunks_path = OUTPUT_DIR / "chunks_B.json"
    raw = json.loads(chunks_path.read_text())
    chunks = [Chunk.model_validate(item) for item in raw]
    logger.info("Loaded %d Config B chunks", len(chunks))

    # Generate QA pairs
    qa_pairs = generate_synthetic_qa(chunks)

    # Save
    save_qa_pairs(qa_pairs)

    # Quality report
    report = compute_qa_quality(qa_pairs, total_chunks=len(chunks))
    save_qa_report(report)

    # Print summary
    print(f"\n{'='*60}")
    print("Synthetic QA Generation Complete")
    print(f"{'='*60}")
    print(f"Total questions: {report.total_questions}")
    print(f"\nBy strategy:")
    for strategy, count in sorted(report.questions_per_strategy.items()):
        print(f"  {strategy}: {count}")
    print(f"\nBy type:")
    for q_type, count in sorted(report.questions_per_type.items()):
        print(f"  {q_type}: {count}")
    print(f"\nBy hierarchy:")
    for hierarchy, count in sorted(report.questions_per_hierarchy.items()):
        print(f"  {hierarchy}: {count}")
    print(f"\nChunk coverage: {report.chunk_coverage_percent:.1f}%")
    print(f"Overlap questions: {report.overlap_question_count}")
    print(f"Avg questions/chunk: {report.avg_questions_per_chunk:.3f}")
