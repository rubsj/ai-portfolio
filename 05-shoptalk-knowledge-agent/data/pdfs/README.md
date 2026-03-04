# P5 Test Corpus — PDF Documents

> **Purpose:** These PDFs form the fixed corpus for all 35+ experiment configurations.
> Every chunking strategy, embedding model, and retrieval method is tested against this same set.
> The corpus stays constant across experiments — only pipeline components vary.

---

## Selection Criteria

| Criterion | Rationale |
|-----------|-----------|
| **Single domain (AI/ML)** | Developer can accurately grade ground truth on 4-level NDCG scale (0-3). Cross-domain corpora require domain experts for reliable grading. |
| **Single-column layout** | PyMuPDF extracts cleanly. Multi-column PDFs produce interleaved text that corrupts chunks. |
| **Well-structured headings** | Tests heading-semantic chunker. Papers with clear Abstract → Methods → Results → Conclusion structure produce meaningful heading-based boundaries. |
| **Content developer knows deeply** | Ground truth requires distinguishing grade-3 (directly answers) from grade-2 (same section, contextually relevant). This judgment requires domain expertise. |
| **Varied internal structure** | Mix of papers with tables, equations, prose-heavy sections, and lists — exercises all chunking strategies differently. |
| **~50-150 total pages** | Produces 500+ chunks across all strategies. Large enough for meaningful retrieval experiments, small enough for 8GB M2 memory budget. |

---

## Corpus (4 Documents)

### 1. Attention Is All You Need (Vaswani et al., 2017)

| Field | Value |
|-------|-------|
| **arXiv ID** | 1706.03762 |
| **PDF URL** | https://arxiv.org/pdf/1706.03762 |
| **Pages** | 15 |
| **License** | arXiv.org perpetual, non-exclusive license to distribute |
| **Local filename** | `attention-is-all-you-need.pdf` |

**Why this paper:**
- Foundation of modern NLP — every P5 component (embeddings, transformers, attention) traces back here
- Dense with tables (BLEU scores, architecture comparisons), equations (scaled dot-product attention), and figures
- Well-structured: Introduction → Background → Model Architecture → Training → Results → Conclusion
- Tests chunking strategies: tables and equations challenge fixed-size chunkers; clear sections benefit heading-semantic
- You can write ground truth questions about self-attention, positional encoding, multi-head attention, and training details with confidence

---

### 2. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)

| Field | Value |
|-------|-------|
| **arXiv ID** | 2005.11401 |
| **PDF URL** | https://arxiv.org/pdf/2005.11401 |
| **Pages** | 15 |
| **License** | arXiv.org perpetual, non-exclusive license to distribute |
| **Local filename** | `rag-lewis-et-al.pdf` |

**Why this paper:**
- THE foundational RAG paper — P5 literally implements what this paper describes
- Self-referential: "I built a RAG system and tested it on the paper that invented RAG" is a memorable interview story
- Covers DPR (Dense Passage Retrieval), BART generator, marginalization over documents — all concepts P5 implements
- Clean experimental tables comparing RAG-Sequence vs RAG-Token across Open-Domain QA, Jeopardy, MS-MARCO
- Tests heading-semantic chunker well: Abstract → Introduction → Methods → Experiments → Results sections clearly delineated

---

### 3. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks (Reimers & Gurevych, 2019)

| Field | Value |
|-------|-------|
| **arXiv ID** | 1908.10084 |
| **PDF URL** | https://arxiv.org/pdf/1908.10084 |
| **Pages** | 11 |
| **License** | CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike) |
| **Local filename** | `sentence-bert.pdf` |

**Why this paper:**
- Directly relevant: MiniLM and mpnet (P5's embedding models) are descendants of SBERT architecture
- Explains siamese networks, triplet loss, pooling strategies — concepts the embedding-semantic chunker relies on
- Rich comparison tables across STS benchmarks and transfer learning tasks
- Shorter paper (11 pages) provides contrast in document length for chunking experiments
- CC BY-SA 4.0 license — most permissive license in the corpus

---

### 4. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., 2019)

| Field | Value |
|-------|-------|
| **arXiv ID** | 1810.04805 |
| **PDF URL** | https://arxiv.org/pdf/1810.04805 |
| **Pages** | 16 |
| **License** | arXiv.org perpetual, non-exclusive license to distribute |
| **Local filename** | `bert-devlin-et-al.pdf` |

**Why this paper:**
- Foundational to the entire embedding pipeline — every SentenceTransformer model is BERT-derived
- Longest paper in the corpus (16 pages) — tests how chunking strategies handle larger documents
- Mix of dense prose (pre-training methodology), tables (GLUE/SQuAD benchmarks), and appendix sections
- Detailed ablation studies provide excellent ground truth material: "What happens when you remove NSP?" has a precise, verifiable answer
- Complex structure with appendices — tests whether heading-semantic chunker handles multi-level document hierarchy

---

## Corpus Summary

| Document | Pages | License | Structure | Key Challenge for Chunkers |
|----------|-------|---------|-----------|---------------------------|
| Attention Is All You Need | 15 | arXiv non-exclusive | Tables + equations heavy | Math notation in fixed-size chunks |
| RAG (Lewis et al.) | 15 | arXiv non-exclusive | Clean section hierarchy | Dense methodology sections |
| Sentence-BERT | 11 | CC BY-SA 4.0 | Comparison tables | Shorter doc, many small tables |
| BERT (Devlin et al.) | 16 | arXiv non-exclusive | Prose + appendices | Long doc with multi-level headings |
| **Total** | **57 pages** | | | |

**Expected chunks:** ~500-800 across all 5 chunking strategies (varies by chunk_size and overlap settings).

---

## Download Instructions

### Manual Download (Day 0 — before Day 1 session)

```bash
# From the project root: 05-shoptalk-knowledge-agent/
mkdir -p data/pdfs

# 1. Attention Is All You Need
curl -L -o data/pdfs/attention-is-all-you-need.pdf https://arxiv.org/pdf/1706.03762

# 2. RAG (Lewis et al.)
curl -L -o data/pdfs/rag-lewis-et-al.pdf https://arxiv.org/pdf/2005.11401

# 3. Sentence-BERT
curl -L -o data/pdfs/sentence-bert.pdf https://arxiv.org/pdf/1908.10084

# 4. BERT (Devlin et al.)
curl -L -o data/pdfs/bert-devlin-et-al.pdf https://arxiv.org/pdf/1810.04805
```

### Verification

```bash
# Verify all 4 PDFs downloaded correctly
ls -lh data/pdfs/*.pdf
# Expected: 4 files, each 200KB-1.5MB

# Quick validation — each should output page count
python -c "
import fitz
for f in ['attention-is-all-you-need', 'rag-lewis-et-al', 'sentence-bert', 'bert-devlin-et-al']:
    doc = fitz.open(f'data/pdfs/{f}.pdf')
    print(f'{f}.pdf: {len(doc)} pages')
    doc.close()
"
# Expected output:
# attention-is-all-you-need.pdf: 15 pages
# rag-lewis-et-al.pdf: 15 pages
# sentence-bert.pdf: 11 pages
# bert-devlin-et-al.pdf: 16 pages
```

---

## License Compliance

All papers are sourced from arXiv.org for **non-commercial, educational, personal research use**.

| Paper | License Type | Reuse Rights |
|-------|-------------|-------------|
| Attention Is All You Need | arXiv perpetual non-exclusive | arXiv distributes; reuse requires author permission. Personal/educational use is standard academic practice. |
| RAG (Lewis et al.) | arXiv perpetual non-exclusive | Same as above. |
| Sentence-BERT | CC BY-SA 4.0 | Free to share and adapt with attribution. Derivative works must use same license. |
| BERT (Devlin et al.) | arXiv perpetual non-exclusive | Same as Attention paper. |

**Note:** These PDFs are used as input data for a personal portfolio project. They are not redistributed, modified, or used commercially. The PDFs themselves are NOT committed to the Git repository — only this README is committed. Each developer downloads their own copies from arXiv.

---

## .gitignore Entry

Add to `05-shoptalk-knowledge-agent/.gitignore`:

```
# Source PDFs — download locally, do not commit
data/pdfs/*.pdf
```

---

## Ground Truth Preview

These papers enable diverse question types for the 15 curated evaluation queries:

**Factual (grade 3 — directly answered):**
- "What scaling factor does the Transformer use in scaled dot-product attention?"
- "How many inference computations does BERT require to find the most similar sentence pair in 10,000 sentences?"

**Analytical (grade 2-3 — requires reasoning across sections):**
- "Why does RAG-Token outperform RAG-Sequence on certain tasks?"
- "What is the computational advantage of SBERT's siamese architecture over cross-encoder BERT?"

**Cross-document (grade 1-2 — topically related across papers):**
- "How does the attention mechanism in the Transformer relate to the retrieval component in RAG?"
- "Why is BERT's bidirectional pre-training important for the sentence embeddings used in SBERT?"

These examples are illustrative — the actual 15 curated queries will be generated by LLM (30 candidates) and manually curated (15 selected) during Day 3.
