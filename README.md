# AI Portfolio — 9 Projects in 8 Weeks

> Production-grade AI systems with measurable results, not toy demos.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C)
![FAISS](https://img.shields.io/badge/FAISS-Meta-lightgrey)
![RAGAS](https://img.shields.io/badge/RAGAS-Eval-F97316)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-DC2626)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063)

---

## About

A focused 8-week engineering sprint (Feb–Apr 2026) building nine production-grade AI systems end-to-end — from synthetic data pipelines to multi-agent orchestration and DevOps automation. Each project targets a distinct layer of the modern AI engineering stack: synthetic data generation, RAG evaluation, embedding fine-tuning, agentic pipelines, and production deployment. Designed to demonstrate the depth and rigor expected at Tier-1 engineering organizations, not just proof-of-concept familiarity.

---

## Projects

| # | Project | Status | Key Techniques | Stack |
|---|---------|--------|----------------|-------|
| 1 | [Synthetic Data — Home DIY Repair](./01-synthetic-data-home-diy/) | ✅ Complete | Structured generation, LLM-as-Judge, correction loop, Jaccard similarity | Pydantic, OpenAI, Instructor, Streamlit |
| 2 | [Evaluating RAG for Any PDF](./02-rag-evaluation/) | ✅ Complete | 16-config grid search, chunking strategies, hybrid reranking, RAGAS metrics | LangChain, FAISS, RAGAS, Cohere, Braintrust, Click |
| 3 | [Contrastive Embedding Fine-Tuning](./03-fine-tuning-guardrails/) | ✅ Complete | CosineSimilarityLoss, LoRA (PEFT), 8-metric evaluation, UMAP, HDBSCAN | Sentence-Transformers, PEFT/LoRA, scikit-learn, scipy, Click |
| 4 | [AI-Powered Resume Coach](./04-resume-coach/) | ⏳ Upcoming | JD analysis, controlled fit levels, LLM-as-Judge evaluation | OpenAI, Pydantic, FastAPI |
| 5 | [Production RAG System](./05-production-rag/) | ⏳ Upcoming | Multi-strategy chunking, hybrid search, Cohere reranking, REST API | LangChain, FAISS, Cohere, FastAPI, Click |
| 6 | [Digital Writing Clone — 5-Agent](./06-digital-writing-clone/) | ⏳ Upcoming | StyleAnalyzer, RAG, Evaluator, Fallback, Planner agents; style-matched generation | CrewAI, LangChain, OpenAI |
| 7 | [Customer Feedback Intelligence](./07-feedback-intelligence/) | ⏳ Upcoming | CrewAI pipeline: Sentiment, Theme, Mapping, Gap agents | CrewAI, OpenAI, Pydantic |
| 8 | [Jira AI Agent](./08-jira-ai-agent/) | ⏳ Upcoming | Semantic search, duplicate detection, sprint planning | OpenAI, FAISS, FastAPI, Click |
| 9 | [DevOps AI Assistant (Capstone)](./09-devops-assistant/) | ⏳ Upcoming | 5-agent system: CI/CD monitoring, log analysis, root cause, remediation | CrewAI, OpenAI, FastAPI |

---

## Completed Project Highlights

### P1 — Synthetic Data: Home DIY Repair

**30 structured QA records** across 5 repair categories × 2 formats × 3 difficulty levels.

- **100% generation success rate** with zero schema validation failures (Pydantic + Instructor)
- **LLM-as-Judge calibration**: GPT-4o judge tuned from 0% → 20% failure detection via prompt engineering
- **Correction loop**: 78% failure reduction (V1 → V2 templates) — upstream prompt improvement outperformed downstream correction
- **7 publication-quality charts**, 6-page Streamlit app with Story Mode narrative, 4 ADRs

→ [Project README](./01-synthetic-data-home-diy/README.md)

---

### P2 — Evaluating RAG for Any PDF

**Systematic benchmarking** of 16 retrieval configurations across a real PDF corpus.

- **Best config**: semantic chunking + OpenAI embeddings → Recall@5 = 0.625; +19.5% with Cohere reranking (→ 0.747)
- **BM25 baseline beaten** by 10 of 15 vector configs; OpenAI embeddings outperform local models by 26% for $0.02/1M tokens
- **Key findings**: 50% chunk overlap underperforms 25% by 13%; faithfulness gap of 0.511 shows retrieval ≠ generation quality; 39% LLM refusal rate misclassified as hallucinations
- **95% test coverage** (384+ tests), 12 charts, 7-page Streamlit dashboard, Click CLI with Rich formatting, 5 ADRs, ~20 PRs

→ [Project README](./02-rag-evaluation/README.md)

---

### P3 — Contrastive Embedding Fine-Tuning

**Flipped inverted embeddings** from Spearman -0.22 to +0.85 using contrastive fine-tuning, then compared standard vs. LoRA approaches.

- **1,238% margin improvement**: Baseline margin -0.083 → +0.940 after standard fine-tuning (AUC-ROC 0.994, Cohen's d 7.73)
- **LoRA efficiency**: 96.9% of standard performance with only 0.32% trainable parameters (73K vs 22.7M) and a 300x smaller model file
- **8-metric evaluation framework**: Spearman, Margin, Cohen's d, AUC-ROC, F1, Cluster Purity, False Positive Analysis, Category Metrics
- **97.8% false positive reduction**: 137 → 3 FPs after standard fine-tuning; self-contained HTML comparison report with 8 charts
- **112 tests**, 3 ADRs, memory-constrained pipeline running on 8GB MacBook Air M2

> [Project README](./03-fine-tuning-guardrails/README.md)

---

## Repository Structure

```
ai-portfolio/
├── 01-synthetic-data-home-diy/   # P1 — Complete
│   ├── src/                      # schemas, templates, generator, validator, evaluator, analysis, corrector
│   ├── tests/
│   ├── data/
│   ├── results/
│   ├── docs/
│   ├── streamlit_app.py
│   └── pyproject.toml
├── 02-rag-evaluation/            # P2 — Complete
│   ├── src/                      # chunker, embedder, index_builder, retrieval_evaluator, judge, reranker, cli, ...
│   ├── tests/
│   ├── data/
│   ├── results/
│   ├── docs/
│   ├── streamlit_app.py
│   └── pyproject.toml
├── 03-fine-tuning-guardrails/    # P3 — Complete
│   ├── src/                      # trainer, lora_trainer, post_training_eval, comparison, metrics, cli
│   ├── tests/
│   ├── data/
│   ├── training/
│   ├── eval/
│   ├── docs/
│   └── pyproject.toml
├── 04-resume-coach/              # P4 — Upcoming (Week 3)
├── 05-production-rag/            # P5 — Upcoming (Week 4)
├── 06-digital-writing-clone/     # P6 — Upcoming (Week 4–5)
├── 07-feedback-intelligence/     # P7 — Upcoming (Week 5)
├── 08-jira-ai-agent/             # P8 — Upcoming (Week 6)
├── 09-devops-assistant/          # P9 — Upcoming (Week 7)
├── shared/                       # Cross-project utilities (llm_client, cache, eval_helpers)
├── docs/                         # Cross-project ADRs, monthly learning notes
└── CLAUDE.md                     # Persistent engineering context
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **LLM & AI** | OpenAI API (GPT-4o-mini / GPT-4o), Sentence-Transformers, Instructor, Cohere Rerank |
| **Data & Evaluation** | RAGAS, Braintrust, FAISS, Pydantic v2 |
| **Frameworks** | LangChain, CrewAI, FastAPI |
| **Frontend & Viz** | Streamlit, Rich, Matplotlib, Seaborn |
| **DevOps & Testing** | Click CLI, pytest, ruff, uv, GitHub Actions |

---

## Engineering Practices

- **Architecture Decision Records** — every significant design choice documented per project (4 in P1, 5 in P2, 3 in P3)
- **95%+ test coverage targets** — pytest with parametrized cases covering happy path and failure modes (500+ tests across 3 projects)
- **5-layer validation methodology** — schema validation, semantic checks, LLM-as-Judge, correction loops, re-evaluation
- **LLM cost management** — MD5-keyed disk cache for all LLM calls; estimated cost logged per call
- **Clean PR history** — feature branches, atomic commits, ~20 PRs merged across projects
- **Reproducible results** — all generation seeds, model versions, and evaluation configs pinned and committed

---

Each project includes comprehensive documentation and reproducible results.

**Connect:** [LinkedIn](https://linkedin.com/in/jharuby) · [Portfolio Site](https://yourportfolio.dev)
