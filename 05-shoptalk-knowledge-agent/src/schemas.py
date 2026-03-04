"""Pydantic data models for the ShopTalk Knowledge Agent.

All data models used across the pipeline. No raw dicts cross module boundaries.

Models:
    Document        — extracted PDF with metadata (source, title, author, page_count, pages)
    Chunk           — atomic retrieval unit with embedding (optional)
    RetrievalResult — single retrieved chunk + score + retriever type + rank
    QAResponse      — complete query-answer output with citations and latency
    Citation        — traceable source reference parsed from [N] markers in the answer
    ExperimentConfig — one experiment specification (all config dimensions + validators)
    ExperimentResult — results for one config (config + per-query metrics + judge scores + perf)
"""
