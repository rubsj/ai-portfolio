"""Streamlit web UI for interactive RAG question-answering.

5 panels:
  1. PDF upload — drag-and-drop, triggers ingestion pipeline
  2. Config sidebar — chunker, embedder, retriever, reranker selection
  3. Question input — free-text query box
  4. Answer + citations — generated answer with [N] citations expanded
  5. Source viewer — retrieved chunks with relevance scores and page numbers

Why Streamlit over FastAPI+React: spec-required. P4 proved FastAPI skills.
P5's value is the RAG pipeline, not another REST wrapper.
"""
