"""CLI: PDF ingestion pipeline — PDF → chunk → embed → FAISS index → save to disk.

Usage: python scripts/ingest.py --pdf data/pdfs/attention-is-all-you-need.pdf
                                 --config experiments/configs/recursive_minilm_dense.yaml
                                 --output data/indices/

Deliverable D1 (PRD Section 7a): verifiable by index files appearing on disk.
"""
