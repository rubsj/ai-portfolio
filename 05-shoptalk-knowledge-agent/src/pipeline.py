"""End-to-end RAG pipeline orchestrator.

Ties together all pipeline phases:
  Phase 1 (Ingestion):  PDF → Load → Chunk → Embed → Index → Save
  Phase 2 (Query):      Question → Embed → Retrieve → [Rerank] → Generate → QAResponse
  Phase 3 (Evaluation): Ground truth queries → run config → metrics → ExperimentResult

All components injected via factories.py from YAML config — no hardcoded
class names in this module.
"""