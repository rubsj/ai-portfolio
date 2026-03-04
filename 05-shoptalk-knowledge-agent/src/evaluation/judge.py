"""5-axis LLM-as-Judge for generation quality evaluation.

Axes (1–5 scale each):
    Relevance       — answer addresses the question
    Accuracy        — every claim verifiable in provided context
    Completeness    — answer is thorough
    Conciseness     — not unnecessarily verbose
    Citation Quality — proper [N] source attribution

Uses Instructor for structured output (same pattern as P1/P4 judge).
Default model: gpt-4o (evaluation requires stronger reasoning than generation).

Why not RAGAS: P2 already used RAGAS. Custom judge gives per-axis diagnostics
and demonstrates understanding of what RAGAS does internally.
"""
