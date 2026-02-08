# CLAUDE.md — P1: Synthetic Data, Home DIY Repair

> Project-specific instructions. Read the root CLAUDE.md first, then this.

## Project Overview

Generate synthetic Home DIY Repair Q&A data using LLMs, validate with Pydantic, evaluate quality,
and analyze failure modes. This is the foundation project — patterns built here (caching, validation,
LLM-as-Judge) will be reused in P2–P9.

## Domain: Home DIY Repair Categories

1. **Appliance Repair** — refrigerators, washing machines, dryers, dishwashers, ovens
2. **Plumbing Repair** — leaks, clogs, fixture repairs, pipe problems
3. **Electrical Repair** — outlets, switches, light fixtures (SAFE homeowner-level only)
4. **HVAC Maintenance** — filters, thermostats, vents, basic troubleshooting
5. **General Home Repair** — drywall, doors/windows, flooring, basic carpentry

## Data Schema (Target JSON Structure)

Each generated record should contain:
```json
{
  "question": "How do I fix a leaking kitchen faucet?",
  "answer": "Detailed step-by-step answer...",
  "category": "plumbing_repair",
  "difficulty": "beginner|intermediate|advanced",
  "equipment_problem": "Leaking single-handle kitchen faucet",
  "tools_required": ["adjustable wrench", "plumber's tape", ...],
  "steps": ["Turn off water supply...", "Remove handle...", ...],
  "safety_info": "Always turn off water supply before...",
  "tips": "Take a photo of the faucet assembly before..."
}
```

## Architecture

```
generator.py  →  schemas.py  →  validator.py  →  evaluator.py  →  analysis.py
   │                  │                │                │               │
   │ Calls GPT-4o-    │ Pydantic       │ Categorizes    │ LLM-as-Judge  │ Heatmaps,
   │ mini, gets JSON  │ validates      │ errors into    │ (GPT-4o)      │ Jaccard,
   │ responses         │ structure      │ 6 failure      │ scores quality│ correlations
   │                  │                │ modes          │               │
   ▼                  ▼                ▼                ▼               ▼
data/generated/   Valid records    data/failures/   data/eval/      results/charts/
```

## File Responsibilities

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `src/schemas.py` | Pydantic models for all data structures | `Tool`, `SafetyPrecaution`, `RepairStep`, `DIYRepairRecord` |
| `src/generator.py` | LLM generation pipeline with caching | `generate_record()`, `generate_batch()`, `load_cache()`, `save_cache()` |
| `src/validator.py` | Validation loop + error categorization | `validate_record()`, `categorize_failures()`, `FailureMode` enum |
| `src/evaluator.py` | LLM-as-Judge evaluation | `evaluate_with_llm()`, `JudgeResult` model |
| `src/analysis.py` | Statistical analysis + visualization | `jaccard_similarity()`, `build_heatmap()`, `correlation_analysis()` |
| `tests/test_schemas.py` | Schema validation tests | Happy path + edge cases + intentional failures |

## Failure Modes to Track

| Code | Failure Mode | What to Look For |
|------|-------------|------------------|
| `incomplete_answer` | Answer lacks sufficient detail | < 3 sentences, missing key information |
| `safety_violations` | Missing or inadequate safety info | Electrical work without power-off warning |
| `unrealistic_tools` | Tools a homeowner wouldn't have | Suggesting oscilloscope for outlet repair |
| `overcomplicated_solution` | Solution too complex for stated difficulty | Professional-grade steps for "beginner" task |
| `missing_context` | Missing equipment_problem or category context | Generic answer not tied to specific problem |
| `poor_quality_tips` | Tips are obvious, unhelpful, or dangerous | "Just wing it" or contradicts safety_info |

## Generation Parameters

- **Model**: GPT-4o-mini (cheap: ~$0.15/1M input tokens)
- **Temperature**: 0.7 (balance creativity vs consistency)
- **Batch size**: 2 per category × 5 categories × 3 difficulties = 30 records (minimum)
- **Stretch**: 4 per combo = 60 records
- **Response format**: `response_format={"type": "json_object"}` with schema in system prompt
- **Cache**: JSON file in `data/cache/`, keyed on MD5 hash of full prompt

## Evaluation (GPT-4o as Judge)

- Use GPT-4o (the expensive model) to judge quality of GPT-4o-mini outputs
- Score each record on: completeness (1-5), safety accuracy (1-5), difficulty appropriateness (1-5), practical usefulness (1-5)
- Flag failure modes as binary labels
- This pattern (cheap model generates, expensive model evaluates) is reused in P2-P9

## Success Criteria

- Generation success rate > 90% (valid JSON that passes Pydantic)
- Validation pass rate tracked and visualized
- Jaccard similarity scores computed and charted
- Correction loop reduces failure rate by >80%
- All deliverables: README, ADR, Loom, Streamlit demo

## Dependencies (pyproject.toml)

```toml
[project]
dependencies = [
    "openai",
    "pydantic",
    "pandas",
    "matplotlib",
    "seaborn",
    "python-dotenv",
    "streamlit",
]

[tool.uv]
dev-dependencies = [
    "pytest",
    "ruff",
]
```

## Current State

- [x] Project scaffolded with uv
- [x] All src/ and tests/ files created (empty)
- [x] .env ready for OPENAI_API_KEY
- [ ] Pydantic schemas (schemas.py) — Sunday Feb 8
- [ ] Generation pipeline (generator.py) — Sunday Feb 8
- [ ] Tests (test_schemas.py) — Sunday Feb 8
- [ ] First batch generated (30-60 records) — Sunday Feb 8
- [ ] Validation loop (validator.py) — Monday Feb 9
- [ ] Analysis + evaluation (evaluator.py, analysis.py) — Tuesday Feb 10
- [ ] Documentation + Streamlit — Wednesday Feb 11

## Key Decisions Made

- Pydantic v2 over dataclasses: runtime validation catches LLM hallucinations
- GPT-4o-mini for generation (cost), GPT-4o for evaluation (quality)
- 5 categories (not 8 as originally discussed) — matches project requirements doc
- JSON file cache over SQLite cache: simpler for first project, can upgrade later
