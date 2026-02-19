# CLAUDE.md — P3: Contrastive Embedding Fine-Tuning

> **Read this file + PRD.md at the start of EVERY session.**
> This is your persistent memory across sessions. Update the "Current State" section before ending each session.

---

## Project Identity

- **Project:** P3 — Contrastive Embedding Fine-Tuning (Dating Compatibility)
- **Location:** `03-fine-tuning-guardrails/` within `ai-portfolio` monorepo
- **Timeline:** Feb 18–20, 2026 (3 sessions × 4h = 12h)
- **PRD:** `PRD.md` in this directory — the implementation contract
- **Concepts Primer:** `p3-concepts-primer.html` in project root — read for contrastive learning, LoRA, UMAP, HDBSCAN theory

---

## Model Routing Protocol (CRITICAL)

This project uses the **Opus-plans, Sonnet-executes** workflow:

### When to use Opus (Planning)
- **Start of each day:** Read PRD tasks for the day, create a detailed implementation plan
- **Architecture decisions:** If something in the PRD is ambiguous, reason through it
- **Debugging:** When stuck on a non-trivial bug (not typos — conceptual issues)
- **Evaluation interpretation:** Analyzing results, deciding what charts to generate
- **Instruction:** Plan the exact files, classes, functions, and interfaces before writing code

### When to use Sonnet (Execution)
- **All code writing** — implement what the Opus plan specified
- **File creation** — pyproject.toml, __init__.py, data loading, tests
- **Running commands** — uv sync, pytest, python scripts
- **Routine edits** — fixing imports, adjusting parameters, formatting
- **Chart generation** — matplotlib/seaborn/plotly code (follow the plan)

### Session Workflow
```
1. Switch to Opus
2. "Read CLAUDE.md and PRD.md. Today is Day [N]. Plan tasks T[X.Y] through T[X.Z]."
3. Opus produces: file-by-file plan, function signatures, key logic, validation criteria
4. Switch to Sonnet
5. "Execute the plan. Start with [first file]."
6. Sonnet implements, tests, commits
7. If blocked → switch to Opus for debugging/replanning
8. At session end → use Sonnet for git commit, journal entry, CLAUDE.md update
```

### Why This Split
- **Cost efficiency:** Opus costs ~15× more than Sonnet per token. Planning is <5% of tokens but needs the strongest reasoning. Implementation is >95% of tokens and Sonnet handles it well.
- **Quality:** Opus catches design issues before code is written. Sonnet doesn't re-debate the architecture — it executes.
- **Speed:** Sonnet is faster for code generation. Opus takes longer but produces better plans.

---

## Developer Context

- **Background:** Java/TypeScript developer learning Python. Completed P1 (Synthetic Data) + P2 (RAG Evaluation).
- **P1 patterns to reuse:** Pydantic validation, JSON caching, matplotlib/seaborn charts, Rich console output
- **P2 patterns to reuse:** SentenceTransformer model loading/unloading, `gc.collect()` memory management, interactive HTML reports with plotly, Click CLI, comparison heatmaps
- **Python comfort level:** Intermediate-to-comfortable after P1 + P2 — Pydantic, type hints, list comprehensions, ThreadPoolExecutor, generators, numpy arrays, FAISS. **New concepts for P3:** PyTorch training loops, DataLoader, loss functions, PEFT/LoRA, UMAP, HDBSCAN
- **IDE:** VS Code + Claude Code terminal
- **Hardware:** MacBook Air M2, 8GB RAM — this is a HARD constraint

---

## Architecture Rules (Do NOT Re-Debate)

These decisions are FINAL. Refer to PRD Section 2 for full rationale.

1. **`all-MiniLM-L6-v2`** as base model — 22.7M params, 384 dims, ~90MB. Fits M2 with room to spare.
2. **Provided data, not synthetic** — 1,195 train + 295 eval pairs from `data/raw/`. Run `SyntheticDataEvaluator` to validate but do NOT generate new data.
3. **`CosineSimilarityLoss`** from sentence-transformers — maps label=1 → cosine sim 1.0, label=0 → cosine sim 0.0.
4. **Two training paths:** Standard (full parameter) + LoRA via PEFT. Compare results. Same hyperparameters for both.
5. **NO QLoRA / bitsandbytes** — CUDA dependency, incompatible with M2. Document in ADR.
6. **NO LLM API calls** — entire project runs locally. No OpenAI key needed.
7. **Instructional fine-tuning** is a STRETCH GOAL — only if core is done by Day 3.
8. **All 8 evaluation metrics** from requirements — no shortcuts. Evaluation depth is the portfolio differentiator.
9. **UMAP random_state=42 everywhere** — reproducibility is non-negotiable.
10. **LoRA is a bonus, not a blocker** — if PEFT has M2 compatibility issues, skip it. Focus on standard fine-tuning + evaluation. Document LoRA in ADR regardless.

---

## Memory Management Protocol (CRITICAL — 8GB M2)

Claude Code MUST follow this when working with models:

```
RULE 1: Never load two SentenceTransformer models simultaneously.
  - Load base model → generate embeddings → save embeddings to disk → del model → gc.collect()
  - Load fine-tuned model → generate embeddings → save embeddings to disk → del model → gc.collect()

RULE 2: Training cleanup after each training run.
  - After standard training: save model → del model → del dataloader → gc.collect()
  - After LoRA training: save adapter → del model → gc.collect()
  - Then load fine-tuned model fresh for evaluation.

RULE 3: batch_size=16, not 32.
  - Monitor with `htop` during first epoch.
  - If memory pressure: reduce to 8.

RULE 4: Save embeddings to disk (numpy .npy files).
  - Don't recompute embeddings for every evaluation step.
  - Baseline embeddings: data/embeddings/baseline_{train|eval}.npy
  - Fine-tuned embeddings: data/embeddings/finetuned_{train|eval}.npy

RULE 5: Close non-essential apps during training.
  - Chrome tabs are the biggest RAM competitor.

NEVER:
  - Load base AND fine-tuned models at same time
  - Keep training DataLoader in memory during evaluation
  - Run UMAP on >2000 points without checking memory first
```

---

## Notion Integration

Claude Code can write to Notion via MCP. Use these IDs:

| Resource | ID / URL |
|----------|----------|
| Command Center | `https://www.notion.so/2ffdb630640a81f58df5f5802aa51550` |
| Project Tracker (data source) | `collection://4eb4a0f8-83c5-4a78-af3a-10491ba75327` |
| P3 Tracker Card | *(create on Day 1 — update this field with the page ID)* |
| Learning Journal (data source) | `collection://c707fafc-4c0e-4746-a3bc-6fc4cd962ce5` |
| ADR Log (data source) | `collection://629d4644-ca7a-494f-af7c-d17386e1189b` |
| Chat Index | `303db630640a81ccb026f767597b023f` |

### Journal Entry Template

At the end of each session, create a journal entry in the Learning Journal:

```
Properties:
  - Title: "P3 Day [N] — [summary]"
  - Project: P3
  - Date: [today]
  - Hours: [session hours]

Content:
  ## What I Built
  [files created/modified, key functionality]

  ## What I Learned
  [concepts understood, Python patterns, surprises]

  ## What Blocked Me
  [issues, workarounds, things deferred]

  ## Python Pattern of the Day
  [one specific Python pattern with Java/TS comparison]

  ## Tomorrow's Plan
  [next session tasks from PRD]
```

---

## Code Conventions

### From P1/P2 (continue these):
- **Comment with "WHY" not "what"** — `# WHY: CosineSimilarityLoss expects float labels in [0, 1], not int`
- **Type hints everywhere** — `def compute_margin(similarities: np.ndarray, labels: list[int]) -> float:`
- **Pydantic for all data models** — no raw dicts. `DatingPair`, `BaselineMetrics`, `ComparisonResult`
- **f-strings for everything** — prompts, log messages, file paths
- **pathlib.Path** over os.path — `Path("data/raw/dating_pairs.jsonl")`

### New for P3:
- **PyTorch DataLoader** — like Java's `Iterator<Batch>`. Handles shuffling, batching, and memory-efficient loading. Sonnet: use sentence-transformers' built-in DataLoader, not raw PyTorch.
- **InputExample** — sentence-transformers' pair format. `InputExample(texts=[text_1, text_2], label=float(label))`
- **Model save/load pattern** — `model.save("path/to/model")` then `SentenceTransformer("path/to/model")` to reload. Like Java serialization but for neural network weights.
- **PEFT wrapping** — `get_peft_model(model, lora_config)` wraps model in-place. Like the Decorator pattern — same interface, modified behavior.
- **numpy for embeddings** — `model.encode(texts)` returns numpy ndarray. All metric computations use numpy/scipy, not Python loops.
- **UMAP + HDBSCAN pipeline** — reduce dims first (UMAP), then cluster (HDBSCAN). Always on the reduced 2D space, not the original 384D.

### Test Conventions:
- `pytest` for all tests
- Test file per module: `test_data_loader.py`, `test_evaluator.py`, `test_metrics.py`
- Focus tests on: Pydantic validation (valid + invalid), metric correctness (known inputs → expected outputs), data loading edge cases
- Don't test training convergence — that's what evaluation is for

---

## File Structure

```
03-fine-tuning-guardrails/
├── CLAUDE.md                          # THIS FILE
├── PRD.md                             # Implementation contract
├── README.md                          # Portfolio README with results
├── pyproject.toml                     # uv project config
├── data/
│   ├── raw/
│   │   ├── dating_pairs.jsonl         # 1,195 training pairs (provided)
│   │   ├── eval_pairs.jsonl           # 295 evaluation pairs (provided)
│   │   ├── dating_pairs_metadata.json # Generation metadata
│   │   └── eval_pairs_metadata.json   # Eval metadata
│   ├── evaluation/
│   │   ├── data_quality_report.json   # SyntheticDataEvaluator output
│   │   └── data_quality_summary.txt   # Human-readable findings
│   └── embeddings/                    # Cached embeddings (numpy .npy)
│       ├── baseline_train.npy
│       ├── baseline_eval.npy
│       ├── finetuned_train.npy
│       ├── finetuned_eval.npy
│       ├── lora_train.npy             # If LoRA succeeds
│       └── lora_eval.npy
├── src/
│   ├── __init__.py
│   ├── models.py                      # ALL Pydantic schemas from PRD Section 6
│   ├── data_loader.py                 # Load JSONL, validate with Pydantic
│   ├── data_evaluator.py              # SyntheticDataEvaluator (5 dimensions)
│   ├── baseline_analysis.py           # Pre-training: embed, cosine, UMAP, HDBSCAN
│   ├── trainer.py                     # Standard fine-tuning (CosineSimilarityLoss)
│   ├── lora_trainer.py                # LoRA fine-tuning via PEFT
│   ├── post_training_eval.py          # Post-training metrics (reuses baseline code)
│   ├── comparison.py                  # Before/after comparison + chart generation
│   ├── metrics.py                     # Shared metric functions (margin, cohen_d, roc, etc.)
│   └── visualizations.py              # All chart generation (matplotlib/seaborn/plotly)
├── training/
│   ├── model/
│   │   ├── standard_model/            # Full fine-tuned weights
│   │   └── lora_model/                # LoRA adapter weights
│   ├── standard_training_info.json
│   └── lora_training_info.json
├── eval/
│   ├── visualizations/
│   │   ├── baseline/                  # Pre-training charts
│   │   └── comparison/                # Before/after charts
│   ├── baseline_metrics.json
│   ├── finetuned_metrics.json
│   ├── lora_metrics.json
│   ├── baseline_report.html           # Interactive baseline dashboard
│   └── comparison_report.html         # Interactive comparison dashboard
├── docs/
│   ├── adr-001-lora-vs-standard.md
│   ├── adr-002-qlora-skip.md
│   └── adr-003-cosine-similarity-loss.md
└── tests/
    ├── test_data_loader.py
    ├── test_evaluator.py
    └── test_metrics.py
```

---

## Environment Setup (Day 1 First Task)

```bash
# From monorepo root
cd 03-fine-tuning-guardrails

# Initialize project
uv init

# Core ML
uv add sentence-transformers torch peft

# Evaluation & Visualization
uv add umap-learn hdbscan scikit-learn scipy
uv add matplotlib seaborn plotly

# Data & Validation
uv add pydantic

# CLI & Formatting
uv add click rich

# Dev tools
uv add pytest ruff --dev

# Copy provided data files
mkdir -p data/raw
cp <path-to-uploads>/dating_pairs.jsonl data/raw/
cp <path-to-uploads>/eval_pairs.jsonl data/raw/
cp <path-to-uploads>/dating_pairs_metadata.json data/raw/
cp <path-to-uploads>/eval_pairs_metadata.json data/raw/
```

**No `.env` file needed** — this project has zero API calls. Everything runs locally.

---

## Session Protocol

### Starting a session (Opus):
```
Read CLAUDE.md and PRD.md. Today is Day [N].
Here's where I left off: [paste handoff from previous session]
Create an implementation plan for tasks T[N.X] through T[N.Y] from PRD Section 9.
For each task, specify: files to create/modify, key functions, input/output, validation criteria.
```

### Switching to execution (Sonnet):
```
Execute the plan from Opus. Start with [first file].
After each file: run ruff check, run relevant tests, commit.
Do NOT re-debate architecture — follow the plan.
```

### Ending a session (Sonnet):
1. **Git commit and push** all work
2. **Update CLAUDE.md** "Current State" section below
3. **Write journal entry** to Notion Learning Journal via MCP
4. **Produce handoff summary** in this format:

```
## P3 Handoff — Session End [Date]

### Branch / Commit
- Branch: `feat/p3-[description]`
- Working tree: [clean/dirty]

### What's Done
[list of completed PRD tasks with task numbers]

### Key Files Created/Modified
[file list with brief description]

### Key Metrics (if any)
[baseline margin, spearman, cohen_d — whatever was measured today]

### What's Next
[next session's tasks from PRD Section 9]

### Blockers / Open Questions
[anything unresolved — flag for Opus planning chat]
```

---

## Current State

> **Claude Code: UPDATE this section at the end of every session.**

### Day 0 (Pre-start)
- [ ] Project directory created
- [ ] Dependencies installed (`uv sync` passes)
- [ ] Data files copied to `data/raw/`
- [ ] P3 card created in Notion Project Tracker

### Day 1 — Data Evaluation + Baseline Analysis (Tue Feb 18)
- [ ] T1.1: Project setup (directory, pyproject.toml, uv sync)
- [ ] T1.2: Data loading with Pydantic validation
- [ ] T1.3: SyntheticDataEvaluator (5-dimension scoring, target ≥60%)
- [ ] T1.4: Generate baseline embeddings (all-MiniLM-L6-v2)
- [ ] T1.5: Baseline cosine similarity analysis (distributions, margins, Cohen's d, ROC)
- [ ] T1.6: UMAP visualization + HDBSCAN clustering
- [ ] T1.7: Category-wise + pair-type-wise baseline breakdown
- [ ] T1.8: Save baseline_metrics.json + baseline_report.html
- [ ] **Checkpoint:** Full baseline established. Every metric has a "before" number.

### Day 2 — Fine-Tuning: Standard + LoRA (Wed Feb 19)
- [ ] T2.1: Implement trainer.py (DataLoader, CosineSimilarityLoss, evaluator)
- [ ] T2.2: Run standard training (4 epochs, monitor Spearman)
- [ ] T2.3: Implement lora_trainer.py (PEFT LoraConfig)
- [ ] T2.4: Run LoRA training (same hyperparams)
- [ ] T2.5: Compare training curves (standard vs LoRA)
- [ ] **Checkpoint:** Fine-tuned model(s) saved. Training curves logged.

### Day 3 — Post-Training Evaluation + Documentation (Thu Feb 20)
- [ ] T3.1: Post-training evaluation (all 8 metrics on fine-tuned model(s))
- [ ] T3.2: Before/after comparison charts (8 charts minimum)
- [ ] T3.3: Generate comparison_report.html
- [ ] T3.4: False positive deep dive (pair_type/category breakdown)
- [ ] T3.5: ADRs (LoRA vs standard, QLoRA skip, loss function choice)
- [ ] T3.6: README with Mermaid diagram + results
- [ ] T3.7: Loom recording
- [ ] T3.8: Notion journal update
- [ ] **Stretch:** T3.S1: Instructional fine-tuning variant
- [ ] **P3 COMPLETE**

---

## P1/P2 Patterns to Reuse

Reference these from completed projects when implementing P3:

| Pattern | Source | P3 Usage |
|---------|--------|----------|
| Pydantic models with validators | P1 `src/schemas.py` | `src/models.py` — DatingPair, BaselineMetrics, ComparisonResult |
| Rich console output | P2 `src/cli.py` | Data evaluator progress, metric summaries |
| matplotlib/seaborn charts | P2 `src/visualization.py` | Baseline + comparison charts (histograms, heatmaps, bar charts) |
| plotly interactive HTML | P2 `eval/reports/` | baseline_report.html, comparison_report.html |
| SentenceTransformer load/unload | P2 `src/embedder.py` | Same pattern — load, encode, del, gc.collect() |
| JSON metrics files | P2 `eval/baseline_metrics.json` | Same pattern — save all metrics to JSON for comparison |
| Click CLI structure | P2 `src/cli.py` | `python -m src.cli evaluate`, `python -m src.cli train` |
| ADR template | P1/P2 `docs/adr/` | Same structure — Context, Decision, Alternatives, Consequences |

---

## Key Concepts Quick Reference

(For deep explanation, read `p3-concepts-primer.html`)

- **Contrastive learning:** Train with pairs — pull compatible closer, push incompatible apart. Like sorting by a custom comparator.
- **CosineSimilarityLoss:** Loss function mapping label=1 → cosine sim 1.0, label=0 → 0.0. Gradients reshape the embedding space.
- **InputExample:** sentence-transformers pair format. `texts=[text_1, text_2], label=float(label)`.
- **EmbeddingSimilarityEvaluator:** Monitors Spearman rank correlation during training. Best checkpoint saved based on this.
- **LoRA:** Low-Rank Adaptation. Freezes base model, trains tiny adapter matrices (A × B). Trainable params drop from 22.7M to ~200K.
- **PEFT:** HuggingFace library for LoRA wrapping. `get_peft_model(model, config)` — like the Decorator pattern.
- **Compatibility margin:** Compatible mean cosine - Incompatible mean cosine. Pre-trained ~0.04, target ≥0.20.
- **Cohen's d:** Effect size measuring separation between two distributions. <0.2 = negligible, 0.5 = medium, >0.8 = large.
- **UMAP:** Projects 384d → 2d for visualization. Preserves local + global structure. Set `random_state=42`.
- **HDBSCAN:** Density-based clustering. Auto-detects cluster count + identifies noise. Like GROUP BY with auto-detected categories.
- **Spearman ρ:** Rank correlation between predicted cosine similarities and ground truth labels. Requirements target ≥0.86.
- **AUC-ROC:** Area under ROC curve. 0.5 = random, 1.0 = perfect. Measures discrimination ability across all thresholds.

---

## Training Hyperparameters Reference

Both standard and LoRA training use identical hyperparameters (controlled experiment):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 4 | Small dataset — more risks overfitting |
| Batch size | 16 | Fits in 8GB M2 RAM |
| Learning rate | 2e-5 | Standard for sentence-transformers |
| Warmup steps | 100 | Prevents early destabilization |
| Eval steps | 500 | Monitors Spearman during training |
| Optimizer | AdamW | Weight decay prevents overfitting |
| Scheduler | WarmupLinear | Ramp up, then linear decay |

LoRA-specific:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rank (r) | 8 | Good balance for small models |
| Alpha | 16 | 2×r is standard scaling |
| Dropout | 0.1 | Prevents overfitting on 1,195 examples |
| Target modules | query, value | Attention matrices that determine "what to attend to" |

---

## Troubleshooting Guide

### "PEFT LoRA won't load on M2"
- PEFT may try to use MPS backend. Force CPU: `export PYTORCH_ENABLE_MPS_FALLBACK=1`
- If still fails: skip LoRA entirely. This is a bonus comparison, not a blocker.
- Document in ADR-001 regardless.

### "Training is very slow (>2h for 4 epochs)"
- Confirm batch_size=16 not 32
- Check `htop` — if swap is being used, close Chrome/VS Code extensions
- Reduce to 3 epochs if >90 min. 3 epochs is enough if Spearman is improving.

### "UMAP produces different results each run"
- Check `random_state=42` is set. UMAP is stochastic by default.
- If still non-deterministic, also set `n_jobs=1` (parallel execution introduces randomness).

### "HDBSCAN finds no clusters (all noise)"
- Lower `min_cluster_size` from 10 → 5
- Lower `min_samples` from 5 → 3
- If still all noise on baseline: that's actually a valid finding — document it ("pre-trained model doesn't create dating-specific clusters")

### "Cohen's d is negative"
- This means incompatible pairs have HIGHER cosine similarity than compatible pairs. Shouldn't happen after fine-tuning. Debug: check labels aren't flipped, check model loaded correctly.

### "Spearman doesn't improve during training"
- Learning rate may be too low or too high. Try 1e-5 or 5e-5.
- Check data loading: are labels converted to float? (0.0 and 1.0, not 0 and 1)
- Verify eval_pairs.jsonl is being used for evaluation, not training data.
