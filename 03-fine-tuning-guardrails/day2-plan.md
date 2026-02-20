# P3 Day 2 Plan: Fine-Tuning Standard + LoRA (T2.1–T2.5)

## Context

Day 1 established a baseline where the pre-trained `all-MiniLM-L6-v2` model is **inverted** — incompatible pairs have higher cosine similarity than compatible ones (margin: -0.083, Spearman: -0.219). Day 2 fine-tunes this model using contrastive loss to flip these numbers positive, then optionally compares with LoRA parameter-efficient training.

**Branch:** `feat/p3-day2-training` from `origin/main`

---

## Execution Order (Critical Path)

```
[0:00–0:30]  Implement trainer.py + TrainingResult model + tests
[0:30–0:40]  Implement cli.py
[0:40–0:45]  Commit: "feat(p3): standard trainer + CLI"

[0:45]       *** START standard training in background terminal ***
             python -m src.cli train --mode standard  (45–90 min)

[0:45–1:15]  *** WHILE training runs: implement lora_trainer.py + tests ***
[1:15–1:20]  Commit: "feat(p3): LoRA trainer implementation"

[1:20–2:15]  Wait for standard training. If >90min, reduce epochs to 3.

[2:15–2:25]  Verify standard model, commit with Spearman number
[2:25]       *** START LoRA training in background ***
[2:25–2:45]  Add plot_training_curves() to visualizations.py

[2:45–3:30]  Wait for LoRA. If PEFT fails: MAX 30 min debug, then skip.
[3:30–3:45]  Generate training curves chart (T2.5)
[3:45–4:00]  Final commit, update CLAUDE.md
```

---

## File 1: `src/models.py` — Add TrainingResult (modify existing)

Add after `ComparisonResult` at line 99:

```python
class TrainingResult(BaseModel):
    """Training hyperparameters, timing, and Spearman progression."""
    model_type: Literal["standard", "lora"]   # Pydantic validates at construction
    epochs: int
    batch_size: int
    learning_rate: float
    warmup_steps: int
    evaluation_steps: int
    training_time_seconds: float
    final_spearman: float
    spearman_history: list[tuple[int, float]] = Field(default_factory=list)
    output_path: str
    trainable_parameters: int
    total_parameters: int
    # LoRA-specific (None for standard)
    lora_rank: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_target_modules: list[str] | None = None
```

---

## File 2: `src/trainer.py` — StandardTrainer (create new)

### Constants
```python
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EVALUATION_STEPS = 500
OUTPUT_DIR = Path("training/model/standard_model")
TRAINING_INFO_PATH = Path("training/standard_training_info.json")
```

### Class: `StandardTrainer`

```python
class StandardTrainer:
    def __init__(
        self,
        train_pairs: list[DatingPair],
        eval_pairs: list[DatingPair],
        output_dir: Path = OUTPUT_DIR,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        warmup_steps: int = WARMUP_STEPS,
        evaluation_steps: int = EVALUATION_STEPS,
    ) -> None: ...

    def prepare_data(self) -> tuple[DataLoader, EmbeddingSimilarityEvaluator]:
        """Convert DatingPair → InputExample(texts=[t1,t2], label=float(label)).
        Return DataLoader(shuffle=True, batch_size=16) + evaluator from eval pairs."""

    def train(self) -> TrainingResult:
        """Load model → prepare_data → CosineSimilarityLoss → model.fit() → cleanup.
        model.fit() params: train_objectives, evaluator, epochs=4, evaluation_steps=500,
        warmup_steps=100, output_path, optimizer_params={"lr": 2e-5},
        scheduler="WarmupLinear", save_best_model=True, show_progress_bar=True.
        After: parse Spearman CSV, save training_info.json, del model + gc.collect()."""

    def _parse_evaluator_csv(self) -> list[tuple[int, float]]:
        """Glob for **/*similarity_evaluation*results.csv under output_dir.
        Don't hardcode path — sentence-transformers versions vary on subdirectory structure.
        Returns list of (step, cosine_spearman). Empty list if no CSV found."""
```

**Key details:**
- Heavy imports (sentence_transformers, torch) inside methods, not module-level — keeps tests fast
- `float(pair.label)` — CosineSimilarityLoss expects float in [0,1], not int
- Reuse: `from src.models import DatingPair, TrainingResult`
- Memory: `del model; del train_dataloader; del train_loss; gc.collect()`
- **save_best_model=True** in model.fit() is critical — without it, only the final epoch model is saved, not the best Spearman checkpoint

---

## File 2b: `.gitignore` — Exclude model weights (modify existing)

Add to `.gitignore`:
```
# Training model weights (~90MB each, must not be committed)
training/model/*/
# DO commit training/*.json (training info) and training/*.png (charts)
```

---

## File 3: `src/lora_trainer.py` — LoRATrainer (create new)

### Constants (import shared hyperparams from trainer.py)
```python
from src.trainer import MODEL_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, WARMUP_STEPS, EVALUATION_STEPS

OUTPUT_DIR = Path("training/model/lora_model")
TRAINING_INFO_PATH = Path("training/lora_training_info.json")
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]
```

### Class: `LoRATrainer`

```python
class LoRATrainer:
    def __init__(
        self,
        train_pairs: list[DatingPair],
        eval_pairs: list[DatingPair],
        output_dir: Path = OUTPUT_DIR,
        # ... same hyperparams as StandardTrainer ...
        lora_rank: int = LORA_RANK,
        lora_alpha: int = LORA_ALPHA,
        lora_dropout: float = LORA_DROPOUT,
        lora_target_modules: list[str] | None = None,
    ) -> None: ...

    def _apply_lora(self, model) -> tuple[int, int]:
        """Wrap model[0].auto_model with get_peft_model(auto_model, LoraConfig(...)).
        LoraConfig: r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["query","value"], task_type=TaskType.FEATURE_EXTRACTION.
        Returns (total_params, trainable_params)."""

    def train(self) -> TrainingResult:
        """Same as StandardTrainer.train() but with _apply_lora() before model.fit().
        IMPORTANT: EmbeddingSimilarityEvaluator must use name='dating-eval-lora' (not 'dating-eval')
        to avoid overwriting the standard evaluator's CSV file.
        Saves adapter weights + lora_training_info.json. Cleanup: del model, gc.collect()."""

    def _parse_evaluator_csv(self) -> list[tuple[int, float]]:
        """Same glob pattern as StandardTrainer — **/*similarity_evaluation*results.csv."""
```

**PEFT integration approach:**
1. `model = SentenceTransformer(MODEL_NAME)` — loads base model
2. `model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)` — wraps inner BertModel
3. `model.fit(...)` — sentence-transformers handles training; gradients only flow to LoRA adapters
4. If `model.fit()` breaks with PEFT wrapper → fallback to manual PyTorch loop (Approach B)

**CRITICAL FALLBACK:** Max 30 min debugging PEFT issues. If unresolved, skip LoRA entirely. Document in ADR.

---

## File 4: `src/cli.py` — Click CLI (create new)

```python
@click.group()
def cli(): ...

@cli.command()
@click.option("--mode", type=click.Choice(["standard", "lora"]), required=True)
def train(mode: str):
    """Load data → create trainer → train() → print summary with Rich."""

@cli.command()
def baseline():
    """Re-run baseline analysis (Day 1 pipeline)."""

if __name__ == "__main__":
    cli()
```

Usage: `python -m src.cli train --mode standard`

---

## File 5: `src/visualizations.py` — Add training curves function (modify existing)

```python
def plot_training_curves(
    standard_history: list[tuple[int, float]],
    lora_history: list[tuple[int, float]] | None = None,
    save_path: Path | None = None,
) -> Path:
    """Both models on same axes. Baseline reference at -0.219, target at 0.86.
    Standard = blue, LoRA = orange. Save to training/training_curves.png."""
```

---

## File 6: `tests/test_trainer.py` — 9 tests (create new)

Mock strategy: `monkeypatch.setitem(sys.modules, "sentence_transformers", mock_st_module)` (same pattern as `test_baseline_analysis.py`)

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_prepare_data_creates_input_examples` | InputExample(texts=[t1,t2], label=float) |
| 2 | `test_prepare_data_labels_are_float` | label=0 → 0.0, label=1 → 1.0 |
| 3 | `test_train_hyperparams_match_prd` | epochs=4, batch=16, lr=2e-5, warmup=100, eval_steps=500 |
| 4 | `test_train_calls_model_fit_with_correct_args` | All model.fit() kwargs verified |
| 5 | `test_train_returns_training_result` | Returns TrainingResult, model_type=="standard" |
| 6 | `test_train_saves_training_info_json` | JSON file written with correct fields |
| 7 | `test_train_cleans_up_memory` | gc.collect() called |
| 8 | `test_parse_evaluator_csv_valid` | Parses (step, spearman) from CSV |
| 9 | `test_parse_evaluator_csv_missing` | Returns [] without crash |

---

## File 7: `tests/test_lora_trainer.py` — 6 tests (create new)

Additional mocks: `monkeypatch.setitem(sys.modules, "peft", mock_peft_module)`

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_lora_config_matches_prd` | r=8, alpha=16, dropout=0.1, target=["query","value"] |
| 2 | `test_apply_lora_calls_get_peft_model` | Called with correct config |
| 3 | `test_apply_lora_replaces_auto_model` | model[0].auto_model is swapped |
| 4 | `test_train_returns_lora_result` | model_type=="lora", lora fields populated |
| 5 | `test_hyperparams_same_as_standard` | All shared hyperparams identical |
| 6 | `test_saves_training_info_json` | JSON includes lora_rank, lora_alpha |

---

## File 8: `tests/test_cli.py` — 3 tests (create new)

Use Click's `CliRunner` for invocation tests.

| # | Test | Validates |
|---|------|-----------|
| 1 | `test_cli_has_train_command` | "train" registered |
| 2 | `test_cli_has_baseline_command` | "baseline" registered |
| 3 | `test_train_mode_validation` | --mode=invalid fails, standard/lora pass |

---

## Verification Checklist

### After T2.1 (trainer.py):
- [ ] `uv run ruff check src/trainer.py` — clean
- [ ] `uv run pytest tests/test_trainer.py -v` — all pass
- [ ] `python -c "from src.trainer import StandardTrainer"` — imports without loading torch

### After T2.2 (standard training complete):
- [ ] `training/model/standard_model/` contains model files
- [ ] `training/standard_training_info.json` has final_spearman > 0
- [ ] Quick-test: load model, encode compatible pair, cosine > 0.7

### After T2.3 (lora_trainer.py):
- [ ] `uv run ruff check src/lora_trainer.py` — clean
- [ ] `uv run pytest tests/test_lora_trainer.py -v` — all pass

### After T2.4 (LoRA training complete, or skipped):
- [ ] If success: `training/lora_training_info.json` has trainable_params << total_params
- [ ] If skipped: issue documented, LoRA noted as bonus

### After T2.5 (training curves):
- [ ] `training/training_curves.png` exists with standard curve (+ LoRA if available)
- [ ] Chart shows upward Spearman trend from negative baseline

---

## Git Commits (3 planned)

1. `feat(p3): standard trainer + CLI` — trainer.py, cli.py, models.py update, tests
2. `feat(p3): LoRA trainer implementation` — lora_trainer.py, tests
3. `feat(p3): fine-tuning complete — standard spearman=X.XX, lora spearman=X.XX` — training outputs, curves chart, CLAUDE.md update
