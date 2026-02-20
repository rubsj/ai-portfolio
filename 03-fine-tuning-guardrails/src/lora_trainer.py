"""LoRA fine-tuning trainer for dating compatibility embeddings.

WHY LoRA: Parameter-Efficient Fine-Tuning (PEFT) — trains only ~1% of parameters
while achieving comparable performance to full fine-tuning. Critical for resource-
constrained environments and faster iteration.
"""

from __future__ import annotations

import csv
import gc
import time
from pathlib import Path

from src.models import DatingPair, TrainingResult

# WHY import shared constants: LoRA uses identical hyperparams except learning_rate
from src.trainer import (
    BATCH_SIZE,
    EPOCHS,
    EVALUATION_STEPS,
    MODEL_NAME,
    WARMUP_STEPS,
)

# LoRA-specific configuration
OUTPUT_DIR = Path("training/model/lora_model")
TRAINING_INFO_PATH = Path("training/lora_training_info.json")

# WHY 2e-4 not 2e-5: LoRA adapters are only 0.32% of total params (73K/22.7M)
# Higher LR needed for small parameter count to learn effectively. Standard uses 2e-5
# for full 22.7M params. 10x higher LR compensates for ~32x fewer trainable params.
LORA_LEARNING_RATE = 2e-4

# WHY r=8: balance between expressiveness and efficiency, standard LoRA default
# WHY alpha=16: scaling factor (alpha/r = 2.0), controls magnitude of LoRA updates
# WHY dropout=0.1: regularization to prevent overfitting on small adapter layers
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# WHY ["query", "value"]: attention mechanism Q/V matrices most impactful for embeddings
# K (key) often frozen as it's more task-agnostic. FFN layers less critical for contrastive loss.
LORA_TARGET_MODULES = ["query", "value"]


class LoRATrainer:
    """Trains sentence-transformers model with LoRA adapters on dating pairs.

    WHY LoRA: Adds low-rank decomposition matrices (A, B) to transformer layers.
    Instead of updating full weight matrix W, trains W + BA where A and B are small.
    For r=8 and d=384 (MiniLM), BA has 384×8 + 8×384 = 6,144 params vs W's 147,456.
    """

    def __init__(
        self,
        train_pairs: list[DatingPair],
        eval_pairs: list[DatingPair],
        output_dir: Path = OUTPUT_DIR,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LORA_LEARNING_RATE,
        warmup_steps: int = WARMUP_STEPS,
        evaluation_steps: int = EVALUATION_STEPS,
        lora_rank: int = LORA_RANK,
        lora_alpha: int = LORA_ALPHA,
        lora_dropout: float = LORA_DROPOUT,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        self.train_pairs = train_pairs
        self.eval_pairs = eval_pairs
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # WHY default to module constant: allows override for experimentation
        self.lora_target_modules = lora_target_modules or LORA_TARGET_MODULES

    def _apply_lora(self, model) -> tuple[int, int]:
        """Wrap model's transformer with LoRA adapters.

        WHY model[0].auto_model: SentenceTransformer wraps a HuggingFace model.
        model[0] is the Transformer module, auto_model is the actual BertModel.
        LoRA needs to wrap the BertModel directly.

        WHY PEFT get_peft_model: HuggingFace PEFT library handles:
        - Freezing base model weights (requires_grad=False)
        - Adding trainable LoRA adapters to target modules
        - Merging adapters back into base weights at inference

        Returns:
            (total_params, trainable_params) — trainable << total for LoRA
        """
        # WHY import here: keeps module-level imports light, tests don't load torch/PEFT
        from peft import LoraConfig, TaskType, get_peft_model

        # WHY TaskType.FEATURE_EXTRACTION: tells PEFT this is not seq2seq/classification
        # Embedding models need all hidden states, not just CLS token
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        # WHY wrap inner model: sentence-transformers model is a pipeline,
        # LoRA adapters go on the transformer component
        base_model = model[0].auto_model
        peft_model = get_peft_model(base_model, lora_config)

        # WHY replace in-place: sentence-transformers fit() method uses model[0].auto_model
        model[0].auto_model = peft_model

        # WHY count parameters: demonstrates LoRA efficiency (trainable ~1-5% of total)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return total_params, trainable_params

    def prepare_data(self) -> tuple:
        """Convert DatingPair → InputExample, create DataLoader and evaluator.

        WHY identical to StandardTrainer: LoRA changes training mechanism, not data format.
        """
        # WHY import here: keeps module-level imports light
        from sentence_transformers import InputExample
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        from torch.utils.data import DataLoader

        train_examples = [
            InputExample(texts=[pair.text_1, pair.text_2], label=float(pair.label))
            for pair in self.train_pairs
        ]

        eval_examples = [
            InputExample(texts=[pair.text_1, pair.text_2], label=float(pair.label))
            for pair in self.eval_pairs
        ]

        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=self.batch_size
        )

        # WHY name='dating-eval-lora': prevents overwriting standard trainer's CSV
        # Each trainer needs separate CSV for Spearman history tracking
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            eval_examples, name="dating-eval-lora"
        )

        return train_dataloader, evaluator

    def train(self) -> TrainingResult:
        """Execute LoRA training loop, save adapters, return metrics.

        WHY same as StandardTrainer except _apply_lora(): LoRA is a drop-in replacement,
        model.fit() works identically because PEFT wraps the PyTorch module transparently.

        Returns:
            TrainingResult with LoRA-specific fields populated
        """
        # WHY import here: keeps module-level imports light
        from sentence_transformers import SentenceTransformer, losses

        start_time = time.time()

        model = SentenceTransformer(MODEL_NAME)

        # WHY apply LoRA before fit(): freezes base weights, adds trainable adapters
        total_params, trainable_params = self._apply_lora(model)

        train_dataloader, evaluator = self.prepare_data()

        train_loss = losses.CosineSimilarityLoss(model)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # WHY identical hyperparams: isolates LoRA effect from other variables
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.epochs,
            evaluation_steps=self.evaluation_steps,
            warmup_steps=self.warmup_steps,
            output_path=str(self.output_dir),
            optimizer_params={"lr": self.learning_rate},
            scheduler="WarmupLinear",
            save_best_model=True,
            show_progress_bar=True,
        )

        training_time = time.time() - start_time

        spearman_history = self._parse_evaluator_csv()
        final_spearman = spearman_history[-1][1] if spearman_history else 0.0

        result = TrainingResult(
            model_type="lora",
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            evaluation_steps=self.evaluation_steps,
            training_time_seconds=training_time,
            final_spearman=final_spearman,
            spearman_history=spearman_history,
            output_path=str(self.output_dir),
            trainable_parameters=trainable_params,
            total_parameters=total_params,
            # LoRA-specific fields
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_target_modules=self.lora_target_modules,
        )

        # WHY save JSON: training metrics needed for comparison with standard
        TRAINING_INFO_PATH.write_text(result.model_dump_json(indent=2))

        # WHY explicit cleanup: prevents OOM on 8GB M2 MacBook Air
        del model
        del train_dataloader
        del train_loss
        gc.collect()

        return result

    def _parse_evaluator_csv(self) -> list[tuple[int, float]]:
        """Parse Spearman correlation history from evaluator CSV.

        WHY same glob pattern as StandardTrainer: handles version differences.
        """
        csv_files = list(self.output_dir.glob("**/*similarity_evaluation*results.csv"))

        if not csv_files:
            return []

        csv_path = csv_files[0]

        history = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # WHY int(float()): CSV epoch is "1.0" not "1", must convert float→int
                step = int(float(row["epoch"]))
                spearman = float(row["cosine_spearman"])
                history.append((step, spearman))

        return history
