"""Standard contrastive fine-tuning trainer for dating compatibility embeddings.

WHY separate file: isolates heavy sentence-transformers imports from rest of codebase.
Imports torch/transformers only inside methods to keep test suite fast.
"""

from __future__ import annotations

import csv
import gc
import time
from pathlib import Path

from src.models import DatingPair, TrainingResult

# WHY module constants: PRD specifies exact hyperparameters, centralizing prevents drift
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5
WARMUP_STEPS = 100
EVALUATION_STEPS = 500
OUTPUT_DIR = Path("training/model/standard_model")
TRAINING_INFO_PATH = Path("training/standard_training_info.json")


class StandardTrainer:
    """Trains sentence-transformers model with CosineSimilarityLoss on dating pairs.

    WHY contrastive loss: compatible pairs should have high cosine similarity (→1),
    incompatible pairs should have low cosine similarity (→0). Loss directly optimizes
    the metric we'll evaluate on (cosine distance between embeddings).
    """

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
    ) -> None:
        self.train_pairs = train_pairs
        self.eval_pairs = eval_pairs
        self.output_dir = output_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.evaluation_steps = evaluation_steps

    def prepare_data(self) -> tuple:
        """Convert DatingPair → InputExample, create DataLoader and evaluator.

        WHY InputExample: sentence-transformers expects texts=[text1, text2], label=float.
        WHY float(label): CosineSimilarityLoss expects continuous values in [0,1], not int.
        WHY shuffle=True: prevents model from learning order patterns in training data.
        WHY EmbeddingSimilarityEvaluator: computes Spearman correlation during training,
        allows early stopping on best checkpoint.

        Returns:
            (DataLoader, EmbeddingSimilarityEvaluator)
        """
        # WHY import here: keeps module-level imports light, tests don't load torch
        from sentence_transformers import InputExample
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
        from torch.utils.data import DataLoader

        # WHY extract gender prefix: downstream bias analysis needs "boy:" vs "girl:"
        train_examples = [
            InputExample(texts=[pair.text_1, pair.text_2], label=float(pair.label))
            for pair in self.train_pairs
        ]

        eval_examples = [
            InputExample(texts=[pair.text_1, pair.text_2], label=float(pair.label))
            for pair in self.eval_pairs
        ]

        # WHY DataLoader: handles batching, shuffling, multi-worker loading
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=self.batch_size
        )

        # WHY name='dating-eval': used in CSV filename for Spearman history
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
            eval_examples, name="dating-eval"
        )

        return train_dataloader, evaluator

    def train(self) -> TrainingResult:
        """Execute training loop, save model, return metrics.

        WHY save_best_model=True: critical — without it, only final epoch is saved,
        not the checkpoint with best Spearman. Overfitting can degrade final epoch.

        WHY WarmupLinear scheduler: gradual learning rate ramp prevents early instability,
        then linear decay helps convergence.

        WHY gc.collect(): M2 MacBook Air has 8GB RAM, model + optimizer states consume ~2GB,
        explicit cleanup prevents OOM when running LoRA training immediately after.

        Returns:
            TrainingResult with hyperparams, timing, Spearman history
        """
        # WHY import here: keeps module-level imports light
        from sentence_transformers import SentenceTransformer, losses

        start_time = time.time()

        # WHY sentence-transformers wrapper: handles tokenization, pooling, normalization
        model = SentenceTransformer(MODEL_NAME)

        train_dataloader, evaluator = self.prepare_data()

        # WHY CosineSimilarityLoss: directly optimizes cosine similarity between embeddings
        # based on label (1 → similar, 0 → dissimilar)
        train_loss = losses.CosineSimilarityLoss(model)

        # WHY output_path as str: sentence-transformers.fit() doesn't accept Path objects
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # WHY all hyperparams explicit: prevents accidental changes from library version updates
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=self.epochs,
            evaluation_steps=self.evaluation_steps,
            warmup_steps=self.warmup_steps,
            output_path=str(self.output_dir),
            optimizer_params={"lr": self.learning_rate},
            scheduler="WarmupLinear",
            save_best_model=True,  # WHY critical: saves best Spearman checkpoint, not just final epoch
            show_progress_bar=True,
        )

        training_time = time.time() - start_time

        # WHY parse CSV: sentence-transformers writes Spearman history to CSV, not returned
        spearman_history = self._parse_evaluator_csv()
        final_spearman = spearman_history[-1][1] if spearman_history else 0.0

        # WHY count parameters: needed for comparison with LoRA (trainable << total)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        result = TrainingResult(
            model_type="standard",
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
        )

        # WHY save JSON: training metrics needed for analysis, model weights separate
        TRAINING_INFO_PATH.write_text(result.model_dump_json(indent=2))

        # WHY explicit cleanup: prevents OOM on 8GB M2 MacBook Air
        del model
        del train_dataloader
        del train_loss
        gc.collect()

        return result

    def _parse_evaluator_csv(self) -> list[tuple[int, float]]:
        """Parse Spearman correlation history from evaluator CSV.

        WHY glob pattern: sentence-transformers versions vary on subdirectory structure,
        some write to eval/, some to output_dir directly. Glob finds the file regardless.

        Returns:
            List of (step, cosine_spearman) tuples, empty if no CSV found
        """
        # WHY glob: handles version differences in sentence-transformers CSV paths
        csv_files = list(self.output_dir.glob("**/*similarity_evaluation*results.csv"))

        if not csv_files:
            return []

        # WHY first match: should only be one CSV, but if multiple, take first
        csv_path = csv_files[0]

        history = []
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # WHY int/float conversion: CSV stores as strings
                step = int(row["epoch"])
                spearman = float(row["cosine_spearman"])
                history.append((step, spearman))

        return history
