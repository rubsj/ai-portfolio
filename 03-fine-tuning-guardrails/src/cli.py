"""CLI for P3 fine-tuning pipeline."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.baseline_analysis import run_full_baseline
from src.data_loader import load_pairs
from src.trainer import StandardTrainer

console = Console()

# WHY hardcoded paths: CLI uses same data as baseline analysis
TRAIN_DATA_PATH = Path("data/raw/dating_pairs.jsonl")
EVAL_DATA_PATH = Path("data/raw/eval_pairs.jsonl")


@click.group()
def cli() -> None:
    """P3: Contrastive Embedding Fine-Tuning CLI."""
    pass


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["standard", "lora"]),
    required=True,
    help="Training mode: standard (full fine-tuning) or lora (parameter-efficient)",
)
def train(mode: str) -> None:
    """Train embedding model on dating compatibility pairs.

    WHY separate command: training takes 45-90 minutes, CLI allows background execution
    with progress monitoring.
    """
    console.print(f"\n[bold blue]Starting {mode} training...[/bold blue]\n")

    # WHY load from raw JSONL: baseline already created train/eval split
    train_pairs = load_pairs(TRAIN_DATA_PATH)
    eval_pairs = load_pairs(EVAL_DATA_PATH)

    if mode == "standard":
        trainer = StandardTrainer(train_pairs, eval_pairs)
        result = trainer.train()

        # WHY Rich table: structured output easier to read than raw JSON
        table = Table(title="Standard Training Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Final Spearman", f"{result.final_spearman:.4f}")
        table.add_row("Training Time", f"{result.training_time_seconds:.1f}s")
        table.add_row("Trainable Params", f"{result.trainable_parameters:,}")
        table.add_row("Total Params", f"{result.total_parameters:,}")
        table.add_row("Output Path", result.output_path)

        console.print(table)

    elif mode == "lora":
        # WHY deferred import: LoRA implementation may not exist yet
        from src.lora_trainer import LoRATrainer

        trainer = LoRATrainer(train_pairs, eval_pairs)
        result = trainer.train()

        table = Table(title="LoRA Training Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Final Spearman", f"{result.final_spearman:.4f}")
        table.add_row("Training Time", f"{result.training_time_seconds:.1f}s")
        table.add_row("Trainable Params", f"{result.trainable_parameters:,}")
        table.add_row("Total Params", f"{result.total_parameters:,}")
        # WHY show efficiency: key LoRA benefit is trainable << total
        efficiency = (result.trainable_parameters / result.total_parameters) * 100
        table.add_row("Trainable %", f"{efficiency:.2f}%")
        table.add_row("LoRA Rank", str(result.lora_rank))
        table.add_row("Output Path", result.output_path)

        console.print(table)


@cli.command()
def baseline() -> None:
    """Re-run baseline analysis (Day 1 pipeline).

    WHY separate command: allows re-running evaluation after code changes without
    re-training models.
    """
    console.print("\n[bold blue]Running baseline analysis...[/bold blue]\n")

    # WHY load from raw JSONL: baseline already created train/eval split
    train_pairs = load_pairs(TRAIN_DATA_PATH)
    eval_pairs = load_pairs(EVAL_DATA_PATH)

    metrics = run_full_baseline(train_pairs, eval_pairs)

    # WHY print key metrics: full report saved to JSON, CLI shows summary
    console.print(
        f"[green]Baseline Spearman:[/green] {metrics.spearman_correlation:.4f}"
    )
    console.print(
        f"[green]Compatibility Margin:[/green] {metrics.compatibility_margin:.4f}"
    )
    console.print(f"[green]Cohen's d:[/green] {metrics.cohens_d:.4f}")


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["standard", "lora", "all"]),
    default="all",
    help="Evaluation mode: standard, lora, or all (default: all)",
)
def evaluate(mode: str) -> None:
    """Run post-training evaluation for fine-tuned models.

    WHY separate from training: enables re-evaluation with different metrics
    or after fixing bugs without re-training (45-90 min saved).

    WHY mode parameter: allows running only standard or LoRA if one model
    already has cached embeddings.
    """
    # WHY deferred import: avoid circular dependency with post_training_eval
    from src.post_training_eval import run_post_training_evaluation

    console.print(f"\n[bold blue]Running {mode} model evaluation...[/bold blue]\n")

    # WHY always run full orchestrator: it automatically skips embedding generation
    # if .npz files already exist, making mode distinction unnecessary for caching
    run_post_training_evaluation()

    console.print("\n[green]✓ Evaluation complete![/green]")
    console.print("Metrics saved to:")
    console.print("  • eval/finetuned_metrics.json (standard model)")
    console.print("  • eval/lora_metrics.json (LoRA model)")


@cli.command()
def compare() -> None:
    """Generate comparison charts and HTML report.

    WHY separate from evaluate: comparison works from cached embeddings
    and metrics JSONs, enabling fast re-runs when tweaking visualizations
    without re-running evaluation.
    """
    # WHY deferred import: avoid loading matplotlib/seaborn until needed
    from src.comparison import run_comparison

    console.print("\n[bold blue]Generating comparison analysis...[/bold blue]\n")

    run_comparison()

    console.print("\n[green]✓ Comparison complete![/green]")
    console.print("Outputs:")
    console.print("  • eval/comparison_report.html (self-contained with charts)")
    console.print("  • eval/visualizations/comparison/*.png (8 charts)")
    console.print("  • eval/false_positive_analysis.txt")
    console.print("  • eval/comparison_result.json")


if __name__ == "__main__":
    cli()
