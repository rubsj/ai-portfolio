"""CLI for P2 RAG Evaluation Benchmarking Framework.

WHY CLI: Quick interface for running grid search, viewing results, and comparing
configs without launching Streamlit. Uses Click for arg parsing and Rich for
formatted tables and progress bars.

Commands:
  - rag-eval run: Execute full grid search pipeline
  - rag-eval report: Display results summary with Rich tables
  - rag-eval compare: Compare specific configs side-by-side

Java/TS parallel: Similar to Spring Boot CLI or npm scripts, provides command-line
interface to core functionality for developer/researcher workflows.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.config import METRICS_DIR, REPORTS_DIR

console = Console()


@click.group()
def cli():
    """P2 RAG Evaluation Benchmarking Framework CLI.

    Evaluate 16 RAG configurations across 5 evaluation layers.
    """
    pass


@cli.command()
@click.option(
    "--skip-reranking",
    is_flag=True,
    help="Skip reranking evaluation (Layer 3)",
)
@click.option(
    "--skip-ragas",
    is_flag=True,
    help="Skip RAGAS generation quality evaluation (Layer 4)",
)
@click.option(
    "--skip-judge",
    is_flag=True,
    help="Skip LLM-as-Judge evaluation (Layer 5)",
)
def run(skip_reranking: bool, skip_ragas: bool, skip_judge: bool):
    """Run full grid search pipeline (all 5 evaluation layers).

    WHY separate flags: Allow skipping expensive LLM-based evaluations
    during development/debugging. Layers 1-2 (retrieval metrics + BM25)
    are always executed.
    """
    console.print("\n[bold cyan]P2 RAG Evaluation Grid Search[/bold cyan]")
    console.print("=" * 60)

    # Import here to avoid slow startup for help commands
    from src.grid_search import main as run_grid_search

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Task 1: Grid Search (Layers 1-2)
        task1 = progress.add_task("[cyan]Running grid search (16 configs)...", total=None)
        try:
            run_grid_search()
            progress.update(task1, completed=True)
            console.print("✅ Grid search complete (Layers 1-2)")
        except Exception as e:
            console.print(f"[red]❌ Grid search failed: {e}[/red]")
            sys.exit(1)

        # Task 2: Reranking (Layer 3)
        if not skip_reranking:
            task2 = progress.add_task("[cyan]Running reranking evaluation...", total=None)
            try:
                from src.reranker import evaluate_reranking
                evaluate_reranking()
                progress.update(task2, completed=True)
                console.print("✅ Reranking evaluation complete (Layer 3)")
            except Exception as e:
                console.print(f"[yellow]⚠️  Reranking failed: {e}[/yellow]")

        # Task 3: RAGAS (Layer 4)
        if not skip_ragas:
            task3 = progress.add_task("[cyan]Running RAGAS evaluation...", total=None)
            try:
                from src.generation_evaluator import evaluate_generation_quality
                evaluate_generation_quality()
                progress.update(task3, completed=True)
                console.print("✅ RAGAS evaluation complete (Layer 4)")
            except Exception as e:
                console.print(f"[yellow]⚠️  RAGAS failed: {e}[/yellow]")

        # Task 4: LLM Judge (Layer 5)
        if not skip_judge:
            task4 = progress.add_task("[cyan]Running LLM-as-Judge evaluation...", total=None)
            try:
                from src.judge import evaluate_with_judge
                evaluate_with_judge()
                progress.update(task4, completed=True)
                console.print("✅ LLM Judge evaluation complete (Layer 5)")
            except Exception as e:
                console.print(f"[yellow]⚠️  Judge evaluation failed: {e}[/yellow]")

    console.print("\n[bold green]✅ Pipeline complete![/bold green]")
    console.print(f"Results saved to: [cyan]{METRICS_DIR}[/cyan]")
    console.print("\nNext steps:")
    console.print("  • Run [cyan]rag-eval report[/cyan] to view results")
    console.print("  • Run [cyan]streamlit run streamlit_app.py[/cyan] for interactive demo")


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format (table or json)",
)
@click.option(
    "--top-n",
    type=int,
    default=5,
    help="Show top N configs (default: 5)",
)
def report(format: str, top_n: int):
    """Display grid search results summary.

    WHY Rich tables: Formatted output for terminal viewing. Better than
    raw JSON for quick inspection of results.
    """
    # Load results
    grid_path = METRICS_DIR / "grid_search_results.json"
    if not grid_path.exists():
        console.print(f"[red]❌ No results found at {grid_path}[/red]")
        console.print("Run [cyan]rag-eval run[/cyan] first to generate results.")
        sys.exit(1)

    data = json.loads(grid_path.read_text())

    # Sort by Recall@5 (descending)
    sorted_data = sorted(data, key=lambda x: x["avg_recall_at_5"], reverse=True)

    if format == "json":
        # JSON output (useful for piping to jq)
        # WHY print instead of console.print: raw JSON for piping, no ANSI codes
        print(json.dumps(sorted_data[:top_n], indent=2))
        return

    console.print("\n[bold cyan]RAG Evaluation Results Summary[/bold cyan]")
    console.print("=" * 60)

    # Table output
    table = Table(title=f"Top {top_n} Configurations by Recall@5")
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Config ID", style="magenta")
    table.add_column("Chunk Config", style="green")
    table.add_column("Embedding", style="blue")
    table.add_column("R@5", justify="right", style="yellow")
    table.add_column("P@5", justify="right", style="yellow")
    table.add_column("MRR@5", justify="right", style="yellow")

    for rank, item in enumerate(sorted_data[:top_n], start=1):
        table.add_row(
            str(rank),
            item["config_id"],
            item["chunk_config"],
            item["embedding_model"],
            f"{item['avg_recall_at_5']:.3f}",
            f"{item['avg_precision_at_5']:.3f}",
            f"{item['avg_mrr_at_5']:.3f}",
        )

    console.print(table)

    # Show BM25 baseline
    bm25_items = [x for x in data if x["retrieval_method"] == "bm25"]
    if bm25_items:
        bm25 = bm25_items[0]
        console.print(f"\n[bold]BM25 Baseline:[/bold] R@5={bm25['avg_recall_at_5']:.3f}, P@5={bm25['avg_precision_at_5']:.3f}")

    # Show reranking impact if available
    rerank_path = METRICS_DIR / "reranking_results.json"
    if rerank_path.exists():
        console.print("\n[bold cyan]Reranking Impact:[/bold cyan]")
        rerank_data = json.loads(rerank_path.read_text())

        rerank_table = Table()
        rerank_table.add_column("Config", style="magenta")
        rerank_table.add_column("Before R@5", justify="right", style="yellow")
        rerank_table.add_column("After R@5", justify="right", style="green")
        rerank_table.add_column("Improvement", justify="right", style="cyan")

        for item in rerank_data:
            improvement = item["recall_improvement_pct"]
            rerank_table.add_row(
                item["config_id"],
                f"{item['recall_at_5_before']:.3f}",
                f"{item['recall_at_5_after']:.3f}",
                f"+{improvement:.1f}%",
            )

        console.print(rerank_table)

    # Show RAGAS scores if available
    ragas_path = METRICS_DIR / "ragas_results.json"
    if ragas_path.exists():
        console.print("\n[bold cyan]RAGAS Generation Quality (E-openai):[/bold cyan]")
        ragas_data = json.loads(ragas_path.read_text())

        ragas_table = Table()
        ragas_table.add_column("Metric", style="magenta")
        ragas_table.add_column("Score", justify="right", style="green")

        ragas_table.add_row("Faithfulness", f"{ragas_data['faithfulness']:.3f}")
        ragas_table.add_row("Answer Relevancy", f"{ragas_data['answer_relevancy']:.3f}")
        ragas_table.add_row("Context Recall", f"{ragas_data['context_recall']:.3f}")
        ragas_table.add_row("Context Precision", f"{ragas_data['context_precision']:.3f}")

        console.print(ragas_table)

    # Show QA dataset quality if available
    qa_report_path = REPORTS_DIR / "qa_dataset_report.json"
    if qa_report_path.exists():
        console.print("\n[bold cyan]QA Dataset Quality:[/bold cyan]")
        qa_data = json.loads(qa_report_path.read_text())

        qa_table = Table()
        qa_table.add_column("Metric", style="magenta")
        qa_table.add_column("Value", justify="right", style="green")

        qa_table.add_row("Total Questions", str(qa_data["total_questions"]))
        qa_table.add_row("Chunk Coverage", f"{qa_data['chunk_coverage_percent']:.1f}%")
        qa_table.add_row("Avg Questions/Chunk", f"{qa_data['avg_questions_per_chunk']:.3f}")

        console.print(qa_table)


@cli.command()
@click.argument("config_ids", nargs=-1, required=True)
def compare(config_ids: tuple[str, ...]):
    """Compare specific configs side-by-side.

    Example:
        rag-eval compare E-openai B-openai A-openai

    WHY side-by-side comparison: Easy to see relative performance across
    chunk strategies or embedding models.
    """
    console.print(f"\n[bold cyan]Comparing {len(config_ids)} Configurations[/bold cyan]")
    console.print("=" * 60)

    # Load results
    grid_path = METRICS_DIR / "grid_search_results.json"
    if not grid_path.exists():
        console.print(f"[red]❌ No results found at {grid_path}[/red]")
        sys.exit(1)

    data = json.loads(grid_path.read_text())

    # Filter to requested configs
    config_map = {item["config_id"]: item for item in data}
    selected = []
    missing = []

    for config_id in config_ids:
        if config_id in config_map:
            selected.append(config_map[config_id])
        else:
            missing.append(config_id)

    if missing:
        console.print(f"[yellow]⚠️  Configs not found: {', '.join(missing)}[/yellow]")

    if not selected:
        console.print("[red]❌ No valid configs to compare[/red]")
        sys.exit(1)

    # Build comparison table
    table = Table(title="Config Comparison")
    table.add_column("Metric", style="cyan")

    for item in selected:
        table.add_column(item["config_id"], style="magenta", justify="right")

    # Add rows for each metric
    metrics = [
        ("Chunk Config", "chunk_config"),
        ("Embedding Model", "embedding_model"),
        ("Retrieval Method", "retrieval_method"),
        ("Recall@1", "avg_recall_at_1"),
        ("Recall@3", "avg_recall_at_3"),
        ("Recall@5", "avg_recall_at_5"),
        ("Precision@1", "avg_precision_at_1"),
        ("Precision@3", "avg_precision_at_3"),
        ("Precision@5", "avg_precision_at_5"),
        ("MRR@1", "avg_mrr_at_1"),
        ("MRR@3", "avg_mrr_at_3"),
        ("MRR@5", "avg_mrr_at_5"),
    ]

    for label, field in metrics:
        row = [label]
        for item in selected:
            value = item.get(field, "N/A")
            # Format numbers to 3 decimal places
            if isinstance(value, float):
                row.append(f"{value:.3f}")
            else:
                row.append(str(value))
        table.add_row(*row)

    console.print(table)

    # Highlight winner for key metrics
    console.print("\n[bold]Winners:[/bold]")

    # Best Recall@5
    best_recall = max(selected, key=lambda x: x["avg_recall_at_5"])
    console.print(f"  🏆 Best Recall@5: [green]{best_recall['config_id']}[/green] ({best_recall['avg_recall_at_5']:.3f})")

    # Best Precision@5
    best_precision = max(selected, key=lambda x: x["avg_precision_at_5"])
    console.print(f"  🏆 Best Precision@5: [green]{best_precision['config_id']}[/green] ({best_precision['avg_precision_at_5']:.3f})")

    # Best MRR@5
    best_mrr = max(selected, key=lambda x: x["avg_mrr_at_5"])
    console.print(f"  🏆 Best MRR@5: [green]{best_mrr['config_id']}[/green] ({best_mrr['avg_mrr_at_5']:.3f})")


if __name__ == "__main__":
    cli()
