content = '''"""Typer-based command-line interface for EvalForge."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)

import typer
from rich.console import Console
from rich.table import Table

from evalforge.reports.generator import ReportGenerator
from evalforge.runner.executor import EvalExecutor
from evalforge.storage.results import get_run_stats
from evalforge.storage.runs import list_runs as db_list_runs
from evalforge.tasks.registry import TaskRegistry

app = typer.Typer(help="EvalForge evaluation CLI")

_ALLOWED_PROVIDERS = frozenset({"anthropic", "openai", "gemini", "ollama"})
console = Console()


@app.command()
def run(
    task: str = typer.Option(..., "--task", help="Task name (tasks/{task}.yaml)"),
    provider: str = typer.Option(..., "--provider", help="LLM provider"),
    model: str = typer.Option(..., "--model", help="Model name"),
    model_version: Optional[str] = typer.Option(None, "--model-version", help="Optional model version label"),
) -> None:
    """Run an evaluation task against a model."""
    task_names = TaskRegistry().list_tasks()
    if task not in task_names:
        console.print(f"[bold red]Error:[/bold red] unknown task [cyan]{task}[/cyan]. Available: {', '.join(task_names) or '(none)'}")
        raise typer.Exit(code=1)

    if provider not in _ALLOWED_PROVIDERS:
        console.print(f"[bold red]Error:[/bold red] provider must be one of {', '.join(sorted(_ALLOWED_PROVIDERS))}; got [cyan]{provider}[/cyan]")
        raise typer.Exit(code=1)

    async def _execute() -> tuple[str, dict]:
        with console.status("[bold green]Running evaluation..."):
            run_id = await EvalExecutor().run(task, provider, model, model_version)
        stats = await get_run_stats(run_id)
        return run_id, stats

    try:
        run_id, stats = asyncio.run(_execute())
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc

    total = int(stats.get("total_items") or 0)
    passed = int(stats.get("passed_items") or 0)
    avg_comp = float(stats.get("avg_composite_score") or 0.0)
    pct = (100.0 * passed / total) if total else 0.0

    console.print(f"Run ID: {run_id}")
    console.print(f"Pass rate: {passed}/{total} ({pct:.1f}%)")
    console.print(f"Avg composite score: {avg_comp:.4f}")


@app.command("list-runs")
def list_runs_cmd(limit: int = typer.Option(20, "--limit", min=1)) -> None:
    """List recent evaluation runs."""
    rows = asyncio.run(db_list_runs(limit))
    table = Table(show_header=True, header_style="bold")
    table.add_column("Run ID")
    table.add_column("Task")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Pass Rate")
    table.add_column("Started At")

    for row in rows:
        rid = row.get("id") or ""
        short_id = rid[:8] if rid else ""
        total = row.get("total_items")
        passed = row.get("passed_items")
        pr = f"{int(passed or 0)}/{int(total)}" if total else "-"
        table.add_row(short_id, str(row.get("task_name") or ""), str(row.get("provider") or ""), str(row.get("model_name") or ""), str(row.get("status") or ""), pr, str(row.get("started_at") or ""))

    console.print(table)


@app.command("list-tasks")
def list_tasks_cmd() -> None:
    """List task definitions found in tasks/."""
    names = TaskRegistry().list_tasks()
    if not names:
        console.print("No tasks found in tasks/")
        return
    for name in names:
        console.print(name)


@app.command("report")
def report_cmd(run_id: str = typer.Option(..., "--run-id", help="Eval run UUID")) -> None:
    """Generate an HTML report for a completed run."""
    try:
        path = asyncio.run(ReportGenerator().build(run_id))
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc
    console.print(f"Report saved: {path}")
'''

with open("evalforge/cli.py", "w") as f:
    f.write(content)
print("Done")