"""Renders HTML and related artifacts from stored run results."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from evalforge.storage.results import get_results_for_run, get_run_stats
from evalforge.storage.runs import get_run


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ReportGenerator:
    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or (_project_root() / "data" / "reports")

    async def build(self, run_id: str) -> str:
        run = await get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")
        results = await get_results_for_run(run_id)
        stats = await get_run_stats(run_id)

        tmpl_dir = Path(__file__).resolve().parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(tmpl_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template("report.html")
        html = template.render(run=run, results=results, stats=stats)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{run_id}.html"
        out_path.write_text(html, encoding="utf-8")
        return str(out_path.resolve())
