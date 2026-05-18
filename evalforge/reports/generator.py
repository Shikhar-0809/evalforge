"""Renders HTML and related artifacts from stored run results."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from evalforge.storage.results import get_results_for_run, get_run_stats
from evalforge.storage.runs import get_run


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _float_score(row: dict, key: str) -> float:
    v = row.get(key)
    if v is None:
        return 0.0
    return float(v)


def _score_distribution(results: list[dict]) -> list[dict]:
    labels = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    counts = [0, 0, 0, 0, 0]
    for r in results:
        s = _float_score(r, "composite_score")
        if s >= 1.0:
            idx = 4
        else:
            idx = min(int(s / 0.2), 4)
        counts[idx] += 1
    total = sum(counts) or 1
    return [
        {"label": labels[i], "count": counts[i], "pct": 100.0 * counts[i] / total}
        for i in range(5)
    ]


class ReportGenerator:
    def __init__(self, output_dir: Path | None = None) -> None:
        self._output_dir = output_dir or (_project_root() / "data" / "reports")

    async def build(self, run_id: str) -> str:
        run = await get_run(run_id)
        if run is None:
            raise ValueError(f"Run not found: {run_id}")

        raw_results = await get_results_for_run(run_id)
        stats = await get_run_stats(run_id)

        results = sorted(raw_results, key=lambda r: _float_score(r, "composite_score"))
        failures = results[:5]

        total_items = int(stats.get("total_items") or 0)
        passed_items = int(stats.get("passed_items") or 0)
        stats_render = {
            **stats,
            "failed_items": max(0, total_items - passed_items),
            "pass_rate_pct": (100.0 * passed_items / total_items) if total_items else 0.0,
            "distribution": _score_distribution(raw_results),
        }

        tmpl_dir = Path(__file__).resolve().parent / "templates"
        env = Environment(
            loader=FileSystemLoader(str(tmpl_dir)),
            autoescape=select_autoescape(["html", "xml"]),
        )
        template = env.get_template("report.html")
        html = template.render(
            run=run,
            stats=stats_render,
            results=results,
            failures=failures,
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / f"{run_id}.html"
        out_path.write_text(html, encoding="utf-8")
        return str(out_path.resolve())
