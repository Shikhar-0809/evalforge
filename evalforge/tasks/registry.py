"""Registry of eval tasks loaded from the local tasks directory."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from evalforge.tasks.models import TaskConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class TaskRegistry:
    @staticmethod
    def load(task_name: str) -> TaskConfig:
        root = _project_root()
        rel_display = f"tasks/{task_name}.yaml"
        path = root / "tasks" / f"{task_name}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"Task YAML not found: {rel_display}")

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        try:
            return TaskConfig.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = first.get("loc", ())
            field = ".".join(str(part) for part in loc) if loc else "model"
            raise ValueError(field) from exc

    @staticmethod
    def list_tasks() -> list[str]:
        tasks_dir = _project_root() / "tasks"
        if not tasks_dir.is_dir():
            return []
        names = sorted(path.stem for path in tasks_dir.glob("*.yaml"))
        return names
