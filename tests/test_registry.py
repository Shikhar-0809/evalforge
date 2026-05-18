"""Tests for task registration, discovery, and task definition parsing."""

from __future__ import annotations

import pytest
import yaml

from evalforge.tasks.registry import TaskRegistry


def test_load_valid_task() -> None:
    cfg = TaskRegistry().load("code_explanation")
    assert cfg.name == "code_explanation"
    assert cfg.version == "1.0"
    assert cfg.pass_threshold == pytest.approx(0.65)


def test_load_missing_task() -> None:
    with pytest.raises(FileNotFoundError):
        TaskRegistry().load("nonexistent_task")


def test_weights_not_summing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    bad = {
        "name": "bad",
        "version": "1.0",
        "description": "d",
        "dataset": {"path": "p.jsonl", "input_field": "x"},
        "prompt_template": "ok {input}",
        "scoring": {
            "semantic": {"weight": 0.5},
            "keyword": {
                "weight": 0.3,
                "required_keywords": [],
                "min_coverage": 0.0,
            },
            "structured": {
                "weight": 0.5,
                "type": "length_check",
                "min_length": 1,
                "max_length": 10,
            },
        },
        "pass_threshold": 0.5,
    }
    (tasks_dir / "bad_weights.yaml").write_text(
        yaml.safe_dump(bad, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evalforge.tasks.registry._project_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError):
        TaskRegistry().load("bad_weights")


def test_list_tasks_returns_four() -> None:
    names = TaskRegistry().list_tasks()
    assert len(names) == 4
    assert set(names) == {
        "code_explanation",
        "factual_qa",
        "json_extraction",
        "summarization",
    }
