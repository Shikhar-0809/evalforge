"""Tests for task registration, discovery, and task definition parsing."""

from __future__ import annotations

import pytest

from evalforge.tasks.registry import TaskRegistry


def test_load_valid_yaml_returns_task_config() -> None:
    cfg = TaskRegistry().load("code_explanation")
    assert isinstance(cfg.name, str)
    assert cfg.name == "code_explanation"
    assert cfg.version == "1.0"
    assert cfg.pass_threshold == pytest.approx(0.65)
    assert cfg.dataset.path == "datasets/code_explanation.jsonl"
    assert cfg.dataset.input_field == "code"
    assert cfg.dataset.reference_field == "explanation"
    total_w = (
        cfg.scoring.semantic.weight
        + cfg.scoring.keyword.weight
        + cfg.scoring.structured.weight
    )
    assert abs(total_w - 1.0) < 1e-9


def test_missing_yaml_raises_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        TaskRegistry().load("nonexistent_task")


def test_weights_not_summing_to_one_raises_value_error_dr01(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tasks_dir / "bad_weights.yaml").write_text(
        """
name: bad_weights
version: "1.0"
description: test
dataset:
  path: p.jsonl
  input_field: x
prompt_template: "ok {input}"
scoring:
  semantic:
    weight: 0.5
  keyword:
    weight: 0.3
    required_keywords: []
    min_coverage: 0.0
  structured:
    weight: 0.5
    type: length_check
    min_length: 1
    max_length: 10
pass_threshold: 0.5
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "evalforge.tasks.registry._project_root",
        lambda: tmp_path,
    )
    with pytest.raises(ValueError):
        TaskRegistry().load("bad_weights")


def test_list_tasks_returns_four_example_names() -> None:
    names = TaskRegistry().list_tasks()
    assert names == sorted(names)
    assert len(names) == 4
    assert set(names) == {
        "code_explanation",
        "factual_qa",
        "json_extraction",
        "summarization",
    }
