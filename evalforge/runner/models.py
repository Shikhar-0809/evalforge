"""Data models for evaluation runs, steps, and aggregated status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunConfig:
    run_id: str
    task_name: str
    provider: str
    model_name: str
    model_version: Optional[str]
    config_hash: str
    started_at: str
    status: str = "running"


@dataclass
class EvalResult:
    id: str
    run_id: str
    item_index: int
    input_text: str
    reference_answer: Optional[str]
    model_response: str
    semantic_score: float
    keyword_score: float
    structured_score: float
    composite_score: float
    passed: int
    latency_ms: Optional[int]
    error: Optional[str]
    created_at: str
