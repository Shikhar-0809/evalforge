"""Pydantic models describing task definitions and inputs."""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator

StructuredType = Literal["exact_match", "regex", "json_fields", "length_check"]


class SemanticConfig(BaseModel):
    weight: float
    model: str = "all-MiniLM-L6-v2"


class KeywordConfig(BaseModel):
    weight: float
    required_keywords: list[str]
    min_coverage: float


class StructuredConfig(BaseModel):
    weight: float
    type: StructuredType
    pattern: Optional[str] = None
    required_fields: Optional[list[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class ScoringConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    semantic: SemanticConfig
    keyword: KeywordConfig
    structured: StructuredConfig

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> ScoringConfig:
        total = self.semantic.weight + self.keyword.weight + self.structured.weight
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "scoring weights must sum to 1.0 "
                "(semantic.weight + keyword.weight + structured.weight)"
            )
        return self


class DatasetConfig(BaseModel):
    path: str
    input_field: str
    reference_field: Optional[str] = None


class TaskConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    description: str
    dataset: DatasetConfig
    prompt_template: str
    scoring: ScoringConfig
    pass_threshold: float
    max_concurrent: int = 5
    timeout_seconds: int = 30
    max_tokens: int = 1024
