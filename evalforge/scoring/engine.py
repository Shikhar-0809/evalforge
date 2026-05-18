"""Applies configured scorers and aggregates evaluation outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from evalforge.scoring.keyword import KeywordScorer
from evalforge.scoring.semantic import SemanticScorer
from evalforge.scoring.structured import StructuredScorer
from evalforge.tasks.models import StructuredConfig, TaskConfig

logger = logging.getLogger(__name__)


@dataclass
class CompositeScore:
    semantic_score: float
    keyword_score: float
    structured_score: float
    composite_score: float
    passed: bool


def _semantic_computable(reference: Optional[str]) -> bool:
    return reference is not None and reference.strip() != ""


def _structured_computable(cfg: StructuredConfig, reference: Optional[str]) -> bool:
    if cfg.type == "exact_match":
        return reference is not None
    if cfg.type == "regex":
        return bool(cfg.pattern)
    if cfg.type == "json_fields":
        return True
    if cfg.type == "length_check":
        return cfg.min_length is not None and cfg.max_length is not None
    logger.warning("Unknown structured scorer type for composability: %s", cfg.type)
    return False


class ScoreEngine:
    def __init__(self) -> None:
        self._semantic_by_model: dict[str, SemanticScorer] = {
            "all-MiniLM-L6-v2": SemanticScorer("all-MiniLM-L6-v2"),
        }
        self.semantic_scorer = self._semantic_by_model["all-MiniLM-L6-v2"]
        self.keyword_scorer = KeywordScorer()
        self.structured_scorer = StructuredScorer()

    def _semantic_for(self, model_name: str) -> SemanticScorer:
        if model_name not in self._semantic_by_model:
            self._semantic_by_model[model_name] = SemanticScorer(model_name)
        return self._semantic_by_model[model_name]

    async def score(
        self,
        response: str,
        reference: Optional[str],
        task_config: TaskConfig,
    ) -> CompositeScore:
        cfg = task_config.scoring
        w_sem, w_key, w_str = cfg.semantic.weight, cfg.keyword.weight, cfg.structured.weight

        c_sem = _semantic_computable(reference)
        c_key = True
        c_str = _structured_computable(cfg.structured, reference)

        if c_sem:
            semantic_raw = await self._semantic_for(cfg.semantic.model).score(
                response,
                reference,
            )
        else:
            semantic_raw = 0.0

        keyword_raw = self.keyword_scorer.score(
            response,
            cfg.keyword.required_keywords,
        )

        if c_str:
            structured_raw = self.structured_scorer.score(
                response,
                cfg.structured,
                reference,
            )
        else:
            structured_raw = 0.0

        active: list[tuple[str, float, float, bool]] = [
            ("semantic", w_sem, semantic_raw, c_sem),
            ("keyword", w_key, keyword_raw, c_key),
            ("structured", w_str, structured_raw, c_str),
        ]

        active_idxs = [i for i, (_, w, _, c) in enumerate(active) if w > 0.0 and c]
        inactive_weight = sum(w for _, w, _, c in active if w > 0.0 and not c)

        if not active_idxs:
            composite = 0.0
        else:
            base_sum = sum(active[i][1] for i in active_idxs)
            if base_sum <= 0.0:
                composite = 0.0
            else:
                total = 0.0
                for i in active_idxs:
                    _, w_i, s_i, _ = active[i]
                    adj_w = w_i + inactive_weight * (w_i / base_sum)
                    total += adj_w * s_i
                composite = total

        composite_rounded = round(composite, 4)
        passed = composite_rounded >= task_config.pass_threshold

        return CompositeScore(
            semantic_score=semantic_raw,
            keyword_score=keyword_raw,
            structured_score=structured_raw,
            composite_score=composite_rounded,
            passed=passed,
        )
