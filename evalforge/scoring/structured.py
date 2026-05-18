"""Structured, schema-aware scoring for JSON and typed responses."""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from evalforge.tasks.models import StructuredConfig

logger = logging.getLogger(__name__)


class StructuredScorer:
    @staticmethod
    def score(
        response: str,
        config: StructuredConfig,
        reference: Optional[str],
    ) -> float:
        t = config.type
        if t == "exact_match":
            if reference is None:
                return 0.0
            return 1.0 if response.strip() == reference.strip() else 0.0
        if t == "regex":
            if not config.pattern:
                return 0.0
            return 1.0 if re.search(config.pattern, response) else 0.0
        if t == "json_fields":
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                return 0.0
            fields = config.required_fields or []
            if not fields:
                return 1.0
            matched = sum(1 for f in fields if f in data and data[f] is not None)
            return matched / len(fields)
        if t == "length_check":
            if config.min_length is None or config.max_length is None:
                return 0.0
            return (
                1.0
                if config.min_length <= len(response) <= config.max_length
                else 0.0
            )
        logger.warning("Unknown structured scorer type: %s", t)
        return 0.0
