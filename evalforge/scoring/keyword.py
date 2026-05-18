"""Keyword and overlap-based scoring for model outputs."""

from __future__ import annotations


class KeywordScorer:
    @staticmethod
    def score(response: str, required_keywords: list[str]) -> float:
        if not required_keywords:
            return 1.0
        haystack = response.lower()
        matched = sum(1 for kw in required_keywords if kw.lower() in haystack)
        return matched / len(required_keywords)
