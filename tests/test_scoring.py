"""Tests for scoring engines and individual scorer implementations."""

from __future__ import annotations

import pytest

from evalforge.scoring.engine import ScoreEngine
from evalforge.scoring.keyword import KeywordScorer
from evalforge.scoring.semantic import SemanticScorer
from evalforge.scoring.structured import StructuredScorer
from evalforge.tasks.models import (
    DatasetConfig,
    KeywordConfig,
    ScoringConfig,
    SemanticConfig,
    StructuredConfig,
    TaskConfig,
)

# —— SemanticScorer (async) ——


@pytest.mark.asyncio
async def test_semantic_identical_strings_high_score() -> None:
    scorer = SemanticScorer()
    s = await scorer.score("hello world", "hello world")
    assert s >= 0.99


@pytest.mark.asyncio
async def test_semantic_unrelated_strings_low_score() -> None:
    scorer = SemanticScorer()
    s = await scorer.score(
        "the cat sat on the mat",
        "quantum physics equations",
    )
    assert s < 0.3


@pytest.mark.asyncio
async def test_semantic_none_reference_is_zero() -> None:
    scorer = SemanticScorer()
    assert await scorer.score("any response text", None) == 0.0


# —— KeywordScorer ——


def test_keyword_all_keywords_present() -> None:
    s = KeywordScorer.score(
        "The function returns a parameter in a loop",
        ["function", "returns"],
    )
    assert s == 1.0


def test_keyword_no_keywords_present() -> None:
    s = KeywordScorer.score("hello world", ["missing", "also_missing"])
    assert s == 0.0


def test_keyword_half_keywords_present() -> None:
    s = KeywordScorer.score(
        "alpha and gamma here",
        ["alpha", "beta", "gamma", "delta"],
    )
    assert s == pytest.approx(0.5)


def test_keyword_case_insensitive_matching() -> None:
    s = KeywordScorer.score("HELLO World", ["hello", "world"])
    assert s == 1.0


# —— StructuredScorer: exact_match ——


def test_structured_exact_match_identical_strings() -> None:
    cfg = StructuredConfig(weight=1.0, type="exact_match")
    assert StructuredScorer.score("same text", cfg, "same text") == 1.0


def test_structured_exact_match_mismatch() -> None:
    cfg = StructuredConfig(weight=1.0, type="exact_match")
    assert StructuredScorer.score("a", cfg, "b") == 0.0


# —— StructuredScorer: regex ——


def test_structured_regex_pattern_found() -> None:
    cfg = StructuredConfig(weight=1.0, type="regex", pattern=r"\d{3}")
    assert StructuredScorer.score("code 999 ok", cfg, None) == 1.0


def test_structured_regex_pattern_not_found() -> None:
    cfg = StructuredConfig(weight=1.0, type="regex", pattern=r"ZZZ")
    assert StructuredScorer.score("no match here", cfg, None) == 0.0


# —— StructuredScorer: json_fields ——


def test_structured_json_fields_all_required_present() -> None:
    cfg = StructuredConfig(
        weight=1.0,
        type="json_fields",
        required_fields=["a", "b"],
    )
    assert StructuredScorer.score('{"a": 1, "b": 2}', cfg, None) == 1.0


def test_structured_json_fields_half_present() -> None:
    cfg = StructuredConfig(
        weight=1.0,
        type="json_fields",
        required_fields=["a", "b"],
    )
    assert StructuredScorer.score('{"a": 1}', cfg, None) == 0.5


def test_structured_json_fields_malformed_json() -> None:
    cfg = StructuredConfig(
        weight=1.0,
        type="json_fields",
        required_fields=["a"],
    )
    assert StructuredScorer.score("not json", cfg, None) == 0.0


# —— StructuredScorer: length_check ——


def test_structured_length_check_in_range() -> None:
    cfg = StructuredConfig(
        weight=1.0,
        type="length_check",
        min_length=2,
        max_length=10,
    )
    assert StructuredScorer.score("hello", cfg, None) == 1.0


def test_structured_length_check_too_short() -> None:
    cfg = StructuredConfig(
        weight=1.0,
        type="length_check",
        min_length=10,
        max_length=20,
    )
    assert StructuredScorer.score("short", cfg, None) == 0.0


def test_structured_length_check_too_long() -> None:
    cfg = StructuredConfig(
        weight=1.0,
        type="length_check",
        min_length=1,
        max_length=3,
    )
    assert StructuredScorer.score("abcd", cfg, None) == 0.0


def _task(scoring: ScoringConfig, *, pass_threshold: float = 0.5) -> TaskConfig:
    return TaskConfig(
        name="test_task",
        version="1.0",
        description="test",
        dataset=DatasetConfig(path="datasets/dummy.jsonl", input_field="x"),
        prompt_template="P {input}",
        scoring=scoring,
        pass_threshold=pass_threshold,
        max_concurrent=5,
        timeout_seconds=30,
        max_tokens=256,
    )


@pytest.mark.asyncio
async def test_engine_dr02_redistributes_when_semantic_not_computable() -> None:
    """Semantic has positive weight but no reference → weight moves to keyword + structured."""
    w_sem, w_key, w_str = 0.3, 0.5, 0.2
    inactive = w_sem
    base = w_key + w_str
    adj_key = w_key + inactive * (w_key / base)
    adj_str = w_str + inactive * (w_str / base)
    s_key = KeywordScorer.score("a only", ["a", "b"])
    assert s_key == 0.5
    s_str = StructuredScorer.score(
        "a",
        StructuredConfig(
            weight=w_str,
            type="length_check",
            min_length=1,
            max_length=100,
        ),
        None,
    )
    assert s_str == 1.0
    expected = round(adj_key * s_key + adj_str * s_str, 4)

    engine = ScoreEngine()
    task = _task(
        ScoringConfig(
            semantic=SemanticConfig(weight=w_sem),
            keyword=KeywordConfig(
                weight=w_key,
                required_keywords=["a", "b"],
                min_coverage=0.0,
            ),
            structured=StructuredConfig(
                weight=w_str,
                type="length_check",
                min_length=1,
                max_length=100,
            ),
        ),
    )
    result = await engine.score("a only", None, task)
    assert result.composite_score == pytest.approx(expected)
    assert result.semantic_score == 0.0


@pytest.mark.asyncio
async def test_engine_dr02_redistributes_when_scorer_weight_zero() -> None:
    """Keyword weight 0: only structured is active; semantic weight pools to structured."""
    w_sem, w_key, w_str = 0.4, 0.0, 0.6
    inactive = w_sem
    base = w_str
    adj_str = w_str + inactive * (w_str / base)
    assert adj_str == pytest.approx(1.0)
    response = "x" * 50
    s_str = StructuredScorer.score(
        response,
        StructuredConfig(
            weight=w_str,
            type="length_check",
            min_length=1,
            max_length=100,
        ),
        None,
    )
    assert s_str == 1.0
    expected = round(adj_str * s_str, 4)

    engine = ScoreEngine()
    task = _task(
        ScoringConfig(
            semantic=SemanticConfig(weight=w_sem),
            keyword=KeywordConfig(
                weight=w_key,
                required_keywords=["nope"],
                min_coverage=0.0,
            ),
            structured=StructuredConfig(
                weight=w_str,
                type="length_check",
                min_length=1,
                max_length=100,
            ),
        ),
    )
    result = await engine.score(response, None, task)
    assert result.composite_score == pytest.approx(expected)
    assert result.keyword_score == 0.0


@pytest.mark.asyncio
async def test_engine_dr03_pass_threshold_passes() -> None:
    engine = ScoreEngine()
    threshold = 0.1
    task = _task(
        ScoringConfig(
            semantic=SemanticConfig(weight=0.0),
            keyword=KeywordConfig(
                weight=0.8,
                required_keywords=["hello"],
                min_coverage=0.0,
            ),
            structured=StructuredConfig(
                weight=0.2,
                type="length_check",
                min_length=1,
                max_length=100,
            ),
        ),
        pass_threshold=threshold,
    )
    result = await engine.score("hello there", None, task)
    assert result.composite_score >= threshold
    assert result.passed is True


@pytest.mark.asyncio
async def test_engine_dr03_pass_threshold_fails() -> None:
    engine = ScoreEngine()
    task = _task(
        ScoringConfig(
            semantic=SemanticConfig(weight=0.0),
            keyword=KeywordConfig(
                weight=0.8,
                required_keywords=["hello"],
                min_coverage=0.0,
            ),
            structured=StructuredConfig(
                weight=0.2,
                type="length_check",
                min_length=1,
                max_length=100,
            ),
        ),
        pass_threshold=0.99,
    )
    result = await engine.score("goodbye short", None, task)
    assert result.passed is False
    assert result.composite_score < task.pass_threshold
