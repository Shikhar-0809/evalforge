"""Persistence layer for per-item scoring results and raw outputs."""

from __future__ import annotations

import aiosqlite

from evalforge.database import get_db, write_lock
from evalforge.runner.models import EvalResult


def _row_to_dict(row: aiosqlite.Row) -> dict:
    return {k: row[k] for k in row.keys()}


async def save_result(result: EvalResult) -> None:
    async with write_lock:
        async with get_db() as db:
            await db.execute(
                """
                INSERT INTO eval_results (
                    id, run_id, item_index, input_text, reference_answer,
                    model_response, semantic_score, keyword_score, structured_score,
                    composite_score, passed, latency_ms, error, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    result.run_id,
                    result.item_index,
                    result.input_text,
                    result.reference_answer,
                    result.model_response,
                    result.semantic_score,
                    result.keyword_score,
                    result.structured_score,
                    result.composite_score,
                    result.passed,
                    result.latency_ms,
                    result.error,
                    result.created_at,
                ),
            )
            await db.commit()


async def get_results_for_run(run_id: str) -> list[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM eval_results
            WHERE run_id = ?
            ORDER BY item_index ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]


async def get_run_stats(run_id: str) -> dict:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT
                COUNT(*) AS total_items,
                COALESCE(SUM(CASE WHEN passed = 1 THEN 1 ELSE 0 END), 0) AS passed_items,
                COALESCE(AVG(composite_score), 0.0) AS avg_composite_score,
                COALESCE(AVG(semantic_score), 0.0) AS avg_semantic_score,
                COALESCE(AVG(keyword_score), 0.0) AS avg_keyword_score,
                COALESCE(AVG(structured_score), 0.0) AS avg_structured_score
            FROM eval_results
            WHERE run_id = ?
            """,
            (run_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return {
            "total_items": 0,
            "passed_items": 0,
            "avg_composite_score": 0.0,
            "avg_semantic_score": 0.0,
            "avg_keyword_score": 0.0,
            "avg_structured_score": 0.0,
        }
    return _row_to_dict(row)
