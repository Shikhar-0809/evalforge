"""Persistence layer for evaluation run metadata and status."""

from __future__ import annotations

from datetime import datetime

import aiosqlite

from evalforge.database import get_db, write_lock
from evalforge.runner.models import RunConfig


def _row_to_dict(row: aiosqlite.Row) -> dict:
    return {k: row[k] for k in row.keys()}


async def create_run(run: RunConfig) -> None:
    async with write_lock:
        async with get_db() as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO eval_runs (
                    id, task_name, provider, model_name, model_version,
                    config_hash, started_at, completed_at, status,
                    total_items, passed_items, avg_composite_score, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, NULL, NULL)
                """,
                (
                    run.run_id,
                    run.task_name,
                    run.provider,
                    run.model_name,
                    run.model_version,
                    run.config_hash,
                    run.started_at,
                    run.status,
                ),
            )
            await db.commit()


async def update_run_status(run_id: str, status: str, stats: dict) -> None:
    completed_at = datetime.utcnow().isoformat()
    async with write_lock:
        async with get_db() as db:
            await db.execute(
                """
                UPDATE eval_runs
                SET status = ?,
                    completed_at = ?,
                    total_items = ?,
                    passed_items = ?,
                    avg_composite_score = ?
                WHERE id = ?
                """,
                (
                    status,
                    completed_at,
                    stats.get("total_items"),
                    stats.get("passed_items"),
                    stats.get("avg_composite_score"),
                    run_id,
                ),
            )
            await db.commit()


async def get_run(run_id: str) -> dict | None:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM eval_runs WHERE id = ?",
            (run_id,),
        )
        row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(row)


async def list_runs(limit: int = 50) -> list[dict]:
    async with get_db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM eval_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
    return [_row_to_dict(r) for r in rows]
