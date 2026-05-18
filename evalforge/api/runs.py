"""REST endpoints for creating and listing evaluation runs."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from evalforge.runner.executor import EvalExecutor
from evalforge.storage.results import get_results_for_run
from evalforge.storage.runs import get_run, list_runs as db_list_runs

router = APIRouter()


class RunRequest(BaseModel):
    task_name: str
    provider: str
    model_name: str
    model_version: Optional[str] = None


async def _execute_run(
    task_name: str,
    provider: str,
    model_name: str,
    model_version: Optional[str],
    run_id: str,
) -> None:
    await EvalExecutor().run(
        task_name,
        provider,
        model_name,
        model_version,
        run_id=run_id,
    )


@router.get("")
async def list_runs_endpoint() -> list[dict]:
    return await db_list_runs(50)


@router.post("")
async def start_run(body: RunRequest, background_tasks: BackgroundTasks) -> dict:
    run_id = str(uuid.uuid4())
    background_tasks.add_task(
        _execute_run,
        body.task_name,
        body.provider,
        body.model_name,
        body.model_version,
        run_id,
    )
    return {"run_id": run_id}


@router.get("/{run_id}/export")
async def export_run_results(run_id: str) -> list[dict]:
    if await get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return await get_results_for_run(run_id)


@router.get("/{run_id}")
async def get_run_detail(run_id: str) -> dict:
    row = await get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    results = await get_results_for_run(run_id)
    return {**row, "results": results}
