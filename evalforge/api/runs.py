"""REST endpoints for creating and listing evaluation runs."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict

from evalforge.runner.executor import EvalExecutor
from evalforge.storage.results import get_results_for_run
from evalforge.storage.runs import get_run, list_runs as db_list_runs

logger = logging.getLogger(__name__)

router = APIRouter()


class RunCreateRequest(BaseModel):
    task_name: str
    provider: str
    model_name: str


class RunIdResponse(BaseModel):
    run_id: str


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    task_name: str
    provider: str
    model_name: str
    model_version: str | None = None
    config_hash: str
    started_at: str
    completed_at: str | None = None
    status: str
    total_items: int | None = None
    passed_items: int | None = None
    avg_composite_score: float | None = None


class EvalResultRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    run_id: str
    item_index: int
    input_text: str
    reference_answer: str | None = None
    model_response: str
    semantic_score: float | None = None
    keyword_score: float | None = None
    structured_score: float | None = None
    composite_score: float
    passed: int
    latency_ms: int | None = None
    error: str | None = None
    created_at: str


class RunDetailResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    task_name: str
    provider: str
    model_name: str
    model_version: str | None = None
    config_hash: str
    started_at: str
    completed_at: str | None = None
    status: str
    total_items: int | None = None
    passed_items: int | None = None
    avg_composite_score: float | None = None
    results: list[EvalResultRow]


async def _execute_run(
    task_name: str,
    provider: str,
    model_name: str,
    run_id: str,
) -> None:
    try:
        await EvalExecutor().run(
            task_name,
            provider,
            model_name,
            None,
            run_id=run_id,
        )
    except Exception:
        logger.exception("Background eval run %s failed", run_id)


@router.get("", response_model=list[RunSummary])
async def list_runs_endpoint() -> list[RunSummary]:
    rows = await db_list_runs(50)
    return [RunSummary.model_validate(r) for r in rows]


@router.post("", response_model=RunIdResponse)
async def start_run(
    body: RunCreateRequest,
    background_tasks: BackgroundTasks,
) -> RunIdResponse:
    run_id = str(uuid.uuid4())
    background_tasks.add_task(
        _execute_run,
        body.task_name,
        body.provider,
        body.model_name,
        run_id,
    )
    logger.info("Scheduled eval run %s task=%s", run_id, body.task_name)
    return RunIdResponse(run_id=run_id)


@router.get("/{run_id}/export", response_model=list[EvalResultRow])
async def export_run_results(run_id: str) -> list[EvalResultRow]:
    if await get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    rows = await get_results_for_run(run_id)
    return [EvalResultRow.model_validate(r) for r in rows]


@router.get("/{run_id}", response_model=RunDetailResponse)
async def get_run_detail(run_id: str) -> RunDetailResponse:
    row = await get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    results_raw = await get_results_for_run(run_id)
    results = [EvalResultRow.model_validate(r) for r in results_raw]
    return RunDetailResponse.model_validate({**row, "results": results})
