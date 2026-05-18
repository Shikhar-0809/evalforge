"""REST endpoints for discovering and inspecting registered tasks."""

from __future__ import annotations

from fastapi import APIRouter

from evalforge.tasks.registry import TaskRegistry

router = APIRouter()


@router.get("")
async def list_tasks() -> list[str]:
    return TaskRegistry().list_tasks()
