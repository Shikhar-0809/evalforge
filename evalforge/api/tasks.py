"""REST endpoints for discovering and inspecting registered tasks."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import RootModel

from evalforge.tasks.registry import TaskRegistry

logger = logging.getLogger(__name__)

router = APIRouter()


class TaskNameList(RootModel[list[str]]):
    """JSON array of task names (stems of tasks/*.yaml)."""


@router.get("", response_model=TaskNameList)
async def list_tasks() -> TaskNameList:
    names = TaskRegistry().list_tasks()
    logger.debug("list_tasks: %d tasks", len(names))
    return TaskNameList(names)
