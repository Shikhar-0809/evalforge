"""ASGI application entry and lifespan wiring for EvalForge."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from evalforge.api.runs import router as runs_router
from evalforge.api.tasks import router as tasks_router
from evalforge.database import init_db

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("EvalForge startup: database initialized, reports directory ready")
    yield


app = FastAPI(title="EvalForge Dashboard", lifespan=lifespan)

app.include_router(runs_router, prefix="/runs")
app.include_router(tasks_router, prefix="/tasks")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/reports",
    StaticFiles(directory=str(REPORTS_DIR)),
    name="reports",
)
