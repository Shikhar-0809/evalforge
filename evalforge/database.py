"""SQLite database connection and session helpers for EvalForge."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

from evalforge.config import settings

write_lock = asyncio.Lock()

_SCHEMA_PATH = Path(__file__).resolve().parent / "storage" / "schema.sql"


async def init_db() -> None:
    """Create the database file (and parent directories) and apply ``schema.sql``."""
    db_path = Path(settings.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.executescript(schema_sql)
        await db.commit()


@asynccontextmanager
async def get_db():
    """Yield an async SQLite connection for the configured ``DATABASE_PATH``."""
    db_path = Path(settings.DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        yield db
