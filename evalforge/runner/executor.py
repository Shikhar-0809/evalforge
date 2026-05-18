"""Orchestrates running evaluations against registered tasks and LLMs."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from evalforge.llm import LLMError, LLMRouter
from evalforge.runner.models import EvalResult, RunConfig
from evalforge.scoring.engine import CompositeScore, ScoreEngine
from evalforge.storage.results import get_run_stats, save_result
from evalforge.storage.runs import create_run, update_run_status
from evalforge.tasks.models import TaskConfig
from evalforge.tasks.registry import TaskRegistry

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _task_yaml_path(task_name: str) -> Path:
    return _project_root() / "tasks" / f"{task_name}.yaml"


def _resolve_dataset_path(task_config: TaskConfig) -> Path:
    raw = Path(task_config.dataset.path)
    if raw.is_absolute():
        return raw
    return _project_root() / raw


def _config_hash_from_yaml(task_name: str) -> str:
    body = _task_yaml_path(task_name).read_text(encoding="utf-8")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class EvalExecutor:
    def __init__(
        self,
        *,
        registry: Optional[TaskRegistry] = None,
        router: Optional[LLMRouter] = None,
        score_engine: Optional[ScoreEngine] = None,
    ) -> None:
        self._registry = registry or TaskRegistry()
        self._router = router or LLMRouter()
        self._score_engine = score_engine or ScoreEngine()

    async def run(
        self,
        task_name: str,
        provider: str,
        model: str,
        model_version: Optional[str] = None,
    ) -> str:
        task_config = self._registry.load(task_name)
        config_hash = _config_hash_from_yaml(task_name)
        run_id = str(uuid.uuid4())
        started_at = datetime.utcnow().isoformat()
        run_cfg = RunConfig(
            run_id=run_id,
            task_name=task_name,
            provider=provider,
            model_name=model,
            model_version=model_version,
            config_hash=config_hash,
            started_at=started_at,
        )
        await create_run(run_cfg)

        logger.info(
            "Starting eval run %s task=%s provider=%s model=%s",
            run_id,
            task_name,
            provider,
            model,
        )

        dataset_path = _resolve_dataset_path(task_config)
        items: list[dict[str, Any]] = []
        with dataset_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))

        input_field = task_config.dataset.input_field
        ref_field = task_config.dataset.reference_field
        sem = asyncio.Semaphore(task_config.max_concurrent)

        async def process_item(item_index: int, row: dict[str, Any]) -> None:
            input_text = str(row[input_field])
            if ref_field:
                ref_val = row.get(ref_field)
                reference: Optional[str] = (
                    None if ref_val is None else str(ref_val)
                )
            else:
                reference = None

            prompt = task_config.prompt_template.replace("{input}", input_text)
            error: Optional[str] = None

            async with sem:
                try:
                    response_text, latency_ms = await self._router.complete(
                        prompt,
                        provider,
                        model,
                        task_config.max_tokens,
                    )
                except LLMError as exc:
                    logger.error(
                        "LLM error for run %s item %s: %s",
                        run_id,
                        item_index,
                        exc,
                    )
                    response_text = ""
                    latency_ms = None
                    error = str(exc)
                    comp = CompositeScore(
                        semantic_score=0.0,
                        keyword_score=0.0,
                        structured_score=0.0,
                        composite_score=0.0,
                        passed=False,
                    )
                else:
                    comp = await self._score_engine.score(
                        response_text,
                        reference,
                        task_config,
                    )

            logger.debug(
                "Run %s item %s scores semantic=%.4f keyword=%.4f structured=%.4f "
                "composite=%.4f passed=%s",
                run_id,
                item_index,
                comp.semantic_score,
                comp.keyword_score,
                comp.structured_score,
                comp.composite_score,
                comp.passed,
            )

            result = EvalResult(
                id=str(uuid.uuid4()),
                run_id=run_id,
                item_index=item_index,
                input_text=input_text,
                reference_answer=reference,
                model_response=response_text,
                semantic_score=comp.semantic_score,
                keyword_score=comp.keyword_score,
                structured_score=comp.structured_score,
                composite_score=comp.composite_score,
                passed=1 if comp.passed else 0,
                latency_ms=latency_ms,
                error=error,
                created_at=datetime.utcnow().isoformat(),
            )
            await save_result(result)

        await asyncio.gather(
            *(process_item(i, row) for i, row in enumerate(items)),
        )

        stats = await get_run_stats(run_id)
        await update_run_status(run_id, "completed", stats)
        return run_id
