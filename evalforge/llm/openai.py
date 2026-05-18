"""OpenAI Chat Completions API adapter for EvalForge."""

from __future__ import annotations

import logging
import time

from openai import AsyncOpenAI

from evalforge.config import settings
from evalforge.llm.errors import LLMError

logger = logging.getLogger(__name__)


class OpenAIProvider:
    @staticmethod
    async def complete(prompt: str, model: str, max_tokens: int) -> tuple[str, int]:
        if not settings.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set", "openai", model)
        try:
            t0 = time.monotonic()
            async with AsyncOpenAI(api_key=settings.OPENAI_API_KEY) as client:
                response = await client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
            latency_ms = int((time.monotonic() - t0) * 1000)
            choice = response.choices[0]
            text = choice.message.content or ""
            return text, latency_ms
        except Exception as exc:
            logger.exception("OpenAI completion failed")
            raise LLMError(str(exc), "openai", model) from exc
