"""Anthropic Messages API adapter for EvalForge."""

from __future__ import annotations

import logging
import time

from anthropic import AsyncAnthropic

from evalforge.config import settings
from evalforge.llm.errors import LLMError

logger = logging.getLogger(__name__)


class AnthropicProvider:
    @staticmethod
    def _extract_text(content: object) -> str:
        parts: list[str] = []
        for block in content:
            btype = getattr(block, "type", None)
            if btype == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)

    @staticmethod
    async def complete(prompt: str, model: str, max_tokens: int) -> tuple[str, int]:
        if not settings.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY is not set", "anthropic", model)
        try:
            t0 = time.monotonic()
            async with AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY) as client:
                response = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
            latency_ms = int((time.monotonic() - t0) * 1000)
            text = AnthropicProvider._extract_text(response.content)
            return text, latency_ms
        except Exception as exc:
            logger.exception("Anthropic completion failed")
            raise LLMError(str(exc), "anthropic", model) from exc
