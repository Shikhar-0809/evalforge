"""Google Gemini API adapter for EvalForge."""

from __future__ import annotations

import asyncio
import logging
import time

from google import genai
from google.genai import types

from evalforge.config import settings
from evalforge.llm.errors import LLMError

logger = logging.getLogger(__name__)


class GeminiProvider:
    async def complete(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
    ) -> tuple[str, int]:
        if not settings.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set", "gemini", model)

        def _sync_call() -> str:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            response = client.models.generate_content(
    		model=model,
    		contents=prompt,
    		config={"temperature": 0.0, "max_output_tokens": max_tokens},
	    )
            return response.text

        try:
            t0 = time.monotonic()
            text = await asyncio.to_thread(_sync_call)
            latency_ms = int((time.monotonic() - t0) * 1000)
            return text, latency_ms
        except Exception as exc:
            logger.exception("Gemini completion failed")
            raise LLMError(str(exc), provider="gemini", model=model) from exc
